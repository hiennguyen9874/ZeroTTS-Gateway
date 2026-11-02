"""Benchmark one zerotts_server weights directory at engine level (issue #13).

Loads `ZeroTTSEngine` as the server does (same threads, sampling, segmentation), then runs
short/medium/long engine texts offline (RTF) and streaming (TTFA), timing every
`session.run()` per graph. Run once per weights directory, in its own process, so the
peak RSS belongs to that one worker:

    uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS --out out/fp32.json
    uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS-int8 --out out/int8.json

The first run of each case is dropped from the medians. The output JSON holds every run,
the load average around each case, load/warmup times and RSS. No audio is written.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import onnxruntime

from zerotts_server.app import split_segments
from zerotts_server.config import Sampling
from zerotts_server.engine import SAMPLE_RATE, ZeroTTSEngine

# Engine text (already what the gateway sends): no digits or abbreviations left.
CASES = {
    # The 2026-10-28 speed report's sentence, as the gateway reads it.
    "short": (
        "hẹn gặp lúc tám giờ ba mươi phút ngày hai mươi lăm tháng mười hai, phí năm mươi nghìn."
    ),
    "medium": (
        "Xin chào quý khách, cảm ơn quý khách đã gọi đến tổng đài chăm sóc khách hàng. "
        "Đơn hàng của quý khách đã được xác nhận và sẽ được giao trong vòng hai ngày làm việc."
    ),
    "long": (
        "Theo thông báo mới nhất, hệ thống sẽ tạm ngừng hoạt động để bảo trì từ mười giờ tối "
        "thứ bảy đến sáu giờ sáng chủ nhật. Trong thời gian này, quý khách vẫn có thể tra cứu "
        "thông tin tài khoản nhưng không thực hiện được giao dịch chuyển tiền. Chúng tôi "
        "khuyến nghị quý khách hoàn tất các giao dịch quan trọng trước thời điểm bảo trì. "
        "Mọi thắc mắc xin vui lòng liên hệ tổng đài hoặc đến chi nhánh gần nhất để được hỗ "
        "trợ. Sau khi bảo trì, ứng dụng sẽ có giao diện mới, tốc độ xử lý nhanh hơn và thêm "
        "tính năng thanh toán hóa đơn tự động. Xin chân thành cảm ơn quý khách đã tin tưởng "
        "và đồng hành cùng chúng tôi trong suốt thời gian qua."
    ),
}

GRAPHS = ("text_encoder", "prefix_step", "local_frame_decode", "codec_full", "codec_step")


class _Timed:
    """Proxy for an onnxruntime session that adds each run()'s wall time to a profile."""

    def __init__(self, session, name: str, profile: dict):
        self._session, self._name, self._profile = session, name, profile

    def run(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return self._session.run(*args, **kwargs)
        finally:
            entry = self._profile.setdefault(self._name, {"calls": 0, "seconds": 0.0})
            entry["calls"] += 1
            entry["seconds"] += time.perf_counter() - t

    def __getattr__(self, name):
        return getattr(self._session, name)


def _rss_mib() -> tuple[float, float]:
    """(current RSS, peak RSS) of this process in MiB."""
    current = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) / 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return round(current, 1), round(peak, 1)


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)


def _run(engine, segments, voice, seed, sampling, stream, profile) -> dict:
    profile.clear()
    samples, ttfa, digest = 0, None, hashlib.sha256()
    t = time.perf_counter()
    for chunk in engine.synthesize(segments, voice, seed, sampling, stream=stream):
        if ttfa is None and chunk.size:
            ttfa = time.perf_counter() - t
        samples += chunk.size
        digest.update(chunk.tobytes())
    total = time.perf_counter() - t
    audio = samples / SAMPLE_RATE
    out = {
        "total_s": round(total, 4),
        "audio_s": round(audio, 4),
        "rtf": round(total / audio, 4),
        # Same seed and weights must give the same audio on every run.
        "audio_sha256": digest.hexdigest(),
        "graphs": {
            k: {"calls": v["calls"], "seconds": round(v["seconds"], 4)} for k, v in profile.items()
        },
        "loadavg_1m": round(os.getloadavg()[0], 2),
    }
    if stream:
        out["ttfa_s"] = round(ttfa, 4)
    return out


def _summary(runs: list[dict]) -> dict:
    kept = runs[1:] or runs
    keys = ["total_s", "audio_s", "rtf"] + (["ttfa_s"] if "ttfa_s" in kept[0] else [])
    out = {k: round(statistics.median(r[k] for r in kept), 4) for k in keys}
    out["graphs_s"] = {
        g: round(statistics.median(r["graphs"].get(g, {}).get("seconds", 0.0) for r in kept), 4)
        for g in GRAPHS
        if any(g in r["graphs"] for r in kept)
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="result JSON (refuses to overwrite)")
    ap.add_argument("--threads", type=int, default=8, help="ONNX intra-op threads (config: 8)")
    ap.add_argument("--runs", type=int, default=4, help="runs per case and mode; first dropped")
    ap.add_argument("--voice", default="baotrang")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"{args.out} exists")

    model_dir = args.model_dir
    manifest = model_dir / "quantization.json"
    sampling = Sampling()
    result = {
        "model_dir": str(model_dir),
        "weights": "int8" if manifest.is_file() else "fp32",
        "threads": args.threads,
        "voice": args.voice,
        "seed": args.seed,
        "sampling": asdict(sampling),
        "runs_per_case": args.runs,
        "env": {
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "cpu_count": os.cpu_count(),
            "loadavg_start": [round(x, 2) for x in os.getloadavg()],
        },
        "sizes_bytes": {
            "model_dir": _dir_bytes(model_dir),
            **{p.name: p.stat().st_size for p in sorted((model_dir / "onnx").glob("*.onnx"))},
            "codec": _dir_bytes(model_dir / "onnx" / "codec"),
        },
    }

    from zerotts_server.model.synthesizer import ZeroTTS

    rss_before = _rss_mib()[0]
    warm = {}
    original_warmup = ZeroTTS.warmup

    def timed_warmup(self):
        t = time.perf_counter()
        original_warmup(self)
        warm["s"] = time.perf_counter() - t

    ZeroTTS.warmup = timed_warmup
    engine = ZeroTTSEngine(model_dir, model_dir / "voices", args.threads)
    t = time.perf_counter()
    engine.load()
    load_total = time.perf_counter() - t
    ZeroTTS.warmup = original_warmup
    result["load"] = {
        "load_and_warmup_s": round(load_total, 3),
        "warmup_s": round(warm["s"], 3),
        "load_s": round(load_total - warm["s"], 3),
        "rss_before_mib": rss_before,
        "rss_after_load_mib": _rss_mib()[0],
    }

    profile: dict = {}
    tts = engine._frames.tts  # onnx-cpu runtime; its codec is the engine's
    tts.text_encoder_sess = _Timed(tts.text_encoder_sess, "text_encoder", profile)
    tts.prefix_step_sess = _Timed(tts.prefix_step_sess, "prefix_step", profile)
    # Inside the engine's _SeededSession, which must stay outermost to receive the seeded RNG.
    seeded = tts.local_frame_decode_sess
    seeded.session = _Timed(seeded.session, "local_frame_decode", profile)
    tts.codec._decode_full_sess = _Timed(tts.codec._decode_full_sess, "codec_full", profile)
    tts.codec._decode_step_sess = _Timed(tts.codec._decode_step_sess, "codec_step", profile)

    result["cases"] = {}
    for case in args.cases:
        segments = split_segments(CASES[case])
        entry = {"chars": len(CASES[case]), "segments": len(segments)}
        for mode, stream in (("offline", False), ("streaming", True)):
            load_before = round(os.getloadavg()[0], 2)
            runs = [
                _run(engine, segments, args.voice, args.seed, sampling, stream, profile)
                for _ in range(args.runs)
            ]
            entry[mode] = {"loadavg_1m_before": load_before, "runs": runs, "median": _summary(runs)}
            m = entry[mode]["median"]
            print(
                f"{result['weights']} {case} {mode}: total={m['total_s']}s audio={m['audio_s']}s "
                f"rtf={m['rtf']}" + (f" ttfa={m['ttfa_s']}s" if stream else ""),
                file=sys.stderr,
            )
        result["cases"][case] = entry

    current, peak = _rss_mib()
    result["rss_end_mib"], result["rss_peak_mib"] = current, peak
    result["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["load"]) + f" peak_rss_mib={peak}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
