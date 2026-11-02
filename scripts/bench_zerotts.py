"""Benchmark one zerotts_server weights directory at engine level (issues #13, #21).

Loads `ZeroTTSEngine` as the server does (same threads, sampling, segmentation), then runs
short/medium/long engine texts offline (RTF) and streaming (TTFA), timing every
`session.run()` per graph (onnx-cpu) or every frame step, segment start and codec call
(torch-cuda). Run once per configuration, in its own process, so the peak RSS and VRAM
belong to it:

    uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS --out out/fp32.json
    uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS-int8 --out out/int8.json
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/bench_zerotts.py --runtime torch-cuda \
        --model-dir models/ZeroTTS-torch --out out/torch-fp16.json

torch-cuda runs as the server does (fp16, CUDA Graphs) unless `--dtype float32` or `--eager`.
`--workers N` loads N workers on the device and also runs every case on all of them at once
(throughput). The first run of each case is dropped from the medians. The output JSON holds
every run, the load average around each case, load/warmup times, RSS and (torch-cuda) VRAM.
No audio is written.
"""

import argparse
import functools
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
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
# torch-cuda: segment starts (launch only; their GPU time lands in the next frame), frame
# steps (each waits for its codes) and codec calls.
TORCH_STEPS = ("begin", "frame", "codec")


class _Timed:
    """Proxy for an onnxruntime session that adds each run()'s wall time to a profile."""

    def __init__(self, session, name: str, profile: dict):
        self._session, self._name, self._profile = session, name, profile

    def run(self, *args, **kwargs):
        return _timed(self._session.run, self._name, self._profile)(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _timed(fn, name: str, profile: dict):
    """fn, adding each call's wall time to profile[name] (and to its per-call list)."""

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t
            entry = profile.setdefault(name, {"calls": 0, "seconds": 0.0, "each": []})
            entry["calls"] += 1
            entry["seconds"] += elapsed
            entry["each"].append(elapsed)

    return wrapper


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
            k: {
                "calls": v["calls"],
                "seconds": round(v["seconds"], 4),
                "median_ms": round(statistics.median(v["each"]) * 1000, 3),
            }
            for k, v in profile.items()
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
        for g in GRAPHS + TORCH_STEPS
        if any(g in r["graphs"] for r in kept)
    }
    if all("frame" in r["graphs"] for r in kept):
        out["frame_ms"] = round(
            statistics.median(r["graphs"]["frame"]["median_ms"] for r in kept), 3
        )
    return out


def _gpu_env(device) -> dict:
    import torch

    query = "name,driver_version,memory.total"
    smi = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "nvidia_smi": smi.splitlines(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _vram_mib(device) -> dict:
    """Device memory in use (all processes, incl. CUDA contexts) and this process's allocator."""
    import torch

    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_used_mib": round((total - free) / 2**20, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
    }


def _concurrent(engines, segments, voice, seed, sampling, stream, runs) -> dict:
    """Every engine synthesizes the same request `runs` times in its own thread, all at once."""
    results: list[list[dict]] = [[] for _ in engines]

    def work(i, engine):
        for _ in range(runs):
            results[i].append(_run(engine, segments, voice, seed, sampling, stream, {}))

    t = time.perf_counter()
    threads = [threading.Thread(target=work, args=(i, e)) for i, e in enumerate(engines)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t
    flat = [r for per_engine in results for r in per_engine]
    audio = sum(r["audio_s"] for r in flat)
    out = {
        "requests": len(flat),
        "wall_s": round(wall, 4),
        "audio_s": round(audio, 4),
        "audio_s_per_s": round(audio / wall, 3),
        "rtf_median": round(statistics.median(r["rtf"] for r in flat), 4),
        "rtf_max": round(max(r["rtf"] for r in flat), 4),
        # Every worker must still produce the single-worker audio.
        "audio_sha256": sorted({r["audio_sha256"] for r in flat}),
    }
    if stream:
        out["ttfa_s_median"] = round(statistics.median(r["ttfa_s"] for r in flat), 4)
        out["ttfa_s_max"] = round(max(r["ttfa_s"] for r in flat), 4)
    return out


def _load_onnx(args, model_dir, profile: dict) -> tuple[list, dict]:
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
    load = {
        "load_and_warmup_s": round(load_total, 3),
        "warmup_s": round(warm["s"], 3),
        "load_s": round(load_total - warm["s"], 3),
        "rss_before_mib": rss_before,
        "rss_after_load_mib": _rss_mib()[0],
    }

    tts = engine._frames.tts  # onnx-cpu runtime; its codec is the engine's
    tts.text_encoder_sess = _Timed(tts.text_encoder_sess, "text_encoder", profile)
    tts.prefix_step_sess = _Timed(tts.prefix_step_sess, "prefix_step", profile)
    # Inside the engine's _SeededSession, which must stay outermost to receive the seeded RNG.
    seeded = tts.local_frame_decode_sess
    seeded.session = _Timed(seeded.session, "local_frame_decode", profile)
    tts.codec._decode_full_sess = _Timed(tts.codec._decode_full_sess, "codec_full", profile)
    tts.codec._decode_step_sess = _Timed(tts.codec._decode_step_sess, "codec_step", profile)
    return [engine], load


def _load_torch(args, model_dir, profile: dict) -> tuple[list, dict]:
    """N workers as the server loads them; each worker's load, warmup and VRAM."""
    import torch

    from zerotts_server import runtime, torch_runtime

    device = torch_runtime.resolve_device(args.device)
    model_cls = torch_runtime.TorchZeroTTS
    # load_runtime builds the model as the server does; the bench may override its dtype and
    # graphs (load_runtime reads both module attributes at call time).
    server_dtype = runtime.TORCH_DTYPE
    runtime.TORCH_DTYPE = args.dtype
    torch_runtime.TorchZeroTTS = functools.partial(model_cls, cuda_graph=not args.eager)
    warm = []
    original_warmup = model_cls.warmup

    def timed_warmup(self):
        t = time.perf_counter()
        original_warmup(self)
        warm.append(time.perf_counter() - t)

    model_cls.warmup = timed_warmup
    torch.cuda.init()
    load = {"vram_before": _vram_mib(device), "workers": []}
    engines = []
    try:
        for _ in range(args.workers):
            engine = ZeroTTSEngine(
                model_dir, model_dir / "voices", args.threads, "torch-cuda", args.device
            )
            t = time.perf_counter()
            engine.load()
            total = time.perf_counter() - t
            engines.append(engine)
            load["workers"].append(
                {
                    "load_and_warmup_s": round(total, 3),
                    "warmup_s": round(warm[-1], 3),
                    "load_s": round(total - warm[-1], 3),
                    "vram_after": _vram_mib(device),
                }
            )
    finally:
        model_cls.warmup = original_warmup
        torch_runtime.TorchZeroTTS = model_cls
        runtime.TORCH_DTYPE = server_dtype
    model = engines[0]._frames
    assert model.dtype == getattr(torch, args.dtype) and bool(model._graphs) != args.eager
    used = [w["vram_after"]["device_used_mib"] for w in load["workers"]]
    load["vram_per_worker_mib"] = [
        round(b - a, 1)
        for a, b in zip([load["vram_before"]["device_used_mib"], *used[:-1]], used, strict=True)
    ]
    load["rss_after_load_mib"] = _rss_mib()[0]

    codec = engines[0]._codec
    model._start = _timed(model._start, "begin", profile)
    model._next_frame = _timed(model._next_frame, "frame", profile)
    codec.decode = _timed(codec.decode, "codec", profile)
    codec.decode_chunk = _timed(codec.decode_chunk, "codec", profile)
    return engines, load


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="result JSON (refuses to overwrite)")
    ap.add_argument("--runtime", default="onnx-cpu", choices=("onnx-cpu", "torch-cuda"))
    ap.add_argument("--threads", type=int, default=8, help="ONNX intra-op threads (config: 8)")
    ap.add_argument("--device", default="cuda:0", help="torch-cuda device")
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--eager", action="store_true", help="torch-cuda without CUDA Graphs")
    ap.add_argument("--workers", type=int, default=1, help="torch-cuda workers on the device")
    ap.add_argument("--runs", type=int, default=4, help="runs per case and mode; first dropped")
    ap.add_argument("--voice", default="baotrang")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"{args.out} exists")
    torch_cuda = args.runtime == "torch-cuda"
    if args.workers > 1 and not torch_cuda:
        ap.error("--workers applies to torch-cuda")

    model_dir = args.model_dir
    manifest = model_dir / "quantization.json"
    sampling = Sampling()
    if torch_cuda:
        label = f"torch-cuda {args.dtype} {'eager' if args.eager else 'graph'}"
    else:
        label = "int8" if manifest.is_file() else "fp32"
    result = {
        "model_dir": str(model_dir),
        "runtime": args.runtime,
        "weights": "int8" if manifest.is_file() else "fp32",
        "label": label,
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
    }
    if torch_cuda:
        result.update(
            device=args.device, dtype=args.dtype, cuda_graph=not args.eager, workers=args.workers
        )
        result["sizes_bytes"] = {
            p.name: p.stat().st_size for p in sorted(model_dir.glob("*.safetensors"))
        }
    else:
        result["threads"] = args.threads
        result["sizes_bytes"] = {
            "model_dir": _dir_bytes(model_dir),
            **{p.name: p.stat().st_size for p in sorted((model_dir / "onnx").glob("*.onnx"))},
            "codec": _dir_bytes(model_dir / "onnx" / "codec"),
        }

    profile: dict = {}
    if torch_cuda:
        engines, result["load"] = _load_torch(args, model_dir, profile)
        result["env"].update(_gpu_env(engines[0]._frames.device))
    else:
        engines, result["load"] = _load_onnx(args, model_dir, profile)
    engine = engines[0]

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
                f"{label} {case} {mode}: total={m['total_s']}s audio={m['audio_s']}s "
                f"rtf={m['rtf']}"
                + (f" ttfa={m['ttfa_s']}s" if stream else "")
                + (f" frame={m['frame_ms']}ms" if "frame_ms" in m else ""),
                file=sys.stderr,
            )
            if len(engines) > 1:
                entry[mode]["concurrent"] = c = _concurrent(
                    engines, segments, args.voice, args.seed, sampling, stream, args.runs
                )
                print(
                    f"  {len(engines)} workers: {c['audio_s_per_s']} audio s/s, "
                    f"rtf median={c['rtf_median']} max={c['rtf_max']}"
                    + (f" ttfa median={c['ttfa_s_median']}s" if stream else ""),
                    file=sys.stderr,
                )
        result["cases"][case] = entry

    current, peak = _rss_mib()
    result["rss_end_mib"], result["rss_peak_mib"] = current, peak
    if torch_cuda:
        result["vram_end"] = _vram_mib(engine._frames.device)
    result["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["load"]) + f" peak_rss_mib={peak}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
