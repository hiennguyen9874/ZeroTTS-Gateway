#!/usr/bin/env python3
"""Giải mã 25 giọng preset của VieNeu-TTS v3 Turbo thành WAV để làm voice clone cho OmniVoice.

Preset v3 Turbo không kèm audio: mỗi giọng chỉ có mã MOSS-Audio-Tokenizer-Nano (T x 16, 12.5
frame/s) và một speaker embedding (không dùng được cho OmniVoice). Script này:

  1. đọc `codes` trong voices_v3_turbo.json, bỏ frame đệm 455 ở đuôi (như VieNeu khi nạp preset);
  2. giải mã bằng MOSS-Audio-Tokenizer-Nano -> 48 kHz, gộp về mono;
  3. đổi sang WAV mono 24 kHz, cắt lặng đầu/cuối (giống sample_voices.py);
  4. (mặc định) chạy ASR để có transcript làm `reference_text`; nên nghe và sửa lại tay;
  5. gộp vào voices_candidates/candidates.json + index.html với dataset "vieneu_v3".

Sau đó nghe ở voices_candidates/index.html và cài giọng đã chọn như các nguồn khác:

  CUDA_VISIBLE_DEVICES=1 uv run --no-project \\
    --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \\
    --with torch --with "transformers==4.57.6" --with numpy \\
    python scripts/decode_vieneu_voices.py            # thêm --no-asr nếu chưa cần transcript
  # (wheel torch mặc định của PyPI không nhận driver CUDA 12.6 ở đây và rơi về CPU)
  uv run --with pyarrow --with huggingface_hub \\
    python scripts/sample_voices.py install vieneu_v3_Hai_Dang

License: preset v3 Turbo là Apache-2.0 (xem references/VieNeu-TTS-v3-Turbo/README.md), giữ ghi
công pnnbao-ump và OpenMOSS-Team khi phân phối lại. Yêu cầu: ffmpeg/ffprobe trong PATH.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_voices import DEFAULT_OUT, ROOT, convert, log, safe_id, write_index  # noqa: E402

DATASET = "vieneu_v3"
REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
DEFAULT_PRESETS = ROOT / "references/VieNeu-TTS/src/vieneu/assets/voices_v3_turbo.json"
MOSS = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CODEC_SR = 48_000
N_VQ = 16
ENCODER_PAD_CODE = 455  # frame đuôi do encoder MOSS tự đệm (vieneu_utils/core_utils.py)


def load_presets(path: Path) -> list[tuple[str, dict, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for name, v in data["presets"].items():
        codes = v.get("codes")
        if codes is None:
            log(f"! {name}: không có codes, bỏ qua")
            continue
        arr = np.asarray(json.loads(codes) if isinstance(codes, str) else codes, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != N_VQ:
            log(f"! {name}: codes có shape {arr.shape}, cần (T, {N_VQ}), bỏ qua")
            continue
        if arr.shape[0] > 1 and int(arr[-1, 0]) == ENCODER_PAD_CODE:
            arr = arr[:-1]
        out.append((name, v, arr))
    return out


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def read_16k(path: Path) -> np.ndarray:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--presets", default=str(DEFAULT_PRESETS), help="voices_v3_turbo.json")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="thư mục voices_candidates")
    p.add_argument("--device", default=None, help="cuda | cpu (mặc định: cuda nếu có)")
    p.add_argument("--no-asr", action="store_true", help="không chạy ASR, để transcript trống")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--only", nargs="*", help="chỉ giải mã các giọng có tên này (vd. 'Mai Anh')")
    a = p.parse_args()

    import torch
    from transformers import AutoModel

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    presets = load_presets(Path(a.presets))
    if a.only:
        presets = [x for x in presets if x[0] in set(a.only)]
    log(f"{len(presets)} giọng, thiết bị {device}")

    out = Path(a.out)
    ddir = out / DATASET
    ddir.mkdir(parents=True, exist_ok=True)

    codec = AutoModel.from_pretrained(MOSS, trust_remote_code=True).to(device).eval()
    asr = None
    if not a.no_asr:
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=a.asr_model,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    got = []
    for name, v, codes in presets:
        vid = f"{DATASET}_{safe_id(name)}"
        dst = ddir / f"{vid}.wav"
        c = torch.from_numpy(codes).permute(1, 0).unsqueeze(1).to(device)  # (n_vq, 1, T)
        with torch.no_grad():
            dec = codec.decode(c, return_dict=True)
        audio = dec.audio[0].mean(0).float().cpu().numpy()
        dur = convert(wav_bytes(audio, CODEC_SR), dst)
        if dur is None:
            log(f"! {name}: ffmpeg lỗi")
            continue
        text = ""
        if asr is not None:
            res = asr(
                {"raw": read_16k(dst), "sampling_rate": 16000},
                generate_kwargs={"language": "vi", "task": "transcribe"},
            )
            text = " ".join(res["text"].replace("|", " ").split())
        got.append(
            {
                "id": vid,
                "dataset": DATASET,
                "repo": REPO,
                "speaker": name,
                "gender": v.get("gender"),
                "region": v.get("region"),
                "style": v.get("style"),
                "frames": int(codes.shape[0]),
                "duration": round(dur, 2),
                "text": text,
                "text_source": "asr" if asr is not None else None,
                "wav": str(dst.relative_to(out)),
            }
        )
        warn = "  (quá ngắn cho voice clone)" if dur < 3.0 else ""
        log(f"  + {vid} ({codes.shape[0]} frame, {dur:.1f}s){warn} {text[:70]}")

    cand = out / "candidates.json"
    old = json.loads(cand.read_text(encoding="utf-8")) if cand.exists() else []
    done = {it["id"] for it in got}
    items = [it for it in old if it["id"] not in done] + got
    write_index(out, items)
    log(f"\nXong: {len(got)} giọng -> {ddir}/, tổng {len(items)} ứng viên trong {out}/index.html")


if __name__ == "__main__":
    main()
