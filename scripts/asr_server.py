#!/usr/bin/env python3
"""OpenAI-style transcription server around gipformer (Vietnamese RNN-T ASR, sherpa-onnx).

Used to score listening runs (scripts/render_samples.py, OMNIVOICE_ASR_URL). Not part of the
gateway; needs the "asr" dependency group and the gipformer ONNX files:

  uv run hf download g-group-ai-lab/gipformer1.5-68M-rnnt encoder.int8.onnx decoder.int8.onnx \\
      joiner.int8.onnx tokens.txt --local-dir models/gipformer1.5-68M-rnnt
  uv run --group asr python scripts/asr_server.py --port 8090
  OMNIVOICE_ASR_URL=http://127.0.0.1:8090 uv run python scripts/render_samples.py ...

Routes:

  POST /v1/audio/transcriptions  multipart: file (WAV, PCM s16/s32/u8, any rate and channels),
                                 model, language (only "vi"), response_format (json | text);
                                 answers {"text": ...}
  GET  /v1/models                the one served model
  GET  /health

Errors use the shape {"error": {"message", "type"}}. Logs carry no audio or text.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import wave
from pathlib import Path
from typing import Annotated

import numpy as np
import sherpa_onnx
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "gipformer1.5-68M-rnnt"
SAMPLE_RATE = 16000
FEATURE_DIM = 80


class AsrError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        self.status, self.kind, self.message = status, kind, message


def main() -> None:
    args = parse_args()
    files = model_files(args.model_dir, args.precision)
    missing = [p for p in files.values() if not p.is_file() or p.stat().st_size < 1024]
    if missing:
        sys.exit(f"missing or Git LFS pointer model files: {', '.join(map(str, missing))}")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        **{k: str(v) for k, v in files.items()},
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method=args.decoding_method,
        provider=args.provider,
    )
    app = create_app(recognizer, args.model_id, args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    ap.add_argument("--model-id", default="gipformer", help="id answered by /v1/models")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument(
        "--decoding-method",
        choices=("greedy_search", "modified_beam_search"),
        default="modified_beam_search",
    )
    ap.add_argument("--provider", default="cpu", help="onnxruntime provider: cpu or cuda")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("OMNIVOICE_ASR_API_KEY") or None,
        help="require Authorization: Bearer <key> (env OMNIVOICE_ASR_API_KEY)",
    )
    return ap.parse_args()


def model_files(model_dir: Path, precision: str) -> dict[str, Path]:
    suffix = ".int8.onnx" if precision == "int8" else ".onnx"
    return {
        "encoder": model_dir / f"encoder{suffix}",
        "decoder": model_dir / f"decoder{suffix}",
        "joiner": model_dir / f"joiner{suffix}",
        "tokens": model_dir / "tokens.txt",
    }


def create_app(recognizer, model_id: str, api_key: str | None) -> FastAPI:
    app = FastAPI(title="gipformer ASR")

    @app.exception_handler(AsrError)
    async def asr_error(request: Request, exc: AsrError) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": exc.message, "type": exc.kind}}, status_code=exc.status
        )

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if (
            api_key
            and request.url.path != "/health"
            and request.headers.get("authorization") != f"Bearer {api_key}"
        ):
            return JSONResponse(
                {"error": {"message": "invalid API key", "type": "authentication_error"}},
                status_code=401,
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "task": "asr", "language": "vi"}],
        }

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = model_id,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        # Sync handler: FastAPI runs it in a worker thread, so decoding does not block the loop.
        if model != model_id:
            raise AsrError(404, "model_not_found", f"model {model!r} is not served")
        if language and language.lower().split("-")[0] != "vi":
            raise AsrError(400, "invalid_request_error", "only language 'vi' is supported")
        if response_format not in ("json", "text"):
            raise AsrError(400, "invalid_request_error", "response_format must be json or text")
        samples, rate = read_wav(file.file.read())
        stream = recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return PlainTextResponse(text) if response_format == "text" else {"text": text}

    return app


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate of a PCM WAV."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AsrError(400, "invalid_request_error", "file is not a PCM WAV") from exc
    if width == 1:
        samples = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        dtype = np.int16 if width == 2 else np.int32
        samples = np.frombuffer(frames, dtype).astype(np.float32) / np.iinfo(dtype).max
    else:
        raise AsrError(400, "invalid_request_error", "unsupported WAV sample width")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(1)
    if not len(samples):
        raise AsrError(400, "invalid_request_error", "file has no audio")
    return np.ascontiguousarray(samples), rate


if __name__ == "__main__":
    main()
