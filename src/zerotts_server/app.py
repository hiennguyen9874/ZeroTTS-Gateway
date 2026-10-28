"""HTTP app: the part of audio.cpp's API the gateway uses, served by a ZeroTTS engine."""

import asyncio
import base64
import io
import json
import logging
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .engine import SAMPLE_RATE, Engine, ZeroTTSEngine

log = logging.getLogger("zerotts_server")

HTTP_ERRORS = {
    404: ("not_found", "route not found"),
    405: ("method_not_allowed", "method not allowed for this route"),
}
# Fields ZeroTTS cannot honour; rejected instead of silently ignored (ADR 0005).
UNSUPPORTED_FIELDS = ("voice_ref", "reference_text", "instruct", "instructions")
MAX_SEED = 2**64


class ServerError(Exception):
    def __init__(self, status: int, type_: str, message: str):
        self.status, self.type, self.message = status, type_, message


def error_response(status: int, type_: str, message: str) -> JSONResponse:
    # audio.cpp's shape; message must never contain request text.
    return JSONResponse({"error": {"message": message, "type": type_}}, status_code=status)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RequestLog:
    """Logs one content-free record per request, after the response is sent."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        fields = {}
        scope.setdefault("state", {})["log"] = fields
        status = None

        async def send_and_record(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_record)
        finally:
            fields = {
                "method": scope["method"],
                "path": scope["path"],
                "status": status,
                "total_ms": elapsed_ms(start),
                **fields,
            }
            log.info("request", extra={"fields": fields})


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav16(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return buffer.getvalue()


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    engine = engine or ZeroTTSEngine(settings.model_dir, settings.voices_path, settings.threads)
    # "loading" until engine.load() returns, then "ready" or "failed".
    state = {"status": "loading"}

    def load() -> None:
        start = time.perf_counter()
        try:
            engine.load()
            voices = engine.voices()
            for model_id, model in settings.models.items():
                if model.default_voice not in voices:
                    raise ValueError(f"default_voice of model {model_id} is not a known voice")
        except Exception as exc:
            state["status"] = "failed"
            log.exception("load failed", extra={"fields": {"error": type(exc).__name__}})
            return
        state["status"] = "ready"
        log.info("ready", extra={"fields": {"load_ms": elapsed_ms(start), "voices": len(voices)}})

    @asynccontextmanager
    async def lifespan(_app):
        # Load in the background so /health can answer 503 while the model warms up.
        task = asyncio.create_task(asyncio.to_thread(load))
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ServerError)
    async def handle_server_error(request: Request, exc: ServerError):
        request.state.log["error"] = exc.type
        return error_response(exc.status, exc.type, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, exc: StarletteHTTPException):
        type_, message = HTTP_ERRORS.get(
            exc.status_code, ("invalid_request_error", "invalid request")
        )
        response = error_response(exc.status_code, type_, message)
        response.headers.update(exc.headers or {})
        return response

    app.add_middleware(RequestLog)

    def require_ready() -> None:
        if state["status"] == "failed":
            raise ServerError(503, "model_load_failed", "model failed to load")
        if state["status"] != "ready":
            raise ServerError(503, "model_loading", "model is still loading")

    @app.get("/health")
    async def health():
        require_ready()
        return {"status": "ok", "models": len(settings.models)}

    @app.get("/v1/models")
    async def models():
        loaded = state["status"] == "ready"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "engine",
                    "family": "zerotts",
                    "mode": "offline",
                    "loaded": loaded,
                }
                for model_id in settings.models
            ],
        }

    @app.get("/v1/audio/voices")
    async def voices():
        require_ready()
        found = sorted(engine.voices().values(), key=lambda v: v.name)
        # `voices` is audio.cpp's list of names; `details` adds what a voice picker shows.
        return {
            "voices": [v.name for v in found],
            "details": [
                {
                    "name": v.name,
                    "display_name": v.display_name,
                    "description": v.description,
                    "gender": v.gender,
                    "language": v.language,
                }
                for v in found
            ],
        }

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await read_body(request)
        fields = request.state.log
        model_id = body.get("model")
        if not isinstance(model_id, str) or model_id not in settings.models:
            raise ServerError(400, "invalid_request_error", "unknown model id")
        model = settings.models[model_id]
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ServerError(400, "invalid_request_error", "'input' must be a non-empty string")
        check_supported(body)
        response_format = body.get("response_format") or "wav"
        if response_format not in ("wav", "pcm", "json", "b64_json"):
            raise ServerError(
                400, "invalid_request_error", "response_format must be wav, pcm or json"
            )
        seed = read_seed(body)
        require_ready()
        voice = body.get("voice")
        if voice is None:
            voice = model.default_voice
        if not isinstance(voice, str) or voice not in engine.voices():
            raise ServerError(400, "unknown_voice", "unknown voice")
        fields.update(model=model_id, voice=voice, seed=seed, input_chars=len(text))

        start = time.perf_counter()
        try:
            pcm = await asyncio.to_thread(
                lambda: b"".join(
                    pcm16(chunk) for chunk in engine.synthesize(text, voice, seed, model.sampling)
                )
            )
        except Exception as exc:
            fields["engine_error"] = type(exc).__name__
            raise ServerError(500, "server_error", "synthesis failed") from None
        wall_ms = elapsed_ms(start)
        audio_ms = round(len(pcm) / 2 / SAMPLE_RATE * 1000, 1)
        rtf = round(wall_ms / audio_ms, 3) if audio_ms else 0.0
        fields.update(synth_ms=wall_ms, audio_ms=audio_ms, rtf=rtf)

        if response_format in ("json", "b64_json"):
            timing = {"wall_ms": wall_ms, "audio_duration_ms": audio_ms, "rtf": rtf}
            return {
                "audio": base64.b64encode(wav16(pcm)).decode(),
                "format": "wav",
                "timing": timing,
            }
        headers = {
            "X-AudioCPP-Wall-Ms": str(wall_ms),
            "X-AudioCPP-Audio-Duration-Ms": str(audio_ms),
            "X-AudioCPP-RTF": str(rtf),
        }
        if response_format == "pcm":
            return Response(pcm, media_type="audio/pcm", headers=headers)
        return Response(wav16(pcm), media_type="audio/wav", headers=headers)

    return app


async def read_body(request: Request) -> dict:
    try:
        body = json.loads(await request.body())
    except ValueError:
        raise ServerError(400, "invalid_request_error", "request body must be valid JSON") from None
    if not isinstance(body, dict):
        raise ServerError(400, "invalid_request_error", "request body must be a JSON object")
    return body


def check_supported(body: dict) -> None:
    for key in UNSUPPORTED_FIELDS:
        if body.get(key) is not None:
            raise ServerError(400, "unsupported_option", f"'{key}' is not supported by ZeroTTS")
    for key in ("speed", "speaking_rate"):
        speed = body.get(key)
        # 1 is accepted as omitted, as audio.cpp does for models without speed support.
        if speed is not None and (isinstance(speed, bool) or speed != 1):
            raise ServerError(400, "unsupported_option", f"'{key}' other than 1 is not supported")
    if body.get("stream_format") is not None or body.get("stream") is True:
        raise ServerError(400, "unsupported_option", "streaming is not supported by this model")
    options = body.get("options")
    if options is not None and not isinstance(options, dict):
        raise ServerError(400, "invalid_request_error", "'options' must be an object")
    if options and set(options) - {"seed"}:
        # Sampling belongs to the operator; the caller only picks the seed.
        raise ServerError(400, "unsupported_option", "options other than 'seed' are not supported")


def read_seed(body: dict) -> int | None:
    """Top-level `seed` (audio.cpp) or `options.seed`; a decimal string carries full uint64."""
    options = body.get("options") or {}
    parsed = set()
    for seed in (body.get("seed"), options.get("seed")):
        if seed is None:
            continue
        if isinstance(seed, str) and seed.isascii() and seed.isdigit():
            seed = int(seed)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
            raise ServerError(
                400, "invalid_request_error", "'seed' must be an integer in [0, 2^64)"
            )
        parsed.add(seed)
    if len(parsed) > 1:
        raise ServerError(400, "invalid_request_error", "'seed' and 'options.seed' differ")
    return parsed.pop() if parsed else None
