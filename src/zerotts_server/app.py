"""HTTP app: the part of audio.cpp's API the gateway uses, served by a ZeroTTS engine."""

import asyncio
import base64
import io
import json
import logging
import threading
import time
import unicodedata
import wave
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from zerotts_server.model.chunking import chunk_text

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
# Longest estimated audio per segment; ZeroTTS truncates at 512 tokens and stops at
# 1500 frames (120 s), so long engine text is split by sentence (ADR 0001, ADR 0005).
MAX_SEGMENT_SEC = 15.0


class ServerError(Exception):
    def __init__(self, status: int, type_: str, message: str):
        self.status, self.type, self.message = status, type_, message


class StreamAborted(Exception):
    """Raised inside a started raw audio stream: the server drops the connection.

    A raw PCM caller has no error event, so a clean end would look like complete audio.
    """


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


def sse_event(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


class WorkerPool:
    """Engines that each run one request at a time; a request waits up to busy_timeout."""

    def __init__(self, engines: list[Engine], busy_timeout: float):
        self.engines, self.busy_timeout = engines, busy_timeout
        self._free: asyncio.Queue | None = None

    def open(self) -> None:
        """Bind the free list to the running event loop; call once, from the loop."""
        self._loop = asyncio.get_running_loop()
        self._free = asyncio.Queue()
        for engine in self.engines:
            self._free.put_nowait(engine)

    async def acquire(self) -> Engine:
        # wait_for(..., 0) times out even when a worker is free.
        with suppress(asyncio.QueueEmpty):
            return self._free.get_nowait()
        try:
            return await asyncio.wait_for(self._free.get(), self.busy_timeout)
        except TimeoutError:
            raise ServerError(503, "server_busy", "all workers are busy") from None

    def release(self, engine: Engine) -> None:
        """Thread-safe: called by the thread that ran the engine, once it has stopped."""
        with suppress(RuntimeError):  # the event loop is gone (shutdown)
            self._loop.call_soon_threadsafe(self._free.put_nowait, engine)


class ThreadedIterator:
    """Runs a blocking iterator in its own thread from creation, handing items to the loop.

    close(), or the consumer stopping (the caller disconnected), makes the thread stop after
    the item in progress and close the iterator. `on_done` then runs in the thread, so the
    worker is returned only once the engine has really stopped.
    """

    _END = object()

    def __init__(self, make: Callable[[], Iterator], on_done: Callable[[], None]):
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop = threading.Event()
        try:
            threading.Thread(target=self._run, args=(make, on_done), daemon=True).start()
        except BaseException:
            on_done()
            raise

    def _put(self, item) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # the event loop is gone (shutdown)
            self._stop.set()

    def _run(self, make: Callable[[], Iterator], on_done: Callable[[], None]) -> None:
        items = None
        try:
            items = iter(make())
            for item in items:
                if self._stop.is_set():
                    break
                self._put(item)
            self._put(self._END)
        except Exception as exc:
            self._put(exc)
        finally:
            try:
                close = getattr(items, "close", None)
                if close:
                    close()
            finally:
                on_done()

    def close(self) -> None:
        self._stop.set()

    async def __aiter__(self) -> AsyncIterator:
        try:
            while (item := await self._queue.get()) is not self._END:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.close()


class EngineStreamingResponse(StreamingResponse):
    """Stops the engine thread however the response ends, even if the body never started."""

    def __init__(self, items: ThreadedIterator, content, **kwargs):
        super().__init__(content, **kwargs)
        self.items = items

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.items.close()


def split_segments(text: str) -> list[str]:
    """Split by sentence, then comma, then word, without rewriting meaningful symbols."""
    # Match the tokenizer's NFC normalization before measuring or splitting characters.
    text = unicodedata.normalize("NFC", text)
    if not any(char.isalnum() for char in text):
        return []
    return chunk_text(text, max_chunk_sec=MAX_SEGMENT_SEC)


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    if engine is None:
        engines = [
            ZeroTTSEngine(settings.model_dir, settings.voices_path, settings.threads)
            for _ in range(settings.workers)
        ]
    else:
        # An injected engine serves every worker slot, so it must be thread-safe.
        engines = [engine] * settings.workers
    engine = engines[0]  # voices are the same in every worker
    pool = WorkerPool(engines, settings.busy_timeout)
    # "loading" until engine.load() returns, then "ready" or "failed".
    state = {"status": "loading"}

    def load() -> None:
        start = time.perf_counter()
        try:
            fields = {"model_dir": str(settings.model_dir), "weights": "fp32"}
            manifest_path = Path(settings.model_dir) / "quantization.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                fields.update(
                    weights="int8",
                    source_revision=manifest["source_revision"],
                    sha256={name: graph["sha256"] for name, graph in manifest["graphs"].items()},
                )
            log.info("weights", extra={"fields": fields})
            for worker in dict.fromkeys(engines):
                worker.load()
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
        pool.open()
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
                    "mode": model.mode,
                    "loaded": loaded,
                }
                for model_id, model in settings.models.items()
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
        streaming = body.get("stream_format") is not None or body.get("stream") is True
        if streaming and model.mode != "streaming":
            raise ServerError(
                400,
                "invalid_request_error",
                "speech streaming requires a model configured with mode=streaming",
            )
        if not streaming and model.mode == "streaming":
            raise ServerError(
                400, "invalid_request_error", "this model only streams; set stream_format"
            )
        if streaming:
            stream_format = body.get("stream_format")
            if stream_format is None:
                stream_format = "sse"
            if stream_format not in ("sse", "audio"):
                raise ServerError(
                    400,
                    "invalid_request_error",
                    "streaming speech stream_format must be sse or audio",
                )
            response_format = body.get("response_format")
            if response_format is not None and response_format != "pcm":
                raise ServerError(
                    400,
                    "invalid_request_error",
                    "streaming speech currently supports response_format=pcm",
                )
        else:
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
        segments = split_segments(text)
        if not segments:
            raise ServerError(400, "invalid_request_error", "'input' has no speakable text")
        fields.update(
            model=model_id, voice=voice, seed=seed, input_chars=len(text), segments=len(segments)
        )

        start = time.perf_counter()
        try:
            worker = await pool.acquire()
        finally:
            fields["wait_ms"] = elapsed_ms(start)
        start = time.perf_counter()
        # From here the thread running the engine returns the worker when it stops.
        items = ThreadedIterator(
            lambda: worker.synthesize(segments, voice, seed, model.sampling, stream=streaming),
            lambda: pool.release(worker),
        )
        if streaming:
            return stream_speech(items, stream_format == "sse", fields, start)

        try:
            pcm = b"".join([pcm16(chunk) async for chunk in items])
        except Exception as exc:
            fields["engine_error"] = type(exc).__name__
            raise ServerError(500, "server_error", "synthesis failed") from None
        finally:
            items.close()
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


def stream_speech(
    items: ThreadedIterator, sse: bool, fields: dict, start: float
) -> StreamingResponse:
    """audio.cpp's streamed speech: SSE events, or raw PCM over chunked transfer."""

    async def body():
        samples = 0
        ttfa_ms = None
        try:
            async for chunk in items:
                pcm = pcm16(chunk)
                if not pcm:
                    continue
                if ttfa_ms is None:
                    ttfa_ms = fields["ttfa_ms"] = elapsed_ms(start)
                samples += len(pcm) // 2
                if sse:
                    yield sse_event(
                        {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
                    )
                else:
                    yield pcm
            if ttfa_ms is None:
                raise RuntimeError("engine produced no audio")
        except Exception as exc:
            fields.update(engine_error=type(exc).__name__, error="server_error")
            if not sse:
                raise StreamAborted("synthesis failed mid-stream") from None
            # audio.cpp ends a failed SSE stream with an error event, without done or [DONE].
            yield sse_event(
                {"type": "error", "error": {"message": "synthesis failed", "type": "server_error"}}
            )
            return
        finally:
            wall_ms = elapsed_ms(start)
            audio_ms = round(samples / SAMPLE_RATE * 1000, 1)
            fields.update(
                synth_ms=wall_ms,
                audio_ms=audio_ms,
                rtf=round(wall_ms / audio_ms, 3) if audio_ms else 0.0,
            )
        if sse:
            yield sse_event({"type": "speech.audio.done", "timing": {"ttft_ms": ttfa_ms}})
            yield sse_event("[DONE]")

    if sse:
        return EngineStreamingResponse(
            items, body(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )
    return EngineStreamingResponse(items, body(), media_type="audio/pcm")


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
            try:
                seed = int(seed)
            except ValueError:
                raise ServerError(
                    400, "invalid_request_error", "'seed' must be an integer in [0, 2^64)"
                ) from None
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
            raise ServerError(
                400, "invalid_request_error", "'seed' must be an integer in [0, 2^64)"
            )
        parsed.add(seed)
    if len(parsed) > 1:
        raise ServerError(400, "invalid_request_error", "'seed' and 'options.seed' differ")
    return parsed.pop() if parsed else None
