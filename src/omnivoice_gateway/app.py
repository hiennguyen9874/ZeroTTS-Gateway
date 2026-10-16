"""HTTP app: audio.cpp-compatible routes in front of the upstream."""

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import text as pipeline
from .config import ModelDefaults, Settings

log = logging.getLogger("omnivoice_gateway")

READY_TIMEOUT_S = 5.0


class GatewayError(Exception):
    def __init__(self, status: int, type_: str, message: str):
        self.status, self.type, self.message = status, type_, message


class StreamAborted(Exception):
    """Raised inside a started response body.

    The server then drops the connection instead of ending the stream.
    """


class RequestLog:
    """Logs one content-free record per request.

    The record is written after the response body has been fully sent (or aborted).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        fields = {}
        # Routes add fields through request.state.log, which is backed by scope["state"].
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


class UpstreamStreamingResponse(StreamingResponse):
    """Streams an upstream body.

    Always closes the upstream response, also when the caller disconnects.
    """

    def __init__(self, upstream: httpx.Response, body, fields: dict, headers: dict[str, str]):
        super().__init__(
            body, upstream.status_code, headers, media_type=upstream.headers.get("content-type")
        )
        self.upstream, self.fields = upstream, fields

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # The body generator records how it ended; if it did not, the caller went away first.
            self.fields.setdefault("stream_end", "caller_disconnected")
            await self.upstream.aclose()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def upstream_error(exc: httpx.TransportError) -> str:
    """Content-free log value for a transport failure."""
    return "connect_timeout" if isinstance(exc, httpx.ConnectTimeout) else "transport_error"


def error_response(status: int, type_: str, message: str) -> JSONResponse:
    # Same shape as audio.cpp; message must never contain request text.
    return JSONResponse({"error": {"message": message, "type": type_}}, status_code=status)


def create_app(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    # Loaded once; a lexicon change takes effect on restart and changes the pipeline version.
    text_pipeline = pipeline.Pipeline(
        pipeline.load_lexicon(settings.lexicon_path) if settings.lexicon_path else None,
        max_chars=settings.max_input_chars,
    )
    # No read/write/pool timeout: an offline model may send its first byte only after synthesizing
    # the whole text, so the absolute deadline is the only limit after connecting.
    client = httpx.AsyncClient(
        base_url=settings.upstream_url,
        transport=transport,
        timeout=httpx.Timeout(None, connect=settings.connect_timeout_s),
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_request: Request, exc: GatewayError):
        return error_response(exc.status, exc.type, exc.message)

    app.add_middleware(RequestLog)

    async def require_auth(authorization: str | None = Header(default=None)):
        if settings.api_key is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode(), settings.api_key.encode()
        ):
            raise GatewayError(401, "unauthorized", "missing or invalid API key")

    async def forward(
        request: Request, method: str, path: str, body: dict | None = None
    ) -> Response:
        if request.url.query:
            path = f"{path}?{request.url.query}"
        try:
            async with asyncio.timeout(settings.deadline_s):
                upstream = await client.request(method, path, json=body)
        except TimeoutError:
            raise GatewayError(
                504, "upstream_timeout", "upstream did not respond before the deadline"
            ) from None
        except httpx.TransportError as exc:
            request.state.log["upstream_error"] = upstream_error(exc)
            raise GatewayError(
                502, "upstream_unavailable", "could not connect to upstream"
            ) from None
        request.state.log["upstream_status"] = upstream.status_code
        return Response(
            upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type")
        )

    async def stream(request: Request, path: str, body: dict, headers: dict[str, str]) -> Response:
        """One upstream request, no retry.

        Each chunk is passed on as it arrives; `headers` go on success only.

        The deadline is absolute: before the first body byte it becomes a 504; after that the
        caller connection is aborted, as it is when the upstream breaks mid-stream.
        """
        fields = request.state.log
        deadline = asyncio.get_running_loop().time() + settings.deadline_s
        start = time.perf_counter()
        try:
            async with asyncio.timeout_at(deadline):
                upstream = await client.send(
                    client.build_request("POST", path, json=body), stream=True
                )
                try:
                    fields["upstream_status"] = upstream.status_code
                    if not upstream.is_success:
                        # Upstream errors are small JSON bodies; pass them through whole.
                        content = await upstream.aread()
                        return Response(
                            content,
                            upstream.status_code,
                            media_type=upstream.headers.get("content-type"),
                        )
                    chunks = upstream.aiter_raw()
                    first = await anext(chunks, b"")
                except BaseException:
                    await upstream.aclose()
                    raise
        except TimeoutError:
            raise GatewayError(
                504, "upstream_timeout", "upstream did not respond before the deadline"
            ) from None
        except httpx.TransportError as exc:
            request.state.log["upstream_error"] = upstream_error(exc)
            raise GatewayError(
                502, "upstream_unavailable", "could not connect to upstream"
            ) from None
        fields["upstream_ttfb_ms"] = elapsed_ms(start)

        async def body():
            if first:
                yield first
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    fields["stream_end"] = "complete"
                    return
                except TimeoutError:
                    fields["stream_end"] = "upstream_timeout"
                    raise StreamAborted("deadline exceeded mid-stream") from None
                except httpx.TransportError:
                    fields["stream_end"] = "upstream_disconnected"
                    raise StreamAborted("upstream broke mid-stream") from None
                yield chunk

        return UpstreamStreamingResponse(upstream, body(), fields, headers)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        try:
            async with asyncio.timeout(READY_TIMEOUT_S):
                for path in ("/health", "/v1/models"):
                    if not (await client.get(path)).is_success:
                        raise GatewayError(
                            503, "upstream_unavailable", f"upstream {path} is not OK"
                        )
        except (TimeoutError, httpx.TransportError) as exc:
            request.state.log["upstream_error"] = (
                upstream_error(exc) if isinstance(exc, httpx.TransportError) else "ready_timeout"
            )
            raise GatewayError(503, "upstream_unavailable", "upstream is not reachable") from None
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models(request: Request):
        return await forward(request, "GET", "/v1/models")

    @app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
    async def voices(request: Request):
        return await forward(request, "GET", "/v1/audio/voices")

    def prepare_text(
        request: Request, body: dict, names: tuple[str, ...] = ("input",)
    ) -> tuple[list[str], str | None]:
        """Replace each of `names` present in body with its engine text, in order.

        Returns (warnings of "input" only, effective language). Input errors are raised for the
        first failing field with the same code whichever field it is.
        """
        defaults = settings.models.get(body.get("model")) or ModelDefaults()
        language = body.get("language")
        if language is None:
            language = defaults.language
        normalize = body.get("text_processing", {}).get("normalize", defaults.normalize)
        vietnamese = normalize and pipeline.is_vietnamese(language)
        fields = request.state.log
        fields.update(model=body.get("model"), language=language, input_chars=len(body["input"]))
        start = time.perf_counter()
        warnings = []
        try:
            for name in names:
                if body.get(name) is not None:
                    body[name], field_warnings = text_pipeline.process(
                        body[name], vietnamese=vietnamese
                    )
                    if name == "input":
                        warnings = field_warnings
        except pipeline.TextError as exc:
            fields["text_error"] = exc.type
            raise GatewayError(400, exc.type, str(exc)) from None
        finally:
            fields["pipeline_ms"] = elapsed_ms(start)
        fields.update(engine_chars=len(body["input"]), warnings=warnings)
        return warnings, language

    def apply_short_text_speed(request: Request, body: dict) -> None:
        """Operator option (ADR 0004): slow short engine text down so it is not cut short.

        audio.cpp sizes the audio from character weight, with no time for the pause the model puts
        before the first phrase and at each comma; on short text some voices then drop the first
        phrase (issue #18). Only when the caller set neither `speed` nor `options.duration_sec`,
        so caller generation parameters still pass through untouched.
        """
        defaults = settings.models.get(body.get("model")) or ModelDefaults()
        if defaults.short_text_speed is None or "speed" in body:
            return
        options = body.get("options")
        if isinstance(options, dict) and ("duration_sec" in options or "duration" in options):
            return
        if pipeline.count_syllables(body["input"]) > defaults.short_text_max_syllables:
            return
        body["speed"] = defaults.short_text_speed
        request.state.log["short_text_speed"] = defaults.short_text_speed

    @app.post("/v1/text/normalize", dependencies=[Depends(require_auth)])
    async def normalize(request: Request):
        body = await read_body(request, required=("input",))
        warnings, language = prepare_text(request, body)
        return {
            "engine_text": body["input"],
            "warnings": warnings,
            "language": language,
            "pipeline_version": text_pipeline.version,
        }

    @app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
    async def speech(request: Request):
        body = await read_body(request, required=("model", "input"))
        check_voice_ref(body)
        # reference_text (the voice clone transcript) goes through the same pipeline; its warnings
        # are not reported because they describe the reference audio, not the caller's input.
        warnings, _ = prepare_text(request, body, ("input", "reference_text"))
        body.pop("text_processing", None)
        apply_short_text_speed(request, body)
        headers = {"X-Text-Pipeline-Version": text_pipeline.version}
        if warnings:
            headers["X-Text-Warnings"] = ",".join(warnings)
        return await stream(request, "/v1/audio/speech", body, headers)

    return app


async def read_body(request: Request, required: tuple[str, ...]) -> dict:
    try:
        body = json.loads(await request.body())
    except ValueError:
        raise GatewayError(400, "invalid_request", "request body must be valid JSON") from None
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request", "request body must be a JSON object")
    for key in ("model", "input", "language", "reference_text"):
        if (key in required or body.get(key) is not None) and not isinstance(body.get(key), str):
            raise GatewayError(400, "invalid_request", f"'{key}' must be a string")
    if body.get("text_processing") is None:
        body.pop("text_processing", None)
    options = body.get("text_processing", {})
    if not isinstance(options, dict) or not isinstance(options.get("normalize", True), bool):
        raise GatewayError(
            400, "invalid_request", "'text_processing' must be an object with boolean 'normalize'"
        )
    return body


def check_voice_ref(body: dict) -> None:
    if "voice_ref" not in body:
        return
    ref = body["voice_ref"]
    # Server-side paths would let callers read arbitrary files on the upstream host.
    if isinstance(ref, str) or (isinstance(ref, dict) and ref.get("type") == "path"):
        raise GatewayError(
            400, "voice_ref_path_not_allowed", 'voice_ref paths are not allowed; use type "base64"'
        )
    if not (isinstance(ref, dict) and ref.get("type") == "base64"):
        raise GatewayError(400, "invalid_request", 'voice_ref must be an object with type "base64"')
