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
    """Raised inside a started response body so the server drops the connection instead of ending the stream."""


class RequestLog:
    """Logs one content-free record per request after the response body has been fully sent (or aborted)."""

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
    """Streams an upstream body; always closes the upstream response, also when the caller disconnects."""

    def __init__(self, upstream: httpx.Response, body, fields: dict, headers: dict[str, str]):
        super().__init__(body, upstream.status_code, headers, media_type=upstream.headers.get("content-type"))
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


def error_response(status: int, type_: str, message: str) -> JSONResponse:
    # Same shape as audio.cpp; message must never contain request text.
    return JSONResponse({"error": {"message": message, "type": type_}}, status_code=status)


def create_app(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    client = httpx.AsyncClient(base_url=settings.upstream_url, transport=transport, timeout=None)

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
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), settings.api_key.encode()):
            raise GatewayError(401, "unauthorized", "missing or invalid API key")

    async def forward(request: Request, method: str, path: str, body: dict | None = None) -> Response:
        if request.url.query:
            path = f"{path}?{request.url.query}"
        try:
            async with asyncio.timeout(settings.deadline_s):
                upstream = await client.request(method, path, json=body)
        except TimeoutError:
            raise GatewayError(504, "upstream_timeout", "upstream did not respond before the deadline")
        except httpx.TransportError:
            raise GatewayError(502, "upstream_unavailable", "could not connect to upstream")
        request.state.log["upstream_status"] = upstream.status_code
        return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type"))

    async def stream(request: Request, path: str, body: dict, headers: dict[str, str]) -> Response:
        """One upstream request, no retry. Each chunk is passed on as it arrives; `headers` go on success only.

        The deadline is absolute: before the first body byte it becomes a 504; after that the
        caller connection is aborted, as it is when the upstream breaks mid-stream.
        """
        fields = request.state.log
        deadline = asyncio.get_running_loop().time() + settings.deadline_s
        start = time.perf_counter()
        try:
            async with asyncio.timeout_at(deadline):
                upstream = await client.send(client.build_request("POST", path, json=body), stream=True)
                try:
                    fields["upstream_status"] = upstream.status_code
                    if not upstream.is_success:
                        # Upstream errors are small JSON bodies; pass them through whole.
                        content = await upstream.aread()
                        return Response(content, upstream.status_code, media_type=upstream.headers.get("content-type"))
                    chunks = upstream.aiter_raw()
                    first = await anext(chunks, b"")
                except BaseException:
                    await upstream.aclose()
                    raise
        except TimeoutError:
            raise GatewayError(504, "upstream_timeout", "upstream did not respond before the deadline")
        except httpx.TransportError:
            raise GatewayError(502, "upstream_unavailable", "could not connect to upstream")
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
    async def ready():
        try:
            async with asyncio.timeout(READY_TIMEOUT_S):
                for path in ("/health", "/v1/models"):
                    if not (await client.get(path)).is_success:
                        raise GatewayError(503, "upstream_unavailable", f"upstream {path} is not OK")
        except (TimeoutError, httpx.TransportError):
            raise GatewayError(503, "upstream_unavailable", "upstream is not reachable")
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models(request: Request):
        return await forward(request, "GET", "/v1/models")

    @app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
    async def voices(request: Request):
        return await forward(request, "GET", "/v1/audio/voices")

    def prepare_text(request: Request, body: dict) -> tuple[str, list[str], str | None]:
        """Run the text pipeline on body["input"]; returns (engine text, warnings, effective language)."""
        defaults = settings.models.get(body.get("model")) or ModelDefaults()
        language = body.get("language")
        if language is None:
            language = defaults.language
        normalize = body.get("text_processing", {}).get("normalize", defaults.normalize)
        fields = request.state.log
        fields.update(model=body.get("model"), language=language, input_chars=len(body["input"]))
        start = time.perf_counter()
        try:
            engine_text, warnings = pipeline.process(
                body["input"],
                max_chars=settings.max_input_chars,
                vietnamese=normalize and pipeline.is_vietnamese(language),
            )
        except pipeline.TextError as exc:
            fields["text_error"] = exc.type
            raise GatewayError(400, exc.type, str(exc))
        finally:
            fields["pipeline_ms"] = elapsed_ms(start)
        fields.update(engine_chars=len(engine_text), warnings=warnings)
        return engine_text, warnings, language

    @app.post("/v1/text/normalize", dependencies=[Depends(require_auth)])
    async def normalize(request: Request):
        body = await read_body(request, required=("input",))
        engine_text, warnings, language = prepare_text(request, body)
        return {
            "engine_text": engine_text,
            "warnings": warnings,
            "language": language,
            "pipeline_version": pipeline.PIPELINE_VERSION,
        }

    @app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
    async def speech(request: Request):
        body = await read_body(request, required=("model", "input"))
        check_voice_ref(body)
        body["input"], warnings, _ = prepare_text(request, body)
        body.pop("text_processing", None)
        headers = {"X-Text-Pipeline-Version": pipeline.PIPELINE_VERSION}
        if warnings:
            headers["X-Text-Warnings"] = ",".join(warnings)
        return await stream(request, "/v1/audio/speech", body, headers)

    return app


async def read_body(request: Request, required: tuple[str, ...]) -> dict:
    try:
        body = json.loads(await request.body())
    except ValueError:
        raise GatewayError(400, "invalid_request", "request body must be valid JSON")
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request", "request body must be a JSON object")
    for key in ("model", "input", "language"):
        if (key in required or body.get(key) is not None) and not isinstance(body.get(key), str):
            raise GatewayError(400, "invalid_request", f"'{key}' must be a string")
    if body.get("text_processing") is None:
        body.pop("text_processing", None)
    options = body.get("text_processing", {})
    if not isinstance(options, dict) or not isinstance(options.get("normalize", True), bool):
        raise GatewayError(400, "invalid_request", "'text_processing' must be an object with boolean 'normalize'")
    return body


def check_voice_ref(body: dict) -> None:
    if "voice_ref" not in body:
        return
    ref = body["voice_ref"]
    # Server-side paths would let callers read arbitrary files on the upstream host.
    if isinstance(ref, str) or (isinstance(ref, dict) and ref.get("type") == "path"):
        raise GatewayError(400, "voice_ref_path_not_allowed", "voice_ref paths are not allowed; use type \"base64\"")
    if not (isinstance(ref, dict) and ref.get("type") == "base64"):
        raise GatewayError(400, "invalid_request", "voice_ref must be an object with type \"base64\"")
