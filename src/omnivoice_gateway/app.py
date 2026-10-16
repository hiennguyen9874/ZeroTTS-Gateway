"""HTTP app: audio.cpp-compatible routes in front of the upstream."""

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from .config import Settings

log = logging.getLogger("omnivoice_gateway")

READY_TIMEOUT_S = 5.0


class GatewayError(Exception):
    def __init__(self, status: int, type_: str, message: str):
        self.status, self.type, self.message = status, type_, message


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

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        start = time.perf_counter()
        request.state.log = {}
        response = await call_next(request)
        fields = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
            **request.state.log,
        }
        log.info("request", extra={"fields": fields})
        return response

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

    @app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
    async def speech(request: Request):
        try:
            body = json.loads(await request.body())
        except ValueError:
            raise GatewayError(400, "invalid_request", "request body must be valid JSON")
        if not isinstance(body, dict):
            raise GatewayError(400, "invalid_request", "request body must be a JSON object")
        for key in ("model", "input"):
            if not isinstance(body.get(key), str):
                raise GatewayError(400, "invalid_request", f"'{key}' must be a string")
        request.state.log.update(model=body["model"], input_chars=len(body["input"]))
        check_voice_ref(body)
        body.pop("text_processing", None)
        return await forward(request, "POST", "/v1/audio/speech", body)

    return app


def check_voice_ref(body: dict) -> None:
    if "voice_ref" not in body:
        return
    ref = body["voice_ref"]
    # Server-side paths would let callers read arbitrary files on the upstream host.
    if isinstance(ref, str) or (isinstance(ref, dict) and ref.get("type") == "path"):
        raise GatewayError(400, "voice_ref_path_not_allowed", "voice_ref paths are not allowed; use type \"base64\"")
    if not (isinstance(ref, dict) and ref.get("type") == "base64"):
        raise GatewayError(400, "invalid_request", "voice_ref must be an object with type \"base64\"")
