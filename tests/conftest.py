"""Test harness: the gateway's HTTP API with a fake upstream injected via httpx transport."""

import inspect
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from omnivoice_gateway import create_app
from omnivoice_gateway.config import Settings

UPSTREAM_URL = "http://upstream.test"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes"

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


class FakeUpstream:
    """Records every request; routes are (method, path) -> handler. `down` simulates a refused connection."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.down = False
        self.routes: dict[tuple[str, str], Handler] = {
            ("GET", "/health"): lambda r: httpx.Response(200, json={"status": "ok"}),
            ("GET", "/v1/models"): lambda r: httpx.Response(200, json={"object": "list", "data": []}),
            ("GET", "/v1/audio/voices"): lambda r: httpx.Response(200, json={"voices": []}),
            ("POST", "/v1/audio/speech"): lambda r: httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"}),
        }

    def on(self, method: str, path: str, handler: Handler) -> None:
        self.routes[(method, path)] = handler

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def last_json(self, path: str = "/v1/audio/speech"):
        return json.loads(self.calls("POST", path)[-1].content)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, json={"error": {"message": "not found", "type": "not_found"}})
        response = handler(request)
        return await response if inspect.isawaitable(response) else response


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def make_client(upstream):
    """Build a gateway client; keyword args override Settings fields."""
    clients = []

    def make(settings: Settings | None = None, **overrides) -> TestClient:
        settings = settings or Settings(upstream_url=UPSTREAM_URL, **overrides)
        client = TestClient(create_app(settings, transport=httpx.MockTransport(upstream.handle)))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
