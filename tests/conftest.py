"""Test harness: the gateway's HTTP API with a fake upstream injected via httpx transport."""

import asyncio
import inspect
import json
import socket
import threading
import time
from collections.abc import Awaitable, Callable

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from omnivoice_gateway import create_app
from omnivoice_gateway.config import Settings

UPSTREAM_URL = "http://upstream.test"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes"

class ScriptedStream(httpx.AsyncByteStream):
    """Fake upstream body that plays a script.

    Steps: bytes are sent as one chunk, a number sleeps that many seconds, a threading.Event
    blocks until set, an exception is raised (the connection breaks).
    """

    def __init__(self, *steps):
        self.steps = steps
        self.finished = False
        self.closed = threading.Event()

    async def __aiter__(self):
        for step in self.steps:
            if isinstance(step, bytes):
                yield step
            elif isinstance(step, int | float):
                await asyncio.sleep(step)
            elif isinstance(step, threading.Event):
                while not step.is_set():
                    await asyncio.sleep(0.005)
            else:
                raise step
        self.finished = True

    async def aclose(self):
        self.closed.set()


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

    def stream(self, *steps, content_type: str = "audio/pcm") -> ScriptedStream:
        """Make the next speech request answer 200 with a scripted streaming body."""
        body = ScriptedStream(*steps)
        self.on("POST", "/v1/audio/speech", lambda r: httpx.Response(200, headers={"content-type": content_type}, stream=body))
        return body

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
        response = await response if inspect.isawaitable(response) else response
        if isinstance(response.stream, ScriptedStream):
            return response
        # httpx pre-reads responses built from bytes; a network transport hands back an unread stream.
        return httpx.Response(response.status_code, headers=response.headers, stream=ScriptedStream(response.content))


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


@pytest.fixture
def live_gateway(upstream):
    """Run the gateway on a real uvicorn server (TestClient buffers whole bodies); returns its base URL."""
    servers = []

    def start(**overrides) -> str:
        app = create_app(Settings(upstream_url=UPSTREAM_URL, **overrides), transport=httpx.MockTransport(upstream.handle))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, lifespan="on", log_config=None))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        servers.append((server, thread))
        deadline = time.monotonic() + 5
        while not server.started:
            assert time.monotonic() < deadline, "gateway did not start"
            time.sleep(0.01)
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(5)
