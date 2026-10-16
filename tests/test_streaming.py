"""Streaming passthrough over a real HTTP connection: timing, byte fidelity, aborts, deadline."""

import json
import logging
import threading

import httpx
import pytest

SPEECH = "/v1/audio/speech"
BODY = {"model": "g-omnivoice-stream", "input": "Xin chào.", "response_format": "pcm", "stream_format": "audio"}
CLOSE_WAIT_S = 2

SSE = (
    b'data: {"type":"speech.audio.delta","audio":"AAEC"}\n\n'
    b'data: {"type":"speech.audio.delta","audio":"AwQF"}\n\n'
    b'data: {"type":"error","error":{"message":"generation failed","type":"server_error"}}\n\n'
    b'data: {"type":"speech.audio.done","usage":{"input_tokens":3}}\n\n'
    b"data: [DONE]\n\n"
)


def split_odd(data: bytes, sizes=(1, 3, 7, 13)) -> list[bytes]:
    """Split into odd-sized chunks that straddle SSE event and PCM sample boundaries."""
    chunks, i = [], 0
    while i < len(data):
        size = sizes[len(chunks) % len(sizes)]
        chunks.append(data[i : i + size])
        i += size
    return chunks


def test_first_byte_reaches_caller_before_upstream_finishes(live_gateway, upstream):
    gate = threading.Event()
    stream = upstream.stream(b"\x01\x02\x03", gate, b"\x04\x05")
    base = live_gateway()

    with httpx.Client(base_url=base, timeout=5) as caller, caller.stream("POST", SPEECH, json=BODY) as response:
        chunks = response.iter_raw()
        assert response.status_code == 200
        assert next(chunks) == b"\x01\x02\x03"
        assert not stream.finished
        gate.set()
        assert b"".join(chunks) == b"\x04\x05"

    assert stream.finished


@pytest.mark.parametrize(
    ("content_type", "data"),
    [("text/event-stream", SSE), ("audio/pcm", bytes(range(256)) * 3 + b"\x7f")],
)
def test_odd_chunks_and_sse_events_arrive_intact(live_gateway, upstream, content_type, data):
    upstream.stream(*split_odd(data), content_type=content_type)
    base = live_gateway()

    response = httpx.post(base + SPEECH, json={**BODY, "stream_format": "sse"}, timeout=5)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.content == data
    assert len(upstream.calls("POST", SPEECH)) == 1


def read_until_abort(base: str) -> bytes:
    received = b""
    with pytest.raises(httpx.RemoteProtocolError):
        with httpx.stream("POST", base + SPEECH, json=BODY, timeout=5) as response:
            assert response.status_code == 200
            for chunk in response.iter_raw():
                received += chunk
    return received


def test_upstream_break_mid_stream_aborts_caller(live_gateway, upstream):
    stream = upstream.stream(b"\x01\x02", 0.05, b"\x03", httpx.ReadError("upstream connection reset"))
    base = live_gateway()

    assert read_until_abort(base) == b"\x01\x02\x03"
    assert stream.closed.wait(CLOSE_WAIT_S)
    assert len(upstream.calls("POST", SPEECH)) == 1


def test_caller_disconnect_closes_upstream(live_gateway, upstream):
    stream = upstream.stream(b"\x01", threading.Event())  # never finishes on its own
    base = live_gateway()

    with httpx.Client(base_url=base, timeout=5) as caller, caller.stream("POST", SPEECH, json=BODY) as response:
        assert next(response.iter_raw()) == b"\x01"

    assert stream.closed.wait(CLOSE_WAIT_S)
    assert not stream.finished


def test_deadline_before_first_byte_returns_504(live_gateway, upstream):
    stream = upstream.stream(1.0, b"\x01")  # headers arrive at once, body is late
    base = live_gateway(deadline_s=0.2)

    response = httpx.post(base + SPEECH, json=BODY, timeout=5)

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "upstream_timeout"
    assert response.json()["error"]["message"]
    assert stream.closed.wait(CLOSE_WAIT_S)
    assert len(upstream.calls("POST", SPEECH)) == 1


def test_deadline_mid_stream_aborts_caller(live_gateway, upstream):
    stream = upstream.stream(b"\x01", 1.0, b"\x02")
    base = live_gateway(deadline_s=0.3)

    assert read_until_abort(base) == b"\x01"
    assert stream.closed.wait(CLOSE_WAIT_S)
    assert not stream.finished
    assert len(upstream.calls("POST", SPEECH)) == 1


def test_stream_log_has_ttfb_and_total_without_text(client, upstream, caplog):
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")
    secret = "Nội dung bí mật của người dùng"
    upstream.stream(0.05, b"\x01", 0.1, b"\x02")

    response = client.post(SPEECH, json={**BODY, "input": secret})

    assert response.content == b"\x01\x02"
    fields = [r.fields for r in caplog.records if r.name == "omnivoice_gateway"][-1]
    assert fields["status"] == 200
    assert 50 <= fields["upstream_ttfb_ms"] < fields["total_ms"]
    assert fields["total_ms"] >= 150
    assert fields["stream_end"] == "complete"
    assert secret not in caplog.text + json.dumps(fields, ensure_ascii=False)


@pytest.mark.parametrize(
    ("steps", "stream_end"),
    [((b"\x01", httpx.ReadError("reset")), "upstream_disconnected"), ((b"\x01", 1.0, b"\x02"), "upstream_timeout")],
)
def test_stream_log_records_abort_reason(make_client, upstream, caplog, steps, stream_end):
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")
    upstream.stream(*steps)
    client = make_client(deadline_s=0.3)

    with pytest.raises(Exception):
        client.post(SPEECH, json=BODY)

    fields = [r.fields for r in caplog.records if r.name == "omnivoice_gateway"][-1]
    assert fields["stream_end"] == stream_end
