"""Opt-in smoke test on local ZeroTTS weights (fp32 or int8).

Uses ZEROTTS_SERVER_MODEL_DIR when set, otherwise models/ZeroTTS; skips if absent.
Download the pinned weights: `uv run scripts/download_zerotts.sh` (ADR 0005).
Run explicitly with `uv run python -m pytest -m real_model`.
"""

import base64
import hashlib
import io
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zerotts_server import create_app
from zerotts_server.config import Settings

MODEL_DIR = Path(
    os.environ.get("ZEROTTS_SERVER_MODEL_DIR", Path(__file__).parent.parent / "models" / "ZeroTTS")
)

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not (MODEL_DIR / "onnx" / "prefix_step.onnx").is_file(),
        reason=f"ZeroTTS weights not in {MODEL_DIR}",
    ),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings(model_dir=str(MODEL_DIR)))) as client:
        deadline = time.monotonic() + 300
        while client.get("/health").status_code != 200:
            assert client.get("/health").json()["error"]["type"] == "model_loading"
            assert time.monotonic() < deadline, "model did not load"
            time.sleep(0.5)
        yield client


def speak(client, text="Xin chào.", **extra) -> bytes:
    response = client.post("/v1/audio/speech", json={"model": "zerotts", "input": text, **extra})
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (48000, 1, 2)
        return f.readframes(f.getnframes())


def test_offline_speech_is_nonempty_48k_audio(client):
    assert "maichi" in client.get("/v1/audio/voices").json()["voices"]
    pcm = np.frombuffer(speak(client, voice="baotrang"), dtype="<i2")

    assert len(pcm) > 48000 * 0.3
    assert np.abs(pcm).max() > 1000


def test_long_text_is_read_past_one_segment(client):
    # About 40 s of speech: several segments, one WAV.
    text = " ".join(
        f"Đây là câu số {i}, kể về một ngày làm việc dài ở văn phòng của công ty."
        for i in range(12)
    )
    pcm = np.frombuffer(speak(client, text, seed=5), dtype="<i2")

    assert len(pcm) > 48000 * 25


def test_same_seed_reproduces_audio_regardless_of_global_rng(client):
    np.random.seed(1)
    first = speak(client, seed=1234)
    np.random.seed(2)
    second = speak(client, seed=1234)

    assert first == second
    assert speak(client, seed=99) != first


def stream_sse(client, text: str, **extra) -> tuple[bytes, float]:
    """Stream over SSE; returns the joined PCM and the server's ttft_ms."""
    body = {"model": "zerotts-stream", "input": text, "stream_format": "sse", **extra}
    response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == 200, response.text
    events = [
        json.loads(block.removeprefix("data: "))
        for block in response.text.split("\n\n")
        if block and block != "data: [DONE]"
    ]
    assert response.text.endswith("data: [DONE]\n\n")
    assert [e["type"] for e in events[:-1]] == ["speech.audio.delta"] * (len(events) - 1)
    assert events[-1]["type"] == "speech.audio.done"
    pcm = b"".join(base64.b64decode(e["audio"]) for e in events[:-1])
    return pcm, events[-1]["timing"]["ttft_ms"]


def test_stream_is_nonempty_48k_audio_with_low_ttfa(client):
    text = " ".join(
        f"Đây là câu số {i}, kể về một ngày làm việc dài ở văn phòng của công ty." for i in range(6)
    )
    pcm, ttft_ms = stream_sse(client, text, seed=5)
    samples = np.frombuffer(pcm, dtype="<i2")

    # Several segments in one stream, at 48 kHz (about 13 s of speech per 4 sentences).
    assert len(samples) > 48000 * 10
    assert np.abs(samples).max() > 1000
    print(f"stream ttft_ms={ttft_ms}")
    # Spec: a few hundred ms on CPU; loose bound because the dev machine is shared.
    assert ttft_ms < 2000


@pytest.mark.skipif(
    os.environ.get("ZEROTTS_VENDOR_BASELINE") not in {"fp32", "int8"},
    reason="set ZEROTTS_VENDOR_BASELINE to fp32 or int8 for the recorded environment",
)
def test_vendored_audio_matches_package_baseline(client):
    # These hashes describe the specific weights and platform in the issue #16 report.
    expected = {
        "fp32": (
            "cfa1a2aa11845b5d518094ca14e2b840bf5ef63cc8925b4a649ee4a8216368ff",
            "1cc72857629becb3ec3955e99e1e2597fdffbf99c23a12addbc0a7de5664141f",
        ),
        "int8": (
            "51206719bd450ca93ff8d563a5c8072a9820377be5b891c939718910bd81c9fa",
            "6a1c2dd7d0c74c30b7f35cbd5620adc5345c9f056947fdc6ba43d16aaf62e420",
        ),
    }[os.environ["ZEROTTS_VENDOR_BASELINE"]]
    text = "Xin chào các bạn. Hôm nay trời rất đẹp."
    options = {"voice": "maichi", "seed": 1234}
    response = client.post("/v1/audio/speech", json={"model": "zerotts", "input": text, **options})
    assert response.status_code == 200, response.text
    stream, _ = stream_sse(client, text, **options)
    assert hashlib.sha256(response.content).hexdigest() == expected[0]
    assert hashlib.sha256(stream).hexdigest() == expected[1]
