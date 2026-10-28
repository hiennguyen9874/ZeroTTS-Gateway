"""Opt-in smoke test on the real ZeroTTS weights; skipped unless they are in models/ZeroTTS.

Download: `hf download zeroweight-ai/ZeroTTS --local-dir models/ZeroTTS` (ADR 0005).
Deselect with `-m "not real_model"`.
"""

import io
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zerotts_server import create_app
from zerotts_server.config import Settings

MODEL_DIR = Path(__file__).parent.parent / "models" / "ZeroTTS"

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not (MODEL_DIR / "onnx" / "prefix_step.onnx").is_file(),
        reason="ZeroTTS weights not in models/ZeroTTS",
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


def speak(client, **extra) -> bytes:
    response = client.post(
        "/v1/audio/speech", json={"model": "zerotts", "input": "Xin chào.", **extra}
    )
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (48000, 1, 2)
        return f.readframes(f.getnframes())


def test_offline_speech_is_nonempty_48k_audio(client):
    assert "maichi" in client.get("/v1/audio/voices").json()["voices"]
    pcm = np.frombuffer(speak(client, voice="baotrang"), dtype="<i2")

    assert len(pcm) > 48000 * 0.3
    assert np.abs(pcm).max() > 1000


def test_same_seed_reproduces_audio_regardless_of_global_rng(client):
    np.random.seed(1)
    first = speak(client, seed=1234)
    np.random.seed(2)
    second = speak(client, seed=1234)

    assert first == second
    assert speak(client, seed=99) != first
