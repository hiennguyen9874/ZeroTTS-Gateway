"""zerotts_server through its HTTP API, with a fake engine injected into create_app."""

import base64
import io
import json
import logging
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zerotts_server import create_app, load_settings
from zerotts_server.config import ModelConfig, Sampling, Settings
from zerotts_server.engine import VoiceInfo

SPEECH = "/v1/audio/speech"
TEXT = "Xin chào các bạn, hôm nay trời đẹp."
VOICES = {
    "maichi": VoiceInfo("maichi", "Mai Chi", "nữ, trẻ, kể chuyện", "nữ", "vi"),
    "giahuy": VoiceInfo("giahuy", "Gia Huy", "nam, trẻ, trầm ấm", "nam", "vi"),
}
# Two chunks, including out-of-range samples that must clip rather than wrap.
CHUNKS = [
    np.linspace(-0.5, 0.5, 4800, dtype=np.float32),
    np.array([1.5, -1.5, 0.25, 0.0], dtype=np.float32),
]


def pcm16(chunks) -> bytes:
    return b"".join((np.clip(c, -1, 1) * 32767).astype("<i2").tobytes() for c in chunks)


class FakeEngine:
    """Deterministic chunks; records each synthesize call. load() blocks until `loaded` is set."""

    def __init__(self, voices=VOICES, chunks=CHUNKS, fail_load=False):
        self._voices, self.chunks, self.fail_load = voices, chunks, fail_load
        self.loaded = threading.Event()
        self.loaded.set()
        self.calls = []
        self.error = None

    def load(self):
        self.loaded.wait(5)
        if self.fail_load:
            raise RuntimeError("weights missing")

    def voices(self):
        return self._voices

    def synthesize(self, text, voice, seed, sampling):
        self.calls.append({"text": text, "voice": voice, "seed": seed, "sampling": sampling})
        if self.error:
            raise self.error
        yield from self.chunks


def wait_for_health(client):
    deadline = time.monotonic() + 5
    while True:
        response = client.get("/health")
        if response.json().get("error", {}).get("type") != "model_loading":
            return response
        assert time.monotonic() < deadline, "server did not finish loading"
        time.sleep(0.01)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def make_client(engine):
    clients = []

    def make(settings=None, wait_ready=True, **overrides):
        client = TestClient(create_app(settings or Settings(**overrides), engine=engine))
        client.__enter__()
        clients.append(client)
        if wait_ready:
            response = wait_for_health(client)
            assert response.status_code == 200, response.text
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def speech_body(**extra):
    return {"model": "zerotts", "input": TEXT, **extra}


def assert_error(response, status, type_):
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    error = response.json()["error"]
    assert error["type"] == type_
    assert isinstance(error["message"], str) and error["message"]
    # Error messages never echo request text.
    assert TEXT not in response.text and "Xin" not in error["message"]


def read_wav(data: bytes):
    with wave.open(io.BytesIO(data)) as f:
        return f.getframerate(), f.getnchannels(), f.getsampwidth(), f.readframes(f.getnframes())


# --- response formats ----------------------------------------------------------


@pytest.mark.parametrize("extra", [{}, {"response_format": "wav"}, {"response_format": None}])
def test_speech_returns_48k_mono_16bit_wav_of_engine_audio(client, engine, extra):
    response = client.post(SPEECH, json=speech_body(**extra))

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert read_wav(response.content) == (48000, 1, 2, pcm16(CHUNKS))
    assert float(response.headers["X-AudioCPP-Audio-Duration-Ms"]) == pytest.approx(100.1, 0.01)
    assert [call["text"] for call in engine.calls] == [TEXT]


def test_speech_pcm_returns_raw_samples(client):
    response = client.post(SPEECH, json=speech_body(response_format="pcm"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.content == pcm16(CHUNKS)


@pytest.mark.parametrize("response_format", ["json", "b64_json"])
def test_speech_json_returns_base64_wav_like_audiocpp(client, response_format):
    response = client.post(SPEECH, json=speech_body(response_format=response_format))

    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "wav"
    assert read_wav(base64.b64decode(payload["audio"])) == (48000, 1, 2, pcm16(CHUNKS))
    assert set(payload["timing"]) == {"wall_ms", "audio_duration_ms", "rtf"}


@pytest.mark.parametrize("response_format", ["mp3", "flac", 1])
def test_speech_rejects_other_response_formats(client, engine, response_format):
    response = client.post(SPEECH, json=speech_body(response_format=response_format))

    assert_error(response, 400, "invalid_request_error")
    assert engine.calls == []


# --- voices --------------------------------------------------------------------


def test_missing_voice_uses_the_model_default_voice(make_client, engine):
    client = make_client(models={"zerotts": ModelConfig(default_voice="giahuy")})

    assert client.post(SPEECH, json=speech_body()).status_code == 200
    assert client.post(SPEECH, json=speech_body(voice=None)).status_code == 200
    assert client.post(SPEECH, json=speech_body(voice="maichi")).status_code == 200
    assert [call["voice"] for call in engine.calls] == ["giahuy", "giahuy", "maichi"]


@pytest.mark.parametrize("voice", ["nobody", "", "../maichi", 3, ["maichi"]])
def test_unknown_voice_is_400(client, engine, voice):
    assert_error(client.post(SPEECH, json=speech_body(voice=voice)), 400, "unknown_voice")
    assert engine.calls == []


def test_voices_lists_voice_packs_with_descriptions(client):
    response = client.get("/v1/audio/voices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["voices"] == ["giahuy", "maichi"]
    assert payload["details"][1] == {
        "name": "maichi",
        "display_name": "Mai Chi",
        "description": "nữ, trẻ, kể chuyện",
        "gender": "nữ",
        "language": "vi",
    }


def test_default_voice_missing_from_voice_dir_fails_load(make_client):
    client = make_client(models={"zerotts": ModelConfig(default_voice="nobody")}, wait_ready=False)

    assert_error(wait_for_health(client), 503, "model_load_failed")


# --- unsupported options -------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"voice_ref": {"type": "base64", "data": "UklGRg=="}},
        {"voice_ref": "voices/a.wav"},
        {"reference_text": "Câu tham chiếu."},
        {"instruct": "female, young adult"},
        {"instructions": "speak slowly"},
        {"speed": 1.2},
        {"speed": 0},
        {"speed": "1"},
        {"speed": True},
        {"speaking_rate": 0.9},
        {"options": {"temperature": 0.5}},
        {"options": {"seed": 1, "cfg_scale": 2.0}},
        {"options": {"instruct": "x"}},
        {"stream_format": "sse"},
        {"stream": True},
    ],
)
def test_unsupported_options_are_400(client, engine, extra):
    response = client.post(SPEECH, json=speech_body(**extra))

    assert_error(response, 400, "unsupported_option")
    assert engine.calls == []


@pytest.mark.parametrize(
    "extra",
    [{"speed": 1}, {"speed": 1.0}, {"speaking_rate": 1}, {"voice_ref": None}, {"options": {}}],
)
def test_neutral_values_are_accepted(client, extra):
    assert client.post(SPEECH, json=speech_body(**extra)).status_code == 200


# --- seed and sampling ---------------------------------------------------------


def test_same_seed_reaches_the_engine_unchanged(client, engine):
    for body in (
        speech_body(seed=42),
        speech_body(seed=42),
        speech_body(options={"seed": 42}),
        speech_body(seed="42", options={"seed": 42}),
        speech_body(seed="18446744073709551615"),
        speech_body(),
    ):
        assert client.post(SPEECH, json=body).status_code == 200

    assert [call["seed"] for call in engine.calls] == [42, 42, 42, 42, 2**64 - 1, None]


@pytest.mark.parametrize(
    "extra",
    [
        {"seed": -1},
        {"seed": 2**64},
        {"seed": 1.5},
        {"seed": "abc"},
        pytest.param({"seed": "1" * 5000}, id="oversized-seed"),
        pytest.param({"options": {"seed": "1" * 5000}}, id="oversized-options-seed"),
        {"seed": True},
        {"seed": 1, "options": {"seed": 2}},
    ],
)
def test_invalid_seed_is_400(client, engine, extra):
    assert_error(client.post(SPEECH, json=speech_body(**extra)), 400, "invalid_request_error")
    assert engine.calls == []


def test_operator_sampling_reaches_the_engine(make_client, engine):
    sampling = Sampling(cfg_scale=1.5, audio_temperature=0.6, audio_topk=10)
    client = make_client(models={"zerotts": ModelConfig(sampling=sampling)})

    assert client.post(SPEECH, json=speech_body(seed=7)).status_code == 200
    assert engine.calls[0]["sampling"] == sampling


def test_config_file_sets_models_voice_and_sampling(tmp_path, make_client, engine):
    config = tmp_path / "server.toml"
    config.write_text(
        '[models.zerotts]\ndefault_voice = "giahuy"\n[models.zerotts.sampling]\naudio_topk = 10\n'
    )

    client = make_client(load_settings(config, env={}))

    response = client.get("/v1/models")
    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["zerotts"]
    assert client.post(SPEECH, json=speech_body()).status_code == 200
    assert engine.calls[0]["voice"] == "giahuy"
    assert engine.calls[0]["sampling"] == Sampling(audio_topk=10)


# --- models, health ------------------------------------------------------------


def test_models_lists_zerotts(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["object"] == "list"
    [model] = response.json()["data"]
    assert model["id"] == "zerotts"
    assert model["mode"] == "offline"
    assert model["loaded"] is True


def test_health_is_503_until_warmup_finishes(make_client, engine):
    engine.loaded.clear()
    client = make_client(wait_ready=False)

    assert_error(client.get("/health"), 503, "model_loading")
    assert_error(client.post(SPEECH, json=speech_body()), 503, "model_loading")
    assert_error(client.get("/v1/audio/voices"), 503, "model_loading")
    assert client.get("/v1/models").json()["data"][0]["loaded"] is False

    engine.loaded.set()
    response = wait_for_health(client)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert client.post(SPEECH, json=speech_body()).status_code == 200


def test_failed_load_keeps_health_503(make_client):
    engine = FakeEngine(fail_load=True)
    client = TestClient(create_app(Settings(), engine=engine))
    with client:
        assert_error(wait_for_health(client), 503, "model_load_failed")
        assert_error(client.post(SPEECH, json=speech_body()), 503, "model_load_failed")


# --- request errors and logs ---------------------------------------------------


@pytest.mark.parametrize(
    ("content", "type_"),
    [
        (b"not json", "invalid_request_error"),
        (b"[1]", "invalid_request_error"),
        (json.dumps({"input": TEXT}).encode(), "invalid_request_error"),
        (json.dumps({"model": "g-omnivoice", "input": TEXT}).encode(), "invalid_request_error"),
        (json.dumps({"model": "zerotts"}).encode(), "invalid_request_error"),
        (json.dumps({"model": "zerotts", "input": "  \n"}).encode(), "invalid_request_error"),
        (json.dumps({"model": "zerotts", "input": 5}).encode(), "invalid_request_error"),
        (json.dumps(speech_body(options=[1])).encode(), "invalid_request_error"),
    ],
)
def test_bad_requests_are_400(client, engine, content, type_):
    response = client.post(SPEECH, content=content, headers={"content-type": "application/json"})

    assert_error(response, 400, type_)
    assert engine.calls == []


def test_routing_errors_use_audiocpp_shape(client):
    assert_error(client.get("/v1/nope"), 404, "not_found")
    assert_error(client.get(SPEECH), 405, "method_not_allowed")


def test_engine_failure_is_500_without_text(client, engine):
    engine.error = RuntimeError(f"tokenizer choked on {TEXT}")

    assert_error(client.post(SPEECH, json=speech_body()), 500, "server_error")


def test_logs_carry_no_text_or_audio(client, engine, caplog):
    caplog.set_level(logging.INFO, logger="zerotts_server")
    client.post(SPEECH, json=speech_body(seed=3, voice="giahuy"))
    engine.error = RuntimeError(f"failed on {TEXT}")
    client.post(SPEECH, json=speech_body())

    records = [r for r in caplog.records if r.getMessage() == "request"]
    assert len(records) == 2
    ok, failed = (r.fields for r in records)
    assert ok["status"] == 200
    assert (ok["model"], ok["voice"], ok["seed"], ok["input_chars"]) == (
        "zerotts",
        "giahuy",
        3,
        len(TEXT),
    )
    assert {"synth_ms", "audio_ms", "rtf"} <= set(ok)
    assert (failed["engine_error"], failed["error"]) == ("RuntimeError", "server_error")
    logged = json.dumps([r.fields for r in records], ensure_ascii=False)
    assert "Xin" not in logged and "chào" not in logged
    # Only short scalar values: no audio bytes or base64.
    assert all(len(str(v)) < 64 for fields in (ok, failed) for v in fields.values())
