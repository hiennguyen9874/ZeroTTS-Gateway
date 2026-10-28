"""zerotts_server through its HTTP API, with a fake engine injected into create_app."""

import base64
import io
import json
import logging
import threading
import time
import unicodedata
import wave

import httpx
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
    """Yields `chunks` once per segment; records each call. load() blocks until `loaded` is set.

    `error` is raised after `fail_after` chunks. `delay` sleeps before each chunk; `gate`, when
    set to an Event, holds the run after the first chunk until it is set. `finished` is set when
    a run yields everything, `closed` when a run ends either way.
    """

    def __init__(self, voices=VOICES, chunks=CHUNKS, fail_load=False):
        self._voices, self.chunks, self.fail_load = voices, chunks, fail_load
        self.loaded = threading.Event()
        self.loaded.set()
        self.calls = []
        self.error = None
        self.fail_after = 0
        self.delay = 0.0
        self.gate = None
        self.finished = threading.Event()
        self.closed = threading.Event()

    def load(self):
        self.loaded.wait(5)
        if self.fail_load:
            raise RuntimeError("weights missing")

    def voices(self):
        return self._voices

    def synthesize(self, segments, voice, seed, sampling, stream=False):
        self.calls.append(
            {
                "segments": list(segments),
                "voice": voice,
                "seed": seed,
                "sampling": sampling,
                "stream": stream,
            }
        )
        sent = 0
        try:
            for _ in segments:
                for chunk in self.chunks:
                    if self.error and sent == self.fail_after:
                        raise self.error
                    time.sleep(self.delay)
                    yield chunk
                    sent += 1
                    if sent == 1 and self.gate is not None:
                        assert self.gate.wait(5), "test never opened the gate"
            self.finished.set()
        finally:
            self.closed.set()


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
    assert [call["segments"] for call in engine.calls] == [[TEXT]]


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


# --- long text segmentation --------------------------------------------------

SENTENCES = [
    f"Đây là câu số {i}, kể về một ngày làm việc dài ở văn phòng của công ty." for i in range(12)
]
# ZeroTTS's chunk budget: 15 s at 15 characters per second.
MAX_SEGMENT_CHARS = 225


def test_long_text_reaches_the_engine_as_ordered_sentence_segments(client, engine):
    text = " ".join(SENTENCES)

    response = client.post(SPEECH, json=speech_body(input=text))

    assert response.status_code == 200
    [call] = engine.calls
    segments = call["segments"]
    assert len(segments) > 1
    assert all(len(s) <= MAX_SEGMENT_CHARS for s in segments)
    # Sentence boundaries: every segment is whole sentences, in order, no word lost.
    assert all(s.endswith(".") for s in segments)
    assert " ".join(segments) == text
    # One WAV of every segment's audio, joined in order.
    assert read_wav(response.content) == (48000, 1, 2, pcm16(CHUNKS * len(segments)))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(", ".join(f"phần {i} của một câu rất dài" for i in range(20)), id="commas"),
        pytest.param(" ".join(f"từ{i}" for i in range(120)), id="words"),
    ],
)
def test_overlong_sentence_is_split_by_comma_then_word(client, engine, text):
    assert len(text) > MAX_SEGMENT_CHARS

    assert client.post(SPEECH, json=speech_body(input=text)).status_code == 200

    segments = engine.calls[0]["segments"]
    assert len(segments) > 1
    assert all(len(s) <= MAX_SEGMENT_CHARS for s in segments)
    words = [w.strip(",.") for s in segments for w in s.split()]
    assert words == [w.strip(",.") for w in text.split()]


@pytest.mark.parametrize("long_text", [False, True], ids=["short", "multi-segment"])
def test_decomposed_vietnamese_matches_precomposed_segments(client, engine, long_text):
    text = " ".join(SENTENCES) if long_text else "Tôi."
    for value in (text, unicodedata.normalize("NFD", text)):
        assert client.post(SPEECH, json=speech_body(input=value)).status_code == 200

    assert engine.calls[0]["segments"] == engine.calls[1]["segments"]
    assert " ".join(engine.calls[1]["segments"]) == text


@pytest.mark.parametrize("text", ["C++", "1+1", "Tôi dùng C++; kiểm tra 1+1 = 2."])
@pytest.mark.parametrize("long_text", [False, True], ids=["short", "multi-segment"])
def test_chunking_preserves_meaningful_symbols(client, engine, text, long_text):
    if long_text:
        text = " ".join([text] * 100)

    assert client.post(SPEECH, json=speech_body(input=text)).status_code == 200

    segments = engine.calls[0]["segments"]
    assert len(segments) > 1 if long_text else len(segments) == 1
    assert " ".join(segments) == text


def test_input_with_no_speakable_text_is_400(client, engine):
    assert_error(
        client.post(SPEECH, json=speech_body(input="... !!")), 400, "invalid_request_error"
    )
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
    models = {model["id"]: model for model in response.json()["data"]}
    assert set(models) == {"zerotts", "zerotts-stream"}
    assert models["zerotts"]["mode"] == "offline"
    assert models["zerotts-stream"]["mode"] == "streaming"
    assert all(model["loaded"] is True for model in models.values())


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


def test_logs_count_segments_of_long_text(client, caplog):
    caplog.set_level(logging.INFO, logger="zerotts_server")
    client.post(SPEECH, json=speech_body(input=" ".join(SENTENCES)))

    [record] = [r for r in caplog.records if r.getMessage() == "request"]
    assert record.fields["segments"] > 1
    logged = json.dumps(record.fields, ensure_ascii=False)
    assert "câu" not in logged and "văn phòng" not in logged


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
    assert ok["segments"] == 1
    assert (failed["engine_error"], failed["error"]) == ("RuntimeError", "server_error")
    logged = json.dumps([r.fields for r in records], ensure_ascii=False)
    assert "Xin" not in logged and "chào" not in logged
    # Only short scalar values: no audio bytes or base64.
    assert all(len(str(v)) < 64 for fields in (ok, failed) for v in fields.values())


# --- streaming (zerotts-stream) ----------------------------------------------


def stream_body(**extra):
    return {"model": "zerotts-stream", "input": TEXT, "stream_format": "sse", **extra}


def sse_events(data: bytes) -> list:
    """Parse an SSE body into its `data:` payloads; JSON objects decoded, [DONE] kept as is."""
    assert data.endswith(b"\n\n")
    events = []
    for block in data.decode().split("\n\n")[:-1]:
        assert block.startswith("data: ")
        payload = block.removeprefix("data: ")
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def joined_deltas(events) -> bytes:
    return b"".join(base64.b64decode(e["audio"]) for e in events[:-2])


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"response_format": "pcm"},
        {"response_format": None},
        {"stream_format": None, "stream": True},
    ],
    ids=["sse", "sse-pcm", "sse-null-response-format", "stream-true"],
)
def test_sse_stream_has_audiocpp_events_and_engine_audio(client, engine, extra):
    response = client.post(SPEECH, json=stream_body(**extra))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    events = sse_events(response.content)
    assert [e["type"] for e in events[:-2]] == ["speech.audio.delta"] * len(CHUNKS)
    assert events[-2]["type"] == "speech.audio.done"
    assert events[-2]["timing"]["ttft_ms"] >= 0
    assert events[-1] == "[DONE]"
    assert joined_deltas(events) == pcm16(CHUNKS)
    assert [(c["segments"], c["stream"]) for c in engine.calls] == [([TEXT], True)]


def test_audio_stream_format_returns_raw_pcm(client, engine):
    response = client.post(SPEECH, json=stream_body(stream_format="audio", response_format="pcm"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.content == pcm16(CHUNKS)
    assert engine.calls[0]["stream"] is True


def test_long_text_streams_every_segment_with_one_done(client, engine):
    text = " ".join(SENTENCES)

    events = sse_events(client.post(SPEECH, json=stream_body(input=text, seed=9)).content)

    [call] = engine.calls
    assert len(call["segments"]) > 1 and " ".join(call["segments"]) == text
    assert call["seed"] == 9
    assert [e if e == "[DONE]" else e["type"] for e in events].count("speech.audio.done") == 1
    assert joined_deltas(events) == pcm16(CHUNKS * len(call["segments"]))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(speech_body(stream_format="sse"), id="sse-to-offline"),
        pytest.param(speech_body(stream=True), id="stream-true-to-offline"),
        pytest.param(stream_body(stream_format=None), id="offline-to-stream"),
        pytest.param(stream_body(stream_format=None, response_format="wav"), id="wav-to-stream"),
        pytest.param(stream_body(stream_format="webrtc"), id="unknown-stream-format"),
        pytest.param(stream_body(response_format="wav"), id="stream-wav"),
        pytest.param(stream_body(stream_format="audio", response_format="mp3"), id="stream-mp3"),
    ],
)
def test_stream_and_model_mode_mismatch_is_400(client, engine, body):
    assert_error(client.post(SPEECH, json=body), 400, "invalid_request_error")
    assert engine.calls == []


@pytest.mark.parametrize("field", ["stream_format", "response_format"])
@pytest.mark.parametrize("value", ["", False, 0, [], {}])
def test_invalid_streaming_formats_are_400(client, engine, field, value):
    response = client.post(SPEECH, json=stream_body(**{field: value}))

    assert_error(response, 400, "invalid_request_error")
    assert engine.calls == []


def test_stream_validates_like_offline(client, engine):
    assert_error(client.post(SPEECH, json=stream_body(voice="nobody")), 400, "unknown_voice")
    assert_error(client.post(SPEECH, json=stream_body(speed=1.5)), 400, "unsupported_option")
    assert_error(client.post(SPEECH, json=stream_body(seed=-1)), 400, "invalid_request_error")
    assert engine.calls == []


@pytest.mark.parametrize("fail_after", [0, 1])
def test_engine_failure_ends_sse_with_error_event(client, engine, caplog, fail_after):
    caplog.set_level(logging.INFO, logger="zerotts_server")
    engine.error, engine.fail_after = RuntimeError(f"choked on {TEXT}"), fail_after

    response = client.post(SPEECH, json=stream_body())

    assert response.status_code == 200
    events = sse_events(response.content)
    assert [e["type"] for e in events] == ["speech.audio.delta"] * fail_after + ["error"]
    assert events[-1]["error"]["type"] == "server_error"
    assert "Xin" not in response.text
    [record] = [r for r in caplog.records if r.getMessage() == "request"]
    assert (record.fields["engine_error"], record.fields["error"]) == (
        "RuntimeError",
        "server_error",
    )


def test_stream_log_has_ttfa_and_rtf_without_text(client, caplog):
    caplog.set_level(logging.INFO, logger="zerotts_server")
    client.post(SPEECH, json=stream_body(stream_format="audio"))

    [record] = [r for r in caplog.records if r.getMessage() == "request"]
    fields = record.fields
    assert (fields["status"], fields["model"], fields["segments"]) == (200, "zerotts-stream", 1)
    assert {"ttfa_ms", "synth_ms", "audio_ms", "rtf"} <= set(fields)
    assert fields["audio_ms"] == pytest.approx(len(pcm16(CHUNKS)) / 2 / 48, 0.01)
    assert "Xin" not in json.dumps(fields, ensure_ascii=False)


# Streaming over real HTTP: TestClient buffers whole bodies.


@pytest.fixture
def live_server(engine, serve):
    def start(**overrides) -> str:
        base = serve(create_app(Settings(**overrides), engine=engine))
        deadline = time.monotonic() + 5
        while httpx.get(base + "/health").status_code != 200:
            assert time.monotonic() < deadline, "server did not finish loading"
            time.sleep(0.01)
        return base

    return start


def test_first_delta_arrives_before_engine_finishes(live_server, engine):
    engine.gate = threading.Event()
    base = live_server()

    with httpx.stream("POST", base + SPEECH, json=stream_body(), timeout=5) as response:
        lines = response.iter_lines()
        first = json.loads(next(lines).removeprefix("data: "))
        assert first["type"] == "speech.audio.delta"
        assert base64.b64decode(first["audio"]) == pcm16(CHUNKS[:1])
        assert not engine.finished.is_set()
        engine.gate.set()
        rest = [line for line in lines if line]

    assert json.loads(rest[-2].removeprefix("data: "))["type"] == "speech.audio.done"
    assert rest[-1] == "data: [DONE]"
    assert engine.finished.is_set()


def test_engine_failure_aborts_raw_audio_stream(live_server, engine):
    engine.error, engine.fail_after = RuntimeError("boom"), 1
    base = live_server()
    received = b""

    with (
        pytest.raises(httpx.RemoteProtocolError),
        httpx.stream(
            "POST", base + SPEECH, json=stream_body(stream_format="audio"), timeout=5
        ) as response,
    ):
        assert response.status_code == 200
        for chunk in response.iter_raw():
            received += chunk

    assert received == pcm16(CHUNKS[:1])


def test_caller_disconnect_stops_the_engine(live_server, engine):
    engine.chunks, engine.delay = [np.zeros(480, dtype=np.float32)] * 100, 0.05  # about 5 s
    base = live_server()

    with httpx.stream("POST", base + SPEECH, json=stream_body(), timeout=5) as response:
        next(response.iter_lines())

    assert engine.closed.wait(2)
    assert not engine.finished.is_set()


@pytest.mark.parametrize("extra", [{}, {"stream_format": "audio"}], ids=["sse", "audio"])
def test_gateway_passes_the_stream_through_unchanged(
    live_server, live_gateway, upstream, engine, extra
):
    base = live_server()
    body = stream_body(input="Xin chào.", **extra)

    direct = httpx.post(base + SPEECH, json=body, timeout=5)
    # Replay the server response through the gateway's fake-upstream transport seam.
    upstream.stream(direct.content, content_type=direct.headers["content-type"])
    via_gateway = httpx.post(live_gateway() + SPEECH, json=body, timeout=5)

    assert via_gateway.status_code == direct.status_code == 200
    assert via_gateway.headers["content-type"] == direct.headers["content-type"]
    if extra:
        assert via_gateway.content == direct.content == pcm16(CHUNKS)
    else:
        assert joined_deltas(sse_events(via_gateway.content)) == pcm16(CHUNKS)
        assert sse_events(via_gateway.content)[-1] == "[DONE]"
    assert via_gateway.content == direct.content
    assert [call["stream"] for call in engine.calls] == [True]
    assert upstream.last_json() == body


def test_gateway_forwards_mode_mismatch_400(live_server, live_gateway, upstream):
    body = speech_body(stream_format="sse")
    direct = httpx.post(live_server() + SPEECH, json=body, timeout=5)
    upstream.on(
        "POST",
        SPEECH,
        lambda r: httpx.Response(
            direct.status_code, content=direct.content, headers=direct.headers
        ),
    )

    via_gateway = httpx.post(live_gateway() + SPEECH, json=body, timeout=5)

    assert_error(via_gateway, 400, "invalid_request_error")
    assert via_gateway.content == direct.content
    assert upstream.last_json() == body
