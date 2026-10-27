import asyncio
import json
import logging
import time

import httpx
import pytest

from omnivoice_gateway.config import ModelDefaults, ShortTextSpeedRamp, load_settings

from .conftest import WAV

SPEECH = "/v1/audio/speech"


def speech_body(**extra):
    return {"model": "g-omnivoice", "input": "Xin chào các bạn.", **extra}


def assert_error(response, status, type_):
    assert response.status_code == status
    error = response.json()["error"]
    assert error["type"] == type_
    assert isinstance(error["message"], str) and error["message"]


# --- POST /v1/audio/speech ---------------------------------------------------


def test_speech_forwards_body_without_text_processing(client, upstream):
    body = speech_body(
        voice="vieneu_v3_Mai_Anh",
        language="vi-VN",
        seed=42,
        speed=1.1,
        num_inference_steps=16,
        instructions="female, young adult",
        options={"guidance_scale": 2.0, "postprocess_output": False},
        response_format="wav",
        text_processing={"normalize": False},
    )
    response = client.post(SPEECH, json=body)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == WAV
    del body["text_processing"]
    assert upstream.last_json() == body


def test_speech_json_response_passes_through(client, upstream):
    payload = {"audio": "UklGRg==", "format": "wav", "sample_rate": 24000}
    upstream.on("POST", SPEECH, lambda r: httpx.Response(200, json=payload))

    response = client.post(SPEECH, json=speech_body(response_format="json"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == payload


def test_speech_base64_voice_ref_forwarded_verbatim(client, upstream):
    ref = {"type": "base64", "data": "UklGRiQAAABXQVZF"}
    response = client.post(SPEECH, json=speech_body(voice_ref=ref, reference_text="Xin chào."))

    assert response.status_code == 200
    assert upstream.last_json()["voice_ref"] == ref
    assert upstream.last_json()["reference_text"] == "Xin chào."


@pytest.mark.parametrize(
    "voice_ref",
    [
        "/etc/passwd",
        "voices/vieneu_v3_Mai_Anh.wav",
        {"type": "path", "path": "/etc/passwd"},
        {"type": "path", "data": "x"},
    ],
)
def test_speech_rejects_path_voice_ref_without_calling_upstream(client, upstream, voice_ref):
    response = client.post(SPEECH, json=speech_body(voice_ref=voice_ref))

    assert_error(response, 400, "voice_ref_path_not_allowed")
    assert upstream.requests == []


@pytest.mark.parametrize("voice_ref", [None, 1, {"data": "x"}, {"type": "url", "url": "http://x"}])
def test_speech_rejects_other_non_base64_voice_ref(client, upstream, voice_ref):
    response = client.post(SPEECH, json=speech_body(voice_ref=voice_ref))

    assert_error(response, 400, "invalid_request")
    assert upstream.requests == []


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"",
        b"[1, 2]",
        json.dumps({"input": "hi"}).encode(),
        json.dumps({"model": "g-omnivoice"}).encode(),
        json.dumps({"model": "g-omnivoice", "input": 5}).encode(),
        b"\xff\xfe",
    ],
)
def test_speech_invalid_request(client, upstream, content):
    response = client.post(SPEECH, content=content, headers={"content-type": "application/json"})

    assert_error(response, 400, "invalid_request")
    assert upstream.requests == []


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (503, {"error": {"message": "model is busy", "type": "server_busy"}}),
        (
            400,
            {
                "error": {
                    "message": "unsupported OmniVoice language",
                    "type": "invalid_request_error",
                }
            },
        ),
        (404, {"error": {"message": "unknown model", "type": "not_found"}}),
    ],
)
def test_speech_upstream_error_passes_through(client, upstream, status, body):
    upstream.on("POST", SPEECH, lambda r: httpx.Response(status, json=body))

    response = client.post(SPEECH, json=speech_body())

    assert response.status_code == status
    assert response.json() == body


def test_speech_upstream_unreachable(client, upstream):
    upstream.down = True

    response = client.post(SPEECH, json=speech_body())

    assert_error(response, 502, "upstream_unavailable")


def test_speech_upstream_deadline(make_client, upstream):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, content=WAV)

    upstream.on("POST", SPEECH, slow)
    client = make_client(deadline_s=0.05)

    response = client.post(SPEECH, json=speech_body())

    assert_error(response, 504, "upstream_timeout")


def test_speech_connect_timeout_fails_fast_before_deadline(make_client, upstream):
    upstream.blackhole = True
    client = make_client(deadline_s=5, connect_timeout_s=0.05)

    start = time.perf_counter()
    response = client.post(SPEECH, json=speech_body())

    assert_error(response, 502, "upstream_unavailable")
    assert time.perf_counter() - start < 1


def test_connect_timeout_does_not_limit_slow_upstream_after_connect(make_client, upstream):
    async def slow(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})

    upstream.on("POST", SPEECH, slow)
    client = make_client(connect_timeout_s=0.01)

    response = client.post(SPEECH, json=speech_body())

    assert response.status_code == 200
    assert response.content == WAV


# --- passthrough routes ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "/v1/models",
        "/v1/models?include_params=true",
        "/v1/audio/voices?model=g-omnivoice&x=a%20b&x=c",
    ],
)
def test_passthrough_preserves_query_string_and_response(client, upstream, url):
    path = url.split("?")[0]
    upstream.on(
        "GET", path, lambda r: httpx.Response(207, json={"seen": str(r.url.query, "ascii")})
    )

    response = client.get(url)

    assert response.status_code == 207
    assert response.json() == {"seen": url.partition("?")[2]}


def test_passthrough_upstream_unreachable(client, upstream):
    upstream.down = True

    assert_error(client.get("/v1/models"), 502, "upstream_unavailable")
    assert_error(client.get("/v1/audio/voices"), 502, "upstream_unavailable")


@pytest.mark.parametrize("path", ["/v1/models", "/v1/audio/voices"])
def test_passthrough_connect_timeout_fails_fast_before_deadline(make_client, upstream, path):
    upstream.blackhole = True
    client = make_client(deadline_s=5, connect_timeout_s=0.05)

    start = time.perf_counter()
    response = client.get(path)

    assert_error(response, 502, "upstream_unavailable")
    assert time.perf_counter() - start < 1


# --- health / ready ----------------------------------------------------------


def test_health_does_not_call_upstream(client, upstream):
    upstream.down = True

    response = client.get("/health")

    assert response.status_code == 200
    assert upstream.requests == []


def test_ready_when_upstream_health_and_models_ok(client, upstream):
    response = client.get("/ready")

    assert response.status_code == 200
    assert [r.url.path for r in upstream.requests] == ["/health", "/v1/models"]


def test_ready_503_when_upstream_down(client, upstream):
    upstream.down = True

    assert_error(client.get("/ready"), 503, "upstream_unavailable")


def test_ready_503_on_connect_timeout(make_client, upstream):
    upstream.blackhole = True
    client = make_client(connect_timeout_s=0.05)

    start = time.perf_counter()
    response = client.get("/ready")

    assert_error(response, 503, "upstream_unavailable")
    assert time.perf_counter() - start < 1


@pytest.mark.parametrize("failing", ["/health", "/v1/models"])
def test_ready_503_when_upstream_route_fails(client, upstream, failing):
    upstream.on(
        "GET", failing, lambda r: httpx.Response(500, json={"error": {"message": "x", "type": "y"}})
    )

    assert_error(client.get("/ready"), 503, "upstream_unavailable")


@pytest.mark.parametrize(
    ("method", "path", "status", "type_"),
    [
        ("GET", "/v1/unknown", 404, "not_found"),
        ("POST", "/v1/audio/unknown", 404, "not_found"),
        ("GET", SPEECH, 405, "method_not_allowed"),
        ("POST", "/v1/models", 405, "method_not_allowed"),
    ],
)
def test_routing_errors_use_audio_cpp_error_shape(client, upstream, method, path, status, type_):
    response = client.request(method, path)

    assert_error(response, status, type_)
    assert set(response.json()) == {"error"}
    assert upstream.requests == []
    if status == 405:
        assert response.headers["allow"]


# --- auth --------------------------------------------------------------------

PROTECTED = [("GET", "/v1/models"), ("GET", "/v1/audio/voices"), ("POST", SPEECH)]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
@pytest.mark.parametrize(
    "authorization", [None, "Bearer wrong", "secret", "Basic secret", "Bearer "]
)
def test_auth_rejects_missing_or_wrong_token(make_client, upstream, method, path, authorization):
    client = make_client(api_key="secret")
    headers = {"authorization": authorization} if authorization else {}

    response = client.request(method, path, json=speech_body(), headers=headers)

    assert_error(response, 401, "unauthorized")
    assert upstream.requests == []


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_auth_accepts_correct_token(make_client, upstream, method, path):
    client = make_client(api_key="secret")

    response = client.request(
        method, path, json=speech_body(), headers={"authorization": "Bearer secret"}
    )

    assert response.status_code == 200


def test_auth_not_required_for_health_and_ready(make_client):
    client = make_client(api_key="secret")

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_no_api_key_means_no_auth(client, method, path):
    assert client.request(method, path, json=speech_body()).status_code == 200


# --- short_text_speed (operator option, issue #18, ADR 0004) -----------------
#
# audio.cpp gives short text too little time; the model pauses before the first phrase and some
# voices (Doan_Trang, Quang_Son) drop it. An operator can make the gateway send a slower `speed`
# for short engine text when the caller did not choose one.

SLOW = {"g-omnivoice": ModelDefaults(language="vi", short_text_speed=0.7)}


def test_short_text_speed_off_by_default(client, upstream):
    client.post(SPEECH, json=speech_body(input="Giới thiệu, Nội dung chính."))

    assert "speed" not in upstream.last_json()


def test_short_text_speed_sent_for_short_engine_text(make_client, upstream):
    client = make_client(models=SLOW)

    client.post(SPEECH, json=speech_body(input="# Giới thiệu\nNội dung chính."))

    assert upstream.last_json()["speed"] == 0.7


def test_short_text_speed_only_for_configured_model(make_client, upstream):
    client = make_client(models=SLOW)

    client.post(SPEECH, json=speech_body(model="omnivoice", input="Xin chào."))

    assert "speed" not in upstream.last_json()


@pytest.mark.parametrize(
    "extra",
    [
        {"speed": 1.0},
        {"speed": 1.2},
        {"options": {"duration_sec": 3}},
        {"options": {"duration": 3}},
    ],
)
def test_short_text_speed_never_overrides_caller_parameters(make_client, upstream, extra):
    client = make_client(models=SLOW)
    body = speech_body(input="Xin chào.", **extra)

    client.post(SPEECH, json=body)

    sent = upstream.last_json()
    assert sent.get("speed") == extra.get("speed")
    assert sent.get("options") == extra.get("options")


def test_short_text_speed_threshold_counts_engine_text_syllables(make_client, upstream):
    models = {
        "g-omnivoice": ModelDefaults(
            language="vi", short_text_speed=0.7, short_text_max_syllables=4
        )
    }
    client = make_client(models=models)

    # Four syllables, plus tags that are not read as words.
    client.post(SPEECH, json=speech_body(input="[laughter] Xin chào các bạn. [sigh]"))
    assert upstream.last_json()["speed"] == 0.7

    # "Có 41 người." is three words but reads as five syllables once normalized.
    client.post(SPEECH, json=speech_body(input="Có 41 người."))
    assert "speed" not in upstream.last_json()


def test_short_text_speed_not_sent_for_long_text(make_client, upstream):
    client = make_client(models=SLOW)
    text = "Xin chào, hôm nay chúng ta sẽ cùng nhau tìm hiểu cách cài đặt phần mềm trên máy tính."

    client.post(SPEECH, json=speech_body(input=text))

    assert "speed" not in upstream.last_json()


def test_short_text_speed_from_config_file(make_client, upstream, tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text(
        '[models.g-omnivoice]\nlanguage = "vi"\n'
        "short_text_speed = 0.75\nshort_text_max_syllables = 3\n"
    )
    client = make_client(load_settings(config, env={}))

    client.post(SPEECH, json=speech_body(input="Xin chào."))
    assert upstream.last_json()["speed"] == 0.75
    client.post(SPEECH, json=speech_body(input="Xin chào các bạn."))
    assert "speed" not in upstream.last_json()


@pytest.mark.parametrize(
    "line", ["short_text_speed = 0", "short_text_speed = -1", "short_text_max_syllables = 0"]
)
def test_short_text_speed_invalid_config_rejected(tmp_path, line):
    config = tmp_path / "gateway.toml"
    config.write_text(f"[models.g-omnivoice]\n{line}\n")

    with pytest.raises(ValueError):
        load_settings(config, env={})


def test_short_text_speed_is_logged_without_text(make_client, upstream, caplog):
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")
    client = make_client(models=SLOW)

    client.post(SPEECH, json=speech_body(input="Bí mật."))

    fields = [r for r in caplog.records if r.name == "omnivoice_gateway"][-1].fields
    assert fields["short_text_speed"] == 0.7
    assert "Bí mật" not in caplog.text + json.dumps(fields, ensure_ascii=False)


@pytest.mark.parametrize("model", ["g-omnivoice", "g-omnivoice-stream"])
@pytest.mark.parametrize(
    ("text", "speed"),
    [
        ("Xin chào.", 0.7),
        ("Cần mang, Áo mưa, Đèn pin, Nước uống.", 0.7),  # 8 syllables
        ("Xin chào, hôm nay chúng ta học bài mới.", 0.6),  # 9 syllables, issue #19
        ("Xin chào, hôm nay chúng ta cùng nhau học một bài mới.", 0.6),  # 12
        ("Xin chào, hôm nay chúng ta sẽ cùng nhau học một bài mới.", None),  # 13
    ],
)
def test_shipped_speed_bands(make_client, upstream, model, text, speed):
    client = make_client(load_settings("configs/gateway.toml", env={}))
    body = speech_body(model=model, input=text, text_processing={"normalize": False})

    response = client.post(SPEECH, json=body)

    assert response.status_code == 200
    sent = upstream.last_json()
    assert sent.get("speed") == speed
    assert ("speed" in sent) == (speed is not None)
    assert sent["input"] == text  # No text workaround or extra request.
    assert len(upstream.requests) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"speed": 1.0},
        {"speed": 0.8, "options": {"num_inference_steps": 64}},
        {"speed": None},
        {"options": {"duration_sec": None}},
        {"options": {"duration_sec": 4}},
        {"options": {"duration": 4}},
    ],
)
def test_speed_bands_preserve_caller_parameters(make_client, upstream, extra):
    client = make_client(load_settings("configs/gateway.toml", env={}))

    response = client.post(
        SPEECH, json=speech_body(input="Xin chào, hôm nay chúng ta học bài mới.", **extra)
    )

    assert response.status_code == 200
    sent = upstream.last_json()
    assert sent.get("speed") == extra.get("speed")
    assert sent.get("options") == extra.get("options")


def test_speed_bands_count_engine_text_and_log_selected_speed(
    make_client, upstream, tmp_path, caplog
):
    config = tmp_path / "gateway.toml"
    config.write_text(
        '[models.g-omnivoice]\nlanguage = "vi"\nshort_text_max_syllables = 2\n'
        "short_text_speed_bands = ["
        "{max_syllables = 4, speed = 0.75}, {max_syllables = 5, speed = 0.65}]\n"
    )
    client = make_client(load_settings(config, env={}))
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")

    # Input is three words, engine text is five syllables; tags do not count.
    response = client.post(SPEECH, json=speech_body(input="[laughter] Có 41 người. [sigh]"))

    assert response.status_code == 200
    assert upstream.last_json()["speed"] == 0.65
    fields = [r for r in caplog.records if r.name == "omnivoice_gateway"][-1].fields
    assert fields["short_text_speed"] == 0.65
    assert "Có 41" not in caplog.text + json.dumps(fields, ensure_ascii=False)

    # Bands, not the scalar threshold, also define where the option stops applying.
    client.post(SPEECH, json=speech_body(input="Xin chào tất cả các bạn."))
    assert "speed" not in upstream.last_json()


@pytest.mark.parametrize("model", ["omnivoice", "omnivoice-vietnamese", "unknown-model"])
def test_shipped_speed_bands_do_not_affect_other_models(make_client, upstream, model):
    client = make_client(load_settings("configs/gateway.toml", env={}))

    response = client.post(
        SPEECH, json=speech_body(model=model, input="Xin chào, hôm nay chúng ta học bài mới.")
    )

    assert response.status_code == 200
    assert "speed" not in upstream.last_json()


@pytest.mark.parametrize(
    "bands",
    [
        "[{max_syllables = 0, speed = 0.7}]",
        "[{max_syllables = 2.5, speed = 0.7}]",
        "[{max_syllables = true, speed = 0.7}]",
        "[{max_syllables = 8, speed = true}]",
        '[{max_syllables = 8, speed = "slow"}]',
        "[{max_syllables = 8, speed = 0}]",
        "[{max_syllables = 8, speed = -0.6}]",
        "[{max_syllables = 8, speed = inf}]",
        "[{max_syllables = 8, speed = nan}]",
        "[{max_syllables = 12, speed = 0.7}, {max_syllables = 8, speed = 0.6}]",
        "[{max_syllables = 8, speed = 0.7}, {max_syllables = 8, speed = 0.6}]",
    ],
)
def test_invalid_speed_bands_rejected(tmp_path, bands):
    config = tmp_path / "gateway.toml"
    config.write_text(f"[models.g-omnivoice]\nshort_text_speed_bands = {bands}\n")

    with pytest.raises(ValueError):
        load_settings(config, env={})


@pytest.mark.parametrize("speed", [None, 0.75])
def test_empty_speed_bands_are_disabled(make_client, upstream, tmp_path, speed):
    config = tmp_path / "gateway.toml"
    scalar = "" if speed is None else f"short_text_speed = {speed}\n"
    config.write_text(f"[models.g-omnivoice]\nshort_text_speed_bands = []\n{scalar}")
    client = make_client(load_settings(config, env={}))

    response = client.post(SPEECH, json=speech_body(input="Xin chào."))

    assert response.status_code == 200
    assert upstream.last_json().get("speed") == speed


def test_scalar_speed_and_bands_cannot_be_combined(tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text(
        "[models.g-omnivoice]\nshort_text_speed = 0.7\n"
        "short_text_speed_bands = [{max_syllables = 12, speed = 0.6}]\n"
    )

    with pytest.raises(ValueError):
        load_settings(config, env={})


# --- config ------------------------------------------------------------------


def test_connect_timeout_defaults_to_3s_and_is_configurable(tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text("connect_timeout_s = 0.5\n")

    assert load_settings(env={}).connect_timeout_s == 3
    assert load_settings(config, env={}).connect_timeout_s == 0.5


def test_host_env_overrides_config(tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text('host = "127.0.0.1"\n')

    assert load_settings(config, env={}).host == "127.0.0.1"
    assert load_settings(config, env={"OMNIVOICE_GATEWAY_HOST": "0.0.0.0"}).host == "0.0.0.0"


def test_config_file_with_env_overrides(make_client, upstream, tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text(
        'upstream_url = "http://from-file.test"\ndeadline_s = 12\n'
        '[models.g-omnivoice]\nlanguage = "vi"\n'
    )
    env = {
        "OMNIVOICE_GATEWAY_CONFIG": str(config),
        "OMNIVOICE_GATEWAY_UPSTREAM_URL": "http://from-env.test:9000",
        "OMNIVOICE_GATEWAY_API_KEY": "env-key",
    }
    client = make_client(load_settings(env=env))

    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers={"authorization": "Bearer env-key"})

    assert response.status_code == 200
    assert str(upstream.requests[-1].url) == "http://from-env.test:9000/v1/models"


def test_shipped_config_loads(make_client, upstream):
    client = make_client(load_settings("configs/gateway.toml", env={}))

    assert client.get("/v1/models").status_code == 200
    assert str(upstream.requests[-1].url).startswith("http://127.0.0.1:8080/")


def test_config_lexicon_path_is_relative_to_config_file(make_client, tmp_path, monkeypatch):
    (tmp_path / "gateway.toml").write_text('lexicon_path = "lex.csv"\n')
    (tmp_path / "lex.csv").write_text("word,replacement,note\nAI,ây ai,\n", encoding="utf-8")
    monkeypatch.chdir("/")
    client = make_client(load_settings(tmp_path / "gateway.toml", env={}))

    response = client.post("/v1/text/normalize", json={"input": "Dùng AI.", "language": "vi"})

    assert response.json()["engine_text"] == "dùng ây ai."


# --- logging -----------------------------------------------------------------


def test_logs_are_structured_and_contain_no_text(client, upstream, caplog):
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")
    secret_input = "Nội dung bí mật của người dùng"
    secret_ref = "Transcript bí mật của giọng mẫu"

    client.post(
        SPEECH,
        json=speech_body(
            input=secret_input,
            voice_ref={"type": "base64", "data": "AAAA"},
            reference_text=secret_ref,
        ),
    )

    records = [r for r in caplog.records if r.name == "omnivoice_gateway"]
    assert records
    fields = records[-1].fields
    assert fields["path"] == SPEECH
    assert fields["status"] == 200
    assert fields["model"] == "g-omnivoice"
    assert fields["input_chars"] == len(secret_input)
    assert fields["engine_chars"] == len(secret_input) + 1  # pipeline adds the final period
    assert fields["language"] is None
    assert fields["warnings"] == []
    assert fields["pipeline_ms"] >= 0
    dumped = caplog.text + json.dumps([r.fields for r in records], ensure_ascii=False)
    assert secret_input not in dumped
    assert secret_ref not in dumped


@pytest.mark.parametrize(
    ("method", "path"), [("POST", SPEECH), ("GET", "/v1/models"), ("GET", "/ready")]
)
def test_log_records_connect_timeout_without_text(make_client, upstream, caplog, method, path):
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")
    upstream.blackhole = True
    client = make_client(connect_timeout_s=0.05)
    secret = "Nội dung bí mật của người dùng"

    client.request(method, path, json=speech_body(input=secret) if method == "POST" else None)

    records = [r for r in caplog.records if r.name == "omnivoice_gateway"]
    fields = records[-1].fields
    assert fields["path"] == path
    assert fields["upstream_error"] == "connect_timeout"
    assert secret not in caplog.text + json.dumps([r.fields for r in records], ensure_ascii=False)


# --- short_text_speed_ramp (operator option, issue #22, ADR 0004) ------------
#
# The bands end at 12 syllables, so 12 → 0.6 and 13 → no speed is a large step in the
# per-request duration budget. The ramp carries the speed back to 1.0 linearly after the bands.

RAMP_TOML = (
    '[models.g-omnivoice]\nlanguage = "vi"\n'
    "short_text_speed_bands = ["
    "{max_syllables = 8, speed = 0.7}, {max_syllables = 12, speed = 0.6}]\n"
    "short_text_speed_ramp = "
    "{from_syllables = 13, to_syllables = 24, from_speed = 0.6, to_speed = 1.0}\n"
)


def syllables_text(n: int) -> str:
    return " ".join(["ba"] * n) + "."


@pytest.mark.parametrize(
    ("syllables", "speed"),
    [(8, 0.7), (12, 0.6), (13, 0.6), (18, 0.782), (24, 1.0), (25, None)],
)
def test_speed_ramp_continues_after_the_bands(make_client, upstream, tmp_path, syllables, speed):
    config = tmp_path / "gateway.toml"
    config.write_text(RAMP_TOML)
    client = make_client(load_settings(config, env={}))
    body = speech_body(input=syllables_text(syllables), text_processing={"normalize": False})

    assert client.post(SPEECH, json=body).status_code == 200

    sent = upstream.last_json()
    assert sent.get("speed") == speed
    assert ("speed" in sent) == (speed is not None)


@pytest.mark.parametrize(
    "ramp",
    [
        "{from_syllables = 12, to_syllables = 24, from_speed = 0.6, to_speed = 1.0}",  # overlaps
        "{from_syllables = 24, to_syllables = 24, from_speed = 0.6, to_speed = 1.0}",
        "{from_syllables = 13, to_syllables = 24, from_speed = 0, to_speed = 1.0}",
        "{from_syllables = 13, to_syllables = 24, from_speed = 0.6, to_speed = -1}",
        "{from_syllables = 13, to_syllables = 24, from_speed = 0.6, to_speed = nan}",
        "{from_syllables = 13, to_syllables = 24, from_speed = true, to_speed = 1.0}",
        "{from_syllables = 13.5, to_syllables = 24, from_speed = 0.6, to_speed = 1.0}",
        "{from_syllables = 13, to_syllables = 24, from_speed = 0.6}",
    ],
)
def test_speed_ramp_invalid_config_rejected(tmp_path, ramp):
    config = tmp_path / "gateway.toml"
    config.write_text(RAMP_TOML.replace(RAMP_TOML.split("short_text_speed_ramp = ")[1], ramp))

    with pytest.raises((ValueError, TypeError)):
        load_settings(config, env={})


def test_speed_ramp_after_the_scalar_form_and_alone(make_client, upstream):
    ramp = ShortTextSpeedRamp(from_syllables=5, to_syllables=9, from_speed=0.8, to_speed=1.0)
    scalar = ModelDefaults(
        language="vi", short_text_speed=0.7, short_text_max_syllables=4, short_text_speed_ramp=ramp
    )
    client = make_client(
        models={
            "g-omnivoice": scalar,
            "g-omnivoice-stream": ModelDefaults(language="vi", short_text_speed_ramp=ramp),
        }
    )
    plain = {"text_processing": {"normalize": False}}

    client.post(SPEECH, json=speech_body(input=syllables_text(4), **plain))
    assert upstream.last_json()["speed"] == 0.7
    client.post(SPEECH, json=speech_body(input=syllables_text(7), **plain))
    assert upstream.last_json()["speed"] == 0.9
    for n, speed in [(4, None), (5, 0.8), (10, None)]:
        client.post(
            SPEECH, json=speech_body(model="g-omnivoice-stream", input=syllables_text(n), **plain)
        )
        assert upstream.last_json().get("speed") == speed


@pytest.mark.parametrize(
    "extra",
    [
        {"speed": 1.0},
        {"speed": 0.9},
        {"options": {"duration_sec": 4}},
        {"options": {"duration": 4}},
    ],
)
def test_speed_ramp_never_overrides_caller_parameters(make_client, upstream, tmp_path, extra):
    config = tmp_path / "gateway.toml"
    config.write_text(RAMP_TOML)
    client = make_client(load_settings(config, env={}))

    client.post(
        SPEECH,
        json=speech_body(input=syllables_text(18), text_processing={"normalize": False}, **extra),
    )

    sent = upstream.last_json()
    assert sent.get("speed") == extra.get("speed")
    assert sent.get("options") == extra.get("options")


def test_speed_ramp_counts_engine_text_and_logs_without_text(
    make_client, upstream, tmp_path, caplog
):
    config = tmp_path / "gateway.toml"
    config.write_text(RAMP_TOML)
    client = make_client(load_settings(config, env={}))
    caplog.set_level(logging.INFO, logger="omnivoice_gateway")

    # Twelve input words, but "41" reads as "bốn mươi mốt": 14 engine syllables. Tags don't count.
    text = "[laughter] Hôm nay lớp mình có 41 bạn đến học bài mới rồi. [sigh]"
    assert client.post(SPEECH, json=speech_body(input=text)).status_code == 200

    assert upstream.last_json()["speed"] == 0.636
    fields = [r for r in caplog.records if r.name == "omnivoice_gateway"][-1].fields
    assert fields["short_text_speed"] == 0.636
    assert "Hôm nay" not in caplog.text + json.dumps(fields, ensure_ascii=False)


@pytest.mark.parametrize("model", ["g-omnivoice", "g-omnivoice-stream"])
def test_shipped_config_leaves_the_ramp_off(make_client, upstream, model):
    client = make_client(load_settings("configs/gateway.toml", env={}))
    body = speech_body(model=model, input=syllables_text(18), text_processing={"normalize": False})

    client.post(SPEECH, json=body)

    assert "speed" not in upstream.last_json()
