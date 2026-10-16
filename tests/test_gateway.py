import asyncio
import json
import logging

import httpx
import pytest

from omnivoice_gateway.config import load_settings

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
        voice="demo_nu",
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
    ["/etc/passwd", "voices/demo_nu.wav", {"type": "path", "path": "/etc/passwd"}, {"type": "path", "data": "x"}],
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
        (400, {"error": {"message": "unsupported OmniVoice language", "type": "invalid_request_error"}}),
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


# --- passthrough routes ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["/v1/models", "/v1/models?include_params=true", "/v1/audio/voices?model=g-omnivoice&x=a%20b&x=c"],
)
def test_passthrough_preserves_query_string_and_response(client, upstream, url):
    path = url.split("?")[0]
    upstream.on("GET", path, lambda r: httpx.Response(207, json={"seen": str(r.url.query, "ascii")}))

    response = client.get(url)

    assert response.status_code == 207
    assert response.json() == {"seen": url.partition("?")[2]}


def test_passthrough_upstream_unreachable(client, upstream):
    upstream.down = True

    assert_error(client.get("/v1/models"), 502, "upstream_unavailable")
    assert_error(client.get("/v1/audio/voices"), 502, "upstream_unavailable")


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


@pytest.mark.parametrize("failing", ["/health", "/v1/models"])
def test_ready_503_when_upstream_route_fails(client, upstream, failing):
    upstream.on("GET", failing, lambda r: httpx.Response(500, json={"error": {"message": "x", "type": "y"}}))

    assert_error(client.get("/ready"), 503, "upstream_unavailable")


# --- auth --------------------------------------------------------------------

PROTECTED = [("GET", "/v1/models"), ("GET", "/v1/audio/voices"), ("POST", SPEECH)]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "secret", "Basic secret", "Bearer "])
def test_auth_rejects_missing_or_wrong_token(make_client, upstream, method, path, authorization):
    client = make_client(api_key="secret")
    headers = {"authorization": authorization} if authorization else {}

    response = client.request(method, path, json=speech_body(), headers=headers)

    assert_error(response, 401, "unauthorized")
    assert upstream.requests == []


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_auth_accepts_correct_token(make_client, upstream, method, path):
    client = make_client(api_key="secret")

    response = client.request(method, path, json=speech_body(), headers={"authorization": "Bearer secret"})

    assert response.status_code == 200


def test_auth_not_required_for_health_and_ready(make_client):
    client = make_client(api_key="secret")

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_no_api_key_means_no_auth(client, method, path):
    assert client.request(method, path, json=speech_body()).status_code == 200


# --- config ------------------------------------------------------------------


def test_config_file_with_env_overrides(make_client, upstream, tmp_path):
    config = tmp_path / "gateway.toml"
    config.write_text('upstream_url = "http://from-file.test"\ndeadline_s = 12\n[models.g-omnivoice]\nlanguage = "vi"\n')
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

    client.post(SPEECH, json=speech_body(input=secret_input, voice_ref={"type": "base64", "data": "AAAA"}, reference_text=secret_ref))

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
