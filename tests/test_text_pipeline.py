"""Text pipeline through the HTTP API: goldens via the dry-run route, then speech-route behaviour."""

import json
from pathlib import Path

import pytest

from omnivoice_gateway.config import ModelDefaults, Settings

from .conftest import UPSTREAM_URL
from .test_gateway import assert_error

NORMALIZE = "/v1/text/normalize"
SPEECH = "/v1/audio/speech"
GOLDEN = json.loads((Path(__file__).parent / "golden" / "pipeline.json").read_text("utf-8"))


def golden_request(case: dict) -> dict:
    body = {"input": case["input"], "language": case.get("language", "vi")}
    if "normalize" in case:
        body["text_processing"] = {"normalize": case["normalize"]}
    return body


def test_golden_cases_are_well_formed():
    ids = [case["id"] for case in GOLDEN]
    assert len(ids) == len(set(ids))
    for case in GOLDEN:
        assert case.get("why"), case["id"]
        assert ("error" in case) != ("expected" in case), case["id"]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case["id"])
def test_golden(client, upstream, case):
    response = client.post(NORMALIZE, json=golden_request(case))

    if "error" in case:
        assert_error(response, 400, case["error"])
    else:
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["engine_text"] == case["expected"]
        assert result["warnings"] == case["warnings"]
    assert upstream.requests == []


# --- dry-run route -----------------------------------------------------------


def test_normalize_response_shape(client):
    response = client.post(NORMALIZE, json={"input": "Xin chào", "language": "vi-VN"})

    assert response.status_code == 200
    result = response.json()
    assert result == {"engine_text": "Xin chào.", "warnings": [], "language": "vi-VN", "pipeline_version": result["pipeline_version"]}
    assert isinstance(result["pipeline_version"], str) and result["pipeline_version"]


@pytest.fixture
def model_client(make_client):
    models = {"g-omnivoice": ModelDefaults(language="vi"), "omnivoice": ModelDefaults(normalize=False)}
    return make_client(Settings(upstream_url=UPSTREAM_URL, models=models))


@pytest.mark.parametrize(
    ("body", "language"),
    [
        ({"model": "g-omnivoice"}, "vi"),
        ({"model": "g-omnivoice", "language": "English"}, "English"),
        ({"model": "g-omnivoice", "language": "VI_vn"}, "VI_vn"),
        ({"model": "omnivoice"}, None),
        ({"model": "not-configured"}, None),
        ({}, None),
    ],
)
def test_effective_language_from_request_then_model_default(model_client, body, language):
    response = model_client.post(NORMALIZE, json={"input": "Xin chào.", **body})

    assert response.status_code == 200
    assert response.json()["language"] == language


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"input": 5},
        {"input": "x", "language": 1},
        {"input": "x", "model": ["g-omnivoice"]},
        {"input": "x", "text_processing": "off"},
        {"input": "x", "text_processing": {"normalize": "false"}},
    ],
)
def test_normalize_invalid_request(client, body):
    assert_error(client.post(NORMALIZE, json=body), 400, "invalid_request")


def test_normalize_requires_auth_when_key_set(make_client):
    client = make_client(api_key="secret")

    assert_error(client.post(NORMALIZE, json={"input": "x"}), 401, "unauthorized")
    assert client.post(NORMALIZE, json={"input": "x"}, headers={"authorization": "Bearer secret"}).status_code == 200


# --- length limit ------------------------------------------------------------


def test_length_limit_counts_nfc_characters(make_client):
    client = make_client(max_input_chars=6)

    assert client.post(NORMALIZE, json={"input": "Việt n"}).status_code == 200
    # 6 characters after NFC, 8 code points before.
    assert client.post(NORMALIZE, json={"input": "Vie\u0323\u0302t n"}).status_code == 200
    assert_error(client.post(NORMALIZE, json={"input": "Việt na"}), 400, "text_too_long")


# --- speech route ------------------------------------------------------------

SECRET = "Bí mật riêng tư"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (SECRET + " " + "a" * 100, "text_too_long"),
        (SECRET + " <think>", "reasoning_markup"),
        (SECRET + " <|im_start|>", "model_control"),
        ("😀 " * 5, "empty_text"),
    ],
)
def test_speech_input_errors_do_not_call_upstream(make_client, upstream, text, error):
    client = make_client(max_input_chars=100)

    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": text})

    assert_error(response, 400, error)
    assert SECRET not in response.text
    assert upstream.requests == []


@pytest.mark.parametrize(
    "text",
    [
        "# Tiêu đề\n- **Một** mục, xem https://x.vn 😀\n```\ncode\n```",
        "Tie\u0302\u0301ng Vie\u0323\u0302t <b>hay</b>",
        "Xin chào",
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_speech_sends_dry_run_engine_text(client, upstream, text, normalize):
    body = {"model": "g-omnivoice", "input": text, "language": "vi", "text_processing": {"normalize": normalize}}
    dry_run = client.post(NORMALIZE, json=body).json()

    response = client.post(SPEECH, json=body)

    assert response.status_code == 200
    sent = upstream.last_json()
    assert sent["input"] == dry_run["engine_text"]
    assert "text_processing" not in sent
    assert sent["language"] == "vi"
    assert response.headers["x-text-pipeline-version"] == dry_run["pipeline_version"]
    assert response.headers.get("x-text-warnings", "") == ",".join(dry_run["warnings"])


def test_speech_warning_header_once_each_in_detection_order(client):
    text = "😀 <i>a</i> https://x.vn 🎉 <b>b</b> www.y.vn\n```\nx\n```"

    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": text})

    assert response.headers["x-text-warnings"] == "omitted_code_block,omitted_html,omitted_url,omitted_emoji"


def test_speech_no_warning_header_without_warnings(client):
    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": "Xin chào."})

    assert response.status_code == 200
    assert "x-text-warnings" not in response.headers
    assert response.headers["x-text-pipeline-version"]


def test_speech_streaming_response_has_text_headers(client, upstream):
    upstream.stream(b"\x01", 0.01, b"\x02")

    response = client.post(SPEECH, json={"model": "g-omnivoice-stream", "input": "Chào 😀", "stream_format": "audio"})

    assert response.content == b"\x01\x02"
    assert response.headers["x-text-warnings"] == "omitted_emoji"
    assert response.headers["x-text-pipeline-version"]


def test_speech_upstream_error_passes_through_without_text_headers(client, upstream):
    import httpx

    upstream.on("POST", SPEECH, lambda r: httpx.Response(503, json={"error": {"message": "busy", "type": "server_busy"}}))

    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": "Chào 😀"})

    assert response.status_code == 503
    assert "x-text-warnings" not in response.headers
