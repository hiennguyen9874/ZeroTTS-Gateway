"""Text pipeline through the HTTP API.

Goldens go through the dry-run route, then speech-route behaviour.
"""

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
        assert case.get("group"), case["id"]
        assert ("error" in case) != ("expected" in case), case["id"]
        # A known issue records accepted-but-wrong output and must say what is wrong.
        assert case.get("known_issue", "x"), case["id"]


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
    assert result == {
        "engine_text": "xin chào.",
        "warnings": [],
        "language": "vi-VN",
        "pipeline_version": result["pipeline_version"],
    }
    assert isinstance(result["pipeline_version"], str) and result["pipeline_version"]


@pytest.fixture
def model_client(make_client):
    models = {
        "g-omnivoice": ModelDefaults(language="vi"),
        "omnivoice": ModelDefaults(normalize=False),
    }
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


@pytest.mark.parametrize("model", ["omnivoice", "not-configured"])
def test_vietnamese_steps_skipped_without_vietnamese_default(model_client, model):
    response = model_client.post(NORMALIZE, json={"input": "Giá 50.000đ [ghi chú]", "model": model})

    assert response.json()["engine_text"] == "Giá 50.000đ [ghi chú]."


def test_vietnamese_steps_follow_model_default_language(model_client):
    response = model_client.post(NORMALIZE, json={"input": "Giá 50.000đ", "model": "g-omnivoice"})

    assert response.json()["engine_text"] == "giá năm mươi nghìn đồng."


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
    assert (
        client.post(
            NORMALIZE, json={"input": "x"}, headers={"authorization": "Bearer secret"}
        ).status_code
        == 200
    )


# --- pronunciation lexicon --------------------------------------------------


@pytest.fixture
def lexicon_client(make_client, tmp_path):
    def make(rows: str) -> object:
        path = tmp_path / f"lexicon-{len(list(tmp_path.iterdir()))}.csv"
        path.write_text("word,replacement,note\n" + rows, encoding="utf-8")
        return make_client(lexicon_path=str(path))

    return make


def dry_run(client, text: str, **body) -> dict:
    response = client.post(NORMALIZE, json={"input": text, "language": "vi", **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_lexicon_matches_whole_words_only(lexicon_client):
    client = lexicon_client("AI,ây ai,acronym\n")

    assert (
        dry_run(client, "AI giúp MAI và AIR, còn AI-2 thì sao?")["engine_text"]
        == "ây ai giúp mai và a i rờ, còn ây ai hai thì sao?"
    )


def test_lexicon_prefers_longest_entry(lexicon_client):
    client = lexicon_client("AI,ây ai,\nAI Studio,ây ai xờ tiu đi ô,product name\n")

    assert (
        dry_run(client, "Mở AI Studio rồi hỏi AI.")["engine_text"]
        == "mở ây ai xờ tiu đi ô rồi hỏi ây ai."
    )


def test_lexicon_is_case_sensitive(lexicon_client):
    client = lexicon_client("AI,ây ai,\n")

    assert dry_run(client, "Ai biết Ai?")["engine_text"] == "ai biết ai?"


def test_lexicon_runs_before_normalizer_and_never_inside_tags(lexicon_client):
    client = lexicon_client(
        "iPhone,ai phôn,brand\nlaughter,cười,\nTP.HCM,Sài Gòn,regional preference\n"
    )

    result = dry_run(client, "Mua iPhone ở TP.HCM [laughter] nhé")

    assert result["engine_text"] == "mua ai phôn ở sài gòn [laughter] nhé."


@pytest.mark.parametrize(
    ("body", "expected"),
    [({"language": "en"}, "AI."), ({"text_processing": {"normalize": False}}, "AI.")],
)
def test_lexicon_only_for_vietnamese_normalization(lexicon_client, body, expected):
    client = lexicon_client("AI,ây ai,\n")

    assert dry_run(client, "AI", **body)["engine_text"] == expected


def test_pipeline_version_tracks_lexicon(client, lexicon_client):
    base = dry_run(client, "x")["pipeline_version"]
    one = dry_run(lexicon_client("AI,ây ai,\n"), "x")["pipeline_version"]
    same = dry_run(lexicon_client("AI,ây ai,\n"), "x")["pipeline_version"]
    changed = dry_run(lexicon_client("AI,ây ai,\nAPI,ây pi ai,\n"), "x")["pipeline_version"]

    assert one == same
    assert len({base, one, changed}) == 3


def test_speech_reports_lexicon_pipeline_version(lexicon_client):
    client = lexicon_client("AI,ây ai,\n")

    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": "AI", "language": "vi"})

    assert response.headers["x-text-pipeline-version"] == dry_run(client, "x")["pipeline_version"]


@pytest.mark.parametrize(
    "rows",
    [
        "wrong,header\n",
        "word,replacement,note\nAI,,\n",
        "word,replacement,note\nAI,ây ai,\nAI,ây ai,\n",
    ],
)
def test_invalid_lexicon_fails_at_startup(tmp_path, rows):
    from omnivoice_gateway import create_app

    path = tmp_path / "lexicon.csv"
    path.write_text(rows, encoding="utf-8")

    with pytest.raises(ValueError):
        create_app(Settings(upstream_url=UPSTREAM_URL, lexicon_path=str(path)))


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
    body = {
        "model": "g-omnivoice",
        "input": text,
        "language": "vi",
        "text_processing": {"normalize": normalize},
    }
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

    assert (
        response.headers["x-text-warnings"]
        == "omitted_code_block,omitted_html,omitted_url,omitted_emoji"
    )


def test_speech_no_warning_header_without_warnings(client):
    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": "Xin chào."})

    assert response.status_code == 200
    assert "x-text-warnings" not in response.headers
    assert response.headers["x-text-pipeline-version"]


def test_speech_streaming_response_has_text_headers(client, upstream):
    upstream.stream(b"\x01", 0.01, b"\x02")

    response = client.post(
        SPEECH, json={"model": "g-omnivoice-stream", "input": "Chào 😀", "stream_format": "audio"}
    )

    assert response.content == b"\x01\x02"
    assert response.headers["x-text-warnings"] == "omitted_emoji"
    assert response.headers["x-text-pipeline-version"]


def test_speech_upstream_error_passes_through_without_text_headers(client, upstream):
    import httpx

    upstream.on(
        "POST",
        SPEECH,
        lambda r: httpx.Response(503, json={"error": {"message": "busy", "type": "server_busy"}}),
    )

    response = client.post(SPEECH, json={"model": "g-omnivoice", "input": "Chào 😀"})

    assert response.status_code == 503
    assert "x-text-warnings" not in response.headers


# --- reference_text ----------------------------------------------------------


def test_speech_normalizes_reference_text(client, upstream):
    body = {
        "model": "g-omnivoice",
        "input": "Xin chào",
        "language": "vi",
        "reference_text": "Giá 50.000đ 😀",
    }

    response = client.post(SPEECH, json=body)

    assert response.status_code == 200
    sent = upstream.last_json()
    assert sent["reference_text"] == "giá năm mươi nghìn đồng."
    assert sent["input"] == "xin chào."
    # Warnings describe the caller's input only.
    assert "x-text-warnings" not in response.headers


def test_speech_reference_text_follows_normalize_flag(client, upstream):
    body = {
        "model": "g-omnivoice",
        "input": "Xin chào",
        "language": "vi",
        "reference_text": "Giá 50.000đ",
        "text_processing": {"normalize": False},
    }

    client.post(SPEECH, json=body)

    assert upstream.last_json()["reference_text"] == "Giá 50.000đ."


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (SECRET + " <think>", "reasoning_markup"),
        (SECRET + " <|im_end|>", "model_control"),
        ("😀", "empty_text"),
        (SECRET + " " + "a" * 100, "text_too_long"),
    ],
)
def test_speech_reference_text_errors_match_input_errors(make_client, upstream, reference, error):
    client = make_client(max_input_chars=100)

    response = client.post(
        SPEECH, json={"model": "g-omnivoice", "input": "Xin chào", "reference_text": reference}
    )

    assert_error(response, 400, error)
    assert SECRET not in response.text
    assert upstream.requests == []


def test_speech_reference_text_must_be_string(client, upstream):
    response = client.post(
        SPEECH, json={"model": "g-omnivoice", "input": "Xin chào", "reference_text": ["x"]}
    )

    assert_error(response, 400, "invalid_request")
    assert upstream.requests == []
