"""Listening-run script (scripts/render_samples.py) end to end.

The script runs as a subprocess against a live gateway and a live fake upstream (the direct side),
so the observable behaviour is its exit code, console output and output directory.
"""

import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from zerotts_gateway.config import ModelDefaults, ShortTextSpeedBand

from .conftest import ScriptedStream

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "render_samples.py"
GOLDEN = json.loads((ROOT / "tests" / "golden" / "pipeline.json").read_text("utf-8"))
# The tests render OmniVoice models, so goldens bound to another model are left out.
AUDIO_CASES = [case for case in GOLDEN if "error" not in case and "model" not in case]
LISTENING = json.loads((ROOT / "scripts" / "listening_cases.json").read_text("utf-8"))
SPEECH = "/v1/audio/speech"
RATE = 24000
MODELS = {
    "object": "list",
    "data": [
        {"id": "g-omnivoice", "object": "model", "task": "tts", "mode": "offline"},
        {"id": "g-omnivoice-stream", "object": "model", "task": "tts", "mode": "streaming"},
    ],
}


def pcm(*parts: tuple[float, int]) -> bytes:
    """s16le mono at RATE: (seconds, level) parts; a non-zero level is a square wave of +/-level."""
    out = bytearray()
    for seconds, level in parts:
        for i in range(round(seconds * RATE)):
            out += (level if i % 2 == 0 else -level).to_bytes(2, "little", signed=True)
    return bytes(out)


def wav(samples: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples)
    return buf.getvalue()


def speaker(choose):
    """Speech handler: `choose(body)` gives the audio steps; streamed PCM or else one WAV."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        steps = choose(body)
        if body.get("stream_format") == "audio":
            return httpx.Response(
                200, headers={"content-type": "audio/pcm"}, stream=ScriptedStream(*steps)
            )
        return httpx.Response(
            200, content=wav(b"".join(steps)), headers={"content-type": "audio/wav"}
        )

    return handle


def odd_chunks(data: bytes, size: int = 4097) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def script_env(**extra: str) -> dict:
    """The test process env without an operator's ASR settings, plus extra."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNIVOICE_ASR_")}
    return env | extra


@pytest.fixture
def run(live_gateway, live_upstream, upstream, tmp_path):
    upstream.on("GET", "/v1/models", lambda r: httpx.Response(200, json=MODELS))
    gateway = live_gateway()

    def render(
        *args: str, out: Path | None = None, env: dict | None = None
    ) -> subprocess.CompletedProcess:
        out = out or tmp_path / "out"
        command = [sys.executable, str(SCRIPT), "--gateway", gateway, "--upstream", live_upstream]
        return subprocess.run(
            [*command, "--out", str(out), *args],
            capture_output=True,
            text=True,
            timeout=300,
            env=script_env(**(env or {})),
        )

    render.gateway = gateway
    return render


class FakeAsr:
    """OpenAI-style transcription server. `answer(request)` gives the transcript, or None for a
    500. Each request is recorded as a dict: form fields, filename, size, authorization."""

    def __init__(self):
        self.requests: list[dict] = []
        self.answer = lambda request: ""

    async def __call__(self, scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers = dict(scope["headers"])
        fields = {
            name.decode(): value.decode()
            for name, value in re.findall(rb'name="(\w+)"\r\n\r\n(.*?)\r\n', body)
        }
        filename = re.search(rb'filename="([^"]+)"', body)
        request = {
            **fields,
            "path": scope["path"],
            "filename": filename.group(1).decode() if filename else None,
            "size": len(body),
            "authorization": headers.get(b"authorization", b"").decode(),
        }
        self.requests.append(request)
        text = self.answer(request)
        status, payload = (
            (200, {"text": text})
            if text is not None
            else (500, {"error": {"message": "boom", "type": "server_error"}})
        )
        data = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": data})


@pytest.fixture
def asr(serve):
    fake = FakeAsr()
    fake.url = serve(fake, lifespan="off")
    return fake


def read_rows(out: Path) -> list[dict]:
    with (out / "results.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize(
    ("model", "streaming"), [("g-omnivoice", False), ("g-omnivoice-stream", True)]
)
def test_renders_golden_set_for_listening(run, upstream, tmp_path, model, streaming):
    audio = pcm((0.2, 0), (1.0, 16384), (0.2, 0))
    upstream.on("POST", SPEECH, speaker(lambda body: odd_chunks(audio)))
    out = tmp_path / "out"

    result = run("--model", model, "--voice", "vieneu_v3_Mai_Anh", "--seed", "7")

    assert result.returncode == 0, result.stderr
    rows = read_rows(out)
    assert [(r["case_id"], r["side"]) for r in rows] == [
        (case["id"], side) for case in AUDIO_CASES for side in ("gateway", "direct")
    ]
    warnings = {case["id"]: ",".join(case["warnings"]) for case in AUDIO_CASES}
    by_id = {case["id"]: case for case in AUDIO_CASES}
    for row in rows:
        assert row["result"] == "audio", row
        assert row["review_result"] == row["reviewer"] == row["notes"] == ""
        assert float(row["duration_s"]) == 1.4
        # Square wave at half scale for 1.0 of 1.4 s: -6.0 dBFS peak, -6.0 + 10*log10(1/1.4) rms.
        assert (row["peak_dbfs"], row["rms_dbfs"]) == ("-6.0", "-7.5")
        assert (row["lead_silence_ms"], row["trail_silence_ms"]) == ("200", "200")
        assert (row["clipped_samples"], row["flags"]) == ("0", "")
        assert 0 <= float(row["ttfb_ms"]) <= float(row["total_ms"])
        assert float(row["rtf"]) > 0
        expected = warnings[row["case_id"]] if row["side"] == "gateway" else ""
        assert row["text_warnings"] == expected, row["case_id"]
        # Each side records the text its request carried: engine text via the gateway, the
        # input text sent straight to the upstream.
        sent = row["engine_text"] if row["side"] == "gateway" else by_id[row["case_id"]]["input"]
        assert row["sent_text"] == sent, row["case_id"]
        assert row["normalize"] == (
            "false" if by_id[row["case_id"]].get("normalize") is False else ""
        )
        # No ASR configured: the run is as before, ASR columns blank.
        assert row["asr_text"] == row["word_recall"] == ""
        assert "words_missing" not in row["flags"]
        with wave.open(str(out / row["file"])) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
            assert w.readframes(w.getnframes()) == audio
    assert not list(out.glob("*.part"))

    # One upstream request per (case, side), same voice and seed, the format fitting the mode.
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    assert len(sent) == 2 * len(AUDIO_CASES)
    for body in sent:
        assert (body["model"], body["voice"], body["seed"]) == (model, "vieneu_v3_Mai_Anh", 7)
        if streaming:
            assert (body["response_format"], body["stream_format"]) == ("pcm", "audio")
        else:
            assert body["response_format"] == "wav"
            assert "stream_format" not in body

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    version = httpx.post(run.gateway + "/v1/text/normalize", json={"input": "x"}).json()
    assert manifest["pipeline_version"] == version["pipeline_version"]
    assert (manifest["model"], manifest["voice"], manifest["seed"]) == (
        model,
        "vieneu_v3_Mai_Anh",
        7,
    )
    assert manifest["mode"] == ("streaming" if streaming else "offline")
    assert manifest["cases"] == len(AUDIO_CASES)
    assert manifest["asr"] is None
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", manifest["git_commit"])
    datetime.fromisoformat(manifest["started_utc"])

    page = (out / "index.html").read_text("utf-8")
    for case in AUDIO_CASES:
        assert f'src="{case["id"]}.gateway.wav"' in page
        assert f'src="{case["id"]}.direct.wav"' in page
    assert "ttfb" in page

    # Console carries case ids and metrics only, never text.
    console = result.stdout + result.stderr
    assert "plain-vi" in console
    for case in AUDIO_CASES:
        if len(case["input"].strip()) >= 8:
            assert case["input"].strip() not in console, case["id"]


def test_flags_hint_at_suspicious_audio(run, upstream, tmp_path):
    silent = pcm((0.1, 0))
    clipped = pcm((1.5, 0), (0.05, 32767), (2.0, 0))
    upstream.on(
        "POST", SPEECH, speaker(lambda body: [silent if "Xin chào" in body["input"] else clipped])
    )

    result = run("--id", "plain-vi", "--id", "nfc")

    assert result.returncode == 0, result.stderr
    flags = {(r["case_id"], r["side"]): r["flags"] for r in read_rows(tmp_path / "out")}
    assert flags == {
        ("plain-vi", "gateway"): "silent,very_short",
        ("plain-vi", "direct"): "silent,very_short",
        ("nfc", "gateway"): "clipping,long_lead_silence,long_trail_silence",
        ("nfc", "direct"): "clipping,long_lead_silence,long_trail_silence",
    }
    page = (tmp_path / "out" / "index.html").read_text("utf-8")
    assert "long_trail_silence" in page


def test_stream_broken_mid_way_is_recorded_without_a_truncated_file(run, upstream, tmp_path):
    upstream.on(
        "POST",
        SPEECH,
        speaker(lambda body: [pcm((0.5, 1000)), httpx.ReadError("upstream crashed")]),
    )
    out = tmp_path / "out"

    result = run("--model", "g-omnivoice-stream", "--id", "plain-vi")

    assert result.returncode == 1
    rows = read_rows(out)
    assert [(r["side"], r["result"], r["file"]) for r in rows] == [
        ("gateway", "stream_aborted", ""),
        ("direct", "stream_aborted", ""),
    ]
    assert not list(out.glob("*.wav*"))
    assert len(upstream.calls("POST", SPEECH)) == 2  # never retried
    assert "stream_aborted" in (out / "index.html").read_text("utf-8")
    assert (out / "manifest.json").exists()


def test_refuses_output_dir_with_data_unless_overwrite(run, upstream, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    (out / "stale.gateway.wav").write_bytes(b"old")
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))

    refused = run("--id", "plain-vi")

    assert refused.returncode != 0
    assert "--overwrite" in refused.stderr
    assert not upstream.calls("POST", SPEECH)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt", "stale.gateway.wav"]

    result = run("--id", "plain-vi", "--overwrite")

    assert result.returncode == 0, result.stderr
    assert (out / "keep.txt").read_text() == "mine"
    assert not (out / "stale.gateway.wav").exists()
    assert (out / "plain-vi.gateway.wav").exists()


def test_listening_set_covers_the_five_questions():
    ids = [case["id"] for case in LISTENING]
    assert len(ids) == len(set(ids))
    for case in LISTENING:
        assert case["input"].strip() and case["why"].strip() and case["review_focus"].strip()
        assert set(case["questions"]) <= {1, 2, 3, 4, 5}, case["id"]
    groups = {case["group"] for case in LISTENING}
    assert {"hard", "abbrev-case", "long-text", "tags", "clone"} <= groups
    assert {q for case in LISTENING for q in case["questions"]} == {1, 2, 3, 4, 5}
    # Long enough for at least 30 s of audio and several streaming chunks.
    for case in LISTENING:
        if case["group"] == "long-text":
            assert 600 <= len(case["input"]) <= 800, case["id"]
    transcripts = (ROOT / "voices" / "prompt_text").read_text("utf-8")
    for case in LISTENING:
        if "clone" in case:
            assert (ROOT / "voices" / f"{case['clone']}.wav").exists()
            assert f"{case['clone']}|" in transcripts


def test_renders_listening_set_with_questions_and_clone_sides(run, upstream, tmp_path):
    audio = pcm((0.1, 0), (0.5, 8000), (0.1, 0))
    upstream.on("POST", SPEECH, speaker(lambda body: [audio]))
    out = tmp_path / "out"

    result = run("--set", "listening", "--seed", "3")

    assert result.returncode == 0, result.stderr
    rows = read_rows(out)
    assert [(r["case_id"], r["side"]) for r in rows] == [
        (case["id"], side) for case in LISTENING for side in ("gateway", "direct")
    ]
    by_id = {case["id"]: case for case in LISTENING}
    for row in rows:
        case = by_id[row["case_id"]]
        assert row["result"] == "audio", row
        assert row["questions"] == ",".join(map(str, case["questions"]))
        assert row["review_focus"] == case["review_focus"]
        expected = (
            ("normalized" if row["side"] == "gateway" else "verbatim") if "clone" in case else ""
        )
        assert row["reference_text"] == expected, row["case_id"]
        # Clone direct side sends the gateway's engine text, so both sides sent the same input.
        sent = row["engine_text"] if row["side"] == "gateway" or "clone" in case else case["input"]
        assert row["sent_text"] == sent, row["case_id"]

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["case_set"] == "listening"
    assert manifest["questions"] == {
        str(q): [case["id"] for case in LISTENING if q in case["questions"]] for q in range(1, 6)
    }
    assert manifest["groups"]["clone"] == [c["id"] for c in LISTENING if "clone" in c]

    # Clone: same reference audio, seed and model input on both sides; only reference_text differs.
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    clones = [body for body in sent if "voice_ref" in body]
    clone_cases = [c for c in LISTENING if "clone" in c]
    assert len(clones) == 2 * len(clone_cases)
    engine = {r["case_id"]: r["engine_text"] for r in rows}
    for i, case in enumerate(clone_cases):
        via_gateway, direct = clones[2 * i : 2 * i + 2]
        voice_wav = base64.b64encode(
            (ROOT / "voices" / f"{case['clone']}.wav").read_bytes()
        ).decode()
        for body in (via_gateway, direct):
            assert body["voice_ref"] == {"type": "base64", "data": voice_wav}
            assert "voice" not in body
            assert (body["seed"], body["input"], body["speed"]) == (3, engine[case["id"]], 1.0)
        normalized = httpx.post(
            run.gateway + "/v1/text/normalize",
            json={"input": direct["reference_text"], "language": "vi"},
        ).json()["engine_text"]
        assert via_gateway["reference_text"] == normalized
    for case in LISTENING:
        if "speed" in case:
            pair = [b for b, r in zip(sent, rows, strict=True) if r["case_id"] == case["id"]]
            assert [b["speed"] for b in pair] == [case["speed"], case["speed"]]
            assert all(
                r["requested_speed"] == str(case["speed"])
                for r in rows
                if r["case_id"] == case["id"]
            )
    assert all(body["voice"] == "vieneu_v3_Mai_Anh" for body in sent if "voice_ref" not in body)

    page = (out / "index.html").read_text("utf-8")
    assert "reference_text (normalized): chào mừng đến với" in page
    assert "reference_text (verbatim): Chào mừng đến với" in page
    assert "Q5" in page


def test_clone_and_case_controls_override_operator_speed_bands(
    live_gateway, live_upstream, upstream, tmp_path
):
    upstream.on("GET", "/v1/models", lambda r: httpx.Response(200, json=MODELS))
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))
    gateway = live_gateway(
        models={
            "g-omnivoice": ModelDefaults(
                language="vi", short_text_speed_bands=(ShortTextSpeedBand(12, 0.6),)
            )
        }
    )
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--gateway", gateway, "--upstream", live_upstream,
            "--set", "listening", "--id", "clone-doan-trang", "--id", "tag-case-upper",
            "--id", "tag-case-lower", "--out", str(tmp_path),
        ],
        capture_output=True, text=True, timeout=300, env=script_env(),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    assert len(sent) == 6
    assert all(b["speed"] == 1.0 for b in sent)
    assert sent[0]["input"] == sent[1]["input"] == "Xin chào các bạn [laughter]."
    assert sent[2]["input"] == sent[3]["input"] == "xin chào các bạn [laughter]."
    assert sent[4]["input"] == sent[5]["input"]
    assert sent[4]["voice_ref"] == sent[5]["voice_ref"]
    assert all(r["requested_speed"] == "1.0" for r in read_rows(tmp_path))
    assert "controlled speed: 1.0 (both sides)" in (tmp_path / "index.html").read_text("utf-8")


def test_filters_listening_set_by_question(run, upstream, tmp_path):
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))

    result = run("--set", "listening", "--question", "3", "--question", "5", "--no-direct")

    assert result.returncode == 0, result.stderr
    assert [r["case_id"] for r in read_rows(tmp_path / "out")] == [
        case["id"] for case in LISTENING if {3, 5} & set(case["questions"])
    ]


def test_render_all_runs_every_voice_once_and_clone_cases_once_per_model(
    live_gateway, live_upstream, upstream, tmp_path
):
    upstream.on("GET", "/v1/models", lambda r: httpx.Response(200, json=MODELS))
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))
    gateway = live_gateway()
    out = tmp_path / "all"
    voices = ["vieneu_v3_Mai_Anh", "vieneu_v3_Duc_Tri"]
    command = [
        sys.executable, str(ROOT / "scripts" / "render_all.py"), "--gateway", gateway,
        "--upstream", live_upstream, "--set", "listening", "--model", "g-omnivoice",
        "--no-direct", "--out", str(out), *(a for v in voices for a in ("--voice", v)),
    ]  # fmt: skip

    result = subprocess.run(command, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    clone_ids = [c["id"] for c in LISTENING if "clone" in c]
    voiced_ids = [c["id"] for c in LISTENING if "clone" not in c]
    for voice in voices:
        rows = read_rows(out / "listening" / "g-omnivoice" / voice)
        assert [r["case_id"] for r in rows] == voiced_ids
    assert [r["case_id"] for r in read_rows(out / "listening" / "g-omnivoice" / "_clone")] == (
        clone_ids
    )
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    assert len(sent) == 2 * len(voiced_ids) + len(clone_ids)
    with (out / "all_results.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(sent)
    assert (out / "summary.csv").exists() and (out / "index.html").exists()

    # Rerun resumes: finished runs are skipped, nothing new reaches the upstream.
    again = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stderr
    assert len(upstream.calls("POST", SPEECH)) == len(sent)


def test_asr_scores_both_sides_against_the_gateway_engine_text(run, upstream, asr, tmp_path):
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))
    asr.answer = lambda request: {
        "tag-middle.gateway.wav": "TÔI NÓI THẬT ĐẤY NHƯNG KHÔNG AI TIN",  # no tag/punctuation
        "tag-middle.direct.wav": "tôi nói thật đấy",  # second half lost
        "abbr-vi-admin.gateway.wav": None,  # ASR server error
        "abbr-vi-admin.direct.wav": "",
    }[request["filename"]]
    out = tmp_path / "out"
    env = {
        "OMNIVOICE_ASR_URL": asr.url,
        "OMNIVOICE_ASR_MODEL": "gipformer-test",
        "OMNIVOICE_ASR_API_KEY": "asr-key",
    }

    result = run("--set", "listening", "--id", "tag-middle", "--id", "abbr-vi-admin", env=env)

    # An ASR failure is a missing score, not a failed render.
    assert result.returncode == 0, result.stderr
    rows = {(r["case_id"], r["side"]): r for r in read_rows(out)}
    scores = {key: (r["asr_text"], r["word_recall"], r["flags"]) for key, r in rows.items()}
    assert scores == {
        ("tag-middle", "gateway"): ("TÔI NÓI THẬT ĐẤY NHƯNG KHÔNG AI TIN", "1.0", ""),
        ("tag-middle", "direct"): ("tôi nói thật đấy", "0.5", "words_missing"),
        ("abbr-vi-admin", "gateway"): ("", "", "asr_failed"),
        ("abbr-vi-admin", "direct"): ("", "0.0", "words_missing"),
    }
    # Both sides are scored against the gateway's engine text; the direct side sent raw input.
    middle = next(c for c in LISTENING if c["id"] == "tag-middle")
    assert "[sigh]" in rows[("tag-middle", "gateway")]["engine_text"]
    assert rows[("tag-middle", "direct")]["sent_text"] == middle["input"]

    assert [r["filename"] for r in asr.requests] == [
        "abbr-vi-admin.gateway.wav", "abbr-vi-admin.direct.wav",
        "tag-middle.gateway.wav", "tag-middle.direct.wav",
    ]  # fmt: skip
    for request in asr.requests:
        assert request["path"] == "/v1/audio/transcriptions"
        assert (request["model"], request["language"], request["response_format"]) == (
            "gipformer-test",
            "vi",
            "json",
        )
        assert request["authorization"] == "Bearer asr-key"

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["asr"] == {"url": asr.url, "model": "gipformer-test"}
    assert (manifest["counts"]["words_missing"], manifest["counts"]["asr_failed"]) == (2, 1)
    page = (out / "index.html").read_text("utf-8")
    assert "TÔI NÓI THẬT ĐẤY NHƯNG KHÔNG AI TIN" in page
    assert "word_recall 0.5" in page and "ASR failed" in page
    console = result.stdout + result.stderr
    assert "recall=0.5" in console and "words_missing" in console
    assert "TÔI NÓI" not in console and "tôi nói" not in console


def test_normalize_false_cases_are_marked_and_can_render_as_production(run, upstream, tmp_path):
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))
    case = next(c for c in AUDIO_CASES if c["id"] == "email-kept")
    out = tmp_path / "out"

    result = run("--id", "email-kept", "--no-direct")

    assert result.returncode == 0, result.stderr
    [row] = read_rows(out)
    assert (row["normalize"], row["engine_text"]) == ("false", case["expected"])
    assert "normalize: false" in (out / "index.html").read_text("utf-8")

    result = run("--id", "email-kept", "--no-direct", "--normalize-on", "--overwrite")

    assert result.returncode == 0, result.stderr
    [row] = read_rows(out)
    assert row["normalize"] == ""
    assert "@" not in row["engine_text"]  # normalized as in production
    assert upstream.last_json()["input"] == row["engine_text"] == row["sent_text"]
    assert json.loads((out / "manifest.json").read_text("utf-8"))["normalize_on"] is True
    assert "rendered with normalize on" in (out / "index.html").read_text("utf-8")


def test_render_all_summarizes_missing_words_by_voice_and_case(
    live_gateway, live_upstream, upstream, asr, tmp_path
):
    # Voices are told apart at the ASR by audio length: Duc_Tri gets longer audio.
    upstream.on("GET", "/v1/models", lambda r: httpx.Response(200, json=MODELS))
    upstream.on(
        "POST",
        SPEECH,
        speaker(
            lambda body: [pcm((1.0 if body.get("voice") == "vieneu_v3_Duc_Tri" else 0.5, 1000))]
        ),
    )
    gateway = live_gateway()
    engine = {
        c["id"]: httpx.post(
            gateway + "/v1/text/normalize",
            json={"input": c["input"], "language": c.get("language", "vi")}
            | ({"text_processing": {"normalize": c["normalize"]}} if "normalize" in c else {}),
        ).json()["engine_text"]
        for c in LISTENING
    }

    def answer(request: dict) -> str:
        case_id = request["filename"].split(".")[0]
        duc_tri = request["size"] > 36000
        if case_id == "hard-short-a" or (case_id == "tag-control" and duc_tri):
            return ""
        return engine[case_id]

    asr.answer = answer
    out = tmp_path / "all"
    command = [
        sys.executable, str(ROOT / "scripts" / "render_all.py"), "--gateway", gateway,
        "--upstream", live_upstream, "--set", "listening", "--model", "g-omnivoice",
        "--no-direct", "--out", str(out), "--voice", "vieneu_v3_Mai_Anh",
        "--voice", "vieneu_v3_Duc_Tri",
    ]  # fmt: skip

    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300,
        env=script_env(OMNIVOICE_ASR_URL=asr.url),
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    with (out / "voices.csv").open(encoding="utf-8", newline="") as f:
        voices = {r["voice"]: r for r in csv.DictReader(f)}
    voiced = sum("clone" not in c for c in LISTENING)
    assert {v: (r["asr_scored"], r["words_missing"]) for v, r in voices.items()} == {
        "vieneu_v3_Mai_Anh": (str(voiced), "1"),
        "vieneu_v3_Duc_Tri": (str(voiced), "2"),
        "_clone": (str(len(LISTENING) - voiced), "0"),
    }
    assert float(voices["vieneu_v3_Mai_Anh"]["mean_word_recall"]) > float(
        voices["vieneu_v3_Duc_Tri"]["mean_word_recall"]
    )
    with (out / "cases.csv").open(encoding="utf-8", newline="") as f:
        cases = {r["case_id"]: r for r in csv.DictReader(f)}
    assert (
        cases["hard-short-a"]["words_missing_voices"],
        cases["hard-short-a"]["all_voices_missing"],
    ) == ("2", "True")
    assert (
        cases["tag-control"]["words_missing_voices"],
        cases["tag-control"]["all_voices_missing"],
    ) == ("1", "False")
    assert cases["tag-middle"]["words_missing_voices"] == "0"

    console = result.stdout
    by_voice = console[console.index("by voice") : console.index("by case")]
    assert by_voice.index("vieneu_v3_Mai_Anh") < by_voice.index("vieneu_v3_Duc_Tri")
    assert re.search(r"hard-short-a\s+2/2 ALL", console)
    assert re.search(r"tag-control\s+1/2\n", console)
    page = (out / "index.html").read_text("utf-8")
    assert "By voice" in page and "By case" in page
