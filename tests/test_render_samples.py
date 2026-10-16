"""Listening-run script (scripts/render_samples.py) end to end.

The script runs as a subprocess against a live gateway and a live fake upstream (the direct side),
so the observable behaviour is its exit code, console output and output directory.
"""

import base64
import csv
import io
import json
import re
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from .conftest import ScriptedStream

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "render_samples.py"
GOLDEN = json.loads((ROOT / "tests" / "golden" / "pipeline.json").read_text("utf-8"))
AUDIO_CASES = [case for case in GOLDEN if "error" not in case]
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


@pytest.fixture
def run(live_gateway, live_upstream, upstream, tmp_path):
    upstream.on("GET", "/v1/models", lambda r: httpx.Response(200, json=MODELS))
    gateway = live_gateway()

    def render(*args: str, out: Path | None = None) -> subprocess.CompletedProcess:
        out = out or tmp_path / "out"
        command = [sys.executable, str(SCRIPT), "--gateway", gateway, "--upstream", live_upstream]
        return subprocess.run(
            [*command, "--out", str(out), *args], capture_output=True, text=True, timeout=300
        )

    render.gateway = gateway
    return render


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

    result = run("--model", model, "--voice", "demo_nu", "--seed", "7")

    assert result.returncode == 0, result.stderr
    rows = read_rows(out)
    assert [(r["case_id"], r["side"]) for r in rows] == [
        (case["id"], side) for case in AUDIO_CASES for side in ("gateway", "direct")
    ]
    warnings = {case["id"]: ",".join(case["warnings"]) for case in AUDIO_CASES}
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
        with wave.open(str(out / row["file"])) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
            assert w.readframes(w.getnframes()) == audio
    assert not list(out.glob("*.part"))

    # One upstream request per (case, side), same voice and seed, the format fitting the mode.
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    assert len(sent) == 2 * len(AUDIO_CASES)
    for body in sent:
        assert (body["model"], body["voice"], body["seed"]) == (model, "demo_nu", 7)
        if streaming:
            assert (body["response_format"], body["stream_format"]) == ("pcm", "audio")
        else:
            assert body["response_format"] == "wav"
            assert "stream_format" not in body

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    version = httpx.post(run.gateway + "/v1/text/normalize", json={"input": "x"}).json()
    assert manifest["pipeline_version"] == version["pipeline_version"]
    assert (manifest["model"], manifest["voice"], manifest["seed"]) == (model, "demo_nu", 7)
    assert manifest["mode"] == ("streaming" if streaming else "offline")
    assert manifest["cases"] == len(AUDIO_CASES)
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

    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["case_set"] == "listening"
    assert manifest["questions"] == {
        str(q): [case["id"] for case in LISTENING if q in case["questions"]] for q in range(1, 6)
    }
    assert manifest["groups"]["clone"] == ["clone-plain", "clone-numbers"]

    # Clone: same reference audio, seed and model input on both sides; only reference_text differs.
    voice_wav = base64.b64encode((ROOT / "voices" / "demo_nu.wav").read_bytes()).decode()
    transcript = "Xin chào các bạn, hôm nay trời Hà Nội rất đẹp."
    sent = [json.loads(r.content) for r in upstream.calls("POST", SPEECH)]
    clones = [body for body in sent if "voice_ref" in body]
    assert len(clones) == 4  # two clone cases, gateway then direct
    engine = {r["case_id"]: r["engine_text"] for r in rows}
    for (via_gateway, direct), case_id in zip(
        (clones[0:2], clones[2:4]), ("clone-plain", "clone-numbers"), strict=True
    ):
        for body in (via_gateway, direct):
            assert body["voice_ref"] == {"type": "base64", "data": voice_wav}
            assert "voice" not in body
            assert (body["seed"], body["input"]) == (3, engine[case_id])
        assert direct["reference_text"].startswith(transcript)
        assert via_gateway["reference_text"].startswith(transcript.lower())
    assert all(body["voice"] == "demo_nu" for body in sent if "voice_ref" not in body)

    page = (out / "index.html").read_text("utf-8")
    assert "reference_text (normalized): xin chào các bạn" in page
    assert "reference_text (verbatim): Xin chào các bạn" in page
    assert "Q5" in page


def test_filters_listening_set_by_question(run, upstream, tmp_path):
    upstream.on("POST", SPEECH, speaker(lambda body: [pcm((0.5, 1000))]))

    result = run("--set", "listening", "--question", "3", "--question", "5", "--no-direct")

    assert result.returncode == 0, result.stderr
    assert [r["case_id"] for r in read_rows(tmp_path / "out")] == [
        case["id"] for case in LISTENING if {3, 5} & set(case["questions"])
    ]
