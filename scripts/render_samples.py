#!/usr/bin/env python3
"""Render cases for a listening run: gateway (A) vs audio.cpp direct (B), with metrics.

Needs a running audio.cpp server and a gateway in front of it (docs/LISTENING.md, docs/USAGE.md):

  uv run python scripts/render_samples.py --out outputs/listen-001
  uv run python scripts/render_samples.py --model g-omnivoice-stream --group numbers-money
  uv run python scripts/render_samples.py --no-direct --id plain-vi --overwrite
  uv run python scripts/render_samples.py --set listening --question 3 --out outputs/q3
  # listen: python -m http.server -d outputs/listen-001 9000

Case sets: "golden" (default) is the regression golden set minus error cases; "listening"
(scripts/listening_cases.json) is the listening-only set for the open questions of issue #7.
Same model, voice and seed for both sides, so differences come from the text pipeline. Clone cases
("clone": voice) send that voice's WAV from voices/ as base64 with its transcript from
voices/prompt_text as reference_text: the gateway side normalizes it, the direct side sends it
verbatim with the gateway's engine text as input, so only reference_text differs. Output:

  <case>.<side>.wav   PCM s16le mono, written via a temp file then renamed
  results.csv         one row per (case, side): metrics, flags, text warnings, blank review columns
  manifest.json       git commit, pipeline version, model, voice, seed, run time, counts
  index.html          players side by side with metrics and flags

Streaming models (mode "streaming" in /v1/models) are called with stream_format=audio and
response_format=pcm so TTFB is real; offline models return WAV. A request is never retried once
it is sent; a stream that breaks after audio arrived is recorded as stream_aborted and no file is
kept. Flags are hints for the listener, not a quality verdict. The console prints case ids and
metrics only, never text. Exit status is 1 when any request did not produce audio.

ASR scoring (optional): with an OpenAI-style transcription endpoint configured (OMNIVOICE_ASR_URL,
OMNIVOICE_ASR_MODEL, OMNIVOICE_ASR_API_KEY or the --asr-* options), every WAV is transcribed via
POST <url>/v1/audio/transcriptions. results.csv gets asr_text and word_recall: the share of the
gateway's engine text words heard in order in the transcript (longest common subsequence), with
non-verbal tags, CMU spans and punctuation ignored, for both sides since the engine text is the
intended reading. word_recall < 0.8 adds the flag words_missing. scripts/asr_server.py serves
gipformer this way. Without ASR the run is unchanged and those columns stay blank.
"""

from __future__ import annotations

import argparse
import array
import base64
import csv
import functools
import html
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CASE_SETS = {
    "golden": ROOT / "tests" / "golden" / "pipeline.json",
    "listening": ROOT / "scripts" / "listening_cases.json",
}
VOICES = ROOT / "voices"

SILENCE_LEVEL = 328  # ~-40 dBFS on s16
CLIP_LEVEL = 32767
RECALL_THRESHOLD = 0.8
# Not words to hear: non-verbal tags ("[laughter]") and CMU pronunciation spans ("[B EY1 S]").
UNSPOKEN = re.compile(r"\[[a-z]+(?:-[a-z]+)?\]|\[[A-Z]+[0-2]?(?: [A-Z]+[0-2]?)*\]")
WORD = re.compile(r"[^\W_]+")
REPORTS = ("results.csv", "manifest.json", "index.html")
CSV_FIELDS = [
    "case_id", "group", "questions", "side", "normalize", "reference_text", "result",
    "requested_speed", "http_status", "file", "duration_s", "ttfb_ms", "total_ms", "rtf",
    "peak_dbfs", "rms_dbfs",
    "lead_silence_ms", "trail_silence_ms", "clipped_samples", "flags", "word_recall",
    "text_warnings", "known_issue", "review_focus", "input", "engine_text", "sent_text",
    "asr_text", "review_result", "reviewer", "notes",
]  # fmt: skip


@dataclass
class Sample:
    case: dict
    side: str
    engine_text: str  # the gateway's engine text, the intended reading for both sides
    sent_text: str = ""  # the input text this side's request carried to the model
    normalize: bool | None = None  # text_processing.normalize sent to the gateway, if any
    reference_text: str = ""  # clone cases: the reference_text the model received
    result: str = "audio"  # "audio" or a failure code
    http_status: int | None = None
    file: str = ""
    ttfb_ms: float | None = None
    total_ms: float | None = None
    warnings: str = ""
    metrics: dict = field(default_factory=dict)
    asr_text: str | None = None  # None: not transcribed
    asr_error: str = ""
    word_recall: float | None = None

    @property
    def flags(self) -> str:
        flags = [f for f in self.metrics.get("flags", "").split(",") if f]
        if self.asr_error:
            flags.append("asr_failed")
        elif self.word_recall is not None and self.word_recall < RECALL_THRESHOLD:
            flags.append("words_missing")
        return ",".join(flags)


def main() -> int:
    args = parse_args()
    cases = [c for c in json.loads(CASE_SETS[args.case_set].read_text("utf-8")) if "error" not in c]
    if args.group:
        cases = [c for c in cases if c.get("group") in args.group]
    if args.id:
        cases = [c for c in cases if c["id"] in args.id]
    if args.question:
        cases = [c for c in cases if set(c.get("questions", ())) & set(args.question)]
    if not cases:
        sys.exit(f"no matching {args.case_set} cases")
    prepare_out(args.out, args.overwrite)

    timeout = httpx.Timeout(args.timeout, connect=10)
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    gateway = httpx.Client(base_url=args.gateway, headers=headers, timeout=timeout)
    upstream = None if args.no_direct else httpx.Client(base_url=args.upstream, timeout=timeout)
    asr = None
    if args.asr_url:
        asr_headers = {"authorization": f"Bearer {args.asr_api_key}"} if args.asr_api_key else {}
        asr = httpx.Client(base_url=args.asr_url, headers=asr_headers, timeout=timeout)
    models = gateway.get("/v1/models").raise_for_status().json()["data"]
    mode = next((m.get("mode") for m in models if m["id"] == args.model), None)
    if mode is None:
        sys.exit(f"model {args.model!r} is not in the upstream's /v1/models")
    streaming = mode == "streaming"
    version = (
        gateway.post("/v1/text/normalize", json={"input": "x"})
        .raise_for_status()
        .json()["pipeline_version"]
    )
    started = datetime.now(UTC).isoformat()

    sides = [("gateway", gateway)] + ([("direct", upstream)] if upstream else [])
    samples: list[Sample] = []
    for i, case in enumerate(cases, 1):
        body = {
            "model": args.model,
            "input": case["input"],
            "seed": args.seed,
            "language": case.get("language", "vi"),
        }
        # Experimental controls must be identical on both sides. In clone pairs, an explicit
        # speed disables gateway speed bands so only reference_text normalization differs.
        if "speed" in case or "clone" in case:
            body["speed"] = case.get("speed", 1.0)
        if "clone" in case:
            audio_b64, reference = clone_reference(case["clone"])
            body |= {
                "voice_ref": {"type": "base64", "data": audio_b64},
                "reference_text": reference,
            }
        else:
            body["voice"] = args.voice
        body |= (
            {"response_format": "pcm", "stream_format": "audio"}
            if streaming
            else {"response_format": "wav"}
        )
        normalize_flag = None if args.normalize_on else case.get("normalize")
        processing = (
            {"text_processing": {"normalize": normalize_flag}} if normalize_flag is not None else {}
        )
        engine = normalize(gateway, {**body, **processing})
        for side, client in sides:
            reference = ""
            if side == "gateway":
                payload = {**body, **processing}
                if "clone" in case:
                    reference = normalize(gateway, {**payload, "input": body["reference_text"]})
            elif "clone" in case:
                # Same model input on both sides, so the only difference is reference_text.
                payload = {**body, "input": engine}
                reference = body["reference_text"]
            else:
                payload = body
            # The gateway side's model input is the engine text; the direct side's is its input.
            sent = engine if side == "gateway" else payload["input"]
            sample = Sample(case, side, engine, sent, normalize_flag, reference)
            audio = synthesize(client, payload, sample, streaming, args.sample_rate)
            if audio:
                sample.file = f"{case['id']}.{side}.wav"
                write_wav(args.out / sample.file, *audio)
                if asr:
                    transcribe(asr, args.asr_model, args.out / sample.file, body, sample)
            samples.append(sample)
            print(f"[{i}/{len(cases)}] {case['id']} {side} {summary(sample)}", flush=True)

    counts = {"requests": len(samples)}
    for sample in samples:
        counts[sample.result] = counts.get(sample.result, 0) + 1
    counts["flagged"] = sum(bool(s.flags) for s in samples)
    if asr:
        counts["words_missing"] = sum("words_missing" in s.flags for s in samples)
        counts["asr_failed"] = sum(bool(s.asr_error) for s in samples)
    groups: dict[str, list[str]] = {}
    questions: dict[str, list[str]] = {}
    for case in cases:
        groups.setdefault(case.get("group", ""), []).append(case["id"])
        for q in case.get("questions", ()):
            questions.setdefault(str(q), []).append(case["id"])
    manifest = {
        "case_set": args.case_set,
        "git_commit": git_commit(),
        "pipeline_version": version,
        "model": args.model,
        "mode": mode,
        "voice": args.voice,
        "seed": args.seed,
        "normalize_on": args.normalize_on,
        "asr": {"url": args.asr_url, "model": args.asr_model} if asr else None,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "cases": len(cases),
        "groups": groups,
        "questions": dict(sorted(questions.items())),
        "sides": [side for side, _ in sides],
        "gateway": args.gateway,
        "upstream": None if args.no_direct else args.upstream,
        "counts": counts,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    write_csv(args.out / "results.csv", samples)
    (args.out / "index.html").write_text(render_html(samples, manifest), "utf-8")
    print(json.dumps(counts))
    print(f"open {args.out / 'index.html'}; fill review columns in {args.out / 'results.csv'}")
    return 0 if all(s.result == "audio" for s in samples) else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--upstream", default="http://127.0.0.1:8080", help="audio.cpp, for the direct (B) side"
    )
    ap.add_argument("--no-direct", action="store_true", help="only render through the gateway")
    ap.add_argument("--api-key", help="gateway Bearer key, if set")
    ap.add_argument("--model", default="g-omnivoice")
    ap.add_argument("--voice", default="vieneu_v3_Mai_Anh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--set",
        dest="case_set",
        choices=CASE_SETS,
        default="golden",
        help="golden: regression goldens without error cases; listening: listening-only set",
    )
    ap.add_argument("--group", action="append", help="only these case groups (repeatable)")
    ap.add_argument("--id", action="append", help="only these case ids (repeatable)")
    ap.add_argument(
        "--question",
        type=int,
        action="append",
        choices=range(1, 6),
        help="only cases for these issue #7 questions (repeatable)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "samples")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow an output dir with data; replaces its WAV files and reports",
    )
    ap.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="rate of streamed PCM, which carries no header (OmniVoice: 24000)",
    )
    ap.add_argument("--timeout", type=float, default=600, help="per-request read timeout (s)")
    ap.add_argument(
        "--normalize-on",
        action="store_true",
        help='ignore "normalize": false in cases, so they are rendered as in production',
    )
    ap.add_argument(
        "--asr-url",
        default=os.environ.get("OMNIVOICE_ASR_URL") or None,
        help="OpenAI-style transcription server for ASR scoring (env OMNIVOICE_ASR_URL;"
        " unset: no ASR)",
    )
    ap.add_argument(
        "--asr-model",
        default=os.environ.get("OMNIVOICE_ASR_MODEL") or "gipformer",
        help="transcription model (env OMNIVOICE_ASR_MODEL; default gipformer)",
    )
    ap.add_argument(
        "--asr-api-key",
        default=os.environ.get("OMNIVOICE_ASR_API_KEY") or None,
        help="Bearer key for the ASR server (env OMNIVOICE_ASR_API_KEY)",
    )
    return ap.parse_args()


def normalize(gateway: httpx.Client, body: dict) -> str:
    return gateway.post("/v1/text/normalize", json=body).raise_for_status().json()["engine_text"]


@functools.cache
def clone_reference(voice: str) -> tuple[str, str]:
    """Base64 WAV of voices/<voice>.wav and its transcript from voices/prompt_text (name|text)."""
    audio = base64.b64encode((VOICES / f"{voice}.wav").read_bytes()).decode("ascii")
    for line in (VOICES / "prompt_text").read_text("utf-8").splitlines():
        name, _, text = line.partition("|")
        if name.strip() == voice:
            return audio, text.strip()
    sys.exit(f"no transcript for voice {voice!r} in {VOICES / 'prompt_text'}")


def prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            sys.exit(f"{out} is not empty; pass --overwrite or choose another directory")
        # Only this script's artifacts: a stale WAV would pass for this run's audio.
        for path in [*out.glob("*.wav"), *out.glob("*.wav.part"), *(out / r for r in REPORTS)]:
            path.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)


def synthesize(
    client: httpx.Client, payload: dict, sample: Sample, streaming: bool, pcm_rate: int
) -> tuple[bytes, int] | None:
    """One request, no retry. Fills sample's timing, warnings and metrics, or its failure code.

    Returns (pcm, sample_rate) when the request produced audio.
    """
    start = time.perf_counter()
    data = bytearray()
    try:
        with client.stream("POST", "/v1/audio/speech", json=payload) as response:
            sample.http_status = response.status_code
            if response.status_code != 200:
                response.read()
                sample.result = error_type(response)
                return None
            sample.warnings = response.headers.get("x-text-warnings", "")
            for chunk in response.iter_bytes():
                if chunk and sample.ttfb_ms is None:
                    sample.ttfb_ms = (time.perf_counter() - start) * 1000
                data += chunk
    except httpx.HTTPError as exc:
        # After audio arrived the file would be truncated: record it, keep nothing.
        sample.result = "stream_aborted" if data else f"transport_{type(exc).__name__}"
        return None
    sample.total_ms = (time.perf_counter() - start) * 1000
    if streaming:
        pcm, rate = bytes(data), pcm_rate
        if not pcm or len(pcm) % 2:
            sample.result = "incomplete_pcm"
            return None
    else:
        try:
            with wave.open(io.BytesIO(data)) as w:
                if (w.getnchannels(), w.getsampwidth()) != (1, 2):
                    raise wave.Error("not mono s16")
                pcm, rate = w.readframes(w.getnframes()), w.getframerate()
        except (wave.Error, EOFError):
            sample.result = "bad_wav"
            return None
    sample.metrics = analyze(pcm, rate)
    return pcm, rate


def transcribe(asr: httpx.Client, model: str, path: Path, body: dict, sample: Sample) -> None:
    """One transcription request, no retry; fills asr_text and word_recall, or asr_error."""
    data = {"model": model, "response_format": "json"}
    if isinstance(body.get("language"), str):
        data["language"] = body["language"].split("-")[0]
    try:
        response = asr.post(
            "/v1/audio/transcriptions",
            data=data,
            files={"file": (path.name, path.read_bytes(), "audio/wav")},
        )
    except httpx.HTTPError as exc:
        sample.asr_error = f"transport_{type(exc).__name__}"
        return
    if response.status_code != 200:
        sample.asr_error = error_type(response)
        return
    try:
        sample.asr_text = str(response.json()["text"]).strip()
    except (ValueError, KeyError, TypeError):
        sample.asr_error = "bad_asr_response"
        return
    sample.word_recall = word_recall(sample.engine_text, sample.asr_text)


def words(text: str) -> list[str]:
    text = UNSPOKEN.sub(" ", unicodedata.normalize("NFC", text))
    return WORD.findall(text.lower())


def word_recall(expected: str, heard: str) -> float | None:
    """Share of expected words found in order in heard (LCS length / expected word count)."""
    ref, hyp = words(expected), words(heard)
    if not ref:
        return None
    prev = [0] * (len(hyp) + 1)
    for r in ref:
        cur = [0]
        for j, h in enumerate(hyp):
            cur.append(prev[j] + 1 if r == h else max(prev[j + 1], cur[j]))
        prev = cur
    return round(prev[-1] / len(ref), 3)


def error_type(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["type"])
    except (ValueError, KeyError, TypeError):
        return f"http_{response.status_code}"


def analyze(pcm: bytes, rate: int) -> dict:
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    n = len(samples)
    peak = max((abs(s) for s in samples), default=0)
    rms = math.sqrt(sum(s * s for s in samples) / n) if n else 0.0
    loud = [i for i, s in enumerate(samples) if abs(s) > SILENCE_LEVEL]
    lead = loud[0] if loud else n
    trail = n - 1 - loud[-1] if loud else n
    clipped = sum(1 for s in samples if abs(s) >= CLIP_LEVEL)
    duration = n / rate

    def dbfs(value: float) -> float | None:
        return round(20 * math.log10(value / 32768), 1) if value > 0 else None

    flags = [
        name
        for name, hit in (
            ("silent", not loud),
            ("very_short", duration < 0.3),
            ("clipping", clipped > 0),
            ("long_lead_silence", lead / rate > 1.0),
            ("long_trail_silence", trail / rate > 1.5),
        )
        if hit
    ]
    return {
        "duration_s": round(duration, 3),
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "lead_silence_ms": round(lead * 1000 / rate),
        "trail_silence_ms": round(trail * 1000 / rate),
        "clipped_samples": clipped,
        "flags": ",".join(flags),
    }


def rtf(sample: Sample) -> float | None:
    duration = sample.metrics.get("duration_s")
    return round(sample.total_ms / 1000 / duration, 3) if duration and sample.total_ms else None


def summary(sample: Sample) -> str:
    if sample.result != "audio":
        return f"{sample.result}" + (f" http={sample.http_status}" if sample.http_status else "")
    m = sample.metrics
    text = (
        f"{m['duration_s']:.2f}s ttfb={sample.ttfb_ms or 0:.0f}ms total={sample.total_ms:.0f}ms"
        f" rtf={rtf(sample)} peak={m['peak_dbfs']}dBFS"
    )
    if sample.word_recall is not None:
        text += f" recall={sample.word_recall}"
    return text + (f" flags={sample.flags}" if sample.flags else "")


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    tmp = path.with_suffix(".wav.part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    os.replace(tmp, path)


def write_csv(path: Path, samples: list[Sample]) -> None:
    def num(value: float | None) -> str | int:
        return "" if value is None else round(value)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "case_id": s.case["id"],
                    "group": s.case.get("group", ""),
                    "questions": ",".join(map(str, s.case.get("questions", ()))),
                    "side": s.side,
                    "normalize": "" if s.normalize is None else str(s.normalize).lower(),
                    "reference_text": (
                        "" if "clone" not in s.case
                        else "normalized" if s.side == "gateway" else "verbatim"
                    ),
                    "requested_speed": s.case.get("speed", 1.0 if "clone" in s.case else ""),
                    "result": s.result,
                    "http_status": s.http_status or "",
                    "file": s.file,
                    "ttfb_ms": num(s.ttfb_ms),
                    "total_ms": num(s.total_ms),
                    "rtf": rtf(s) or "",
                    **{k: "" if s.metrics.get(k) is None else s.metrics[k] for k in (
                        "duration_s", "peak_dbfs", "rms_dbfs", "lead_silence_ms",
                        "trail_silence_ms", "clipped_samples",
                    )},
                    "flags": s.flags,
                    "word_recall": "" if s.word_recall is None else s.word_recall,
                    "text_warnings": s.warnings,
                    "known_issue": s.case.get("known_issue", ""),
                    "review_focus": s.case.get("review_focus", ""),
                    "input": s.case["input"],
                    "engine_text": s.engine_text,
                    "sent_text": s.sent_text,
                    "asr_text": s.asr_text or "",
                    "review_result": "",  # pass / fail / ambiguous / not-listened
                    "reviewer": "",
                    "notes": "",
                }
            )  # fmt: skip


def git_commit() -> str | None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def render_html(samples: list[Sample], manifest: dict) -> str:
    e = html.escape
    sides = manifest["sides"]

    def cell(s: Sample) -> str:
        reference = ""
        if "clone" in s.case:
            kind = "normalized" if s.side == "gateway" else "verbatim"
            reference = f'<div class="muted">reference_text ({kind}): {e(s.reference_text)}</div>'
        if s.result != "audio":
            return f'<td class="bad">{e(s.result)}{reference}</td>'
        m = s.metrics
        info = (
            f"{m['duration_s']}s · ttfb {s.ttfb_ms or 0:.0f} ms · total {s.total_ms:.0f} ms"
            f" · rtf {rtf(s)} · peak {m['peak_dbfs']} / rms {m['rms_dbfs']} dBFS"
            f" · lead {m['lead_silence_ms']} / trail {m['trail_silence_ms']} ms"
            f" · clipped {m['clipped_samples']}"
        )
        flags = f'<div class="bad">{e(s.flags)}</div>' if s.flags else ""
        warnings = f'<div class="muted">warnings: {e(s.warnings)}</div>' if s.warnings else ""
        sent = (
            f'<div class="muted">sent_text: {e(s.sent_text)}</div>'
            if s.sent_text != s.engine_text
            else ""
        )
        asr = ""
        if s.asr_error:
            asr = f'<div class="bad">ASR failed: {e(s.asr_error)}</div>'
        elif s.asr_text is not None:
            recall = "–" if s.word_recall is None else s.word_recall
            low = s.word_recall is not None and s.word_recall < RECALL_THRESHOLD
            asr = (
                f'<div class="asr{" bad" if low else ""}"><b>ASR</b> (word_recall {recall}):'
                f" {e(s.asr_text)}</div>"
            )
        return (
            f'<td><audio controls preload="none" src="{e(s.file)}"></audio>{asr}'
            f'<div class="muted">{e(info)}</div>{flags}{warnings}{sent}{reference}</td>'
        )

    by_case: dict[str, dict[str, Sample]] = {}
    for s in samples:
        by_case.setdefault(s.case["id"], {})[s.side] = s
    trs = []
    for row in by_case.values():
        first = next(iter(row.values()))
        c = first.case
        issue = (
            f'<div class="bad">known issue: {e(c["known_issue"])}</div>'
            if c.get("known_issue")
            else ""
        )
        questions = " ".join(f"Q{q}" for q in c.get("questions", ()))
        normalize = ""
        if c.get("normalize") is False:
            normalize = (
                '<div class="warn">normalize: false in the case; rendered with normalize on'
                " (--normalize-on), as in production</div>"
                if manifest["normalize_on"]
                else '<div class="warn">normalize: false (tests cleanup only; production has'
                " normalize on, so symbols such as @ &lt; &gt; reach the model as typed)</div>"
            )
        control_speed = c.get("speed", 1.0 if "clone" in c else None)
        if control_speed is not None:
            normalize += f'<div class="muted">controlled speed: {control_speed} (both sides)</div>'
        focus = (
            f"<div><b>listen for:</b> {e(c['review_focus'])}</div>" if c.get("review_focus") else ""
        )
        trs.append(
            f"<tr><td><b>{e(c['id'])}</b><br><small>{e(c.get('group', ''))}"
            f" {e(questions)}</small>{normalize}</td>"
            f"<td><pre>{e(c['input'])}</pre></td>"
            f"<td><pre>{e(first.engine_text)}</pre><small>{e(c['why'])}</small>{focus}{issue}</td>"
            + "".join(cell(row[side]) for side in sides)
            + "</tr>"
        )
    names = {"gateway": "A: gateway", "direct": "B: direct"}
    head = "".join(f"<th>{names[side]}</th>" for side in sides)
    meta = " · ".join(
        f"{k}={manifest[k]}"
        for k in (
            "case_set",
            "model",
            "mode",
            "voice",
            "seed",
            "normalize_on",
            "pipeline_version",
            "git_commit",
            "started_utc",
        )
    )
    asr_note = (
        f"ASR: {manifest['asr']['model']}; word_recall compares each side's transcript with the"
        f" gateway's engine text; below {RECALL_THRESHOLD} adds words_missing."
        if manifest["asr"]
        else ""
    )
    return f"""<!doctype html><meta charset="utf-8"><title>OmniVoice gateway samples</title>
<style>body{{font-family:sans-serif;margin:1em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px;vertical-align:top}}
pre{{white-space:pre-wrap;margin:0;max-width:28em}}.bad{{color:#b00}}
.muted{{color:#666;font-size:.8em;max-width:22em}}.asr{{font-size:.85em;max-width:22em}}
.warn{{background:#fff3cd;font-size:.8em;max-width:14em;padding:2px}}</style>
<h1>OmniVoice gateway samples</h1><p>{e(meta)}</p>
<p class="muted">Flags are hints, not a quality verdict. Record review results in results.csv.
{e(asr_note)}</p>
<table><tr><th>case</th><th>input text</th><th>engine text</th>{head}</tr>
{chr(10).join(trs)}
</table>
"""


if __name__ == "__main__":
    sys.exit(main())
