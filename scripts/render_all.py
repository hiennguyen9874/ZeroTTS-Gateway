#!/usr/bin/env python3
"""Run render_samples.py for every (case set, model, voice) and build one summary.

Needs zerotts_server and the gateway running (see docs/LISTENING.md). Defaults: both case sets,
every TTS model in /v1/models and its available voices.

  uv run python scripts/render_all.py                                  # everything
  uv run python scripts/render_all.py --set listening --model zerotts --voice maichi
  uv run python scripts/render_all.py --out outputs/render-all-001     # rerun resumes
  uv run python scripts/render_all.py --dry-run                        # print the plan only
  # listen: python -m http.server -d outputs/render-all-001 9000

Layout under --out:

  <set>/<model>/<voice>/   one render_samples.py run (WAVs, results.csv, manifest.json, index.html)
  <set>/<model>/_clone/    groups made only of clone cases ("clone": voice): they carry their own
                           reference audio, so they run once per model instead of once per voice
  all_results.csv          every results.csv row, prefixed with set, model, voice, run
  summary.csv              per (set, model, side): counts, flagged, words_missing, mean word_recall,
                           median rtf / ttfb / duration
  voices.csv               per (set, model, voice, side): words_missing count, mean word_recall;
                           printed sorted best first, to pick the main listening voice
  cases.csv                per (set, model, case, side): voices rendered, voices with
                           words_missing, mean word_recall; printed for cases missing words in
                           some voice, "all" marking cases broken in every voice
  runs.json                each run's exit code and directory
  index.html               links to every run's page with its counts

Models run in the outer loop so audio.cpp (max_loaded_models) loads each model once. A run whose
directory already has manifest.json is skipped unless --overwrite; a directory left half-written
by an interrupted run is re-rendered. A failing run is recorded and the next one still starts.
Exit status is 1 when any run failed. ASR scoring (OMNIVOICE_ASR_* env, see render_samples.py) is
passed through to every run; without it the word_recall columns stay blank.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "scripts" / "render_samples.py"
CASE_SETS = {
    "golden": ROOT / "tests" / "golden" / "pipeline.json",
    "listening": ROOT / "scripts" / "listening_cases.json",
}
VOICES = ROOT / "voices"
CLONE_RUN = "_clone"


def main() -> int:
    args = parse_args()
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    families = gateway_models(args.gateway, headers)
    models = args.model or list(families)
    voices = {}
    for model in models:
        # ZeroTTS voices are the upstream's preset packs; OmniVoice ones are voices/*.wav.
        if args.voice:
            voices[model] = args.voice
        elif families.get(model) == "zerotts":
            voices[model] = gateway_voices(args.gateway, headers)
        else:
            voices[model] = library_voices()
    if not models or not all(voices.values()):
        sys.exit("no models or no voices to render")
    plan = build_plan(args.set or list(CASE_SETS), models, voices, families)
    print(f"{len(plan)} runs: sets={args.set or list(CASE_SETS)} models={models}"
          f" voices={sum(map(len, voices.values()))} -> {args.out}", flush=True)  # fmt: skip
    if args.dry_run:
        for run in plan:
            print(f"  {run['set']}/{run['model']}/{run['voice']} groups={','.join(run['groups'])}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.json"
    runs = json.loads(runs_path.read_text("utf-8")) if runs_path.exists() else {}
    for i, run in enumerate(plan, 1):
        key = f"{run['set']}/{run['model']}/{run['voice']}"
        out = args.out / key
        if (out / "manifest.json").exists() and not args.overwrite:
            print(f"[{i}/{len(plan)}] {key} skip (done)", flush=True)
            continue
        command = [
            sys.executable, str(RENDER), "--gateway", args.gateway, "--upstream", args.upstream,
            "--set", run["set"], "--model", run["model"], "--seed", str(args.seed),
            "--timeout", str(args.timeout), "--out", str(out),
        ]  # fmt: skip
        if args.sample_rate:
            command += ["--sample-rate", str(args.sample_rate)]
        if run["voice"] != CLONE_RUN:
            command += ["--voice", run["voice"]]
        for group in run["groups"]:
            command += ["--group", group]
        if args.api_key:
            command += ["--api-key", args.api_key]
        if args.no_direct:
            command.append("--no-direct")
        if args.normalize_on:
            command.append("--normalize-on")
        if out.exists() and any(out.iterdir()):
            command.append("--overwrite")
        print(f"[{i}/{len(plan)}] {key}", flush=True)
        start = time.perf_counter()
        code = subprocess.run(command, cwd=ROOT).returncode
        runs[key] = {**run, "exit_code": code, "seconds": round(time.perf_counter() - start, 1)}
        runs_path.write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", "utf-8")
        if code:
            print(f"[{i}/{len(plan)}] {key} FAILED exit={code}", flush=True)

    rows, manifests = collect(args.out)
    write_reports(args.out, rows, manifests, runs)
    print_tables(args.out)
    failed = sorted(k for k, r in runs.items() if r["exit_code"])
    print(f"done: {len(manifests)} runs, {len(rows)} rows, {len(failed)} failed runs")
    for key in failed:
        print(f"  failed: {key}")
    print(f"open {args.out / 'index.html'}; all rows in {args.out / 'all_results.csv'}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", help="gateway Bearer key, if set")
    ap.add_argument("--no-direct", action="store_true", help="only render through the gateway")
    ap.add_argument("--set", action="append", choices=CASE_SETS, help="case sets (repeatable)")
    ap.add_argument("--model", action="append", help="models (repeatable; default: all TTS)")
    ap.add_argument("--voice", action="append", help="voices (repeatable; default: all in voices/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600, help="per-request read timeout (s)")
    ap.add_argument(
        "--sample-rate", type=int, help="rate of streamed PCM (default: render_samples.py's)"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / f"render-all-{datetime.now(UTC):%Y%m%d-%H%M%S}",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-render finished runs too")
    ap.add_argument(
        "--normalize-on",
        action="store_true",
        help="pass --normalize-on: send text_processing.normalize=true for every case",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args()


def gateway_models(gateway: str, headers: dict) -> dict[str, str | None]:
    """TTS model ids in the gateway's /v1/models, each with its family."""
    data = httpx.get(f"{gateway}/v1/models", headers=headers, timeout=30).raise_for_status()
    return {m["id"]: m.get("family") for m in data.json()["data"] if m.get("task", "tts") == "tts"}


def gateway_voices(gateway: str, headers: dict) -> list[str]:
    """Voice names in the gateway's /v1/audio/voices (ZeroTTS preset voices)."""
    data = httpx.get(f"{gateway}/v1/audio/voices", headers=headers, timeout=30).raise_for_status()
    return sorted(data.json()["voices"])


def library_voices() -> list[str]:
    """voices/*.wav with a transcript line in voices/prompt_text (as audio.cpp's voice_dir)."""
    lines = (VOICES / "prompt_text").read_text("utf-8").splitlines()
    named = {line.partition("|")[0].strip() for line in lines if "|" in line}
    voices = sorted(p.stem for p in VOICES.glob("*.wav"))
    for voice in voices:
        if voice not in named:
            print(f"warning: {voice}.wav has no transcript in voices/prompt_text; skipped")
    return [v for v in voices if v in named]


def build_plan(
    sets: list[str], models: list[str], voices: dict[str, list[str]], families: dict
) -> list[dict]:
    split = {}
    for name in sets:
        cases = [c for c in json.loads(CASE_SETS[name].read_text("utf-8")) if "error" not in c]
        groups: dict[str, list[dict]] = {}
        for case in cases:
            groups.setdefault(case.get("group", ""), []).append(case)
        clone = sorted(g for g, cs in groups.items() if all("clone" in c for c in cs))
        split[name] = (sorted(set(groups) - set(clone)), clone)
    plan = []
    for model in models:
        for name in sets:
            voiced, clone = split[name]
            if voiced:
                plan += [
                    {"set": name, "model": model, "voice": v, "groups": voiced}
                    for v in voices[model]
                ]
            # ZeroTTS cannot clone (ADR 0005), so clone-only groups are not planned for it.
            if clone and families.get(model) != "zerotts":
                plan.append({"set": name, "model": model, "voice": CLONE_RUN, "groups": clone})
    return plan


def collect(out: Path) -> tuple[list[dict], list[dict]]:
    rows, manifests = [], []
    for manifest_path in sorted(out.glob("*/*/*/manifest.json")):
        run_dir = manifest_path.parent
        name, model, voice = run_dir.relative_to(out).parts
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifests.append({"set": name, "model": model, "voice": voice, **manifest})
        with (run_dir / "results.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                run = f"{name}/{model}/{voice}"
                rows.append({"set": name, "model": model, "voice": voice, "run": run, **row})
    return rows, manifests


def write_reports(out: Path, rows: list[dict], manifests: list[dict], runs: dict) -> None:
    if rows:
        # Runs rendered by older script versions may lack newer columns: union, first-seen order.
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / "all_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    def median(values: list[str]) -> float | str:
        nums = [float(v) for v in values if v not in ("", None)]
        return round(statistics.median(nums), 3) if nums else ""

    summary: dict[tuple, list[dict]] = {}
    for row in rows:
        summary.setdefault((row["set"], row["model"], row["side"]), []).append(row)
    fields = ["set", "model", "side", "requests", "audio", "failed", "flagged", "asr_scored",
              "words_missing", "mean_word_recall", "median_rtf", "median_ttfb_ms",
              "median_duration_s"]  # fmt: skip
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (name, model, side), group in sorted(summary.items()):
            audio = [r for r in group if r["result"] == "audio"]
            writer.writerow(
                {
                    "set": name,
                    "model": model,
                    "side": side,
                    "requests": len(group),
                    "audio": len(audio),
                    "failed": len(group) - len(audio),
                    "flagged": sum(bool(r["flags"]) for r in audio),
                    **recall_stats(audio),
                    "median_rtf": median([r["rtf"] for r in audio]),
                    "median_ttfb_ms": median([r["ttfb_ms"] for r in audio]),
                    "median_duration_s": median([r["duration_s"] for r in audio]),
                }
            )

    voices, cases = recall_tables(rows)
    write_table(out / "voices.csv", voices)
    write_table(out / "cases.csv", cases)

    e = html.escape
    trs = []
    for m in manifests:
        key = f"{m['set']}/{m['model']}/{m['voice']}"
        counts = m.get("counts", {})
        bad = counts.get("requests", 0) - counts.get("audio", 0)
        code = runs.get(key, {}).get("exit_code", "")
        trs.append(
            f"<tr><td>{e(m['set'])}</td><td>{e(m['model'])}</td><td>{e(m['mode'])}</td>"
            f'<td><a href="{e(key)}/index.html">{e(m["voice"])}</a></td>'
            f"<td>{m.get('cases', '')}</td><td>{counts.get('audio', 0)}</td>"
            f'<td class="{"bad" if bad else ""}">{bad}</td><td>{counts.get("flagged", 0)}</td>'
            f"<td>{counts.get('words_missing', '')}</td><td>{e(str(code))}</td></tr>"
        )
    scored = any(v["asr_scored"] for v in voices)
    voice_trs = [
        "<tr>"
        + "".join(f"<td>{e(str(v[k]))}</td>" for k in ("set", "model", "side"))
        + f'<td><a href="{e(v["set"])}/{e(v["model"])}/{e(v["voice"])}/index.html">'
        f"{e(v['voice'])}</a></td>"
        + "".join(
            f"<td>{e(str(v[k]))}</td>"
            for k in ("audio", "asr_scored", "words_missing", "mean_word_recall")
        )
        + "</tr>"
        for v in sorted(voices, key=voice_rank)
    ]
    case_trs = [
        "<tr>"
        + "".join(f"<td>{e(str(c[k]))}</td>" for k in ("set", "model", "side", "case_id"))
        + f"<td>{c['voices']}</td><td>{c['asr_scored']}</td>"
        + f'<td class="{"bad" if c["all_voices_missing"] else ""}">{c["words_missing_voices"]}'
        + f"{' (all)' if c['all_voices_missing'] else ''}</td>"
        + f"<td>{c['mean_word_recall']}</td></tr>"
        for c in sorted(cases, key=case_rank)
        if c["words_missing_voices"]
    ]
    asr_html = (
        f"""<h2>By voice</h2>
<p>words_missing: word_recall below 0.8 (ASR). Fewest words_missing first.</p>
<table><tr><th>set</th><th>model</th><th>side</th><th>voice</th><th>audio</th><th>asr scored</th>
<th>words missing</th><th>mean word_recall</th></tr>
{chr(10).join(voice_trs)}
</table>
<h2>By case</h2>
<p>Cases with words missing in at least one voice; (all) = in every scored voice.</p>
<table><tr><th>set</th><th>model</th><th>side</th><th>case</th><th>voices</th><th>asr scored</th>
<th>voices missing words</th><th>mean word_recall</th></tr>
{chr(10).join(case_trs)}
</table>
"""
        if scored
        else "<p>No ASR scores (set OMNIVOICE_ASR_URL to score runs).</p>"
    )
    (out / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>ZeroTTS render-all</title>
<style>body{{font-family:sans-serif;margin:1em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}.bad{{color:#b00;font-weight:bold}}</style>
<h1>ZeroTTS render-all</h1>
<p>{len(manifests)} runs · {len(rows)} rows · <a href="summary.csv">summary.csv</a> ·
<a href="all_results.csv">all_results.csv</a> · <a href="voices.csv">voices.csv</a> ·
<a href="cases.csv">cases.csv</a></p>
<table><tr><th>set</th><th>model</th><th>mode</th><th>voice</th><th>cases</th><th>audio</th>
<th>failed</th><th>flagged</th><th>words missing</th><th>exit</th></tr>
{chr(10).join(trs)}
</table>
{asr_html}""",
        "utf-8",
    )


def recall_stats(audio: list[dict]) -> dict:
    """ASR columns over rows that produced audio: scored rows, words_missing, mean word_recall."""
    recalls = [float(r["word_recall"]) for r in audio if r.get("word_recall")]
    return {
        "asr_scored": len(recalls),
        "words_missing": sum("words_missing" in r["flags"].split(",") for r in audio),
        "mean_word_recall": round(statistics.mean(recalls), 3) if recalls else "",
    }


def recall_tables(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per-voice and per-case summaries of the ASR word_recall."""
    by_voice: dict[tuple, list[dict]] = {}
    by_case: dict[tuple, list[dict]] = {}
    for row in rows:
        by_voice.setdefault((row["set"], row["model"], row["voice"], row["side"]), []).append(row)
        by_case.setdefault((row["set"], row["model"], row["case_id"], row["side"]), []).append(row)
    voices = []
    for (name, model, voice, side), group in sorted(by_voice.items()):
        audio = [r for r in group if r["result"] == "audio"]
        voices.append(
            {"set": name, "model": model, "voice": voice, "side": side,
             "requests": len(group), "audio": len(audio), **recall_stats(audio)}
        )  # fmt: skip
    cases = []
    for (name, model, case_id, side), group in sorted(by_case.items()):
        audio = [r for r in group if r["result"] == "audio"]
        stats = recall_stats(audio)
        cases.append(
            {
                "set": name,
                "model": model,
                "case_id": case_id,
                "side": side,
                "voices": len(audio),
                "asr_scored": stats["asr_scored"],
                "words_missing_voices": stats["words_missing"],
                "all_voices_missing": bool(stats["asr_scored"])
                and stats["words_missing"] == stats["asr_scored"],
                "mean_word_recall": stats["mean_word_recall"],
            }
        )
    return voices, cases


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def voice_rank(v: dict) -> tuple:
    recall = v["mean_word_recall"]
    # Clone runs have other cases and no library voice to pick: after the voices.
    clone = v["voice"] == CLONE_RUN
    return (v["set"], v["model"], v["side"], clone, v["words_missing"], -(recall or 0), v["voice"])


def case_rank(c: dict) -> tuple:
    return (c["set"], c["model"], c["side"], -c["words_missing_voices"], c["case_id"])


def print_tables(out: Path) -> None:
    """Console summary by voice and by case (ids and numbers only, never text)."""
    voices = read_table(out / "voices.csv")
    cases = read_table(out / "cases.csv")
    if not any(int(v["asr_scored"]) for v in voices):
        return
    print("by voice (fewest words_missing first): set/model/side voice  missing/scored  recall")
    for v in sorted(voices, key=lambda v: voice_rank(numeric(v))):
        print(f"  {v['set']}/{v['model']}/{v['side']} {v['voice']}"
              f"  {v['words_missing']}/{v['asr_scored']}  {v['mean_word_recall']}")  # fmt: skip
    missing = [numeric(c) for c in cases if int(c["words_missing_voices"])]
    print("by case (words missing in some voice): set/model/side case  voices_missing/asr_scored")
    for c in sorted(missing, key=case_rank):
        every = " ALL" if c["all_voices_missing"] else ""
        print(f"  {c['set']}/{c['model']}/{c['side']} {c['case_id']}"
              f"  {c['words_missing_voices']}/{c['asr_scored']}{every}")  # fmt: skip
    if not missing:
        print("  none")


def read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric(row: dict) -> dict:
    """A voices.csv / cases.csv row read back with its counts and recall as numbers."""
    out = dict(row)
    for key in ("requests", "audio", "voices", "asr_scored", "words_missing",
                "words_missing_voices"):  # fmt: skip
        if key in out:
            out[key] = int(out[key])
    if "mean_word_recall" in out:
        out["mean_word_recall"] = float(out["mean_word_recall"] or 0) or ""
    if "all_voices_missing" in out:
        out["all_voices_missing"] = out["all_voices_missing"] == "True"
    return out


if __name__ == "__main__":
    sys.exit(main())
