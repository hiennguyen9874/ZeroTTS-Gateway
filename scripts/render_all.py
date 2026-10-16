#!/usr/bin/env python3
"""Run render_samples.py for every (case set, model, voice) and build one summary.

Needs audio.cpp and the gateway running (see README). Defaults: both case sets, every TTS model in
the gateway's /v1/models, every voices/*.wav that has a line in voices/prompt_text.

  uv run python scripts/render_all.py                                  # everything
  uv run python scripts/render_all.py --set listening --model g-omnivoice --voice vieneu_v3_Mai_Anh
  uv run python scripts/render_all.py --out outputs/render-all-001     # rerun resumes
  uv run python scripts/render_all.py --dry-run                        # print the plan only
  # listen: python -m http.server -d outputs/render-all-001 9000

Layout under --out:

  <set>/<model>/<voice>/   one render_samples.py run (WAVs, results.csv, manifest.json, index.html)
  <set>/<model>/_clone/    groups made only of clone cases ("clone": voice): they carry their own
                           reference audio, so they run once per model instead of once per voice
  all_results.csv          every results.csv row, prefixed with set, model, voice, run
  summary.csv              per (set, model, side): counts, flagged, median rtf / ttfb / duration
  runs.json                each run's exit code and directory
  index.html               links to every run's page with its counts

Models run in the outer loop so audio.cpp (max_loaded_models) loads each model once. A run whose
directory already has manifest.json is skipped unless --overwrite; a directory left half-written
by an interrupted run is re-rendered. A failing run is recorded and the next one still starts.
Exit status is 1 when any run failed.
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
    models = args.model or gateway_models(args.gateway, headers)
    voices = args.voice or library_voices()
    if not models or not voices:
        sys.exit("no models or no voices to render")
    plan = build_plan(args.set or list(CASE_SETS), models, voices)
    print(f"{len(plan)} runs: sets={args.set or list(CASE_SETS)} models={models}"
          f" voices={len(voices)} -> {args.out}", flush=True)  # fmt: skip
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
            "--timeout", str(args.timeout), "--sample-rate", str(args.sample_rate),
            "--out", str(out),
        ]  # fmt: skip
        if run["voice"] != CLONE_RUN:
            command += ["--voice", run["voice"]]
        for group in run["groups"]:
            command += ["--group", group]
        if args.api_key:
            command += ["--api-key", args.api_key]
        if args.no_direct:
            command.append("--no-direct")
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
    ap.add_argument("--sample-rate", type=int, default=24000, help="rate of streamed PCM")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / f"render-all-{datetime.now(UTC):%Y%m%d-%H%M%S}",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-render finished runs too")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args()


def gateway_models(gateway: str, headers: dict) -> list[str]:
    data = httpx.get(f"{gateway}/v1/models", headers=headers, timeout=30).raise_for_status()
    return [m["id"] for m in data.json()["data"] if m.get("task", "tts") == "tts"]


def library_voices() -> list[str]:
    """voices/*.wav with a transcript line in voices/prompt_text (as audio.cpp's voice_dir)."""
    lines = (VOICES / "prompt_text").read_text("utf-8").splitlines()
    named = {line.partition("|")[0].strip() for line in lines if "|" in line}
    voices = sorted(p.stem for p in VOICES.glob("*.wav"))
    for voice in voices:
        if voice not in named:
            print(f"warning: {voice}.wav has no transcript in voices/prompt_text; skipped")
    return [v for v in voices if v in named]


def build_plan(sets: list[str], models: list[str], voices: list[str]) -> list[dict]:
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
                    {"set": name, "model": model, "voice": v, "groups": voiced} for v in voices
                ]
            if clone:
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
        with (out / "all_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def median(values: list[str]) -> float | str:
        nums = [float(v) for v in values if v not in ("", None)]
        return round(statistics.median(nums), 3) if nums else ""

    summary: dict[tuple, list[dict]] = {}
    for row in rows:
        summary.setdefault((row["set"], row["model"], row["side"]), []).append(row)
    fields = ["set", "model", "side", "requests", "audio", "failed", "flagged",
              "median_rtf", "median_ttfb_ms", "median_duration_s"]  # fmt: skip
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
                    "median_rtf": median([r["rtf"] for r in audio]),
                    "median_ttfb_ms": median([r["ttfb_ms"] for r in audio]),
                    "median_duration_s": median([r["duration_s"] for r in audio]),
                }
            )

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
            f"<td>{e(str(code))}</td></tr>"
        )
    (out / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>OmniVoice render-all</title>
<style>body{{font-family:sans-serif;margin:1em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}.bad{{color:#b00;font-weight:bold}}</style>
<h1>OmniVoice render-all</h1>
<p>{len(manifests)} runs · {len(rows)} rows · <a href="summary.csv">summary.csv</a> ·
<a href="all_results.csv">all_results.csv</a></p>
<table><tr><th>set</th><th>model</th><th>mode</th><th>voice</th><th>cases</th><th>audio</th>
<th>failed</th><th>flagged</th><th>exit</th></tr>
{chr(10).join(trs)}
</table>
""",
        "utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
