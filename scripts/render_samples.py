#!/usr/bin/env python3
"""Render golden cases to audio for listening: gateway vs audio.cpp direct (A/B), plus an HTML page.

Needs a running audio.cpp server and a gateway in front of it (see README, docs/USAGE.md):

  uv run python scripts/render_samples.py                       # all golden cases that produce text
  uv run python scripts/render_samples.py --group numbers-money --group dates-times
  uv run python scripts/render_samples.py --no-direct --model g-omnivoice-stream
  # listen: python -m http.server -d outputs/samples  (or open outputs/samples/index.html)

Same model, voice and seed for both sides, so differences come from the text pipeline.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "pipeline.json"


def main() -> int:
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
    ap.add_argument("--voice", default="demo_nu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group", action="append", help="only these golden groups (repeatable)")
    ap.add_argument("--id", action="append", help="only these golden ids (repeatable)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "samples")
    args = ap.parse_args()

    cases = [c for c in json.loads(GOLDEN.read_text("utf-8")) if "error" not in c]
    if args.group:
        cases = [c for c in cases if c.get("group") in args.group]
    if args.id:
        cases = [c for c in cases if c["id"] in args.id]
    if not cases:
        sys.exit("no matching golden cases")

    args.out.mkdir(parents=True, exist_ok=True)
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    gateway = httpx.Client(base_url=args.gateway, headers=headers, timeout=600)
    upstream = None if args.no_direct else httpx.Client(base_url=args.upstream, timeout=600)
    version = (
        gateway.post("/v1/text/normalize", json={"input": "x"})
        .raise_for_status()
        .json()["pipeline_version"]
    )

    rows = []
    for i, case in enumerate(cases, 1):
        body = {
            "model": args.model,
            "input": case["input"],
            "voice": args.voice,
            "seed": args.seed,
            "language": case.get("language", "vi"),
            "response_format": "wav",
        }
        processing = (
            {"text_processing": {"normalize": case["normalize"]}} if "normalize" in case else {}
        )
        engine = (
            gateway.post("/v1/text/normalize", json={**body, **processing})
            .raise_for_status()
            .json()["engine_text"]
        )
        row = {"case": case, "engine": engine, "files": {}}
        sides = [("gateway", gateway, {**body, **processing})] + (
            [("direct", upstream, body)] if upstream else []
        )
        for side, client, payload in sides:
            name = f"{case['id']}.{side}.wav"
            response = client.post("/v1/audio/speech", json=payload)
            if response.status_code == 200:
                (args.out / name).write_bytes(response.content)
                row["files"][side] = name
            else:
                row["files"][side] = None
                print(
                    f"  {side}: HTTP {response.status_code} {response.text[:200]}", file=sys.stderr
                )
        print(f"[{i}/{len(cases)}] {case['id']}", file=sys.stderr)
        rows.append(row)

    meta = f"model={args.model} voice={args.voice} seed={args.seed} pipeline={version}"
    (args.out / "index.html").write_text(render_html(rows, meta, bool(upstream)), "utf-8")
    print(f"wrote {args.out / 'index.html'} ({meta})", file=sys.stderr)
    return 0


def render_html(rows: list[dict], meta: str, direct: bool) -> str:
    e = html.escape

    def audio(name: str | None) -> str:
        return (
            f'<audio controls preload="none" src="{e(name)}"></audio>' if name else "<i>error</i>"
        )

    trs = []
    for row in rows:
        c = row["case"]
        issue = (
            f'<div class="issue">known issue: {e(c["known_issue"])}</div>'
            if c.get("known_issue")
            else ""
        )
        cells = [
            f"<b>{e(c['id'])}</b><br><small>{e(c.get('group', ''))}</small>",
            f"<pre>{e(c['input'])}</pre>",
            f"<pre>{e(row['engine'])}</pre><small>{e(c['why'])}</small>{issue}",
            audio(row["files"].get("gateway")),
        ]
        if direct:
            cells.append(audio(row["files"].get("direct")))
        trs.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    head = "<th>case</th><th>input text</th><th>engine text</th><th>A: gateway</th>" + (
        "<th>B: direct</th>" if direct else ""
    )
    return f"""<!doctype html><meta charset="utf-8"><title>OmniVoice gateway samples</title>
<style>body{{font-family:sans-serif;margin:1em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px;vertical-align:top}}
pre{{white-space:pre-wrap;margin:0;max-width:28em}}.issue{{color:#b00}}</style>
<h1>OmniVoice gateway samples</h1><p>{e(meta)}</p>
<table><tr>{head}</tr>
{chr(10).join(trs)}
</table>
"""


if __name__ == "__main__":
    sys.exit(main())
