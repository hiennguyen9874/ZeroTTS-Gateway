#!/usr/bin/env python3
"""Score text pipeline variants by ASR: WER/CER per case category (issue #9).

Reads the results.csv of render_samples.py runs (a run directory or a render_all.py tree) for
each named variant and scores every transcribed WAV against the case's spoken reading:

  uv run python scripts/ab_report.py --variant a=outputs/ab9/a-s42 --variant a=outputs/ab9/a-s7 \
      --variant b=outputs/ab9/b-s42 --variant c=outputs/ab9/c-s42 --out outputs/ab9/report

The reference is the input text through the full Vietnamese pipeline, once with the lexicon
(configs/lexicon.csv) and once without; a row scores against whichever is closer, so reading
"Google" as "gu gồ" or as "google" both count. Words are compared lowercased, without punctuation,
non-verbal tags or CMU spans, and with regional variants folded ("ngàn" = "nghìn"). Cases whose
language is not Vietnamese are left out (the ASR is Vietnamese only). WER and CER are corpus-level:
edits summed over rows divided by reference words or characters (spaces not counted).

The reference is the gateway's own reading, so a variant that reads a form differently but
correctly ("hai ngàn hai mươi bốn" for 2024) still loses points; listen to the cases listed in
cases.csv before concluding. Output in --out: report.md (tables), cases.csv (per case mean WER
per variant, sorted by spread between variants).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from render_samples import CASE_SETS, words  # noqa: E402

from zerotts_gateway.text import Pipeline, TextError, is_vietnamese, load_lexicon  # noqa: E402

LEXICON = ROOT / "configs" / "lexicon.csv"
# The categories asked for in #9; golden groups not listed keep their own name.
CATEGORIES = {
    "dates-times": "ngày giờ",
    "numbers-money": "số",
    "measurements": "số",
    "codes-phones": "số",
    "abbrev-english": "viết tắt",
    "abbrev-case": "viết tắt",
    "loanwords": "tiếng Anh",
    "markup": "markup",
    "short-sentences": "markup",
}
# abbrev-english cases that are English words or names rather than abbreviations.
ENGLISH_CASES = {
    "vi-acronym-mixed", "vi-english-words", "vi-english-sentence", "vi-ok", "vi-iphone",
    "vi-ipad", "vi-ios", "vi-youtube", "vi-facebook", "vi-google", "vi-python-version",
}  # fmt: skip
# Same word, regional or ASR spelling: folded on both sides before comparing.
FOLD = {"ngàn": "nghìn", "tư": "bốn", "mốt": "một", "lẻ": "linh"}


def main() -> int:
    args = parse_args()
    lexicon = Pipeline(load_lexicon(LEXICON), max_chars=1_000_000)
    plain = Pipeline(None, max_chars=1_000_000)
    cases = {
        (name, case["id"]): case
        for name, path in CASE_SETS.items()
        for case in json.loads(path.read_text("utf-8"))
    }

    scored: list[dict] = []
    skipped = 0
    for variant, directory in args.variant:
        manifests = sorted(directory.glob("**/manifest.json"))
        if not manifests:
            sys.exit(f"{directory}: no render_samples.py run (manifest.json) found")
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            with (manifest_path.parent / "results.csv").open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    case = cases[(manifest["case_set"], row["case_id"])]
                    refs = references(case, lexicon, plain)
                    if not refs:
                        skipped += 1
                        continue
                    side = "" if row["side"] == "gateway" else f"/{row['side']}"
                    scored.append(
                        {
                            "variant": variant + side,
                            "voice": manifest["voice"],
                            "seed": manifest["seed"],
                            "case_id": row["case_id"],
                            "category": category(case),
                            **score(refs, row),
                        }
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(r["variant"] for r in scored))
    report = render_report(scored, variants, skipped)
    (args.out / "report.md").write_text(report, "utf-8")
    write_cases(args.out / "cases.csv", scored, variants)
    print(report)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variant",
        action="append",
        required=True,
        type=lambda s: (s.partition("=")[0], Path(s.partition("=")[2])),
        help="NAME=DIR; repeat a name to pool runs (seeds, voices) into one variant",
    )
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def references(case: dict, lexicon: Pipeline, plain: Pipeline) -> list[list[str]]:
    """The spoken readings a transcript may match, as folded word lists; [] when not scored."""
    if not is_vietnamese(case.get("language", "vi")):
        return []
    refs = []
    for pipeline in (lexicon, plain):
        try:
            text, _ = pipeline.process(case["input"], vietnamese=True, nonverbal_tags=False)
        except TextError:
            return []
        ref = fold(text)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def fold(text: str) -> list[str]:
    return [FOLD.get(w, w) for w in words(text)]


def category(case: dict) -> str:
    if case["id"] in ENGLISH_CASES:
        return "tiếng Anh"
    return CATEGORIES.get(case.get("group", ""), case.get("group", ""))


def score(refs: list[list[str]], row: dict) -> dict:
    """Word and char edits against the closest reference; a row without a transcript failed."""
    if row["result"] != "audio" or row["asr_text"] == "" and not row["word_recall"]:
        ref = refs[0]
        return {"failed": 1, "words": len(ref), "word_edits": len(ref),
                "chars": len("".join(ref)), "char_edits": len("".join(ref))}  # fmt: skip
    hyp = fold(row["asr_text"])
    ref = min(refs, key=lambda r: (edits(r, hyp) / len(r), edits("".join(r), "".join(hyp))))
    return {
        "failed": 0,
        "words": len(ref),
        "word_edits": edits(ref, hyp),
        "chars": len("".join(ref)),
        "char_edits": edits("".join(ref), "".join(hyp)),
    }


def edits(ref, hyp) -> int:
    """Levenshtein distance between two sequences (words or characters)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def rates(rows: list[dict]) -> tuple[float, float]:
    words_ = sum(r["words"] for r in rows)
    chars = sum(r["chars"] for r in rows)
    return (
        sum(r["word_edits"] for r in rows) / words_ if words_ else 0.0,
        sum(r["char_edits"] for r in rows) / chars if chars else 0.0,
    )


def render_report(scored: list[dict], variants: list[str], skipped: int) -> str:
    def table(key: str, title: str) -> list[str]:
        keys = sorted({r[key] for r in scored}, key=str)
        lines = [
            f"### {title}",
            "",
            f"| {key} | rows | " + " | ".join(f"WER {v} | CER {v}" for v in variants) + " |",
            "| --- | --- |" + " --- | --- |" * len(variants),
        ]
        for k in [*keys, "(all)"]:
            cells = []
            rows_n = ""
            for v in variants:
                rows = [r for r in scored if r["variant"] == v and k in ("(all)", r[key])]
                wer, cer = rates(rows)
                rows_n = rows_n or str(len(rows))
                cells.append(f"{wer:.3f} | {cer:.3f}" if rows else "– | –")
            lines.append(f"| {k} | {rows_n} | " + " | ".join(cells) + " |")
        return [*lines, ""]

    failed = {v: sum(r["failed"] for r in scored if r["variant"] == v) for v in variants}
    lines = [
        f"Rows scored: {len(scored)}; rows left out (non-Vietnamese or error cases): {skipped}.",
        "Failed rows (no audio or no transcript, counted as all words wrong): "
        + ", ".join(f"{v} {n}" for v, n in failed.items())
        + ".",
        "",
        *table("category", "By category"),
        *table("voice", "By voice"),
        *table("seed", "By seed"),
    ]
    return "\n".join(lines)


def write_cases(path: Path, scored: list[dict], variants: list[str]) -> None:
    by_case: dict[str, dict[str, list[dict]]] = {}
    category_of = {}
    for r in scored:
        by_case.setdefault(r["case_id"], {}).setdefault(r["variant"], []).append(r)
        category_of[r["case_id"]] = r["category"]
    out = []
    for case_id, per_variant in by_case.items():
        wers = {
            v: round(statistics.mean(r["word_edits"] / r["words"] for r in rows), 3)
            for v, rows in per_variant.items()
        }
        spread = round(max(wers.values()) - min(wers.values()), 3)
        out.append({"case_id": case_id, "category": category_of[case_id], "spread": spread,
                    **{f"wer_{v}": wers.get(v, "") for v in variants}})  # fmt: skip
    out.sort(key=lambda r: (-r["spread"], r["case_id"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "category", "spread", *(f"wer_{v}" for v in variants)]
        )
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    raise SystemExit(main())
