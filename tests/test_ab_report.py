"""A/B scoring script (scripts/ab_report.py) end to end, on render_samples.py output dirs."""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ab_report.py"
FIELDS = ["case_id", "side", "result", "word_recall", "asr_text"]


def run_dir(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """A render_samples.py run on the golden set: rows of (case id, result, ASR transcript)."""
    path.mkdir(parents=True)
    manifest = {"case_set": "golden", "voice": "maichi", "seed": 42}
    (path / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for case_id, result, asr in rows:
            recall = "1.0" if result == "audio" else ""
            writer.writerow({"case_id": case_id, "side": "gateway", "result": result,
                             "word_recall": recall, "asr_text": asr})  # fmt: skip
    return path


def wer_cells(report: str, row: str) -> list[str]:
    """WER cells of one row of the "By category" table (cells alternate WER, CER per variant)."""
    table = report.split("### By category")[1].split("###")[0]
    line = next(line for line in table.splitlines() if line.startswith(f"| {row} |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    return cells[2::2]


def test_scores_each_variant_against_the_closest_spoken_reading(tmp_path):
    a = run_dir(
        tmp_path / "a" / "golden" / "zerotts" / "maichi",
        [
            ("vi-google", "audio", "TÌM TRÊN GU GỒ"),  # the lexicon reading
            ("vi-year", "audio", "NĂM MỘT NGÀN CHÍN TRĂM BẢY MƯƠI LĂM LÀ NĂM LỊCH SỬ"),
            ("english-cleanup", "audio", "HELLO"),  # English: not scored by a Vietnamese ASR
        ],
    )
    c = run_dir(
        tmp_path / "c",
        [
            ("vi-google", "audio", "TÌM TRÊN GOOGLE"),  # the reading without the lexicon
            ("vi-year", "audio", "NĂM MỘT NGHÌN CHÍN TRĂM BẢY MƯƠI LĂM LÀ NĂM"),  # 2 of 12 lost
            ("vi-hour", "transport_ConnectError", ""),  # no audio: all 5 words wrong
        ],
    )
    out = tmp_path / "report"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--variant", f"a={a.parents[2]}", "--variant", f"c={c}",
         "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    report = (out / "report.md").read_text("utf-8")
    assert "rows left out (non-Vietnamese or error cases): 1" in report
    assert "a 0, c 1" in report
    # "Google" heard as "gu gồ" or as "google" both match a reading; "ngàn" folds to "nghìn".
    assert wer_cells(report, "tiếng Anh") == ["0.000", "0.000"]
    assert wer_cells(report, "ngày giờ") == ["0.000", "0.412"]  # (2 + 5) / (12 + 5)
    assert wer_cells(report, "(all)") == ["0.000", "0.350"]  # 7 / (3 + 12 + 5)

    with (out / "cases.csv").open(encoding="utf-8", newline="") as f:
        cases = list(csv.DictReader(f))
    assert cases[0] == {
        "case_id": "vi-year", "category": "ngày giờ", "spread": "0.167",
        "wer_a": "0.0", "wer_c": "0.167",
    }  # fmt: skip
    assert {r["case_id"]: r["wer_a"] for r in cases}["vi-hour"] == ""
