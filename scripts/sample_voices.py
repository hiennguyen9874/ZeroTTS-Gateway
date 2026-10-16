#!/usr/bin/env python3
"""Lấy mẫu vài giọng từ các dataset tiếng Việt trên HF để nghe thử và chọn làm voice clone.

Chỉ đọc một phần nhỏ dataset (theo row group parquet / record batch arrow) qua HfFileSystem,
không tải nguyên shard.

  export HF_TOKEN=hf_xxx   # đã bấm "Agree" trên trang của từng dataset
  uv run --with pyarrow --with huggingface_hub \\
    python scripts/sample_voices.py sample --dataset all -n 10
  # nghe: mở voices_candidates/index.html (hoặc python -m http.server -d voices_candidates)
  uv run --with pyarrow --with huggingface_hub \\
    python scripts/sample_voices.py install vieneu_xxx phoaudiobook_yyy

Yêu cầu: ffmpeg/ffprobe trong PATH.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "voices_candidates"
VOICES = ROOT / "voices"
SMALL_FILE = 200 << 20  # file nhỏ hơn mức này thì tải hẳn vào cache HF
# max-dur riêng (PhoAudiobook toàn clip >= 10s)
MAX_DUR = {"phoaudiobook": 12.5}
# số dòng tối đa đọc mỗi file (VieNeu: 300 dòng = 3 batch arrow, ~90 MB)
MAX_ROWS = {"vieneu": 300}
# speaker id -> giọng thật (VieNeu: "jellyfish1010_0052" -> "jellyfish1010")
SPEAKER_KEY = {"vieneu": lambda s: s.rsplit("_", 1)[0]}

DATASETS = {
    # name: (repo, file pattern, kind, text column, speaker column)
    "phoaudiobook": (
        "thivux/phoaudiobook",
        ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
        "parquet",
        "text",
        "speaker",
    ),
    "vieneu": (
        "pnnbao-ump/VieNeu-TTS-140h",
        [f"data-{i:05d}-of-00049.arrow" for i in range(49)],
        "arrow",
        "text",
        "speaker",
    ),
    "vietspeech": (
        "NhutP/VietSpeech",
        [f"data/train-{i:05d}-of-00027.parquet" for i in range(0, 27, 3)],
        "parquet",
        "transcription",
        None,
    ),
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def safe_id(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:60]


def audio_bytes(cell) -> bytes | None:
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def iter_parquet(fs, path, columns):
    import pyarrow.parquet as pq

    with fs.open(path, "rb", block_size=8 << 20) as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=cols).to_pylist()


def iter_arrow(fs, path, columns):
    import pyarrow as pa

    with fs.open(path, "rb", block_size=8 << 20) as f:
        try:
            reader = pa.ipc.open_stream(f)
            batches = iter(reader)
        except pa.ArrowInvalid:
            f.seek(0)
            r = pa.ipc.open_file(f)
            batches = (r.get_batch(i) for i in range(r.num_record_batches))
        for b in batches:
            cols = [c for c in columns if c in b.schema.names]
            yield from b.select(cols).to_pylist()


def convert(raw: bytes, dst: Path) -> float | None:
    """Đổi sang WAV mono 24 kHz, cắt lặng đầu/cuối. Trả về độ dài (s)."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        tmp.write(raw)
        tmp.flush()
        trim = (
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,"
            "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,areverse"
        )
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                tmp.name,
                "-ac",
                "1",
                "-ar",
                "24000",
                "-af",
                trim,
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            return None
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dst)],
        capture_output=True,
        text=True,
    )
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def est_duration(row, raw: bytes) -> float | None:
    if row.get("duration") is not None:
        return float(row["duration"])
    try:
        with wave.open(io.BytesIO(raw)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001  (không phải WAV PCM, để ffprobe đo sau)
        return None


def open_reader(fs, repo, fname, kind, columns):
    path = f"datasets/{repo}/{fname}"
    if fs.info(path)["size"] < SMALL_FILE:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo, fname, repo_type="dataset")
        import fsspec

        lfs, path = fsspec.filesystem("file"), local
    else:
        lfs = fs
    return (iter_parquet if kind == "parquet" else iter_arrow)(lfs, path, columns)


def sample_dataset(
    name, fs, out: Path, n: int, min_dur: float, max_dur: float, max_rows: int, target: float = 8.0
):
    repo, files, kind, text_col, spk_col = DATASETS[name]
    columns = ["audio", text_col] + [c for c in (spk_col, "gender", "duration") if c]
    got = []
    seen_spk: set[str] = set()
    ddir = out / name
    ddir.mkdir(parents=True, exist_ok=True)

    def emit(spk, row, text, raw):
        key = safe_id(spk) if spk else f"{len(got):03d}_{abs(hash(text)) % 10**6:06d}"
        vid = f"{name}_{key}"
        dst = ddir / f"{vid}.wav"
        dur = convert(raw, dst)
        if dur is None or not (min_dur - 0.5 <= dur <= max_dur):
            dst.unlink(missing_ok=True)
            return
        if spk is not None:
            seen_spk.add(spk)
        got.append(
            {
                "id": vid,
                "dataset": name,
                "repo": repo,
                "speaker": spk,
                "gender": row.get("gender"),
                "duration": round(dur, 2),
                "text": text,
                "wav": str(dst.relative_to(out)),
            }
        )
        log(f"  + {vid} ({dur:.1f}s) {text[:70]}")

    for fname in files:
        if len(got) >= n:
            break
        log(f"[{name}] đọc {fname}")
        best: dict[str, tuple] = {}  # speaker -> (score, row, text, raw), chỉ giữ clip tốt nhất
        rows = file_got = 0
        per_file = -(-n // len(files))
        try:
            for row in open_reader(fs, repo, fname, kind, columns):
                rows += 1
                if rows > max_rows:
                    break
                spk = SPEAKER_KEY.get(name, str)(str(row.get(spk_col))) if spk_col else None
                if spk is not None and spk in seen_spk:
                    continue
                text = " ".join((row.get(text_col) or "").replace("|", " ").split())
                raw = audio_bytes(row.get("audio"))
                if not raw or len(text) < 10:
                    continue
                d = est_duration(row, raw)
                if d is not None and not (min_dur <= d <= max_dur):
                    continue
                score = abs((d if d is not None else target) - target)
                if spk is None:  # không có speaker id: lấy luôn, rải đều qua các file
                    if len(got) < n and file_got < per_file:
                        file_got += 1
                        emit(None, row, text, raw)
                    else:
                        break
                    continue
                if spk not in best or score < best[spk][0]:
                    best[spk] = (score, row, text, raw)
        except Exception as e:  # noqa: BLE001
            log(f"  ! lỗi khi đọc {fname}: {e}")
        for spk, (_, row, text, raw) in best.items():
            if len(got) >= n:
                break
            emit(spk, row, text, raw)
    return got


def write_index(out: Path, items):
    (out / "candidates.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(out / "prompt_text", "w", encoding="utf-8") as f:
        for it in items:
            f.write(f"{it['id']}|{it['text']}\n")
    rows = "\n".join(
        f"<tr><td><code>{html.escape(it['id'])}</code></td><td>{it['dataset']}</td>"
        f"<td>{it.get('gender') or ''}</td><td>{it['duration']}s</td>"
        f"<td><audio controls preload=none src='{html.escape(it['wav'])}'></audio></td>"
        f"<td>{html.escape(it['text'])}</td></tr>"
        for it in items
    )
    (out / "index.html").write_text(
        "<meta charset=utf-8><title>Voice candidates</title><style>body{font-family:sans-serif}"
        "td{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:middle}</style>"
        f"<h2>{len(items)} candidates</h2>"
        "<table><tr><th>id</th><th>dataset</th><th>gender</th><th>dur</th>"
        f"<th>audio</th><th>transcript</th></tr>{rows}</table>",
        encoding="utf-8",
    )


def cmd_sample(a):
    from huggingface_hub import HfFileSystem

    token = os.environ.get("HF_TOKEN") or None
    fs = HfFileSystem(token=token)
    out = Path(a.out)
    names = list(DATASETS) if a.dataset == "all" else a.dataset.split(",")
    old = []
    if (out / "candidates.json").exists() and not a.fresh:
        old = [
            it
            for it in json.loads((out / "candidates.json").read_text())
            if it["dataset"] not in names
        ]
    items = old
    for name in names:
        items += sample_dataset(
            name,
            fs,
            out,
            a.n,
            a.min_dur,
            a.max_dur or MAX_DUR.get(name, 10.0),
            a.max_rows or MAX_ROWS.get(name, 3000),
        )
    write_index(out, items)
    log(f"\nXong: {len(items)} giọng -> {out}/index.html")


def cmd_install(a):
    out = Path(a.out)
    items = {
        it["id"]: it for it in json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    }
    VOICES.mkdir(exist_ok=True)
    pt = VOICES / "prompt_text"
    lines = pt.read_text(encoding="utf-8").splitlines() if pt.exists() else []
    for vid in a.ids:
        it = items.get(vid)
        if not it:
            log(f"! không có {vid}")
            continue
        name = a.rename.get(vid, vid) if a.rename else vid
        shutil.copy(out / it["wav"], VOICES / f"{name}.wav")
        lines = [ln for ln in lines if not ln.startswith(name + "|")] + [f"{name}|{it['text']}"]
        log(f"+ voices/{name}.wav")
    pt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--dataset", default="all", help="all | phoaudiobook,vieneu,vietspeech")
    s.add_argument("-n", type=int, default=10, help="số giọng mỗi dataset")
    s.add_argument("--min-dur", type=float, default=3.0)
    s.add_argument("--max-dur", type=float, default=None, help="mặc định 10 (phoaudiobook 12.5)")
    s.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="số dòng tối đa đọc mỗi file (mặc định 3000, vieneu 300)",
    )
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--fresh", action="store_true", help="xóa danh sách cũ")
    s.set_defaults(func=cmd_sample)
    i = sub.add_parser("install", help="copy giọng đã chọn vào voices/ + prompt_text")
    i.add_argument("ids", nargs="+")
    i.add_argument("--out", default=str(DEFAULT_OUT))
    i.add_argument(
        "--as", dest="rename_list", action="append", default=[], help="old=new để đổi tên voice id"
    )
    i.set_defaults(func=cmd_install)
    a = p.parse_args()
    if a.cmd == "install":
        a.rename = dict(x.split("=", 1) for x in a.rename_list)
    a.func(a)


if __name__ == "__main__":
    main()
