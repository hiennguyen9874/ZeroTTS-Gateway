#!/usr/bin/env python3
"""Create local, self-contained int8 weights; never contacts the Hub.

Run on the host: uv run --group quantize python scripts/quantize_zerotts.py
    --revision <source HF commit>
Based on references/ZeroTTS-repo/tools/quantize_onnx_int8.py.
"""

import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

GRAPHS = ("text_encoder.onnx", "prefix_step.onnx", "local_frame_decode.onnx")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantize_one(src: Path, dst: Path) -> dict:
    dst.chmod(dst.stat().st_mode | stat.S_IWUSR)
    folded = dst.with_suffix(".opt.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    options.intra_op_num_threads = 1
    try:
        # Folding exposes weights behind Identity nodes to the MatMul quantizer.
        ort.InferenceSession(str(src), options, providers=["CPUExecutionProvider"])
        quantize_dynamic(
            str(folded), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
        )
    finally:
        folded.unlink(missing_ok=True)
    count = sum(
        node.op_type == "MatMulInteger"
        for node in onnx.load(str(dst), load_external_data=False).graph.node
    )
    if not count:
        raise ValueError(f"no MatMulInteger nodes in {dst.name}")
    return {
        "source_sha256": sha256(src),
        "sha256": sha256(dst),
        "matmul_integer": count,
        "size_in": src.stat().st_size,
        "size_out": dst.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/ZeroTTS"))
    parser.add_argument("--dst", type=Path, default=Path("models/ZeroTTS-int8"))
    parser.add_argument("--revision", required=True, help="source HF commit (40 hex characters)")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    src, dst = args.model.resolve(), args.dst.absolute()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full HF commit hash")
    resolved_dst = dst.resolve()
    if src == resolved_dst or src in resolved_dst.parents or resolved_dst in src.parents:
        parser.error("source and output directories must not overlap")
    if dst.is_symlink() or (dst.exists() and not dst.is_dir()):
        parser.error("output must be a directory, not a symlink or file")
    if dst.exists() and not args.force:
        parser.error("output already exists; use --force to replace it")
    if (src / "quantization.json").exists():
        parser.error("source must be fp32, not an already quantized directory")
    required = [
        src / "config.json",
        src / "tokenizer.json",
        src / "null_voice_emb.npy",
        src / "silence_frame.npy",
        src / "voices",
        src / "onnx" / "codec",
        *(src / "onnx" / name for name in GRAPHS),
    ]
    if not all(path.exists() for path in required):
        parser.error("source is missing required ZeroTTS weights or runtime assets")
    # If HF download metadata is present, reject a mislabeled or mixed-revision source.
    metadata = src / ".cache" / "huggingface" / "download"
    for path in metadata.rglob("*.metadata"):
        if path.read_text().splitlines()[0] != args.revision:
            parser.error("source HF metadata does not match --revision")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep an existing output intact until every graph and the manifest are complete.
    with tempfile.TemporaryDirectory(prefix=".zerotts-int8-", dir=dst.parent) as tmp:
        output = Path(tmp) / "weights"
        shutil.copytree(src, output, ignore=shutil.ignore_patterns(".cache", ".git"))
        manifest = {
            "source_revision": args.revision,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "graphs": {},
        }
        print("graph                         fp32 bytes    int8 bytes  MatMulInteger", flush=True)
        for name in GRAPHS:
            stats = quantize_one(src / "onnx" / name, output / "onnx" / name)
            manifest["graphs"][f"onnx/{name}"] = stats
            print(
                f"{name:28} {stats['size_in']:12} {stats['size_out']:12} "
                f"{stats['matmul_integer']:14}",
                flush=True,
            )
        (output / "quantization.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if dst.exists():
            shutil.rmtree(dst)
        output.rename(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
