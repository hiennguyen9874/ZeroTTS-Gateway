#!/usr/bin/env python3
"""Build the self-contained weights directory for the torch-cuda runtime.

Run on the host: uv run --group torch-weights python scripts/build_zerotts_torch.py
    --revision <HF commit of --model>
Downloads two pinned files from the Hub (real files, no HF-cache symlinks), checks their
SHA-256, converts them to safetensors and verifies every tensor before writing --dst.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from gguf import GGUFReader
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from safetensors.numpy import load_file, save_file

MODEL_SOURCE = {
    "repo": "zeroweight-ai/ZeroTTS-GGUF",
    "revision": "92ca8651645d4733df56620b3aadc768a76f7c46",
    "file": "gguf/zerotts-f32.gguf",
    "sha256": "9b0490c317b369e9a77a17eb1f77f5d5fb98d6e3fcb82a3002e8ea62ebbab531",
}
CODEC_SOURCE = {
    "repo": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
    "revision": "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
    "file": "model-00001-of-00001.safetensors",
    "sha256": "34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67",
}
ASSETS_REPO = "zeroweight-ai/ZeroTTS"
CODEC_PREFIXES = ("decoder.", "quantizer.")  # the encoder is dropped
CODEC_ONNX = ("moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_step.onnx")
CODEC_TOLERANCE = 1e-6  # weight-norm folded into ONNX constants differs by ~1e-7
TOOLS = ("numpy", "gguf", "safetensors", "onnx", "huggingface_hub")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(source: dict, directory: Path) -> Path:
    path = Path(
        hf_hub_download(
            source["repo"], source["file"], revision=source["revision"], local_dir=directory
        )
    )
    if sha256(path) != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {source['repo']}/{source['file']}")
    return path


def convert_model(gguf_path: Path, dst: Path) -> int:
    reader = GGUFReader(gguf_path)
    tensors = {}
    for tensor in reader.tensors:
        if tensor.tensor_type.name != "F32":
            raise ValueError(f"{tensor.name} is {tensor.tensor_type.name}, expected F32")
        tensors[tensor.name] = np.ascontiguousarray(tensor.data)
    # Keep the hyperparameters (d_ff, local_*, eps, rope_theta) that config.json lacks. One
    # key only: safetensors writes metadata from a hash map, so several keys reorder per run.
    fields = {k: f.contents() for k, f in reader.fields.items() if not k.startswith("GGUF.")}
    save_file(tensors, dst, metadata={"gguf": json.dumps(fields)})
    saved = load_file(dst)
    if saved.keys() != tensors.keys():
        raise ValueError("model safetensors tensor names differ from the GGUF")
    for name, value in tensors.items():
        if saved[name].dtype != np.float32 or not np.array_equal(saved[name], value):
            raise ValueError(f"model tensor {name} differs from the GGUF")
    return len(tensors)


def convert_codec(src: Path, dst: Path) -> int:
    tensors = {k: v for k, v in load_file(src).items() if k.startswith(CODEC_PREFIXES)}
    save_file(tensors, dst)
    saved = load_file(dst)
    if saved.keys() != tensors.keys() or any(
        not np.array_equal(saved[k], v) for k, v in tensors.items()
    ):
        raise ValueError("codec safetensors differs from its source")
    return len(tensors)


def codec_candidates(tensors: dict) -> list[tuple[str, np.ndarray]]:
    """What an ONNX exporter may have turned each codec tensor into."""
    out = []
    for name, value in tensors.items():
        if name.endswith(".parametrizations.weight.original0"):
            continue
        if name.endswith(".parametrizations.weight.original1"):
            # torch weight_norm(dim=0): w = g * v / ||v||, the norm over all other dims.
            g = tensors[name.removesuffix("1") + "0"]
            axes = tuple(range(1, value.ndim))
            folded = g * value / np.sqrt((value * value).sum(axis=axes, keepdims=True))
            out.append((name.split(".parametrizations.")[0] + ".weight (folded)", folded))
            continue
        out.append((name, value))
        if value.ndim == 2:
            out.append((name + " (transposed)", value.T))
    return out


def verify_codec_onnx(codec: Path, onnx_dir: Path) -> dict:
    """Every ONNX codec initializer must equal some kept codec tensor."""
    tensors = load_file(codec)
    candidates = codec_candidates(tensors)
    report = {}
    for graph in CODEC_ONNX:
        model = onnx.load(str(onnx_dir / graph))
        folded, worst = 0, 0.0
        for init in model.graph.initializer:
            value = numpy_helper.to_array(init)
            if init.name.startswith("model.") and init.name[6:] in tensors:
                pool = [(init.name[6:], tensors[init.name[6:]])]
            else:
                pool = candidates
            diffs = [
                (float(np.abs(c.astype(np.float64) - value).max()), label)
                for label, c in pool
                if c.shape == value.shape
            ]
            diff, label = min(
                (item for item in diffs if np.isfinite(item[0])), default=(float("inf"), "")
            )
            if diff > CODEC_TOLERANCE:
                raise ValueError(f"{graph}: initializer {init.name} matches no codec tensor")
            folded += label.endswith("(folded)")
            worst = max(worst, diff)
        report[graph] = {
            "initializers": len(model.graph.initializer),
            "weight_norm": folded,
            "max_abs_diff": worst,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/ZeroTTS"))
    parser.add_argument("--dst", type=Path, default=Path("models/ZeroTTS-torch"))
    parser.add_argument(
        "--revision", required=True, help="HF commit of --model (40 hex characters)"
    )
    parser.add_argument(
        "--download-dir", type=Path, help="keep downloads here for reuse (default: temporary)"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    src, dst = args.model.resolve(), args.dst.absolute()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full HF commit hash")
    resolved_dst = dst.resolve()
    if src == resolved_dst or src in resolved_dst.parents or resolved_dst in src.parents:
        parser.error("source and output directories must not overlap")
    if args.download_dir is not None:
        args.download_dir = args.download_dir.resolve()
        if any(args.download_dir.is_relative_to(path) for path in (src, resolved_dst)):
            parser.error("download directory must not be inside source or output directories")
    if dst.is_symlink() or (dst.exists() and not dst.is_dir()):
        parser.error("output must be a directory, not a symlink or file")
    if dst.exists() and not args.force:
        parser.error("output already exists; use --force to replace it")
    if (src / "quantization.json").exists() or (src / "manifest.json").exists():
        parser.error("source must be the downloaded fp32 ONNX directory")
    required = [
        src / "config.json",
        src / "tokenizer.json",
        src / "silence_frame.npy",
        src / "voices",
        *(src / "onnx" / "codec" / name for name in CODEC_ONNX),
    ]
    if not all(path.exists() for path in required):
        parser.error("source is missing required ZeroTTS runtime assets")
    # If HF download metadata is present, reject a mislabeled or mixed-revision source.
    metadata = src / ".cache" / "huggingface" / "download"
    for path in metadata.rglob("*.metadata"):
        if path.read_text().splitlines()[0] != args.revision:
            parser.error("source HF metadata does not match --revision")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep an existing output intact until every file and the manifest are complete.
    with tempfile.TemporaryDirectory(prefix=".zerotts-torch-", dir=dst.parent) as tmp:
        downloads = args.download_dir or Path(tmp) / "downloads"
        output = Path(tmp) / "weights"
        output.mkdir()
        gguf_path = download(MODEL_SOURCE, downloads)
        codec_src = download(CODEC_SOURCE, downloads)

        for name in ("config.json", "tokenizer.json", "silence_frame.npy"):
            shutil.copy2(src / name, output / name)
        ignore = shutil.ignore_patterns(".cache", ".git")
        shutil.copytree(src / "voices", output / "voices", ignore=ignore)
        # Temporary (#17): the ONNX codec until the PyTorch codec lands (#19 removes it).
        shutil.copytree(src / "onnx" / "codec", output / "onnx" / "codec", ignore=ignore)

        counts = {
            "zerotts.safetensors": convert_model(gguf_path, output / "zerotts.safetensors"),
            "codec.safetensors": convert_codec(codec_src, output / "codec.safetensors"),
        }
        codec_check = verify_codec_onnx(output / "codec.safetensors", output / "onnx" / "codec")

        files = {
            path.relative_to(output).as_posix(): {
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "format": "zerotts-torch",
            "sources": {
                "model": {**MODEL_SOURCE, "size": gguf_path.stat().st_size},
                "codec": {**CODEC_SOURCE, "size": codec_src.stat().st_size},
                "assets": {"repo": ASSETS_REPO, "revision": args.revision},
            },
            "tools": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in TOOLS},
            },
            "tensors": counts,
            "codec_onnx_check": codec_check,
            "files": files,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        print("file                              bytes  tensors", flush=True)
        for name, count in counts.items():
            print(f"{name:22} {files[name]['size']:14} {count:8}", flush=True)
        for graph, stats in codec_check.items():
            print(
                f"{graph}: {stats['initializers']} initializers match "
                f"({stats['weight_norm']} folded weight-norm, "
                f"max |diff| {stats['max_abs_diff']:.1e})",
                flush=True,
            )
        if dst.exists():
            shutil.rmtree(dst)
        output.rename(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
