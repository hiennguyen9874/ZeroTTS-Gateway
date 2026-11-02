"""Runtime seam: how the upstream executes the ZeroTTS model on one kind of hardware (ADR 0007).

A frame generator turns one segment (text, voice latents, sampling, per-request RNG) into codes,
one (1, K) int64 frame at a time; the engine joins it with a codec. `onnx-cpu` runs the vendored
ONNX graphs, `torch-cuda` the PyTorch port in `torch_runtime`. Both draw their random numbers
from the request's RNG in the same order, so equal weights and equal draws give equal codes.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import RUNTIMES, Sampling

# manifest.json "format" of a directory built by scripts/build_zerotts_torch.py.
TORCH_FORMAT = "zerotts-torch"
# torch-cuda compute precision (the weights stay fp32 on disk).
TORCH_DTYPE = "float16"
# zerotts 0.1.5's synthesize defaults.
MIN_FRAMES = 4
MAX_FRAMES = 1500
EOA_EXTRA_FRAMES = 1


def draw_u(rng: np.random.Generator, num_codebooks: int) -> tuple[np.ndarray, np.ndarray]:
    """One frame's sampler draws: the control channel's, then one per codebook."""
    ctrl = rng.random((1,)).astype(np.float32)
    audio = rng.random((1, num_codebooks)).astype(np.float32)
    return ctrl, audio


class FrameGenerator(Protocol):
    n_voice_queries: int

    def warmup(self) -> None:
        """Run the hot path once so first-request setup cost lands at load time."""

    def frames(
        self, text: str, voice_emb: np.ndarray, sampling: Sampling, rng: np.random.Generator
    ) -> Iterator[np.ndarray]:
        """Yield the segment's codes, one (1, K) int64 frame at a time."""


class _SeededSession:
    """Wraps local_frame_decode.onnx so its random draws come from a per-request RNG.

    zerotts draws them from numpy's global RNG, which other requests and libraries share.
    """

    def __init__(self, session):
        self.session = session
        self.rng = np.random.default_rng()

    def run(self, output_names, feeds):
        feeds["ctrl_random_u"], feeds["audio_random_u"] = draw_u(
            self.rng, feeds["audio_random_u"].shape[1]
        )
        return self.session.run(output_names, feeds)

    def __getattr__(self, name):
        return getattr(self.session, name)


class OnnxFrameGenerator:
    """The vendored zerotts frame loop on onnxruntime CPU."""

    def __init__(self, model_dir: Path, threads: int):
        from zerotts_server.model.synthesizer import ZeroTTS

        self.tts = ZeroTTS(model_dir, intra_op_num_threads=threads, warmup=False)
        self._session = _SeededSession(self.tts.local_frame_decode_sess)
        self.tts.local_frame_decode_sess = self._session
        self.n_voice_queries = self.tts.n_voice_queries

    def warmup(self) -> None:
        self.tts.warmup()

    def frames(self, text, voice_emb, sampling, rng):
        self._session.rng = rng
        yield from self.tts._generate_frames(
            text,
            MIN_FRAMES,
            MAX_FRAMES,
            voice_emb=voice_emb,
            eoa_extra_frames=EOA_EXTRA_FRAMES,
            **asdict(sampling),
        )


def load_runtime(model_dir: Path, runtime: str, device: str, threads: int):
    """(frame generator, codec) for one worker. Never falls back to another runtime."""
    if runtime == "onnx-cpu":
        generator = OnnxFrameGenerator(model_dir, threads)
        return generator, generator.tts.codec
    if runtime == "torch-cuda":
        from .torch_runtime import TorchCodec, TorchZeroTTS

        generator = TorchZeroTTS(model_dir, device, dtype=TORCH_DTYPE)
        codec = TorchCodec(model_dir, generator.device, generator.stream, generator.cuda_graph)
        return generator, codec
    raise ValueError(f"runtime must be one of {RUNTIMES}")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def weights_identity(model_dir: str | Path, runtime: str, device: str) -> dict:
    """Check that model_dir holds weights for `runtime`; return the `weights` log record.

    The identity comes from the directory's manifest; this is not an integrity scan.
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    is_torch = manifest.get("format") == TORCH_FORMAT
    if runtime == "torch-cuda":
        if not is_torch:
            raise ValueError(f"runtime torch-cuda needs a {TORCH_FORMAT} weights directory")
        return {
            "model_dir": str(model_dir),
            "runtime": runtime,
            "device": device,
            "weights": "fp32",
            "dtype": TORCH_DTYPE,
            "format": TORCH_FORMAT,
            "source_revisions": {
                name: source["revision"] for name, source in manifest["sources"].items()
            },
            "sha256": {
                name: manifest["files"][name]["sha256"]
                for name in ("zerotts.safetensors", "codec.safetensors")
            },
            "torch": _package_version("torch"),
        }
    if is_torch:
        raise ValueError(f"runtime {runtime} cannot load a {TORCH_FORMAT} weights directory")
    fields = {"model_dir": str(model_dir), "runtime": runtime, "device": "cpu", "weights": "fp32"}
    quantization_path = model_dir / "quantization.json"
    if quantization_path.is_file():
        quantization = json.loads(quantization_path.read_text())
        fields.update(
            weights="int8",
            source_revision=quantization["source_revision"],
            sha256={name: graph["sha256"] for name, graph in quantization["graphs"].items()},
        )
    return fields
