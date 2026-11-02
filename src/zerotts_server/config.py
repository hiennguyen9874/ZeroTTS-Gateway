"""zerotts_server configuration: a TOML file plus environment overrides."""

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENV_CONFIG = "ZEROTTS_SERVER_CONFIG"
ENV_MODEL_DIR = "ZEROTTS_SERVER_MODEL_DIR"
ENV_HOST = "ZEROTTS_SERVER_HOST"
ENV_PORT = "ZEROTTS_SERVER_PORT"
ENV_RUNTIME = "ZEROTTS_SERVER_RUNTIME"
ENV_DEVICE = "ZEROTTS_SERVER_DEVICE"
# How the model runs (ADR 0007); see runtime.py. Kept here so config needs no runtime imports.
RUNTIMES = ("onnx-cpu", "torch-cuda")


@dataclass(frozen=True)
class Sampling:
    """Operator-owned sampling parameters; callers only choose `seed`.

    Defaults are zerotts 0.1.5's benchmarked ones.
    """

    cfg_scale: float = 1.0
    text_temperature: float = 1.0
    text_topk: int = 50
    audio_temperature: float = 0.8
    audio_topk: int = 25
    audio_topp: float = 0.95
    audio_repetition_penalty: float = 1.2


MODES = ("offline", "streaming")


@dataclass(frozen=True)
class ModelConfig:
    # Used when the caller sends no `voice`; never the model's unstable null voice (ADR 0005).
    default_voice: str = "maichi"
    sampling: Sampling = field(default_factory=Sampling)
    # As audio.cpp's entry `mode`: "streaming" models answer only stream requests, and the
    # other way round.
    mode: str = "offline"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"model mode must be one of {MODES}")


@dataclass(frozen=True)
class Settings:
    # Local ZeroTTS weights (`hf download --local-dir`, not the HF cache: ADR 0005).
    model_dir: str = "models/ZeroTTS"
    # Voice packs (<name>/voice.npz + meta.json); defaults to <model_dir>/voices.
    voices_dir: str | None = None
    # Workers, each holding its own model instance and running one request at a time.
    workers: int = 1
    # "onnx-cpu" (onnxruntime on CPU) or "torch-cuda" (PyTorch on one GPU); never falls back.
    runtime: str = "onnx-cpu"
    # GPU for torch-cuda, "cuda" or "cuda:<index>"; unused by onnx-cpu.
    device: str = "cuda:0"
    # onnxruntime intra-op threads per worker (onnx-cpu only).
    threads: int = 8
    # Seconds a request waits for a free worker before it gets 503 server_busy.
    busy_timeout: float = 5.0
    host: str = "127.0.0.1"
    port: int = 8080
    models: Mapping[str, ModelConfig] = field(
        default_factory=lambda: {
            "zerotts": ModelConfig(),
            "zerotts-stream": ModelConfig(mode="streaming"),
        }
    )

    def __post_init__(self):
        for name in ("workers", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if isinstance(self.busy_timeout, bool) or not (
            isinstance(self.busy_timeout, int | float) and self.busy_timeout >= 0
        ):
            raise ValueError("busy_timeout must be a number of seconds >= 0")
        if self.runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {RUNTIMES}")
        if not re.fullmatch(r"cuda(:\d+)?", self.device):
            raise ValueError('device must be "cuda" or "cuda:<index>"')

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir) if self.voices_dir else Path(self.model_dir) / "voices"


def load_settings(path: str | Path | None = None, env: Mapping[str, str] = os.environ) -> Settings:
    path = path or env.get(ENV_CONFIG)
    data = {}
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # Relative paths in the config file are relative to that file.
        for key in ("model_dir", "voices_dir"):
            if data.get(key):
                data[key] = str(Path(path).parent / data[key])
    if "models" in data:
        data["models"] = {
            model_id: ModelConfig(
                **{k: v for k, v in values.items() if k != "sampling"},
                sampling=Sampling(**values.get("sampling", {})),
            )
            for model_id, values in data["models"].items()
        }
    if env.get(ENV_MODEL_DIR):
        data["model_dir"] = env[ENV_MODEL_DIR]
    if env.get(ENV_RUNTIME):
        data["runtime"] = env[ENV_RUNTIME]
    if env.get(ENV_DEVICE):
        data["device"] = env[ENV_DEVICE]
    if env.get(ENV_HOST):
        data["host"] = env[ENV_HOST]
    if env.get(ENV_PORT):
        data["port"] = int(env[ENV_PORT])
    return Settings(**data)
