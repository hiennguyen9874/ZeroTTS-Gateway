"""zerotts_server configuration: a TOML file plus environment overrides."""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENV_CONFIG = "ZEROTTS_SERVER_CONFIG"
ENV_MODEL_DIR = "ZEROTTS_SERVER_MODEL_DIR"
ENV_HOST = "ZEROTTS_SERVER_HOST"


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


@dataclass(frozen=True)
class ModelConfig:
    # Used when the caller sends no `voice`; never the model's unstable null voice (ADR 0005).
    default_voice: str = "maichi"
    sampling: Sampling = field(default_factory=Sampling)


@dataclass(frozen=True)
class Settings:
    # Local ZeroTTS weights (`hf download --local-dir`, not the HF cache: ADR 0005).
    model_dir: str = "models/ZeroTTS"
    # Voice packs (<name>/voice.npz + meta.json); defaults to <model_dir>/voices.
    voices_dir: str | None = None
    # onnxruntime intra-op threads.
    threads: int = 8
    host: str = "127.0.0.1"
    port: int = 8080
    models: Mapping[str, ModelConfig] = field(default_factory=lambda: {"zerotts": ModelConfig()})

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
    if env.get(ENV_HOST):
        data["host"] = env[ENV_HOST]
    return Settings(**data)
