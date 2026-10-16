"""Gateway configuration: a TOML file plus environment overrides."""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENV_CONFIG = "OMNIVOICE_GATEWAY_CONFIG"
ENV_UPSTREAM_URL = "OMNIVOICE_GATEWAY_UPSTREAM_URL"
ENV_API_KEY = "OMNIVOICE_GATEWAY_API_KEY"
ENV_HOST = "OMNIVOICE_GATEWAY_HOST"


@dataclass(frozen=True)
class ModelDefaults:
    language: str | None = None
    normalize: bool = True


@dataclass(frozen=True)
class Settings:
    upstream_url: str = "http://127.0.0.1:8080"
    # Secret: only read from the environment, never from the config file.
    api_key: str | None = None
    deadline_s: float = 300.0
    # Connect only; once connected, deadline_s is the only limit on waiting and streaming.
    connect_timeout_s: float = 3.0
    max_input_chars: int = 5000
    host: str = "127.0.0.1"
    port: int = 8000
    # Pronunciation lexicon CSV; a relative path in the config file is relative to that file.
    lexicon_path: str | None = None
    models: Mapping[str, ModelDefaults] = field(default_factory=dict)


def load_settings(path: str | Path | None = None, env: Mapping[str, str] = os.environ) -> Settings:
    path = path or env.get(ENV_CONFIG)
    data = {}
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    if data.get("lexicon_path"):
        data["lexicon_path"] = str(Path(path).parent / data["lexicon_path"])
    models = {
        model_id: ModelDefaults(**values) for model_id, values in data.pop("models", {}).items()
    }
    if env.get(ENV_UPSTREAM_URL):
        data["upstream_url"] = env[ENV_UPSTREAM_URL]
    if env.get(ENV_HOST):
        data["host"] = env[ENV_HOST]
    return Settings(**data, api_key=env.get(ENV_API_KEY) or None, models=models)
