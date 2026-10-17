"""Gateway configuration: a TOML file plus environment overrides."""

import math
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
class ShortTextSpeedBand:
    """Inclusive upper syllable bound and operator-selected speed (ADR 0004)."""

    max_syllables: int
    speed: float

    def __post_init__(self):
        if type(self.max_syllables) is not int or self.max_syllables < 1:
            raise ValueError("band max_syllables must be a positive integer")
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, (int, float))
            or not math.isfinite(self.speed)
            or self.speed <= 0
        ):
            raise ValueError("band speed must be a finite positive number")


@dataclass(frozen=True)
class ModelDefaults:
    language: str | None = None
    normalize: bool = True
    # Operator option, off by default (ADR 0004): `speed` sent upstream when the caller sends no
    # `speed` and the engine text has at most short_text_max_syllables syllables. audio.cpp gives
    # short text too little time, so some voices drop its first phrase (issue #18).
    short_text_speed: float | None = None
    short_text_max_syllables: int = 12
    # Alternative to the scalar speed: ascending inclusive bounds, no speed above the last band.
    short_text_speed_bands: tuple[ShortTextSpeedBand, ...] = ()

    def __post_init__(self):
        if self.short_text_speed_bands:
            if self.short_text_speed is not None:
                raise ValueError("choose short_text_speed or short_text_speed_bands, not both")
            bounds = [band.max_syllables for band in self.short_text_speed_bands]
            if any(a >= b for a, b in zip(bounds, bounds[1:], strict=False)):
                raise ValueError("short_text_speed_bands must have strictly increasing bounds")
        if self.short_text_speed is not None and not self.short_text_speed > 0:
            raise ValueError("short_text_speed must be a positive number")
        if self.short_text_max_syllables < 1:
            raise ValueError("short_text_max_syllables must be at least 1")


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
    models = {}
    for model_id, values in data.pop("models", {}).items():
        bands = tuple(
            ShortTextSpeedBand(**band) for band in values.pop("short_text_speed_bands", [])
        )
        models[model_id] = ModelDefaults(**values, short_text_speed_bands=bands)
    if env.get(ENV_UPSTREAM_URL):
        data["upstream_url"] = env[ENV_UPSTREAM_URL]
    if env.get(ENV_HOST):
        data["host"] = env[ENV_HOST]
    return Settings(**data, api_key=env.get(ENV_API_KEY) or None, models=models)
