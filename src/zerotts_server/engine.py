"""Engine seam: the narrow interface the HTTP app synthesizes through.

Tests inject a fake engine (`create_app(settings, engine=...)`); production uses ZeroTTSEngine.
"""

import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Sampling

SAMPLE_RATE = 48_000
# Silence between segments; zerotts's own `say --chunk` default (gap_sec).
SEGMENT_GAP_SEC = 0.15


@dataclass(frozen=True)
class VoiceInfo:
    name: str
    display_name: str = ""
    description: str = ""
    gender: str = ""
    language: str = ""


class Engine(Protocol):
    def load(self) -> None:
        """Load weights and warm up; blocking. /health is 200 only after this returns."""

    def voices(self) -> Mapping[str, VoiceInfo]:
        """Voices by name; valid only after load()."""

    def synthesize(
        self, segments: Sequence[str], voice: str, seed: int | None, sampling: Sampling
    ) -> Iterator[np.ndarray]:
        """Synthesize the segments of one engine text in order, as one utterance.

        Yield float32 mono chunks at SAMPLE_RATE. A None seed means fresh entropy.
        """


class _SeededSession:
    """Wraps local_frame_decode.onnx so its random draws come from a per-request RNG.

    zerotts draws them from numpy's global RNG, which other requests and libraries share.
    """

    def __init__(self, session):
        self.session = session
        self.rng = np.random.default_rng()

    def run(self, output_names, feeds):
        for name in ("ctrl_random_u", "audio_random_u"):
            feeds[name] = self.rng.random(feeds[name].shape).astype(np.float32)
        return self.session.run(output_names, feeds)

    def __getattr__(self, name):
        return getattr(self.session, name)


class ZeroTTSEngine:
    """One zerotts.ZeroTTS instance; synthesizes one request at a time."""

    def __init__(self, model_dir: str | Path, voices_dir: str | Path, threads: int):
        self.model_dir, self.voices_dir, self.threads = Path(model_dir), Path(voices_dir), threads
        self._lock = threading.Lock()
        self._voices: dict = {}

    def load(self) -> None:
        from zerotts import ZeroTTS
        from zerotts.voices import discover_packs, load_voice_dir

        tts = ZeroTTS(self.model_dir, intra_op_num_threads=self.threads, warmup=False)
        tts.local_frame_decode_sess = _SeededSession(tts.local_frame_decode_sess)
        # Voices come from voices_dir only, not zerotts's per-user install directory.
        self._voices = {
            name: load_voice_dir(path, name, expect_queries=tts.n_voice_queries)
            for name, path in discover_packs(self.voices_dir).items()
        }
        tts.warmup()
        self._tts = tts

    def voices(self) -> Mapping[str, VoiceInfo]:
        return {
            name: VoiceInfo(name, v.display_name, v.description, v.gender, v.language)
            for name, v in self._voices.items()
        }

    def synthesize(
        self, segments: Sequence[str], voice: str, seed: int | None, sampling: Sampling
    ) -> Iterator[np.ndarray]:
        gap = np.zeros(int(SEGMENT_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
        with self._lock:
            # One RNG per request, so a seed reproduces the whole multi-segment audio.
            self._tts.local_frame_decode_sess.rng = np.random.default_rng(seed)
            for i, segment in enumerate(segments):
                if i:
                    yield gap
                audio = self._tts.synthesize(
                    segment, voice=self._voices[voice].emb, **asdict(sampling)
                )
                yield audio.reshape(-1).astype(np.float32, copy=False)
