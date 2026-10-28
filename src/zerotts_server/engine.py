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
# Streaming decode schedule (spec #1): the first segment ramps 1, 2, 4 ... frames per codec
# call for low time-to-first-audio, capped here; later segments decode at the cap.
STREAM_MAX_CHUNK_FRAMES = 16
# Codec silence frames between streamed segments: about SEGMENT_GAP_SEC at 12.5 frames/s.
STREAM_GAP_FRAMES = 2


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
        self,
        segments: Sequence[str],
        voice: str,
        seed: int | None,
        sampling: Sampling,
        stream: bool = False,
    ) -> Iterator[np.ndarray]:
        """Synthesize the segments of one engine text in order, as one utterance.

        Yield float32 mono chunks at SAMPLE_RATE. A None seed means fresh entropy. With
        `stream`, yield audio as it is generated; closing the iterator stops synthesis.
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
        self._silence_frame = None

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
        silence = self.model_dir / "silence_frame.npy"
        if silence.is_file():
            self._silence_frame = np.load(silence)
        tts.warmup()
        self._tts = tts

    def voices(self) -> Mapping[str, VoiceInfo]:
        return {
            name: VoiceInfo(name, v.display_name, v.description, v.gender, v.language)
            for name, v in self._voices.items()
        }

    def synthesize(
        self,
        segments: Sequence[str],
        voice: str,
        seed: int | None,
        sampling: Sampling,
        stream: bool = False,
    ) -> Iterator[np.ndarray]:
        with self._lock:
            # One RNG per request, so a seed reproduces the whole multi-segment audio.
            self._tts.local_frame_decode_sess.rng = np.random.default_rng(seed)
            if stream:
                yield from self._stream(segments, voice, sampling)
                return
            gap = np.zeros(int(SEGMENT_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
            for i, segment in enumerate(segments):
                if i:
                    yield gap
                audio = self._tts.synthesize(
                    segment, voice=self._voices[voice].emb, **asdict(sampling)
                )
                yield audio.reshape(-1).astype(np.float32, copy=False)

    def _stream(
        self, segments: Sequence[str], voice: str, sampling: Sampling
    ) -> Iterator[np.ndarray]:
        """Like zerotts's webui generate_stream: one codec streaming decoder for all segments.

        A decoder per segment would restart the causal decoder's state and click at each
        boundary. Uses zerotts 0.1.5's frame generator, as its synthesize_stream does.
        """
        from zerotts.synthesizer import DEFAULT_MAX_FRAMES

        tts = self._tts
        voice_emb = tts.resolve_voice(self._voices[voice].emb)
        decoder = tts.codec.streaming_decoder()

        def decode(frames: list) -> np.ndarray:
            audio = decoder.decode_chunk(np.stack(frames, axis=-1))
            return audio.reshape(-1).astype(np.float32, copy=False)

        try:
            for i, segment in enumerate(segments):
                if i:
                    if self._silence_frame is None:
                        yield np.zeros(int(SEGMENT_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
                    else:
                        yield decode([self._silence_frame] * STREAM_GAP_FRAMES)
                target = 1 if i == 0 else STREAM_MAX_CHUNK_FRAMES
                frames: list = []
                for codes in tts._generate_frames(
                    segment, 4, DEFAULT_MAX_FRAMES, voice_emb=voice_emb, **asdict(sampling)
                ):
                    frames.append(codes)
                    if len(frames) >= target:
                        yield decode(frames)
                        frames = []
                        target = min(STREAM_MAX_CHUNK_FRAMES, target * 2)
                if frames:
                    yield decode(frames)
        finally:
            decoder.close()
