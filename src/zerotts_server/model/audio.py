"""ZeroTTS WAV output helper."""

import numpy as np

SAMPLE_RATE = 48_000


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a (..., T) float array (leading dims squeezed) as 16-bit PCM wav."""
    import soundfile as sf

    sf.write(path, np.asarray(audio).squeeze(), sample_rate, subtype="PCM_16")
