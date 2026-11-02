"""CPU checks for the CUDA parity test's acceptance helpers.

These test the acceptance check itself; HTTP cannot expose false positives in that check.
"""

import httpx
import numpy as np
import pytest

from tests import test_torch_cuda
from tests.test_torch_cuda import near_tie


@pytest.mark.parametrize(
    ("reference", "perturbed", "u", "topk", "topp", "ours", "theirs", "expected"),
    [
        ([0.0, -100.0], [0.0, -100.0], 0.5, 2, 1.0, 1, 0, False),
        ([1.0, 0.0, -100.0], [1.0, 0.0, -100.0], 0.5, 3, 0.73105858, 2, 0, False),
        ([1.0, 1.00001], [1.00001, 1.0], 0.1, 1, 1.0, 0, 1, True),
        ([0.0, 0.0], [0.0001, 0.0], 0.50001, 2, 1.0, 0, 1, True),
        ([0.00004, 0.0], [-0.00004, 0.0], 0.9, 2, 0.5, 1, 0, True),
        ([0.0, 0.0], [0.0, 0.0], 0.1, 2, 1.0, 1, 0, False),
        ([1.0, 0.0], [0.0, 1.0], 0.1, 1, 1.0, 1, 0, False),
    ],
)
def test_near_tie_requires_bounded_logits_that_explain_both_picks(
    reference, perturbed, u, topk, topp, ours, theirs, expected
):
    assert (
        near_tie(np.array(reference), np.array(perturbed), u, topk, topp, ours, theirs) == expected
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_codec_metrics_reject_nonfinite_audio(value, side):
    actual = np.ones((1, 3840), dtype=np.float32)
    reference = actual.copy()
    (actual if side == "actual" else reference)[0, 100] = value
    with pytest.raises(AssertionError, match="finite"):
        test_torch_cuda.codec_metrics(actual, reference)


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_codec_metrics_accept_identical_audio_including_silence(value):
    audio = np.full((1, 3840), value, dtype=np.float32)
    assert test_torch_cuda.codec_metrics(audio, audio) == (0.0, np.inf)


def test_codec_metrics_measure_error_and_snr():
    reference = np.ones((1, 3840), dtype=np.float32)
    difference, snr = test_torch_cuda.codec_metrics(reference + 0.125, reference)
    assert difference == 0.125
    assert snr == pytest.approx(18.0617997398)


def test_codec_metrics_reject_noise_against_silence():
    difference, snr = test_torch_cuda.codec_metrics(np.ones((1, 3840)), np.zeros((1, 3840)))
    assert difference == 1.0
    assert snr == -np.inf


def test_smoke_readiness_accepts_loading_followed_by_ready(monkeypatch):
    responses = iter(
        [
            httpx.Response(503, json={"error": {"type": "model_loading"}}),
            httpx.Response(200, json={"status": "ok"}),
        ]
    )

    class LoadingClient:
        def get(self, path):
            assert path == "/health"
            return next(responses)

    monkeypatch.setattr(test_torch_cuda.time, "sleep", lambda seconds: None)

    test_torch_cuda.wait_until_ready(LoadingClient())
    assert next(responses, None) is None
