"""CPU checks for the CUDA parity test's mismatch validator.

These test the acceptance check itself; HTTP cannot expose false positives in that check.
"""

from contextlib import nullcontext

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


def test_smoke_fixture_accepts_loading_followed_by_ready(monkeypatch):
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

    client = LoadingClient()
    monkeypatch.setattr(test_torch_cuda, "TestClient", lambda app: nullcontext(client))
    monkeypatch.setattr(test_torch_cuda.time, "sleep", lambda seconds: None)

    fixture = test_torch_cuda.client.__wrapped__("cuda:0")
    assert next(fixture) is client
    with pytest.raises(StopIteration):
        next(fixture)
