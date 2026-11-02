"""Engine-level benchmark checks; HTTP cannot expose benchmark aggregation failures."""

import numpy as np
import pytest

from scripts import bench_zerotts
from zerotts_server.config import Sampling


class FakeEngine:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def synthesize(self, segments, voice, seed, sampling, *, stream):
        self.calls += 1
        yield np.ones(480, dtype=np.float32)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated synthesis failure")
        yield np.ones(480, dtype=np.float32)


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_benchmark_counts_every_worker_run(stream):
    engines = [FakeEngine(), FakeEngine()]
    result = bench_zerotts._concurrent(engines, ["hello"], "maichi", 42, Sampling(), stream, 3)

    assert result["requests"] == 6
    assert result["audio_s"] == pytest.approx(0.12)
    assert len(result["audio_sha256"]) == 1
    assert [engine.calls for engine in engines] == [3, 3]
    if stream:
        assert result["ttfa_s_max"] >= result["ttfa_s_median"] >= 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fail_after", [0, 1])
def test_concurrent_benchmark_rejects_worker_failure_after_partial_audio(stream, fail_after):
    engines = [FakeEngine(), FakeEngine(fail_after=fail_after)]
    with pytest.raises(RuntimeError, match="simulated synthesis failure"):
        bench_zerotts._concurrent(engines, ["hello"], "maichi", 42, Sampling(), stream, 3)

    # All threads finish before failure is returned; no benchmark work escapes the call.
    assert [engine.calls for engine in engines] == [3, fail_after + 1]
