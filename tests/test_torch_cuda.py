"""Opt-in tests of the torch-cuda runtime on a GPU (marker `cuda`, ADR 0007).

Skipped without torch, a CUDA device or models/ZeroTTS-torch; the parity tests also need the fp32
ONNX weights in models/ZeroTTS. Run on a free GPU, e.g.:

  CUDA_VISIBLE_DEVICES=2 uv run --extra server-cuda --group torch-weights \\
      python -m pytest -m cuda tests/test_torch_cuda.py -s

`torch-weights` provides `onnx`, used to expose the ONNX sampler's logits for the comparison.
ZEROTTS_SERVER_DEVICE picks the device (default cuda:0).
"""

import io
import json
import os
import time
import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zerotts_server import create_app
from zerotts_server.app import split_segments
from zerotts_server.config import Sampling, Settings
from zerotts_server.model.voices import load_voice_dir
from zerotts_server.runtime import EOA_EXTRA_FRAMES, MAX_FRAMES, MIN_FRAMES, draw_u

ROOT = Path(__file__).parent.parent
TORCH_DIR = ROOT / "models" / "ZeroTTS-torch"
ONNX_DIR = ROOT / "models" / "ZeroTTS"
DEVICE = os.environ.get("ZEROTTS_SERVER_DEVICE", "cuda:0")
VOICES = ("maichi", "quangminh")
# fp32 on both sides: |torch - onnx| <= ATOL + RTOL * max|onnx| per tensor. Logits reach ~200,
# where one fp32 ulp is ~1.5e-5. Observed maxima are printed (run with -s).
ATOL = 1e-4
RTOL = 1e-5

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        find_spec("torch") is None or find_spec("safetensors") is None,
        reason="torch-cuda needs the server-cuda extra",
    ),
    pytest.mark.skipif(
        not (TORCH_DIR / "manifest.json").is_file(), reason=f"no torch weights in {TORCH_DIR}"
    ),
]
needs_onnx = pytest.mark.skipif(
    not (ONNX_DIR / "onnx" / "prefix_step.onnx").is_file(), reason=f"no ONNX weights in {ONNX_DIR}"
)


def golden_texts() -> list[str]:
    cases = json.loads((ROOT / "tests" / "golden" / "pipeline.json").read_text("utf-8"))
    return [c["expected"] for c in cases if "error" not in c and split_segments(c["expected"])]


@pytest.fixture(scope="module")
def device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return DEVICE


@pytest.fixture(scope="module")
def torch_model(device):
    from zerotts_server.torch_runtime import TorchZeroTTS

    model = TorchZeroTTS(TORCH_DIR, device)
    model.warmup()
    return model


@pytest.fixture(scope="module")
def onnx_generator():
    from zerotts_server.runtime import OnnxFrameGenerator

    generator = OnnxFrameGenerator(ONNX_DIR, threads=8)
    generator.warmup()
    return generator


def voice(name: str) -> np.ndarray:
    return load_voice_dir(TORCH_DIR / "voices" / name).emb


def segment_codes(generator, segments, voice_emb, seed, sampling=None) -> list[np.ndarray]:
    """Codes of each segment, one RNG across them, as the engine draws them."""
    sampling = sampling or Sampling()
    rng = np.random.default_rng(seed)
    return [np.concatenate(list(generator.frames(s, voice_emb, sampling, rng))) for s in segments]


# --- runtime seam: torch-cuda fp32 against onnx-cpu ------------------------------


def onnx_session_with_logits():
    """local_frame_decode.onnx with every sampler's input (the scaled logits) as an output."""
    onnx = pytest.importorskip("onnx", reason="the torch-weights group provides onnx")
    import onnxruntime as ort

    model = onnx.load(ONNX_DIR / "onnx" / "local_frame_decode.onnx")
    # In graph order: the control channel, then codebooks 0..K-1.
    names = [node.input[0] for node in model.graph.node if node.op_type == "TopK"]
    for name in names:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    return ort.InferenceSession(model.SerializeToString(), options), len(names)


def compare(worst: dict, key: str, ours, theirs) -> None:
    """Record the max abs difference and the max share of the tolerance used."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    diff = np.abs(ours - theirs)
    # A forbidden <eoa> is -FLT_MAX on both sides; it does not set the scale.
    scale = np.abs(theirs[np.abs(theirs) < 1e30]).max()
    worst[key] = max(worst[key], float(diff.max()))
    worst["tolerance_used"] = max(
        worst["tolerance_used"], float(diff.max() / (ATOL + RTOL * scale))
    )


def near_tie(reference, perturbed, u: float, topk: int, topp: float, a: int, b: int) -> bool:
    """Accept a mismatch only if bounded logit noise reproduces both actual picks.

    Replaying both distributions accounts for sort, top-k, nucleus and draw boundaries
    without treating an unrelated boundary elsewhere in the distribution as evidence.
    """
    reference = np.asarray(reference, np.float64)
    perturbed = np.asarray(perturbed, np.float64)
    scale = np.abs(reference[np.abs(reference) < 1e30]).max()
    if np.max(np.abs(perturbed - reference)) > ATOL + RTOL * scale:
        return False

    def pick(logits):
        order = np.argsort(-logits, kind="stable")[: min(max(int(topk), 1), len(logits))]
        values = logits[order]
        p = np.exp(values - values[0])
        p /= p.sum()
        p = np.where(np.cumsum(p) - p <= np.clip(topp, 1e-6, 1.0), p, 0.0)
        cdf = np.cumsum(p / p.sum())
        return int(order[np.argmax(cdf > np.clip(u, 0.0, 0.999999))])

    return pick(perturbed) == a and pick(reference) == b


def lockstep(session, tts, model, segments, emb, seed, worst, sampling=None):
    """Run onnx-cpu frame by frame with torch-cuda teacher-forced to the ONNX codes.

    Compares the global hidden and the sampler's scaled logits at every frame and checks the
    torch pick wherever it differs from ONNX's is a near tie. Returns the ONNX codes of every
    segment and the first near tie as (segment, frame), or None.
    """
    import torch

    sampling = sampling or Sampling()
    K, C = tts.num_codebooks, tts.codebook_size
    rng = np.random.default_rng(seed)
    first_tie = None
    reference = []
    for s, segment in enumerate(segments):
        ids = tts.tokenizer(segment)
        h_o, kv, valid, cross_kv, text_valid = tts._prefix_step_init(
            ids[None].astype(np.int64), np.array([len(ids)]), emb
        )
        h_t = model.begin(ids, emb)
        seen_o = np.zeros((1, K, C), dtype=bool)
        seen_t = torch.zeros(K, C, dtype=torch.bool, device=model.device)
        t, tail, frames = 0, None, []
        while True:
            compare(worst, "hidden", h_t.cpu(), h_o)
            forbid = t < MIN_FRAMES or tail is not None
            ctrl_u, audio_u = draw_u(rng, K)
            is_eoa, codes, *logits = session.run(
                None,
                {
                    "global_hidden": h_o,
                    "forbid_eoa": np.array([forbid]),
                    "text_temperature": np.array([sampling.text_temperature], np.float32),
                    "text_topk": np.array(
                        [sampling.text_topk if sampling.text_topk > 0 else C], np.int64
                    ),
                    "audio_temperature": np.array([sampling.audio_temperature], np.float32),
                    "audio_topk": np.array(
                        [sampling.audio_topk if sampling.audio_topk > 0 else C], np.int64
                    ),
                    "audio_topp": np.array([sampling.audio_topp], np.float32),
                    "audio_repetition_penalty": np.array(
                        [sampling.audio_repetition_penalty], np.float32
                    ),
                    "seen_mask": seen_o,
                    "ctrl_random_u": ctrl_u,
                    "audio_random_u": audio_u,
                    "cfg_scale": np.array([1.0], np.float32),
                },
            )
            codes = codes[0].astype(np.int64)
            forced = torch.as_tensor(codes, device=model.device)
            eoa_t, codes_t, ctrl_t, audio_t = model.frame(
                h_t, forbid, sampling, ctrl_u, audio_u, seen_t, forced=forced
            )
            ctrl_o, audio_o = logits[0][0], np.concatenate(logits[1:])
            ctrl_t, audio_t = ctrl_t.cpu().numpy(), audio_t.cpu().numpy()
            compare(worst, "ctrl_logits", ctrl_t, ctrl_o)
            compare(worst, "audio_logits", audio_t, audio_o)
            text_topk = sampling.text_topk if sampling.text_topk > 0 else C
            audio_topk = sampling.audio_topk if sampling.audio_topk > 0 else C
            picks = [(bool(eoa_t), bool(is_eoa[0]), ctrl_o, ctrl_t, ctrl_u[0], text_topk, 1.0)] + [
                (
                    int(ours),
                    int(theirs),
                    audio_o[c],
                    audio_t[c],
                    audio_u[0, c],
                    audio_topk,
                    sampling.audio_topp,
                )
                for c, (ours, theirs) in enumerate(zip(codes_t.cpu(), codes, strict=True))
            ]
            for ours, theirs, reference_logits, torch_logits, u, topk, topp in picks:
                if ours != theirs:
                    assert near_tie(
                        reference_logits, torch_logits, u, topk, topp, int(ours), int(theirs)
                    ), (s, t)
                    worst["near_ties"] += 1
                    first_tie = first_tie or (s, t)
            seen_o[0, np.arange(K), codes] = True
            seen_t[model.codebooks, forced] = True
            if tail is None and is_eoa[0]:
                tail = EOA_EXTRA_FRAMES
            if (tail is not None and tail <= 0) or t >= MAX_FRAMES:
                break
            frames.append(codes[None])
            if tail is not None:
                tail -= 1
                if tail <= 0:
                    break
            h_o, kv, valid = tts._prefix_step_frame(
                codes[None, None, :],
                np.array([t]),
                kv,
                valid,
                n_voice=emb.shape[1],
                cross_kv=cross_kv,
                text_valid=text_valid,
            )
            h_t = model.advance(forced, t)
            t += 1
        reference.append(np.concatenate(frames))
    return reference, first_tie


@needs_onnx
def test_golden_set_matches_onnx(torch_model, onnx_generator):
    """torch-cuda fp32 against onnx-cpu at the runtime seam, on the golden set.

    Same text, voice and random draws. Lockstep: hidden and logits within tolerance, codes equal
    except at near ties. Free-running (`frames`, as the engine calls it): codes equal ONNX up
    to the first near tie, after which the two runtimes may take different takes.
    """
    session, n_logits = onnx_session_with_logits()
    assert n_logits == onnx_generator.tts.num_codebooks + 1
    worst = dict.fromkeys(("hidden", "ctrl_logits", "audio_logits", "tolerance_used"), 0.0)
    worst["near_ties"] = 0
    texts = golden_texts()
    frames = diverged = 0
    for i, text in enumerate(texts):
        segments = split_segments(text)
        emb = voice(VOICES[i % len(VOICES)])
        reference, first_tie = lockstep(
            session, onnx_generator.tts, torch_model, segments, emb, i, worst
        )
        frames += sum(len(codes) for codes in reference)
        actual = segment_codes(torch_model, segments, emb, seed=i)
        if first_tie is None:
            assert len(actual) == len(reference)
            for s, (ours, theirs) in enumerate(zip(actual, reference, strict=True)):
                assert np.array_equal(ours, theirs), f"case {i} segment {s}"
        else:
            diverged += 1
            s, t = first_tie
            for ours, theirs in zip(actual[:s], reference[:s], strict=True):
                assert np.array_equal(ours, theirs), f"case {i}"
            assert np.array_equal(actual[s][:t], reference[s][:t]), f"case {i} segment {s}"
    print(
        f"\ngolden parity: {len(texts)} cases, {frames} frames, {diverged} cases diverge "
        f"after a near tie; max abs diff {worst}"
    )
    assert worst["tolerance_used"] <= 1, worst


@needs_onnx
@pytest.mark.parametrize(
    "sampling",
    [
        Sampling(text_topk=1.5, audio_topk=1.5),
        Sampling(text_topk=0.5, audio_topk=0.5),
        Sampling(
            text_topk=0,
            audio_topk=64,
            audio_topp=0.8,
            text_temperature=0.9,
            audio_temperature=1.1,
            audio_repetition_penalty=1.05,
        ),
    ],
)
def test_nondefault_sampling_matches_onnx(torch_model, onnx_generator, sampling):
    session, count = onnx_session_with_logits()
    assert count == onnx_generator.tts.num_codebooks + 1
    worst = {"hidden": 0, "ctrl_logits": 0, "audio_logits": 0, "tolerance_used": 0, "near_ties": 0}
    segments, emb = ["Xin chào."], voice("maichi")
    reference, first_tie = lockstep(
        session, onnx_generator.tts, torch_model, segments, emb, 42, worst, sampling
    )
    actual = segment_codes(torch_model, segments, emb, 42, sampling)
    if first_tie is None:
        assert np.array_equal(actual[0], reference[0])
    else:
        _, frame = first_tie
        assert np.array_equal(actual[0][:frame], reference[0][:frame])


# --- HTTP smoke on torch-cuda ---------------------------------------------------


@pytest.fixture(scope="module")
def client(device):
    settings = Settings(model_dir=str(TORCH_DIR), runtime="torch-cuda", device=device)
    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 300
        while (response := client.get("/health")).status_code != 200:
            assert response.json()["error"]["type"] == "model_loading"
            assert time.monotonic() < deadline, "model did not load"
            time.sleep(0.5)
        yield client


def speak(client, **extra) -> bytes:
    body = {"model": "zerotts", "input": "Xin chào các bạn. Hôm nay trời rất đẹp.", **extra}
    response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (48000, 1, 2)
        return f.readframes(f.getnframes())


def test_offline_speech_is_48k_mono_wav(client):
    pcm = np.frombuffer(speak(client, voice="baotrang"), dtype="<i2")

    assert len(pcm) > 48000 * 1.0
    assert np.abs(pcm).max() > 1000


def test_same_seed_reproduces_audio_on_torch_cuda(client):
    np.random.seed(1)
    first = speak(client, seed=1234)
    np.random.seed(2)

    assert speak(client, seed=1234) == first
    assert speak(client, seed=4321) != first


def test_streaming_speech_ends_with_done(client):
    body = {"model": "zerotts-stream", "input": "Xin chào. Hẹn gặp lại.", "stream": True}
    response = client.post("/v1/audio/speech", json=body)

    assert response.status_code == 200
    events = [line[6:] for line in response.text.split("\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["type"] == "speech.audio.done"
    assert sum(json.loads(e)["type"] == "speech.audio.delta" for e in events[:-2]) > 1


def test_missing_gpu_fails_load_instead_of_falling_back(device):
    settings = Settings(model_dir=str(TORCH_DIR), runtime="torch-cuda", device="cuda:99")
    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 60
        while (error := client.get("/health").json()["error"]["type"]) == "model_loading":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert error == "model_load_failed"
