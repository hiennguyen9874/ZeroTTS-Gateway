"""Opt-in tests of the torch-cuda runtime on a GPU (marker `cuda`, ADR 0007).

Skipped without torch, a CUDA device or models/ZeroTTS-torch; the parity tests also need the fp32
ONNX weights in models/ZeroTTS. The parity tests run the fp32 model with its CUDA Graphs (the
engine's `frames()` replays them; lockstep drives the same code eagerly); the HTTP tests run the
server as deployed, fp16 with CUDA Graphs. Run on a free GPU, e.g.:

  CUDA_VISIBLE_DEVICES=2 uv run --extra server-cuda --group torch-weights \\
      python -m pytest -m cuda tests/test_torch_cuda.py -s

`torch-weights` provides `onnx`, used to expose the ONNX sampler's logits for the comparison.
ZEROTTS_SERVER_DEVICE picks the device (default cuda:0).
"""

import base64
import hashlib
import io
import json
import os
import socket
import threading
import time
import wave
from importlib.util import find_spec
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from zerotts_server import create_app
from zerotts_server.app import pcm16, split_segments
from zerotts_server.config import Sampling, Settings
from zerotts_server.engine import STREAM_GAP_FRAMES, STREAM_MAX_CHUNK_FRAMES, ZeroTTSEngine
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
# fp16 against fp32 on the same history: |fp16 - fp32| <= FP16_RTOL * max|fp32| per tensor
# (observed ~0.25%); a wrong position or mask moves them by a whole scale.
FP16_RTOL = 0.01
# PyTorch fp32 codec against ONNX offline on the same codes, audio peaks near 1.
CODEC_ATOL = 1e-4
CODEC_MIN_SNR_DB = 80
# Three segments (the golden set has only single-segment cases), so streaming crosses two
# segment boundaries with codec silence frames between them.
LONG_TEXT = " ".join(
    f"Đây là câu số {i}, kể về một ngày làm việc dài ở văn phòng." for i in range(7)
)
SPEECH = "/v1/audio/speech"

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

    model = TorchZeroTTS(TORCH_DIR, device, dtype="float32")
    model.warmup()
    return model


@pytest.fixture(scope="module")
def torch_model_fp16(device):
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


def test_fp16_tracks_fp32_under_teacher_forcing(torch_model, torch_model_fp16):
    """The served fp16 model against fp32, both fed the fp32 codes: hidden and sampler logits
    stay within FP16_RTOL of each tensor's scale. fp16 codes differ (different takes), so this
    is what can be compared; quality is the ASR/listening A/B's job (#22)."""
    import torch

    K, C = torch_model.num_codebooks, torch_model.codebook_size
    worst = {"hidden": 0.0, "ctrl_logits": 0.0, "audio_logits": 0.0}

    def check(key, fp16, fp32):
        fp16, fp32 = fp16.float(), fp32.float()
        keep = fp32.abs() < 1e30  # a forbidden <eoa> is -FLT_MAX on both sides
        assert torch.isfinite(fp16[keep]).all(), key
        error = float((fp16 - fp32)[keep].abs().max() / fp32[keep].abs().max())
        worst[key] = max(worst[key], error)

    frames = 0
    for i, text in enumerate(golden_texts()[::20]):
        segment, emb = split_segments(text)[0], voice(VOICES[i % len(VOICES)])
        [codes] = segment_codes(torch_model, [segment], emb, seed=i)
        ids = torch_model.tokenizer(segment)
        h32, h16 = torch_model.begin(ids, emb), torch_model_fp16.begin(ids, emb)
        seen = torch.zeros(K, C, dtype=torch.bool, device=torch_model.device)
        rng = np.random.default_rng(i)
        for t, frame_codes in enumerate(codes):
            check("hidden", h16, h32)
            draws = draw_u(rng, K)
            forced = torch.as_tensor(frame_codes, device=torch_model.device)
            forbid = t < MIN_FRAMES
            _, _, c32, a32 = torch_model.frame(h32, forbid, Sampling(), *draws, seen, forced)
            _, _, c16, a16 = torch_model_fp16.frame(h16, forbid, Sampling(), *draws, seen, forced)
            check("ctrl_logits", c16, c32)
            check("audio_logits", a16, a32)
            seen[torch_model.codebooks, forced] = True
            h32, h16 = torch_model.advance(forced, t), torch_model_fp16.advance(forced, t)
        frames += len(codes)
    print(f"\nfp16 vs fp32: {frames} frames, max error / scale {worst}")
    assert max(worst.values()) <= FP16_RTOL, worst


def codec_metrics(actual: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    assert np.isfinite(actual).all(), "actual codec PCM must be finite"
    assert np.isfinite(reference).all(), "reference codec PCM must be finite"
    error = actual.astype(np.float64) - reference
    noise = np.sum(error**2)
    signal = np.sum(reference.astype(np.float64) ** 2)
    with np.errstate(divide="ignore"):
        snr = np.inf if noise == 0 else 10 * np.log10(signal / noise)
    return float(np.abs(error).max()), float(snr)


@pytest.fixture(scope="module")
def golden_codes(torch_model) -> list[list[np.ndarray]]:
    """torch-cuda codes of every golden case's segments: case i uses seed i, alternating voices."""
    return [
        segment_codes(torch_model, split_segments(text), voice(VOICES[i % len(VOICES)]), seed=i)
        for i, text in enumerate(golden_texts())
    ]


@pytest.fixture(scope="module")
def torch_codec(torch_model):
    from zerotts_server.torch_runtime import TorchCodec

    return TorchCodec(TORCH_DIR, torch_model.device, torch_model.stream)


@pytest.fixture(scope="module")
def onnx_codec():
    from zerotts_server.model.codec import MossCodecDecoder

    return MossCodecDecoder(ONNX_DIR / "onnx" / "codec", intra_op_num_threads=8)


@needs_onnx
def test_codec_matches_onnx_offline(golden_codes, torch_codec, onnx_codec):
    """PyTorch fp32 codec against the ONNX codec offline, on the golden set's codes."""
    worst_diff, worst_snr, frames = 0.0, np.inf, 0
    for case in golden_codes:
        for codes in case:
            codes = codes.T[None]  # (T, K) -> (1, K, T), as the engine decodes
            ours, theirs = torch_codec.decode(codes), onnx_codec.decode(codes)
            assert ours.dtype == np.float32
            assert ours.shape == theirs.shape == (1, codes.shape[2] * 3840)
            difference, snr = codec_metrics(ours, theirs)
            worst_diff, worst_snr = max(worst_diff, difference), min(worst_snr, snr)
            frames += codes.shape[2]
    print(
        f"\ncodec parity: {len(golden_codes)} cases, {frames} frames, "
        f"max abs diff {worst_diff:.2e}, min SNR {worst_snr:.1f} dB"
    )
    assert worst_diff <= CODEC_ATOL
    assert worst_snr >= CODEC_MIN_SNR_DB


def stream_chunks(segments: list[np.ndarray], silence: np.ndarray) -> list[np.ndarray]:
    """The (1, K, n) codec calls ZeroTTSEngine._stream makes for these segments' (T, K) codes.

    The first segment ramps 1, 2, 4 ... frames up to the cap, later ones decode at the cap;
    STREAM_GAP_FRAMES codec silence frames go between segments.
    """
    chunks = []
    for i, codes in enumerate(segments):
        if i:
            chunks.append(np.stack([silence] * STREAM_GAP_FRAMES, axis=-1))
        size, start = (1 if i == 0 else STREAM_MAX_CHUNK_FRAMES), 0
        while start < len(codes):
            chunks.append(codes[start : start + size].T[None])
            start += size
            size = min(STREAM_MAX_CHUNK_FRAMES, size * 2)
    return chunks


def stream_decode(codec, chunks: list[np.ndarray]) -> list[np.ndarray]:
    """One streaming decoder across all chunks of a request, as the engine uses it."""
    decoder = codec.streaming_decoder()
    try:
        return [decoder.decode_chunk(chunk) for chunk in chunks]
    finally:
        decoder.close()


@needs_onnx
def test_codec_matches_onnx_streaming(torch_model, golden_codes, torch_codec, onnx_codec):
    """Joined streaming PCM of the PyTorch codec against ONNX streaming on the same codes.

    Golden set plus a three-segment text, decoded with the engine's schedule and silence
    frames; the second pass of each case checks close() reset the causal state.
    """
    silence = np.load(TORCH_DIR / "silence_frame.npy")
    long_case = segment_codes(torch_model, split_segments(LONG_TEXT), voice("maichi"), seed=7)
    assert len(long_case) == 3
    worst_diff, worst_snr, chunk_count = 0.0, np.inf, 0
    for case in [*golden_codes, long_case]:
        chunks = stream_chunks(case, silence)
        ours, theirs = stream_decode(torch_codec, chunks), stream_decode(onnx_codec, chunks)
        expected_shapes = [(1, chunk.shape[2] * 3840) for chunk in chunks]
        assert [c.shape for c in ours] == [c.shape for c in theirs] == expected_shapes
        assert all(c.dtype == np.float32 for c in ours)
        difference, snr = codec_metrics(np.concatenate(ours, axis=1), np.concatenate(theirs, 1))
        worst_diff, worst_snr = max(worst_diff, difference), min(worst_snr, snr)
        chunk_count += len(chunks)
    again = np.concatenate(stream_decode(torch_codec, stream_chunks(long_case, silence)), axis=1)
    assert np.array_equal(again, np.concatenate(ours, axis=1))
    print(
        f"\nstreaming codec parity: {len(golden_codes) + 1} cases, {chunk_count} chunks, "
        f"max abs diff {worst_diff:.2e}, min SNR {worst_snr:.1f} dB"
    )
    assert worst_diff <= CODEC_ATOL
    assert worst_snr >= CODEC_MIN_SNR_DB


def test_weights_have_no_onnx_and_match_manifest():
    manifest = json.loads((TORCH_DIR / "manifest.json").read_text())
    files = {
        path.relative_to(TORCH_DIR).as_posix(): path
        for path in TORCH_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json" and ".cache" not in path.parts
    }

    assert not [name for name in files if name.endswith((".onnx", ".data"))]
    assert files.keys() == manifest["files"].keys()
    for name, path in files.items():
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        assert manifest["files"][name] == {"sha256": digest, "size": path.stat().st_size}, name


# --- HTTP smoke on torch-cuda ---------------------------------------------------


def wait_until_ready(client, timeout: float = 300) -> None:
    """Poll /health, one request per round, until 200; only model_loading may come before."""
    deadline = time.monotonic() + timeout
    while (response := client.get("/health")).status_code != 200:
        assert response.json()["error"]["type"] == "model_loading"
        assert time.monotonic() < deadline, "model did not load"
        time.sleep(0.5)


@pytest.fixture(scope="module")
def torch_engine(device):
    """The server's only worker; tests also call it directly for the reference audio."""
    return ZeroTTSEngine(TORCH_DIR, TORCH_DIR / "voices", 8, "torch-cuda", device)


@pytest.fixture(scope="module")
def client(torch_engine, device):
    """The torch-cuda server over real HTTP: TestClient buffers whole streamed bodies."""
    import onnxruntime
    import uvicorn

    def no_onnx(*args, **kwargs):
        raise AssertionError("torch-cuda must not open an ONNX session")

    settings = Settings(model_dir=str(TORCH_DIR), runtime="torch-cuda", device=device)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    app = create_app(settings, engine=torch_engine)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None))
    # Sessions are opened at load; any ONNX one would fail it, so /health never gets 200.
    patch = pytest.MonkeyPatch()
    patch.setattr(onnxruntime, "InferenceSession", no_onnx)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{sock.getsockname()[1]}"
        with httpx.Client(base_url=base, timeout=300) as client:
            deadline = time.monotonic() + 60
            while not server.started:
                assert time.monotonic() < deadline, "server did not start"
                time.sleep(0.1)
            wait_until_ready(client)
            patch.undo()
            yield client
    finally:
        patch.undo()
        server.should_exit = True
        thread.join(10)


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


STREAM_REQUEST = {"model": "zerotts-stream", "input": LONG_TEXT, "voice": "maichi", "seed": 7}


@pytest.fixture(scope="module")
def stream_reference(client, torch_engine) -> bytes:
    """The engine's own streamed audio for STREAM_REQUEST, as 16-bit PCM."""
    segments = split_segments(LONG_TEXT)
    assert len(segments) == 3
    chunks = torch_engine.synthesize(segments, "maichi", 7, Sampling(), stream=True)
    return b"".join(pcm16(chunk) for chunk in chunks)


def stream(client, stream_format: str) -> tuple[httpx.Response, bytes, float]:
    """POST STREAM_REQUEST; returns the response, its body and the client-side TTFA in ms."""
    body = {**STREAM_REQUEST, "stream_format": stream_format}
    start, first_ms, data = time.perf_counter(), None, b""
    with client.stream("POST", SPEECH, json=body) as response:
        assert response.status_code == 200, response.read()
        for chunk in response.iter_raw():
            if first_ms is None:
                first_ms = (time.perf_counter() - start) * 1000
            data += chunk
    return response, data, first_ms


def test_stream_deltas_join_to_the_engine_audio(client, stream_reference):
    sse, sse_body, sse_first_ms = stream(client, "sse")
    raw, pcm, raw_first_ms = stream(client, "audio")

    assert sse.headers["content-type"].startswith("text/event-stream")
    assert sse_body.endswith(b"data: [DONE]\n\n")
    blocks = sse_body.decode().split("\n\n")[:-2]
    events = [json.loads(block.removeprefix("data: ")) for block in blocks]
    deltas, done = events[:-1], events[-1]
    assert [e["type"] for e in deltas] == ["speech.audio.delta"] * len(deltas)
    assert done["type"] == "speech.audio.done"
    assert len(deltas) > 5  # 1, 2, 4, 8, 16 ... frames, then later segments
    assert b"".join(base64.b64decode(e["audio"]) for e in deltas) == stream_reference
    assert raw.headers["content-type"] == "audio/pcm"
    assert pcm == stream_reference
    audio_s = len(stream_reference) / 2 / 48000
    print(
        f"\nstream: {audio_s:.1f} s audio, server ttft_ms={done['timing']['ttft_ms']}, "
        f"client first delta {sse_first_ms:.0f} ms (sse), first bytes {raw_first_ms:.0f} ms (pcm)"
    )
    assert audio_s > 15


def test_caller_disconnect_frees_the_worker_and_resets_the_codec(client, stream_reference):
    body = {**STREAM_REQUEST, "stream_format": "audio"}
    with client.stream("POST", SPEECH, json=body) as response:
        assert response.status_code == 200
        next(response.iter_raw())

    # Within busy_timeout the worker is free again, and the next stream starts from a fresh
    # codec state: same seed, same audio.
    _, pcm, _ = stream(client, "audio")
    assert pcm == stream_reference


def test_two_workers_on_one_gpu_stream_the_single_worker_audio(device, stream_reference):
    """N workers share the device, each with its own weights, buffers, stream and graphs:
    concurrent streams must not disturb each other."""
    settings = Settings(
        model_dir=str(TORCH_DIR), runtime="torch-cuda", device=device, workers=2, busy_timeout=0
    )
    with TestClient(create_app(settings)) as client:
        wait_until_ready(client)
        results = [None, None]

        def run(i):
            body = {**STREAM_REQUEST, "stream_format": "audio"}
            response = client.post(SPEECH, json=body)
            results[i] = (response.status_code, response.content)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # Same seed, same audio as the single-worker server, whichever worker served each.
    assert results == [(200, stream_reference), (200, stream_reference)]


def test_missing_gpu_fails_load_instead_of_falling_back(device):
    settings = Settings(model_dir=str(TORCH_DIR), runtime="torch-cuda", device="cuda:99")
    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 60
        while (error := client.get("/health").json()["error"]["type"]) == "model_loading":
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert error == "model_load_failed"
