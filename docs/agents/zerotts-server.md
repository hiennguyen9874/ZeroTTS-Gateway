# zerotts_server

The upstream (ADR 0005): runs the ZeroTTS model on a **runtime** (`onnx-cpu` by default, or `torch-cuda`, ADR 0007) and speaks the part of audio.cpp's API the gateway uses. Code: `src/zerotts_server/` (`app.py` routes, `engine.py` synthesis, `runtime.py` the runtime seam, `torch_runtime.py` the PyTorch model, `config.py`). Config keys, defaults and env overrides are documented in `configs/zerotts_server.toml`; relative paths resolve from the config file's directory.

```bash
uv sync                      # dev group pulls in the `server` extra
uv run zerotts-server --config configs/zerotts_server.toml
```

The server listens at once and loads plus warms up the model in the background; `/health` is 503 until done.

## Runtimes

`runtime` (`ZEROTTS_SERVER_RUNTIME`) and `device` (`ZEROTTS_SERVER_DEVICE`) are server-level; callers see no difference and `/v1/models` is unchanged.

- `onnx-cpu` (default): the vendored ONNX graphs on onnxruntime CPU; `threads` per worker. Audio is byte-identical to before the seam (the #16 SHA-256 tests pass).
- `torch-cuda`: `TorchZeroTTS` in `torch_runtime.py`, a PyTorch port of `references/ZeroTTS-repo/cpp/src/zerotts.cpp` reading `models/ZeroTTS-torch`. Eager fp32 (#18), with a preallocated global KV cache written by position. The codec is `TorchCodec` (#19): the vendored PyTorch MOSS-Audio-Tokenizer-Nano (`model/moss/`, verbatim from `6aa02b01`, Apache-2.0) built from `codec_config.json` and `codec.safetensors`, encoder removed, fp32 on the same GPU with cuDNN TF32 turned off (TF32 moved the PCM by ~1e-3). It is never loaded with `from_pretrained`/`trust_remote_code` and the runtime opens no ONNX session; `threads` is unused. Offline and streaming both work (streaming keeps the codec's causal state in its `streaming()` context across chunks; streaming parity is #20). Install with `uv sync --extra server-cuda` (torch 2.14.1+cu126 from the PyTorch index, `safetensors`, `transformers` 5.22.1 which the vendored codec still imports); the default dev sync and the CPU image have no torch.
- Seam: `runtime.FrameGenerator.frames(text, voice_emb, sampling, rng)` yields (1, K) int64 codes per frame; `ZeroTTSEngine` joins it with a codec (offline `decode`, or one `streaming_decoder` across segments). Both runtimes draw the sampler's random numbers with `runtime.draw_u` from the per-request numpy RNG in the same order (control, then K codebooks, per frame). The torch sampler mirrors `local_frame_decode.onnx` op for op: stable full sort, rank ≥ k → -inf, softmax, drop ranks whose preceding mass > top-p, softmax again, first CDF > u. A forbidden `<eoa>` is -FLT_MAX.
- **No fallback to CPU.** Startup fails with `model_load_failed` (`/health` 503) when: the weights directory does not match the runtime (`manifest.json` `format: "zerotts-torch"` ↔ `torch-cuda`; anything else ↔ `onnx-cpu`); any model has `cfg_scale` ≠ 1 under `torch-cuda`; torch is missing, CUDA is unavailable, the device index does not exist, or loading N workers runs out of VRAM. Each worker loads its own weights and caches on `device`. A `device` that is not `cuda`/`cuda:<n>`, or an unknown `runtime`, is a config error at startup. A CUDA error mid-request is an engine error (500, SSE `error`, truncated PCM).
- Seed: same seed, runtime and weights → same audio. `torch-cuda` and `onnx-cpu` give different audio for the same seed: equal draws give equal codes except at fp32 near-ties, after which the takes diverge, and the codec paths differ.

## Weights

`scripts/download_zerotts.sh` downloads the pinned HF revision into `models/ZeroTTS` with `hf download --local-dir`. Real files are required: onnxruntime ≥ 1.25 rejects external data behind HF-cache symlinks. Re-runs skip files already at the revision. To change revision: edit `REVISION`, re-download, run the real-model smoke test.

### Opt-in int8 (host only)

fp32 remains the default. ZeroTTS does not publish int8 weights on HF; create them
from an already downloaded fp32 directory (no Hub access):

```bash
uv run --group quantize python scripts/quantize_zerotts.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 \
  uv run zerotts-server --config configs/zerotts_server.toml
```

The script defaults to `--model models/ZeroTTS --dst models/ZeroTTS-int8`.
`--revision` is required: supply the actual source HF commit, not a new label for
old weights. When HF download metadata is present, it must agree with this commit.
The output is self-contained (config, tokenizer, voices, codec), without HF cache
metadata. Existing output is refused unless `--force` explicitly requests replacement;
the source must be separate and stays untouched. Basic ORT graph folding precedes
dynamic QInt8 MatMul quantization of text_encoder, prefix_step and local_frame_decode;
codec stays fp32. Sizes and MatMulInteger counts are printed per hot graph.

`quantization.json` records `source_revision`, `onnx_version`,
`onnxruntime_version` and `graphs` (relative hot-graph paths with source/output SHA-256,
MatMulInteger count and byte sizes). Startup logs one `weights` record with
`model_dir` and `weights: fp32` or, when the manifest exists, `weights: int8`,
source revision and output hashes. These are manifest identities, not a startup
integrity scan. Keep the manifest with its weights. The engine and `/v1/models`
are unchanged. The pinned `onnx` package is in the separate `quantize` dependency
group, not the server extra or image.

Seed reproducibility applies only to the **same weights** (and runtime).
The same seed produces different audio with fp32 versus int8. Two quantizations
on this dev machine produced identical hot-graph SHA-256 hashes; this does not
guarantee equality across machines or versions. For multiple nodes, prefer copying
one complete int8 directory. Evidence and reproduction:
`docs/reports/issue12-zerotts-int8.md`. Dev-machine speed (int8 about 1.4–1.5× faster RTF):
`docs/reports/issue13-zerotts-int8-speed.md`, measured with `scripts/bench_zerotts.py`
(engine level, one weights directory per process). Quality A/B (#14, ASR WER 0.074 for
both; listening verdict pending): `docs/reports/issue14-zerotts-int8-quality-ab.md`.

### torch-cuda weights (`models/ZeroTTS-torch`, host only)

The `torch-cuda` runtime (ADR 0007, #15) uses its own self-contained directory, built on the
host from pinned revisions:

```bash
uv run --group torch-weights python scripts/build_zerotts_torch.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
```

The script downloads `zeroweight-ai/ZeroTTS-GGUF` `92ca8651…` `gguf/zerotts-f32.gguf` and
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` `6aa02b01…` `model-00001-of-00001.safetensors` and `config.json`
with `hf_hub_download(local_dir=...)` (real files, no HF-cache symlinks) into a temporary
directory, or into `--download-dir` for reuse, and rejects a SHA-256 that differs from the
pinned one. The download directory must not be the source or output directory, or inside
either (including symlink aliases). `--model` (default `models/ZeroTTS`, the downloaded fp32
ONNX directory) supplies config, tokenizer, voices and `silence_frame.npy`;
`--revision` is its HF commit, checked
against HF metadata when present. Output defaults to `--dst models/ZeroTTS-torch`; existing
output is refused unless `--force`, and the source is never touched. The `gguf`,
`safetensors` and `onnx` packages live in the `torch-weights` dependency group only.

| File | Content |
| --- | --- |
| `zerotts.safetensors` | All 377 GGUF tensors, fp32, GGUF names and value-equal. Header `__metadata__.gguf` holds the GGUF key-values (JSON), e.g. `d_ff`, `local_*`, eps, `rope_theta` |
| `codec.safetensors` | Codec `decoder.*` and `quantizer.*` (246 tensors, encoder dropped), original names incl. weight-norm `parametrizations.weight.original0/1` |
| `codec_config.json` | The MOSS `config.json` at `6aa02b01` (codec architecture) |
| `config.json`, `tokenizer.json`, `silence_frame.npy`, `voices/` | Copied from `--model` |

No ONNX file is copied: the directory has no codec ONNX (#19).
| `manifest.json` | See below |

`manifest.json` has `format: "zerotts-torch"` (the weights-kind marker), `sources` (repo,
revision, file, SHA-256, size for `model`, `codec` and `codec_config`; repo and revision for the assets), `tools`
(python, numpy, gguf, safetensors, onnx, huggingface_hub versions), `tensors` (count per
safetensors), `codec_onnx_check` and `files` (SHA-256 and size of every other file). Before
writing, the script re-reads both safetensors and checks every GGUF tensor is equal and every
initializer of both codec ONNX graphs of `--model` (the reference only) equals a codec tensor (direct, transposed or folded
weight-norm, max abs diff ≤ 1e-6; 17 folded, 8.9e-8 observed). Two runs gave identical
SHA-256 for every file: `docs/reports/issue17-zerotts-torch-weights.md`. Serve it with
`ZEROTTS_SERVER_RUNTIME=torch-cuda ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-torch`; the startup
`weights` record then has `runtime`, `device`, `weights: fp32`, `format`, `source_revisions`
(model, codec, codec_config, assets), `sha256` of both safetensors (from the manifest) and the `torch` version.
`onnx-cpu` records carry `runtime: onnx-cpu` and `device: cpu`.

## Workers and voices

- Each worker holds its own model instance and runs one request at a time. With all busy, a request waits up to `busy_timeout`, then gets `503 server_busy` (before any stream starts). No retry; the gateway passes it through. A worker frees only when the engine stops: done, failed, or caller disconnect (after the audio piece in progress).
- Voices are preset packs `<voices_dir>/<name>/voice.npz` (+ optional `meta.json`); add one and restart. No voice cloning. `default_voice` must exist or the model fails to load. `maichi` and `quangminh` ship with the weights; the production default is unsettled until someone listens on real data and nodes.
- Sampling is per model in config; callers control only `seed`.

## Routes

| Route | Behaviour |
| --- | --- |
| `GET /health` | 200 `{"status":"ok"}` after warmup; 503 `model_loading` before, 503 `model_load_failed` on load error |
| `GET /v1/models` | audio.cpp-style ids with `mode` (`offline`/`streaming`) and `loaded` |
| `GET /v1/audio/voices` | `voices` (names, as audio.cpp) plus `details` (`display_name`, `description`, `gender`, `language`) |
| `POST /v1/audio/speech` | Offline (`zerotts`): `response_format` `wav` (default), `pcm`, `json`/`b64_json` (`{"audio": <WAV base64>, "format":"wav", "timing"}`). Streaming (`zerotts-stream`): see below |

Audio is always mono 48 kHz 16-bit little-endian.

## Speech requests

- `voice`: missing → model's `default_voice`; unknown → `400 unknown_voice`.
- `input` is NFC-normalized, then split by `zerotts`'s `chunk_text`: sentences grouped up to ~15 s (225 chars), overlong sentences split at commas, then words. Punctuation is never added or changed and meaningful symbols (`C++`, `1+1`) stay. Segments synthesize sequentially with one seeded RNG, joined by 0.15 s silence into one WAV. Nothing speakable left (`... !!`) → `400 invalid_request_error`.
- `seed` or `options.seed`: integer in [0, 2^64), decimal string accepted; per-request RNG, never numpy's global one. Same seed and same weights/runtime, same audio; fp32 and int8, or `onnx-cpu` and `torch-cuda`, differ.
- `400 unsupported_option`: `voice_ref`, `reference_text`, `instruct`, `instructions`, `speed`/`speaking_rate` other than 1, any `options` key except `seed`.
- Mode mismatch (stream request to an offline model, or non-stream request to a streaming model) → `400 invalid_request_error`. audio.cpp would return WAV for the latter; spec #1 chose to reject.

## Streaming (`zerotts-stream`)

- `stream_format: "sse"` (default with `stream: true`): `text/event-stream`, `X-Accel-Buffering: no`; events `speech.audio.delta` (PCM base64), one `speech.audio.done` with `timing.ttft_ms` for the whole request, then `data: [DONE]`. Concatenated deltas equal the engine audio.
- `stream_format: "audio"`: raw PCM, `audio/pcm`, chunked.
- `response_format` accepts only `pcm`; any other value or `stream_format` → `400 invalid_request_error`.
- One shared codec streaming decoder across segments, so no clicks at boundaries (as ZeroTTS `webui/engine.py`). The first segment decodes at 1, 2, 4, 8, 16 frames for low TTFA, later ones 16 at a time; 2 codec silence frames (`silence_frame.npy`, ~0.16 s) between segments.
- Engine error mid-stream: SSE ends with `{"type":"error","error":{"message","type":"server_error"}}` and no `done`/`[DONE]`; raw PCM drops the connection without the terminating chunk, so truncated audio never looks complete.

## Logs

One JSON line per request: model, voice, seed, char count, `segments`, `wait_ms`, `synth_ms`, `audio_ms`, `rtf`, `ttfa_ms` (streaming), error code. Never text or audio.

## Tests

`tests/test_server.py` uses a fake engine. `tests/test_server_smoke.py` (marker `real_model`) is deselected by default; run `uv run python -m pytest -m real_model`, which still skips without weights in `models/ZeroTTS` (or `ZEROTTS_SERVER_MODEL_DIR` when set). To check int8: `ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 uv run python -m pytest -m real_model tests/test_server_smoke.py`.

`tests/test_torch_cuda.py` (marker `cuda`) is deselected by default and skips without torch, a GPU or `models/ZeroTTS-torch`; the parity test also needs `models/ZeroTTS` and `onnx`. Pin a free GPU:

```bash
uv sync --extra server-cuda --group torch-weights
CUDA_VISIBLE_DEVICES=2 uv run python -m pytest -m cuda tests/test_torch_cuda.py -s
uv sync   # back to the default environment
```

The parity test runs the golden set (non-error cases of `tests/golden/pipeline.json`) through both runtimes with the same voice and seed: lockstep with torch teacher-forced to the ONNX codes (`frame(..., forced=...)`, as `zerotts_frame_forced`) compares hidden and logits within 1e-4 + 1e-5·max|ref| and allows a different pick only at a near-tie; free-running `frames()` must equal ONNX up to the first near-tie. About 11 minutes on an A100 plus CPU. Results: `docs/reports/issue18-torch-cuda-eager.md`. The codec parity test decodes the golden set's torch codes with `TorchCodec` and the ONNX codec offline: max abs diff ≤ 1e-4 and SNR ≥ 80 dB (observed 1.3e-5, 112.5 dB; `docs/reports/issue19-torch-codec.md`). Another case checks `models/ZeroTTS-torch` has no ONNX file and matches `manifest.json` (names, SHA-256, sizes). The HTTP smoke tests load with `onnxruntime.InferenceSession` made to fail (so no ONNX session can open), then cover WAV 48 kHz mono, seed reproducibility, streaming and `cuda:99` → `model_load_failed`. A near-tie exemption must replay both actual picks from the two runtimes' logits within the stated tolerance, not merely find a nearby boundary. Additional CUDA cases cover nondefault sampling, including fractional top-k values coerced to integers as ONNX does. `tests/test_torch_cuda_validation.py` checks the acceptance helper and readiness polling on CPU.

## Vendored model code

`src/zerotts_server/model/` comes from ZeroTTS commit `319c3e0` (PyPI 0.1.5); provenance, licenses and the comparison command are in `model/VENDORED.md`. The server uses local weights only. The quantization script operates on ONNX files and does not import model code. For the opt-in byte-equivalence check and its environment constraints, see `docs/reports/issue16-vendored-zerotts.md`.
