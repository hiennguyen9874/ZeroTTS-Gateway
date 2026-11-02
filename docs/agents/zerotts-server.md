# zerotts_server

The upstream runs ZeroTTS and serves the audio.cpp-style API used by the gateway (ADR 0005). The default runtime is `onnx-cpu`; for the `torch-cuda` runtime, weight building, GPU tests and benchmarks, read `docs/agents/zerotts-cuda.md` as well. Runtime is operator-selected; it does not change the API.

Server configuration, defaults and environment overrides are in `configs/zerotts_server.toml`. Paths in that file are relative to the config directory; environment path overrides are used as given. To run the default CPU server:

```bash
uv sync
uv run scripts/download_zerotts.sh
uv run zerotts-server --config configs/zerotts_server.toml
```

The server listens while it loads and warms up in the background; `/health` returns 503 until ready. The provided container is CPU-only: `docs/agents/deploy.md`.

## Weights

`scripts/download_zerotts.sh` pins the HF revision and uses `hf download --local-dir`: real files are required because onnxruntime rejects external data behind HF-cache symlinks. Re-runs skip files at the pinned revision. To change it, edit the script's `REVISION`, re-download and run the real-model smoke test below.

### Opt-in int8 (host only)

fp32 is the default. To quantize the downloaded fp32 directory locally, set `SOURCE_HF_COMMIT` to its actual 40-character HF revision (the default download's revision is in `scripts/download_zerotts.sh`). If `UV_NO_SYNC=true`, first run `UV_NO_SYNC=false uv sync --group quantize`. Then run:

```bash
uv run --group quantize python scripts/quantize_zerotts.py --revision "$SOURCE_HF_COMMIT"
ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 \
  uv run zerotts-server --config configs/zerotts_server.toml
```

The script's `--help` describes paths and overwrite options. It validates available HF metadata against the revision, keeps source and output separate, and writes `quantization.json` with graph hashes and versions. Keep the manifest with the weights: startup logs identify the weights from it, but do not verify the files' hashes. Seed reproducibility requires the same weights and runtime; fp32 and int8 differ for the same seed. Copy one complete int8 directory to multiple nodes instead of assuming independent conversions match. Reproducibility, speed and quality evidence are in the issue 12–14 int8 reports under `docs/reports/`; the quality comparison still requires listening. For Compose configuration see `docs/agents/deploy.md#opt-in-int8`.

## Workers and voices

- Each worker holds a model instance and runs one request at a time. When all are busy, a request waits up to `busy_timeout` then receives `503 server_busy` before streaming starts. A worker is released only when synthesis stops, including after a caller disconnect.
- Voices are preset packs `<voices_dir>/<name>/voice.npz` with optional `meta.json`; add one and restart. The configured `default_voice` must exist or loading fails. There is no voice cloning. The production default still needs listening on real data and nodes.
- Sampling belongs to the operator per model; callers control only `seed`.

## Routes

| Route | Behaviour |
| --- | --- |
| `GET /health` | 200 `{"status":"ok","models":N}` after warmup; 503 `model_loading` while loading, 503 `model_load_failed` on load error |
| `GET /v1/models` | audio.cpp-style ids with `mode` (`offline`/`streaming`) and `loaded` |
| `GET /v1/audio/voices` | `voices` (names) plus `details` (`display_name`, `description`, `gender`, `language`) |
| `POST /v1/audio/speech` | Offline (`zerotts`): `wav` (default), `pcm`, `json`/`b64_json` (WAV base64 and timing). Streaming (`zerotts-stream`): see below |

Audio is mono 48 kHz 16-bit little-endian.

## Speech requests

- Missing `voice` selects the model's `default_voice`; an unknown voice returns `400 unknown_voice`.
- Input is NFC-normalized, then `chunk_text` groups sentences up to approximately 15 s (225 characters), splitting overlong pieces at commas, then words, then raw characters as a last resort. It does not rewrite punctuation or meaningful symbols (`C++`, `1+1`). Segments synthesize sequentially with one seeded RNG; offline joins them with 0.15 s silence. Nothing speakable left (`... !!`) returns `400 invalid_request_error`.
- `seed` or `options.seed` accepts an integer in [0, 2^64), including a decimal string; conflicting values are rejected. It uses a per-request RNG. The same seed reproduces audio on the same runtime and weights, but changing runtime or weights changes the audio.
- `voice_ref`, `reference_text`, `instruct`, `instructions`, `speed`/`speaking_rate` other than 1, and `options` keys other than `seed` return `400 unsupported_option`.
- A stream request to an offline model, or a non-stream request to a streaming model, returns `400 invalid_request_error`.

## Streaming (`zerotts-stream`)

- `stream_format: "sse"` (default with `stream: true`) returns `speech.audio.delta` events with base64 PCM, one `speech.audio.done` with `timing.ttft_ms`, then `data: [DONE]`. `stream_format: "audio"` returns raw PCM. Streaming accepts only `response_format: "pcm"` or an omitted response format.
- One codec streaming decoder spans all segments to avoid boundary clicks. The first segment decodes at 1, 2, 4, 8, then 16 frames; later segments use 16. Between segments the engine decodes two codec silence frames if `silence_frame.npy` exists; otherwise it emits 0.15 s of zero PCM.
- On an engine failure mid-stream, SSE ends with an `error` event and no `done`/`[DONE]`; raw PCM drops the connection so truncated audio does not look complete. For gateway response headers and proxy buffering, see `docs/agents/gateway.md#forwarding-rules`.

## Tests and vendored code

`tests/test_server.py` uses a fake engine. Default tests exclude the `real_model` and `cuda` markers. The opt-in CPU smoke test skips if ONNX weights are absent:

```bash
uv run python -m pytest -m real_model tests/test_server_smoke.py
ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 \
  uv run python -m pytest -m real_model tests/test_server_smoke.py
```

The optional byte-equivalence check has additional environment constraints: `docs/reports/issue16-vendored-zerotts.md`. For provenance, licenses and comparison instructions for the vendored ZeroTTS model, see `src/zerotts_server/model/VENDORED.md`; the MOSS codec has its own `model/moss/VENDORED.md`.
