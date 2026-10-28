# zerotts_server

The upstream (ADR 0005): wraps `zerotts==0.1.5` on onnxruntime CPU and speaks the part of audio.cpp's API the gateway uses. Code: `src/zerotts_server/` (`app.py` routes, `engine.py` synthesis, `config.py`). Config keys, defaults and env overrides are documented in `configs/zerotts_server.toml`; relative paths resolve from the config file's directory.

```bash
uv sync                      # dev group pulls in the `server` extra
uv run zerotts-server --config configs/zerotts_server.toml
```

The server listens at once and loads plus warms up the model in the background; `/health` is 503 until done.

## Weights

`scripts/download_zerotts.sh` downloads the pinned HF revision into `models/ZeroTTS` with `hf download --local-dir`. Real files are required: onnxruntime ≥ 1.25 rejects external data behind HF-cache symlinks. Re-runs skip files already at the revision. To change revision: edit `REVISION`, re-download, run the real-model smoke test.

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
- `seed` or `options.seed`: integer in [0, 2^64), decimal string accepted; per-request RNG, never numpy's global one. Same seed, same audio.
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

`tests/test_server.py` uses a fake engine. `tests/test_server_smoke.py` (marker `real_model`) is deselected by default; run `uv run python -m pytest -m real_model`, which still skips without weights in `models/ZeroTTS`.
