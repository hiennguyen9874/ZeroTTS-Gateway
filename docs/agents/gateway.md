# Gateway

Code: `src/zerotts_gateway/` (`app.py` routes, `config.py` settings, `text.py` pipeline). Config keys and env overrides are documented in `configs/gateway.toml`; run with `uv run zerotts-gateway --config configs/gateway.toml`.

## Routes

| Route | Auth | Behaviour |
| --- | --- | --- |
| `GET /health` | none | Gateway process alive |
| `GET /ready` | none | 200 only when upstream `/health` and `/v1/models` both succeed within 5 s; otherwise 503 `upstream_unavailable` |
| `GET /v1/models`, `GET /v1/audio/voices` | key | Passed through to the upstream |
| `POST /v1/audio/speech` | key | Text pipeline, then one upstream request; response streamed through chunk by chunk |
| `POST /v1/text/normalize` | key | Dry-run of the text pipeline; never calls the upstream |

"key" means `Authorization: Bearer <key>` is required only when `ZEROTTS_GATEWAY_API_KEY` is set (`401 unauthorized`). The key is read from the environment only, never from the config file.

## Forwarding rules

- `voice_ref` as a string or `{"type":"path"}` → `400 voice_ref_path_not_allowed` (it would let callers read files on the upstream host). Only a `{"type":"base64", ...}` object passes this check; other forms get `400 invalid_request`. The ZeroTTS upstream rejects even base64 references as unsupported.
- `text_processing: {"normalize": bool}` is consumed by the gateway and stripped before forwarding.
- Upstream error responses are passed through whole, status and body unchanged; the gateway retries nothing. Forwarded responses preserve content type, not arbitrary upstream headers: in particular the upstream SSE `X-Accel-Buffering: no` is not forwarded, so configure reverse proxies to disable buffering separately.
- `connect_timeout_s` bounds only the connect: past it, `502 upstream_unavailable`. `deadline_s` is absolute (first byte plus whole body): before the first byte it becomes `504 upstream_timeout`; after that the caller connection is aborted, as when the upstream breaks mid-stream.
- `[models.<id>]` gives per-model defaults (`language`, `normalize`, `nonverbal_tags`); unknown model ids are still forwarded.

## Docker (gateway only)

- `127.0.0.1` inside the container is the container. An upstream on the Docker host is `http://host.docker.internal:8080` and must listen on `0.0.0.0`.
- The image healthcheck hits `/health`, never `/ready`, so a dead upstream does not restart the gateway; load balancers should probe `/ready`.
- `.dockerignore` admits only `pyproject.toml`, `uv.lock`, `README.md`, `src/`; `configs/` is mounted read-only at `/config`, so config and lexicon edits need `docker compose restart`, not a rebuild.

Combined stacks and the upstream container: `docs/agents/deploy.md`.
