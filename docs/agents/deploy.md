# Deployment

The provided Compose files run gateway and `zerotts_server` in separate containers; the upstream image is CPU-only and contains no torch. For the host-run CUDA runtime and its packaging status, see `docs/agents/zerotts-cuda.md`. `docker compose` reads `COMPOSE_FILE` from `.env`; variables are documented in `.env.example`.

For the combined CPU stack, download weights with `uv run scripts/download_zerotts.sh`, copy `.env.example` to `.env`, select its combined `COMPOSE_FILE` and set `ZEROTTS_GATEWAY_API_KEY` if exposed. Then run:

```bash
docker compose up -d --build
docker compose ps                    # both healthy
curl -f localhost:8000/ready         # use GATEWAY_PORT if changed
```

For gateway-only, supply a reachable upstream; for upstream-only, check `http://127.0.0.1:8080/health` instead (use `ZEROTTS_SERVER_PORT` if changed).

## Opt-in int8

Create int8 weights on the host as described in `docs/agents/zerotts-server.md#opt-in-int8-host-only`. Set `model_dir = "../models/ZeroTTS-int8"` in `configs/zerotts_server.toml` and restart the upstream. The existing read-only `./models:/srv/models` mount includes both directories. Alternatively, explicitly pass `ZEROTTS_SERVER_MODEL_DIR=/srv/models/ZeroTTS-int8` in the container environment: a host shell variable alone is not forwarded by the provided Compose file. To revert, select the fp32 directory and remove any model-dir override.

## Gotchas

- In the stack overlay the gateway calls `http://zerotts:${ZEROTTS_SERVER_LISTEN_PORT:-8080}` and ignores `ZEROTTS_GATEWAY_UPSTREAM_URL`; it starts only after `zerotts` is healthy (model loaded and warmed up).
- `zerotts_server` has no auth. It is published on `127.0.0.1:8080` for debugging and listening scripts; open `ZEROTTS_SERVER_BIND` only to a trusted network behind a firewall that admits the gateway alone. Callers talk to the gateway only.
- The upstream image holds no weights or config: `configs/` and `models/` mount read-only at `/srv/configs`, `/srv/models`, mirroring the repo so `model_dir = "../models/ZeroTTS"` works on host and in the container. A missing host directory fails `up` instead of mounting empty.
- Compose uses `GATEWAY_PORT` and `ZEROTTS_SERVER_PORT` for host ports; `ZEROTTS_GATEWAY_PORT` and `ZEROTTS_SERVER_LISTEN_PORT` set container listen ports. Outside Compose, the process overrides are `ZEROTTS_GATEWAY_PORT` and `ZEROTTS_SERVER_PORT`. The image healthchecks use the effective process port.
- Upstream healthcheck is `/health`, 200 only after warmup (~10 s on a dev box; `start-period` 120 s for slow nodes). Keep `workers × threads` within the container's cores.
- Config or voice-pack changes need `docker compose restart zerotts`, not a rebuild. Weight revision changes: see `docs/agents/zerotts-server.md#weights`.
- After a host reboot, `restart: unless-stopped` brings both back without `depends_on` ordering, so the gateway's `/ready` is 503 until the upstream warms up.
- A deploy host without uv can copy `models/ZeroTTS` over, or run the script with `pip install huggingface_hub`.

Gateway container specifics: `docs/agents/gateway.md#docker-gateway-only`.
