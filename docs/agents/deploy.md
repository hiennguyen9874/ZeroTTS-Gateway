# Deployment

Gateway and `zerotts_server` are always separate containers, so each can change, rebuild or scale alone. CPU only: no GPU, NVIDIA driver or container toolkit. `docker compose` reads `COMPOSE_FILE` from `.env`; the three shapes and every variable are documented in `.env.example`, and each compose file explains itself in its header.

| Shape | `COMPOSE_FILE` |
| --- | --- |
| Gateway only, upstream elsewhere | `compose.yaml` |
| Gateway + upstream on one CPU node | `compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml` |
| Upstream only, separate CPU node | `compose.zerotts.yaml` |

```bash
uv run scripts/download_zerotts.sh   # weights first (~860 MB)
cp .env.example .env                 # set COMPOSE_FILE, ZEROTTS_GATEWAY_API_KEY if exposed
docker compose up -d --build
docker compose ps                    # both "healthy"
curl -f localhost:8000/ready
```

## Gotchas

- In the stack overlay the gateway calls `http://zerotts:8080` and ignores `ZEROTTS_GATEWAY_UPSTREAM_URL`; it starts only after `zerotts` is healthy (model loaded and warmed up).
- `zerotts_server` has no auth. It is published on `127.0.0.1:8080` for debugging and listening scripts; open `ZEROTTS_SERVER_BIND` only to a trusted network behind a firewall that admits the gateway alone. Callers talk to the gateway only.
- The upstream image holds no weights or config: `configs/` and `models/` mount read-only at `/srv/configs`, `/srv/models`, mirroring the repo so `model_dir = "../models/ZeroTTS"` works on host and in the container. A missing host directory fails `up` instead of mounting empty.
- `port` comes from the TOML config; changing it means also changing `ports` in the compose file and the image healthcheck.
- Upstream healthcheck is `/health`, 200 only after warmup (~10 s on a dev box; `start-period` 120 s for slow nodes). Keep `workers × threads` within the container's cores.
- Config or voice-pack changes need `docker compose restart zerotts`, not a rebuild. Weight revision changes: see `docs/agents/zerotts-server.md#weights`.
- After a host reboot, `restart: unless-stopped` brings both back without `depends_on` ordering, so the gateway's `/ready` is 503 until the upstream warms up.
- A deploy host without uv can copy `models/ZeroTTS` over, or run the script with `pip install huggingface_hub`.

Gateway container specifics: `docs/agents/gateway.md#docker-gateway-only`.
