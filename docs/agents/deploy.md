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

## Opt-in int8

fp32 is the default. Quantize on the **host**, never in the server image:

```bash
uv run --group quantize python scripts/quantize_zerotts.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
```

Set `model_dir = "../models/ZeroTTS-int8"` in `configs/zerotts_server.toml`,
then restart the upstream. The existing compose mount of `./models` at
`/srv/models` is read-only and includes both directories. Alternatively, pass
`ZEROTTS_SERVER_MODEL_DIR=/srv/models/ZeroTTS-int8` in the container environment
(a host shell variable alone is not forwarded by compose). On the host use
`ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8`. No engine or image change is needed.

To disable int8, restore `model_dir = "../models/ZeroTTS"`, remove any model-dir
environment override, and restart. Retain `quantization.json` beside the int8
weights: startup logs identify the directory, weight type, source revision and
graph hashes. Seed reproducibility is only within the same weights/runtime;
fp32 and int8 produce different audio for the same seed. For multiple nodes,
copy one complete int8 directory (including manifest), mounted read-only on
each node, rather than relying on independent conversions. See
`docs/agents/zerotts-server.md#opt-in-int8-host-only` and the
`docs/reports/issue12-zerotts-int8.md` reproducibility check.

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
