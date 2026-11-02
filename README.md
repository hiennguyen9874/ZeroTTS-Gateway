# ZeroTTS Gateway

Gateway mỏng đứng trước `zerotts_server` (ZeroTTS TTS tiếng Việt, mặc định ONNX CPU, tuỳ chọn PyTorch CUDA). API kiểu audio.cpp: `POST /v1/audio/speech`, `GET /v1/models`, `GET /v1/audio/voices`. Gateway chuẩn hoá văn bản tiếng Việt, chặn `voice_ref` dạng path, hỗ trợ Bearer auth và không ghi text/audio vào log. Audio trả về là mono 48 kHz, PCM 16-bit.

```
client ──► gateway :8000 ──► zerotts_server :8080
```

## Chạy bằng uv

Cần [uv](https://docs.astral.sh/uv/) và Python ≥ 3.12.

```bash
uv sync
uv run scripts/download_zerotts.sh                          # weights -> models/ZeroTTS
uv run zerotts-server --config configs/zerotts_server.toml  # upstream, cổng 8080
uv run zerotts-gateway --config configs/gateway.toml        # terminal khác, cổng 8000
```

Upstream tải và warm up ở nền; `/health` trả 503 cho đến khi sẵn sàng. Kiểm tra gateway thấy upstream:

```bash
curl -f localhost:8000/ready
```

Đổi cổng bằng `ZEROTTS_SERVER_PORT`, `ZEROTTS_GATEWAY_PORT`; nếu đổi cổng upstream, đặt thêm `ZEROTTS_GATEWAY_UPSTREAM_URL`. Cấu hình khác nằm trong `configs/*.toml`.

## Chạy bằng Docker

```bash
cp .env.example .env     # chọn COMPOSE_FILE bên dưới, đặt ZEROTTS_GATEWAY_API_KEY nếu mở ra ngoài
docker compose up -d --build
docker compose ps
curl -f localhost:8000/ready
```

| Kiểu triển khai | `COMPOSE_FILE` |
| --- | --- |
| Gateway + upstream trên một node CPU | `compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml` |
| Chỉ gateway (upstream ở nơi khác) | `compose.yaml` |
| Chỉ upstream (node CPU riêng) | `compose.zerotts.yaml` |

- Tải weights trước (`uv run scripts/download_zerotts.sh`); `models/` và `configs/` được mount read-only. Đổi config hoặc voice: `docker compose restart`, không cần build lại.
- Chỉ gateway: đặt `ZEROTTS_GATEWAY_UPSTREAM_URL` tới upstream; `127.0.0.1` trong container là chính container (dùng `http://host.docker.internal:8080` cho upstream trên host, upstream phải listen `0.0.0.0`).
- Cổng host: `GATEWAY_PORT`, `ZEROTTS_SERVER_PORT`; cổng trong container: `ZEROTTS_GATEWAY_PORT`, `ZEROTTS_SERVER_LISTEN_PORT`.
- `zerotts_server` không có auth; chỉ bind `127.0.0.1` hoặc mạng tin cậy. Caller chỉ gọi gateway.

## Gọi API

```bash
curl --fail-with-body localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' -o out.wav \
  -d '{"model":"zerotts","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"maichi"}'

curl localhost:8000/v1/models
curl localhost:8000/v1/audio/voices
```

- Model: `zerotts` (offline, `wav`/`pcm`/`json`) và `zerotts-stream` (`stream: true`, SSE hoặc PCM thô).
- Có `ZEROTTS_GATEWAY_API_KEY` thì thêm `-H "Authorization: Bearer $ZEROTTS_GATEWAY_API_KEY"` cho `/v1/*`.
- `POST /v1/text/normalize` xem văn bản sau chuẩn hoá mà không gọi upstream.
- `seed` (số nguyên) để tái lập audio trên cùng weights/runtime.

## Tuỳ chọn

- **int8** (nhanh hơn, tạo trên host): `uv run --group quantize python scripts/quantize_zerotts.py --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4`, rồi `ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 uv run zerotts-server --config configs/zerotts_server.toml`. Chi tiết: [docs/agents/deploy.md](docs/agents/deploy.md#opt-in-int8).
- **CUDA** (chạy trên host, chưa có Docker): xem [docs/agents/zerotts-cuda.md](docs/agents/zerotts-cuda.md).

## Test

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
