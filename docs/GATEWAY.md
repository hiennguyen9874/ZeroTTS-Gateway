# Chạy và cấu hình gateway

Gateway đứng trước audio.cpp (upstream) và giữ nguyên API của upstream: caller chỉ đổi base URL. Spec: issue #1. Build và chạy audio.cpp: [INSTALL.md](INSTALL.md), [USAGE.md](USAGE.md).

## Chạy trực tiếp

```bash
uv run omnivoice-gateway --config configs/gateway.toml
```

Biến môi trường:

- `OMNIVOICE_GATEWAY_CONFIG`: đường dẫn config, dùng khi không truyền `--config`
- `OMNIVOICE_GATEWAY_UPSTREAM_URL`: ghi đè `upstream_url`
- `OMNIVOICE_GATEWAY_API_KEY`: đặt thì mọi route trừ `/health`, `/ready` yêu cầu `Authorization: Bearer <key>`
- `OMNIVOICE_GATEWAY_HOST`: ghi đè `host` (địa chỉ lắng nghe)

## Route

| Route | Auth | Ghi chú |
| --- | --- | --- |
| `GET /health` | không | Gateway đang chạy |
| `GET /ready` | không | 200 khi tới được upstream, 503 khi không |
| `GET /v1/models`, `GET /v1/audio/voices` | có (nếu đặt key) | Chuyển nguyên tới upstream |
| `POST /v1/audio/speech` | có (nếu đặt key) | Chạy text pipeline rồi gửi một request lên upstream ([TEXT_PIPELINE.md](TEXT_PIPELINE.md)) |
| `POST /v1/text/normalize` | có (nếu đặt key) | Dry-run text pipeline, không gọi upstream |

Body lỗi theo dạng của audio.cpp: `{"error":{"message","type"}}`.

## Config

`configs/gateway.toml` (chú thích chi tiết ngay trong file):

- `upstream_url`, `host`, `port`: địa chỉ upstream và địa chỉ lắng nghe.
- `deadline_s`: giới hạn tuyệt đối cho mỗi request (chờ byte đầu và stream toàn bộ body).
- `connect_timeout_s`: giới hạn riêng cho bước kết nối upstream; quá thì caller nhận `502 upstream_unavailable` và `/ready` trả 503.
- `max_input_chars`: quá thì trả `400 text_too_long`.
- `lexicon_path`: từ điển phát âm ([TEXT_PIPELINE.md](TEXT_PIPELINE.md#từ-điển-phát-âm)).
- `[models.<id>]`: mặc định theo model: `language`, `normalize`, `short_text_speed_bands` / `short_text_speed`, tuỳ chọn thêm `short_text_speed_ramp` ([ADR 0004](adr/0004-short-text-speed.md)). Model không có trong config vẫn được forward.

## Chạy bằng Docker

Container chỉ chứa gateway. audio.cpp chạy trên host, ở máy riêng, hoặc trong container riêng cùng compose (`compose.audiocpp.yaml` + `compose.stack.yaml`): xem [DEPLOY.md](DEPLOY.md). Phần dưới là cách chỉ chạy gateway (`COMPOSE_FILE=compose.yaml`, mặc định).

```bash
cp .env.example .env        # sửa OMNIVOICE_GATEWAY_UPSTREAM_URL, OMNIVOICE_GATEWAY_API_KEY
docker compose up -d --build
curl localhost:8000/health  # 200
curl localhost:8000/ready   # 200 khi tới được upstream, 503 khi không
```

- `configs/` được mount read-only vào `/config`; gateway đọc `/config/gateway.toml` và từ điển theo `lexicon_path`. Sửa config hoặc từ điển rồi `docker compose restart`, không cần build lại.
- Trong container gateway lắng nghe trên `0.0.0.0:8000` (image đặt `OMNIVOICE_GATEWAY_HOST=0.0.0.0`; `port` vẫn lấy từ config, đổi thì sửa cả `ports` trong `compose.yaml`). Cổng trên host: `GATEWAY_PORT`.
- `127.0.0.1` trong container là chính container. audio.cpp trên host: dùng `http://host.docker.internal:8080` (mặc định, compose đã thêm `host-gateway`) và cho audio.cpp lắng nghe trên `0.0.0.0` thay vì `127.0.0.1`. audio.cpp ở máy khác: dùng host/IP của máy đó.
- Image: `python:3.12-slim`, `uv sync --locked --no-dev`, chạy bằng user không phải root; `.dockerignore` chỉ cho vào `pyproject.toml`, `uv.lock`, `README.md`, `src/`.
- Healthcheck của image gọi `/health` (gateway còn sống), không gọi `/ready`: upstream chết không làm Docker restart gateway. Load balancer nên kiểm tra `/ready`.
