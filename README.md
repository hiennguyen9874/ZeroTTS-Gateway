# ZeroTTS Gateway

Gateway mỏng đứng trước `zerotts_server` (ZeroTTS trên CPU), giữ API kiểu audio.cpp (`/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`). Gateway chuẩn hoá văn bản tiếng Việt trước khi gửi lên upstream, hỗ trợ Bearer auth và không ghi text hay audio vào log. Audio trả về là mono 48 kHz.

## Chạy nhanh

```bash
uv sync
uv run scripts/download_zerotts.sh                          # weights -> models/ZeroTTS
uv run zerotts-server --config configs/zerotts_server.toml  # upstream, cổng 8080
uv run zerotts-gateway --config configs/gateway.toml        # terminal khác, cổng 8000
curl --fail-with-body localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -o out.wav \
  -d '{"model":"zerotts","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"maichi"}'
```

Docker: `cp .env.example .env`, chọn `COMPOSE_FILE` theo hướng dẫn trong file, rồi `docker compose up -d --build`.

## Tài liệu

- Chuyển từ OmniVoice: [docs/MIGRATION.md](docs/MIGRATION.md)
- Quyết định thiết kế: [docs/adr/](docs/adr/); báo cáo đánh giá: [docs/reports/](docs/reports/)
- Tài liệu cho agent (chi tiết hành vi, cấu hình, triển khai): [AGENTS.md](AGENTS.md)

## Test

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
