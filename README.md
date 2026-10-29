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

Khi chạy trực tiếp, đặt `ZEROTTS_SERVER_PORT` và `ZEROTTS_GATEWAY_PORT` để ghi đè cổng lắng nghe trong TOML. Nếu đổi cổng upstream, cũng đặt `ZEROTTS_GATEWAY_UPSTREAM_URL` trỏ tới cổng đó.

Docker: `cp .env.example .env`, chọn `COMPOSE_FILE` theo hướng dẫn trong file, rồi `docker compose up -d --build`. Các cổng host (`GATEWAY_PORT`, `ZEROTTS_SERVER_PORT`) độc lập với cổng process trong container (`ZEROTTS_GATEWAY_PORT`, `ZEROTTS_SERVER_LISTEN_PORT`).

## Weights int8 (tuỳ chọn)

fp32 vẫn là mặc định. Tạo int8 từ weights đã tải, trên host (không tải từ Hub,
không chạy trong image):

```bash
uv run --group quantize python scripts/quantize_zerotts.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 \
  uv run zerotts-server --config configs/zerotts_server.toml
```

Thư mục đích mặc định `models/ZeroTTS-int8` chứa đủ weights, tokenizer, voices,
codec và `quantization.json`; script từ chối ghi đè trừ khi có `--force`.
Có thể đổi `model_dir` trong config thay cho biến môi trường; quay lại fp32 bằng
cách chọn `models/ZeroTTS` và khởi động lại. Docker vẫn mount weights read-only.
Seed chỉ tái lập trên cùng bộ weights/runtime: cùng seed cho audio khác nhau
giữa fp32 và int8. Hai lần quantize trên máy dev có SHA-256 giống nhau, không
đảm bảo giữa các máy/phiên bản; nhiều node nên copy chung một thư mục int8.
Chi tiết: [triển khai](docs/agents/deploy.md#opt-in-int8),
[báo cáo kiểm chứng](docs/reports/issue12-zerotts-int8.md),
[A/B chất lượng fp32 so với int8](docs/reports/issue14-zerotts-int8-quality-ab.md).

## Tài liệu

- Chuyển từ OmniVoice: [docs/MIGRATION.md](docs/MIGRATION.md)
- Quyết định thiết kế: [docs/adr/](docs/adr/); báo cáo đánh giá: [docs/reports/](docs/reports/)
- Tài liệu cho agent (chi tiết hành vi, cấu hình, triển khai): [AGENTS.md](AGENTS.md)

## Test

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
