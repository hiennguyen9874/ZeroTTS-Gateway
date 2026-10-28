# ZeroTTS Gateway

Gateway mỏng đứng trước upstream `zerotts_server` (ZeroTTS trên CPU), cung cấp API kiểu audio.cpp. Gateway chuẩn hoá văn bản tiếng Việt, hỗ trợ Bearer auth, chặn đường dẫn `voice_ref`, kiểm tra upstream qua `/ready` và không ghi text hay audio vào log. Gateway không tổng hợp, chia đoạn, retry hay xếp hàng.

## Bắt đầu nhanh

```bash
uv sync
uv run scripts/download_zerotts.sh
uv run zerotts-server --config configs/zerotts_server.toml
# Trong terminal khác:
uv run zerotts-gateway --config configs/gateway.toml
curl -f localhost:8000/ready
curl --fail-with-body localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -o out.wav \
  -d '{"model":"zerotts","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"maichi"}'
```

Docker và cấu hình từng cách triển khai: [docs/DEPLOY.md](docs/DEPLOY.md). Đặt `ZEROTTS_GATEWAY_API_KEY` để yêu cầu Bearer auth (trừ `/health`, `/ready`).

## Ghi chú chuyển đổi từ OmniVoice

Hướng dẫn từng bước, kiểm tra caller và rollback: [docs/MIGRATION.md](docs/MIGRATION.md).

- Đổi model sang `zerotts` (offline) hoặc `zerotts-stream` (streaming); đổi voice sang tên voice pack ZeroTTS, ví dụ `maichi`. Voice mặc định do operator cấu hình. Xem [khuyến nghị](docs/MODEL_VOICE_RECOMMENDATIONS.md).
- Audio là mono **48 kHz**, thay cho 24 kHz; caller PCM thô phải cập nhật sample rate.
- `voice_ref`, `reference_text`, `instruct`/`instructions`, `speed`/`speaking_rate` khác 1 và các khóa `options` ngoài `seed` bị từ chối với `400 unsupported_option`. ZeroTTS không clone giọng. `speed: 1` vẫn được chấp nhận.
- Lệnh và biến gateway nay là `zerotts-gateway` và `ZEROTTS_GATEWAY_*`. Xem [hợp đồng server](docs/ZEROTTS_SERVER.md#route) cho định dạng và giới hạn đầy đủ.

## Tài liệu

- [Gateway](docs/GATEWAY.md), [upstream ZeroTTS](docs/ZEROTTS_SERVER.md), [triển khai Docker](docs/DEPLOY.md)
- [Text pipeline](docs/TEXT_PIPELINE.md), [nghe thử và ASR](docs/LISTENING.md)
- [Quyết định thiết kế](docs/adr/), [báo cáo](docs/reports/)

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
