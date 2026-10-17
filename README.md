# OmniVoice Gateway

Gateway mỏng đứng trước audio.cpp chạy OmniVoice. Gateway giữ nguyên API của audio.cpp (`/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`), nên caller chỉ cần đổi base URL. Gateway thêm:

- chuẩn hoá văn bản tiếng Việt trước khi gửi lên model (số, ngày, tiền, viết tắt, markdown, …);
- Bearer auth tuỳ chọn, chặn `voice_ref` dạng đường dẫn, `/ready` kiểm tra upstream;
- log không chứa văn bản hay audio.

Gateway không tự sinh audio, không chia/ghép, không retry, không xếp hàng. Spec: issue #1.

## Bắt đầu nhanh

**0. Lấy mã nguồn tham chiếu** (audio.cpp, sea-g2p, … là git submodule trong `references/`):

```bash
git submodule update --init
```

Không cần `--recursive`: submodule lồng của audio.cpp chỉ dùng cho WebUI ([docs/INSTALL.md](docs/INSTALL.md#2-lấy-source)).

**1. Chạy audio.cpp** (build: [docs/INSTALL.md](docs/INSTALL.md); model và server: [docs/USAGE.md](docs/USAGE.md)):

```bash
references/audio.cpp/build/linux-cuda126-release/bin/audiocpp_server --config configs/server.json --no-ui
```

**2. Chạy gateway** (cổng 8000, upstream `http://127.0.0.1:8080`):

```bash
uv run omnivoice-gateway --config configs/gateway.toml
```

Hoặc bằng Docker: `cp .env.example .env && docker compose up -d --build`. Mặc định chỉ chạy gateway; chạy cả audio.cpp trong Docker thì đặt `COMPOSE_FILE` trong `.env` (xem [docs/DEPLOY.md](docs/DEPLOY.md)).

**3. Gọi API** như gọi audio.cpp:

```bash
curl localhost:8000/ready   # 200 khi tới được upstream

curl -s --fail-with-body localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -o out.wav \
  -d '{"model":"g-omnivoice","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"vieneu_v3_Kim_Thanh"}'

# Xem văn bản thật sự gửi lên model (không gọi upstream)
curl -s localhost:8000/v1/text/normalize -H 'Content-Type: application/json' \
  -d '{"input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","language":"vi"}'
```

Đặt `OMNIVOICE_GATEWAY_API_KEY` thì mọi route trừ `/health`, `/ready` cần header `Authorization: Bearer <key>`. Tắt chuẩn hoá cho một request: thêm `"text_processing": {"normalize": false}`.

## Chọn model và giọng

Mặc định nên dùng `g-omnivoice` + `vieneu_v3_Kim_Thanh`; đoạn dài cần phát sớm dùng `g-omnivoice-stream`. `configs/server.json` chỉ khai báo hai entry này; các model khác (`omnivoice-vietnamese`, `omnivoice`) cần thêm lại vào config theo [docs/USAGE.md, mục 4.1.1](docs/USAGE.md#411-thêm-lại-các-model-đã-bỏ). Lý do và giới hạn: [docs/MODEL_VOICE_RECOMMENDATIONS.md](docs/MODEL_VOICE_RECOMMENDATIONS.md).

## Tài liệu

| Tài liệu | Nội dung |
| --- | --- |
| [docs/INSTALL.md](docs/INSTALL.md) | Build audio.cpp với CUDA 12.6 trên A100 |
| [docs/USAGE.md](docs/USAGE.md) | Convert checkpoint sang GGUF, CLI, server audio.cpp, voice clone/design, API |
| [docs/GATEWAY.md](docs/GATEWAY.md) | Chạy gateway, biến môi trường, route, config, Docker |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Triển khai Docker: chỉ gateway, chỉ audio.cpp, hoặc cả hai |
| [docs/TEXT_PIPELINE.md](docs/TEXT_PIPELINE.md) | Text pipeline, cảnh báo và mã lỗi, từ điển phát âm, thêm ca regression |
| [docs/LISTENING.md](docs/LISTENING.md) | Render mẫu nghe thử A/B, bộ câu nghe, chấm bằng ASR, chạy mọi model × giọng |
| [docs/MODEL_VOICE_RECOMMENDATIONS.md](docs/MODEL_VOICE_RECOMMENDATIONS.md) | Đánh giá model, giọng và khuyến nghị sử dụng |
| [docs/adr/](docs/adr/) | Quyết định thiết kế |
| [docs/reports/](docs/reports/) | Báo cáo các lượt nghe và đo |

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
