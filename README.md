# OmniVoice Gateway

Gateway đứng trước audio.cpp, giữ nguyên API `/v1/audio/speech` của upstream. Spec: issue #1.

## Chạy

```bash
uv run omnivoice-gateway --config configs/gateway.toml
```

Biến môi trường:

- `OMNIVOICE_GATEWAY_CONFIG`: đường dẫn config, dùng khi không truyền `--config`
- `OMNIVOICE_GATEWAY_UPSTREAM_URL`: ghi đè `upstream_url`
- `OMNIVOICE_GATEWAY_API_KEY`: đặt thì mọi route trừ `/health`, `/ready` yêu cầu `Authorization: Bearer <key>`

## Text pipeline

`POST /v1/audio/speech` thay `input` bằng engine text trước khi gửi lên upstream. Field riêng `text_processing: {"normalize": bool}` bị gỡ trước khi forward; guard và làm sạch luôn chạy.

- Response thành công có `X-Text-Pipeline-Version` và, khi có cảnh báo, `X-Text-Warnings` (`omitted_code_block`, `omitted_html`, `omitted_url`, `omitted_emoji`).
- Lỗi đầu vào `400`: `text_too_long`, `reasoning_markup`, `model_control`, `empty_text`; trả trước khi gọi upstream.
- Dry-run: `POST /v1/text/normalize` với `{"input", "language"?, "model"?, "text_processing"?}` trả `engine_text`, `warnings`, `language`, `pipeline_version`, không gọi upstream.

Ca golden nằm ở `tests/golden/pipeline.json`, chạy qua route dry-run; mỗi ca có `why`. Đổi luật thì tăng `RULES_VERSION` trong `src/omnivoice_gateway/text.py`.

## Test

```bash
uv run pytest
```
