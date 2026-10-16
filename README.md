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
- `OMNIVOICE_GATEWAY_HOST`: ghi đè `host` (địa chỉ lắng nghe)

## Chạy bằng Docker

Container chỉ chứa gateway; audio.cpp chạy ở nơi khác (trên host hoặc máy riêng).

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

## Text pipeline

`POST /v1/audio/speech` thay `input` bằng engine text trước khi gửi lên upstream. Field riêng `text_processing: {"normalize": bool}` bị gỡ trước khi forward; guard và làm sạch luôn chạy.

- Response thành công có `X-Text-Pipeline-Version` và, khi có cảnh báo, `X-Text-Warnings` (`omitted_code_block`, `omitted_html`, `omitted_url`, `omitted_emoji`).
- Lỗi đầu vào `400`: `text_too_long`, `reasoning_markup`, `model_control`, `empty_text`; trả trước khi gọi upstream.
- Dry-run: `POST /v1/text/normalize` với `{"input", "language"?, "model"?, "text_processing"?}` trả `engine_text`, `warnings`, `language`, `pipeline_version`, không gọi upstream.

Khi effective language là tiếng Việt và normalize bật, pipeline còn: giữ nguyên 13 non-verbal tag (`[laughter]`, `[sigh]`, …) và phiên âm CMU viết hoa (`[B EY1 S]`), bỏ dấu ngoặc của các ngoặc vuông khác; áp từ điển phát âm; chuẩn hoá bằng sea-g2p (`punc_norm=False`); gỡ thẻ `<en>` và ghép chữ cái đánh vần thành viết tắt hoa (`AI`). `reference_text` đi qua cùng pipeline, cùng mã lỗi, không tính vào warnings.

### Từ điển phát âm

`lexicon_path` trong config (tương đối theo file config; bản mặc định `configs/lexicon.csv`). CSV có header `word,replacement,note`:

```csv
word,replacement,note
iPhone,ai phôn,thương hiệu
```

Khớp theo ranh giới từ, mục dài trước, phân biệt hoa thường; chạy trước sea-g2p nên `replacement` phải là dạng nói tiếng Việt. Nạp lúc khởi động (sửa thì restart); file sai định dạng làm gateway không khởi động. Hash của từ điển nằm trong `X-Text-Pipeline-Version` (`<RULES_VERSION>-<hash>`).

### Thêm ca regression

Ca golden nằm ở `tests/golden/pipeline.json`, chạy qua route dry-run. Mỗi khi sửa một lỗi đọc sai:

1. Xem engine text hiện tại: `curl -s localhost:8000/v1/text/normalize -d '{"input":"…","language":"vi"}'`.
2. Thêm một dòng: `id`, `group` (`numbers-money`, `dates-times`, `measurements`, `codes-phones`, `abbrev-english`, `markup`, `tags`, `input-errors`, `short-sentences`, `language`, …), `input`, tuỳ chọn `language` (mặc định `vi`) và `normalize`, rồi `expected` + `warnings` hoặc `error`, và `why` (vì sao đó là cách đọc đúng).
3. Nếu tạm chấp nhận output sai (hành vi của sea-g2p chưa sửa), ghi output hiện tại vào `expected` và thêm `known_issue` mô tả chỗ sai; không xoá ca.
4. Sửa luật hoặc thêm dòng vào từ điển. Đổi luật thì tăng `RULES_VERSION` trong `src/omnivoice_gateway/text.py`.
5. `uv run pytest`.

## Nghe thử mẫu

Cần audio.cpp (`docs/USAGE.md`) và gateway đang chạy. Script đọc các ca golden không lỗi, gọi cả gateway (A) lẫn audio.cpp trực tiếp (B) với cùng model, voice, seed, rồi lưu WAV và trang `index.html` để nghe so sánh:

```bash
uv run python scripts/render_samples.py                      # --model, --voice, --seed, --group, --id, --no-direct
python -m http.server -d outputs/samples 9000                # mở http://localhost:9000
```

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
