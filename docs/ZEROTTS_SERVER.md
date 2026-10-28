# Chạy và cấu hình `zerotts_server`

`zerotts_server` là upstream mới (ADR 0005): bọc `zerotts==0.1.5` trên onnxruntime CPU và nói phần API audio.cpp mà gateway dùng. Spec: issue #1. Có model offline `zerotts` và model streaming `zerotts-stream`; một engine xử lý lần lượt từng request (worker và `busy_timeout`: #6). Docker và script tải weights: #7.

## Weights

Tải vào thư mục cục bộ, không qua symlink của HF cache (onnxruntime ≥ 1.25 từ chối external data nằm sau symlink):

```bash
uv run hf download zeroweight-ai/ZeroTTS --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4 \
  --local-dir models/ZeroTTS \
  --include 'config.json' --include 'tokenizer.json' --include 'null_voice_emb.npy' \
  --include 'silence_frame.npy' --include 'onnx/*' --include 'onnx/codec/*' \
  --include 'voices/index.json' --include 'voices/*/voice.npz' --include 'voices/*/meta.json'
```

## Chạy

```bash
uv sync                      # dev group kéo theo extra `server` (zerotts, onnxruntime, numpy)
uv run zerotts-server --config configs/zerotts_server.toml
```

Chỉ cài cho server: `uv sync --no-dev --extra server`. Server nghe ngay, nạp model và warmup ở nền; `/health` trả 503 tới khi xong.

Biến môi trường:

- `ZEROTTS_SERVER_CONFIG`: đường dẫn config, dùng khi không truyền `--config`
- `ZEROTTS_SERVER_MODEL_DIR`: ghi đè `model_dir`
- `ZEROTTS_SERVER_HOST`: ghi đè `host`

## Config

`configs/zerotts_server.toml` (chú thích ngay trong file). Đường dẫn tương đối tính từ thư mục chứa file config.

- `host`, `port` (mặc định 8080), `threads` (số luồng ONNX, mặc định 8).
- `model_dir`: thư mục weights; `voices_dir`: thư mục voice pack, mặc định `<model_dir>/voices`. Thêm voice: thả một thư mục `<tên>/voice.npz` (kèm `meta.json` nếu muốn có mô tả) vào đó rồi khởi động lại.
- `[models.<id>]`: model id phục vụ. `mode`: `offline` (mặc định) hoặc `streaming`, như entry `mode` của audio.cpp; mặc định có `zerotts` (offline) và `zerotts-stream` (streaming). `default_voice` dùng khi request không có `voice` (phải có trong thư mục voice, nếu không model báo nạp lỗi). `[models.<id>.sampling]`: `cfg_scale`, `text_temperature`, `text_topk`, `audio_temperature`, `audio_topk`, `audio_topp`, `audio_repetition_penalty`; mặc định theo `zerotts`.

## Route

| Route | Ghi chú |
| --- | --- |
| `GET /health` | 200 `{"status":"ok"}` sau warmup; 503 `model_loading` trước đó, 503 `model_load_failed` nếu nạp lỗi |
| `GET /v1/models` | Model id kiểu audio.cpp (`mode`: `offline`/`streaming`, `loaded`) |
| `GET /v1/audio/voices` | `voices`: danh sách tên như audio.cpp; `details`: `display_name`, `description`, `gender`, `language` |
| `POST /v1/audio/speech` | Model offline: `response_format` `wav` (mặc định, 48 kHz mono 16-bit), `pcm` (PCM thô `audio/pcm`), `json`/`b64_json` (`{"audio": <WAV base64>, "format": "wav", "timing"}`). Model streaming: `stream_format` `sse`/`audio` (xem dưới) |

Request `/v1/audio/speech`:

- `voice`: tên voice; thiếu thì dùng `default_voice` của model; voice lạ trả `400 unknown_voice`.
- `input`: engine text được chia thành đoạn theo cách của `zerotts` (ADR 0001, ADR 0005): Chuẩn hóa Unicode về NFC trước khi `chunk_text` chia theo câu rồi gom lại sao cho mỗi đoạn ≤ khoảng 15 s (225 ký tự); câu quá dài thì chia tiếp theo dấu phẩy, rồi theo từ. Không đổi hoặc thêm dấu câu, không xóa ký hiệu có nghĩa như `+`, `=`, `;` (ví dụ `C++`, `1+1` được giữ nguyên). Các đoạn được tổng hợp tuần tự trong cùng request, cùng một RNG theo seed, cách nhau 0,15 s im lặng, rồi trả về một WAV duy nhất. Không còn bị cắt ở 512 token hay 1500 frame (120 s). Input không còn chữ nào để đọc (ví dụ `... !!`) trả `400 invalid_request_error`.
- `seed` (hoặc `options.seed`): số nguyên trong [0, 2^64), có thể gửi dạng chuỗi thập phân. Cùng seed cho cùng audio; seed áp riêng cho từng request, không dùng RNG toàn cục của numpy. Không có seed thì mỗi request ngẫu nhiên.
- Trả `400 unsupported_option`: `voice_ref`, `reference_text`, `instruct`, `instructions`, `speed`/`speaking_rate` khác 1, mọi khóa trong `options` ngoài `seed`. `speed: 1` được chấp nhận.
- Request stream (`stream_format` khác null hoặc `stream: true`) gửi cho model offline, hoặc request không stream gửi cho model streaming, trả `400 invalid_request_error`. audio.cpp trả WAV khi request không stream gửi cho model streaming; ở đây spec #1 chọn từ chối.

### Streaming (`zerotts-stream`)

- `stream_format: "sse"` (mặc định khi chỉ gửi `stream: true`): `text/event-stream`, header `X-Accel-Buffering: no`. Các sự kiện `data: {"type":"speech.audio.delta","audio":<PCM base64>}`, rồi `data: {"type":"speech.audio.done","timing":{"ttft_ms":…}}`, rồi `data: [DONE]`, như audio.cpp. Ghép các delta lại được đúng audio của engine.
- `stream_format: "audio"`: PCM thô 48 kHz mono 16-bit little-endian, `audio/pcm`, chunked transfer.
- `response_format` chỉ nhận `pcm` (mặc định); khác thì trả `400 invalid_request_error`, `stream_format` khác `sse`/`audio` cũng vậy.
- Chia đoạn và seed như offline. Một codec streaming decoder dùng chung cho mọi đoạn để không có tiếng click ở ranh giới đoạn (như `webui/engine.py` của ZeroTTS). Đoạn đầu decode theo nhịp 1, 2, 4, 8, 16 frame để TTFA thấp; các đoạn sau decode 16 frame một lần. Giữa các đoạn chèn 2 frame im lặng của codec (`silence_frame.npy`, khoảng 0,16 s). Chỉ có một `speech.audio.done` cho cả request.
- Engine lỗi giữa stream: SSE kết thúc bằng `data: {"type":"error","error":{"message","type":"server_error"}}`, không có `speech.audio.done` hay `[DONE]`. PCM thô không có chỗ báo lỗi nên server cắt kết nối (không gửi chunk kết thúc), để caller không nhầm audio cụt là audio đủ. Caller ngắt kết nối thì engine dừng sau mẩu audio đang sinh.
- Gateway chuyển stream nguyên vẹn, không cần đổi code.

Body lỗi theo dạng audio.cpp `{"error":{"message","type"}}`; `message` không chứa text. Log là JSON, mỗi request một dòng: model, voice, seed, số ký tự, số đoạn (`segments`), `synth_ms`, `audio_ms`, `rtf`, `ttfa_ms` (khi stream), mã lỗi; không có text hay audio.

## Test

`tests/test_server.py` chạy qua HTTP với engine giả (`create_app(settings, engine=...)`). `tests/test_server_smoke.py` (marker `real_model`) chạy model thật, mặc định không được chọn kể cả khi có weights. Chạy riêng bằng `uv run python -m pytest -m real_model`; test vẫn tự bỏ qua khi không có weights trong `models/ZeroTTS`.
