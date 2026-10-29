# Chuyển từ OmniVoice sang ZeroTTS

Hướng dẫn cho operator đang chạy gateway trước audio.cpp/OmniVoice và caller đang dùng API đó. Upstream mới là `zerotts_server` (ZeroTTS trên onnxruntime CPU), **không phải** một model có thể nạp vào audio.cpp. Gateway và upstream vẫn là hai process riêng; gateway không tổng hợp, chia đoạn, resample, retry hay xếp hàng. Quyết định: [ADR 0005](adr/0005-zerotts-replaces-omnivoice.md).

Mọi lệnh dưới đây chạy từ thư mục gốc repo. Ví dụ `curl` dùng gateway tại `http://localhost:8000`, chưa bật auth. Nếu đặt `ZEROTTS_GATEWAY_API_KEY`, thêm `-H "Authorization: Bearer $ZEROTTS_GATEWAY_API_KEY"` cho các route `/v1/*`; `/health` và `/ready` không cần auth.

## 1. Những thay đổi bắt buộc

| Thành phần | Trước đây | Sau khi chuyển |
| --- | --- | --- |
| Upstream | audio.cpp chạy OmniVoice | `zerotts_server`, CPU, không cần GPU |
| Package / lệnh gateway | `omnivoice_gateway` / `omnivoice-gateway` | `zerotts_gateway` / `zerotts-gateway` |
| Model offline | Ví dụ `g-omnivoice` | `zerotts` |
| Model streaming | Ví dụ `g-omnivoice-stream` | `zerotts-stream` |
| Voice | Voice cũ hoặc audio tham chiếu | Tên voice pack ZeroTTS, ví dụ `maichi`; không clone giọng |
| Audio | Mono 24 kHz | **Mono 48 kHz, PCM 16-bit little-endian** (kể cả dữ liệu trong SSE) |
| Cấu hình upstream | `configs/server.json` và các tùy chọn OmniVoice | `configs/zerotts_server.toml` |
| Triển khai upstream | Compose/Dockerfile audio.cpp cũ | `compose.zerotts.yaml`, `docker/zerotts_server.Dockerfile` |

Caller có thể giữ các route `/v1/models`, `/v1/audio/voices`, `/v1/audio/speech` và bộ phân tích SSE. Không có alias cho model id, voice hay tên biến gateway cũ. Nếu giữ nguyên URL gateway khi chuyển traffic thì không cần đổi URL ở caller; nếu triển khai song song, đổi sang URL gateway mới (SDK dùng base URL có `/v1` thì giữ quy ước đó).

### Request cần sửa

- Xóa `voice_ref`, `reference_text`, `instruct` và `instructions` — ZeroTTS không hỗ trợ clone giọng hay chỉ dẫn giọng.
- Xóa `speed`/`speaking_rate`, hoặc chỉ gửi giá trị `1`. Giá trị khác bị từ chối, không được bỏ qua âm thầm.
- Trong `options`, chỉ giữ `seed`. Sampling thuộc operator, cấu hình theo model tại upstream; không sao chép tùy chọn sampling OmniVoice vào request mới.
- `seed` có thể nằm ở top-level hoặc `options.seed`, là số nguyên trong `[0, 2^64)` (cũng nhận chuỗi thập phân). Dùng để tái lập trong ZeroTTS, không để tái tạo audio OmniVoice.
- Không gửi stream cho `zerotts`, hoặc request offline cho `zerotts-stream`: sai mode trả 400. Streaming chỉ nhận `response_format: "pcm"`.
- Non-verbal tag như `[laughter]` và phiên âm CMU bị gỡ, kèm `omitted_nonverbal_tag` trong `X-Text-Warnings`; không còn tạo tiếng cười theo thẻ.

Upstream từ chối option không hỗ trợ bằng `400 unsupported_option`; gateway vẫn chặn `voice_ref` dạng đường dẫn trước khi chuyển tiếp, nên có thể trả `400 voice_ref_path_not_allowed`. Lỗi giữ dạng `{"error":{"message":"…","type":"…"}}`, không chứa input text. Không coi mọi lỗi 400 đều là lỗi tạm thời để retry.

**Với PCM thô hoặc SSE:** cập nhật bộ phát và mọi bước xử lý audio từ 24.000 lên **48.000 sample/s, 1 kênh, 16-bit**. Gateway không resample; phát dữ liệu mới ở 24 kHz sẽ làm giọng chậm và trầm. Với WAV, đọc sample rate từ header thay vì hard-code giá trị cũ.

## 2. Chuẩn bị trước khi chuyển

1. Sao lưu deployment cũ: commit/image tag, config, `.env`, lexicon và weights/voice cũ. Không xóa chúng trước khi nghiệm thu; phiên bản hiện tại đã gỡ các file triển khai OmniVoice.
2. Chuẩn bị node CPU, Python ≥ 3.12 và `uv` (hoặc Docker Compose v2). Dùng hai deployment/thư mục config riêng nếu cần chạy song song; tránh trùng cổng 8000/8080.
3. Cài dependency và tải weights theo revision đã pin:

   ```bash
   uv sync
   uv run scripts/download_zerotts.sh
   ```

   `uv sync` mặc định cài cả dependency server qua dev group. Nếu cài production không có dev group, dùng `uv sync --no-dev --extra server` cho upstream; gateway chỉ cần `uv sync --no-dev`.

   Weights nằm ở `models/ZeroTTS`, được tải bằng `hf download --local-dir`, không dùng symlink HF cache. Không dùng weights GGUF/audio tokenizer của OmniVoice hoặc ZeroTTS-GGUF. Chi tiết: [agents/zerotts-server.md](agents/zerotts-server.md#weights).

4. Dùng cấu hình mẫu mới thay vì chép nguyên cấu hình cũ:
   - `configs/zerotts_server.toml`: kiểm tra `model_dir`, `default_voice` cho từng model, sampling, `workers`, `threads`, `busy_timeout`. Mặc định là **1 worker × 8 luồng ONNX**, chờ worker tối đa **5 giây**; mỗi worker giữ một instance model riêng.
   - `configs/gateway.toml`: giữ `zerotts`/`zerotts-stream`, `language = "vi"`, `nonverbal_tags = false`. Pipeline đầy đủ hiện được bật sau A/B #9; **không cần thêm `normalize = false`** theo cấu hình chuyển tiếp cũ. Xem [báo cáo A/B](reports/issue9-zerotts-text-pipeline-ab.md).
   - Gỡ `short_text_speed` và các entry model OmniVoice. Rà lại `deadline_s`, `max_input_chars`, lexicon và timeout của caller cho input text thực tế.
   - Thêm voice bằng voice pack `<tên>/voice.npz` (có thể kèm `meta.json`) vào thư mục voice rồi restart upstream. Không đổi tên WAV cũ thành voice pack.

### Đổi biến môi trường gateway

| Tên cũ | Tên mới |
| --- | --- |
| `OMNIVOICE_GATEWAY_CONFIG` | `ZEROTTS_GATEWAY_CONFIG` |
| `OMNIVOICE_GATEWAY_UPSTREAM_URL` | `ZEROTTS_GATEWAY_UPSTREAM_URL` |
| `OMNIVOICE_GATEWAY_API_KEY` | `ZEROTTS_GATEWAY_API_KEY` |
| `OMNIVOICE_GATEWAY_HOST` | `ZEROTTS_GATEWAY_HOST` |

Không có alias cho các biến cũ. Đổi trong service manager, CI/CD, secret store và cấu hình container; giữ lại giá trị Bearer key nếu muốn caller tiếp tục dùng key hiện tại. `ZEROTTS_GATEWAY_CONFIG` trỏ tới config gateway, không phải config upstream.

Các biến upstream mới gồm `ZEROTTS_SERVER_CONFIG`, `ZEROTTS_SERVER_MODEL_DIR`, `ZEROTTS_SERVER_HOST`. Bộ script nghe hiện vẫn dùng `OMNIVOICE_ASR_*` cho dịch vụ ASR; đó không phải biến gateway, không đổi chúng chỉ bằng cách thay tiền tố.

## 3. Khởi động deployment mới

### Chạy trực tiếp

```bash
uv run zerotts-server --config configs/zerotts_server.toml
# In another terminal:
uv run zerotts-gateway --config configs/gateway.toml
# In a third terminal:
curl --fail-with-body localhost:8080/health
curl --fail-with-body localhost:8000/ready
```

Config mặc định kết nối gateway tới `http://127.0.0.1:8080`. Upstream `/health` trả 503 khi đang nạp model/warmup, rồi 200 khi sẵn sàng. Gateway `/health` chỉ xác nhận process gateway còn sống; dùng **`/ready`** để quyết định nhận traffic.

### Docker: cả hai trên cùng host

Tạo `.env` từ `.env.example` nếu chưa có; nếu đã có, sao lưu và sửa, không ghi đè bí mật:

```dotenv
COMPOSE_FILE=compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml
ZEROTTS_GATEWAY_API_KEY=replace-with-a-secret
ZEROTTS_SERVER_BIND=127.0.0.1
ZEROTTS_SERVER_PORT=8080
GATEWAY_PORT=8000
```

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl --fail-with-body localhost:8000/ready
```

Overlay đặt upstream URL thành `http://zerotts:<ZEROTTS_SERVER_LISTEN_PORT>` (mặc định 8080), ghi đè `ZEROTTS_GATEWAY_UPSTREAM_URL`. Weights/config được mount read-only; image không chứa weights. Nếu upstream cũ đang chiếm 8080, đổi `ZEROTTS_SERVER_PORT` sang cổng host còn trống; kết nối nội bộ stack vẫn dùng `ZEROTTS_SERVER_LISTEN_PORT`. Khi chạy song song, cũng đổi `GATEWAY_PORT` và URL kiểm tra tương ứng.

### Docker: upstream ở node khác

- Node CPU: `COMPOSE_FILE=compose.zerotts.yaml`, đặt `ZEROTTS_SERVER_BIND` thành IP nội bộ phù hợp.
- Node gateway: `COMPOSE_FILE=compose.yaml`, đặt `ZEROTTS_GATEWAY_UPSTREAM_URL=http://<node-cpu>:8080` (hoặc cổng host đã đổi).
- Upstream không có auth: dùng firewall chỉ cho gateway truy cập. Caller chỉ gọi gateway.
- `127.0.0.1` trong container là chính container. Upstream chạy trên Docker host: dùng `host.docker.internal` và cho upstream lắng nghe ở địa chỉ container truy cập được.

Chi tiết biến, volume, restart và healthcheck: [agents/deploy.md](agents/deploy.md).

## 4. Cập nhật và kiểm tra caller

Liệt kê model/voice trước khi chọn voice; kiểm tra `default_voice` có trong danh sách:

```bash
curl --fail-with-body localhost:8000/v1/models
curl --fail-with-body localhost:8000/v1/audio/voices
```

### Offline: WAV, PCM hoặc JSON

```bash
curl --fail-with-body localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' -o /tmp/zerotts.wav \
  -d '{"model":"zerotts","input":"Xin chào. Hẹn gặp lại ngày mai.","voice":"maichi","seed":42}'
```

Thiếu `voice` thì dùng `default_voice` của model, không chọn ngẫu nhiên. Voice lạ trả `400 unknown_voice`. `response_format` mặc định là `wav`; dùng `pcm` để nhận PCM thô hoặc `json` để nhận WAV base64 trong trường `audio` (không phải PCM base64).

Kiểm tra header WAV:

```bash
uv run python - <<'PY'
import wave
with wave.open('/tmp/zerotts.wav') as audio:
    assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48000, 1, 2)
    assert audio.getnframes() > 0
PY
```

### Streaming: SSE

```bash
curl --fail-with-body -N localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"zerotts-stream","input":"Xin chào. Hẹn gặp lại ngày mai.","voice":"maichi","stream_format":"sse","response_format":"pcm","seed":42}'
```

Giữ bộ phân tích sự kiện `speech.audio.delta` → `speech.audio.done` → `[DONE]`. Mỗi delta chứa PCM base64 **48 kHz**, không có header WAV. Chỉ có một `done` cho cả request nhiều đoạn. Nếu nhận sự kiện `error` giữa stream, không coi audio đã nhận là kết quả đầy đủ.

### Streaming: PCM thô

```bash
curl --fail-with-body -N localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' -o /tmp/zerotts.pcm \
  -d '{"model":"zerotts-stream","input":"Xin chào. Hẹn gặp lại ngày mai.","voice":"maichi","stream_format":"audio","response_format":"pcm"}'
```

Caller phát các chunk ngay khi nhận, với PCM mono 48 kHz signed 16-bit little-endian. Ranh giới chunk HTTP không phải ranh giới câu hay toàn bộ audio. Khi engine lỗi giữa stream PCM, upstream cắt kết nối; caller cần xử lý lỗi đọc/kết thúc bất thường, không chỉ kiểm tra status 200 ban đầu.

## 5. Nghiệm thu và chuyển traffic

- [ ] `/ready` trả 200 sau warmup; offline WAV, JSON, SSE và PCM dùng trong ứng dụng đều hoạt động qua gateway.
- [ ] Caller phát đúng 48 kHz, voice được chọn đúng; input text nhiều câu được đọc đủ, stream qua ranh giới đoạn không bị cắt.
- [ ] Các option cũ bị từ chối 400; `speed: 1` vẫn chạy; lỗi giữ dạng audio.cpp. Caller không còn phụ thuộc clone, instruct hay điều chỉnh speed.
- [ ] Bearer auth hoạt động khi bật; upstream không bị mở công khai. Log không chứa input text hay audio.
- [ ] Nghe thử dữ liệu thực tế, đặc biệt tên sản phẩm, viết tắt, PIN/OTP và USD. A/B bằng ASR không thay thế quyết định của người nghe ở các ca này.
- [ ] Đo TTFA/RTF trên **node triển khai thật**, cho input ngắn/vừa/dài với 1 và 2 request tổng hợp đồng thời; thử các cách chia worker × luồng và ghi rõ có đạt RTF < 1 cho hai request hay không. Không dùng số đo máy dev để cam kết hiệu năng; #10 vẫn là hạng mục cần hoàn tất.
- [ ] Kiểm tra tình huống mọi worker bận: sau `busy_timeout`, caller nhận `503 server_busy`. Gateway không retry; caller tự quyết định chính sách thử lại/hủy lượt nói.
- [ ] Với Docker, thực sự build/chạy image và kiểm tra stack trên host đích; chỉ kiểm tra Compose hoặc chạy bằng process thường chưa đủ.

Kiểm tra tự động khi có môi trường phát triển và weights:

```bash
uv sync
uv run python -m pytest -q
uv run ruff check .
uv run python -m pytest -m real_model -q
```

Suite mặc định dùng upstream/engine giả; các test `real_model` phải chọn riêng và tự bỏ qua nếu thiếu weights. Test pass không thay thế benchmark hay nghe thử.

Sau khi nghiệm thu, cập nhật cấu hình caller và/hoặc chuyển load balancer sang gateway mới. Khi đổi upstream tại gateway cũ, phải đồng bộ thay đổi model, voice và sample rate ở caller; không chuyển riêng upstream rồi giữ nguyên request OmniVoice. Dừng nhận request mới ở deployment cũ và để request đang chạy kết thúc trước khi dừng process. Chỉ gỡ deployment cũ sau khi xác nhận vận hành ổn định.

## 6. Quay lại khi có sự cố

1. Ngừng chuyển traffic mới sang deployment ZeroTTS; giữ log chỉ gồm mã lỗi và số đo để điều tra, không thêm input text/audio vào log.
2. Khởi động hoặc giữ nguyên deployment OmniVoice đã lưu bằng image/commit và cấu hình cũ. Repo hiện tại không còn các file triển khai audio.cpp cũ.
3. Khôi phục đồng bộ URL, model id, voice, tùy chọn request và cấu hình audio **24 kHz** ở caller cũ; không đưa request ZeroTTS hoặc PCM 48 kHz vào luồng phát cũ.
4. Kiểm tra readiness và một request hoàn chỉnh trước khi chuyển traffic lại. Không rollback chỉ bằng cách đổi `ZEROTTS_GATEWAY_UPSTREAM_URL`.

Không xóa weights/config cũ hoặc đổi pin trong `references/` như một phần của thao tác chuyển traffic. Dùng deployment riêng cho rollback để tránh trộn config, dependency và bí mật giữa hai phiên bản.
