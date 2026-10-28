# Triển khai bằng Docker

Gateway và upstream luôn là hai container riêng: tách ra thì đổi, build lại hay scale từng bên độc lập. Upstream là `zerotts_server` (ZeroTTS trên onnxruntime CPU, ADR 0005, không cần GPU).

| Cách | `COMPOSE_FILE` trong `.env` | Khi nào dùng |
| --- | --- | --- |
| Chỉ gateway | `compose.yaml` (mặc định) | Upstream đã chạy sẵn trên host hoặc ở máy khác |
| Gateway + `zerotts_server` | `compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml` | Một node CPU chạy cả hai (khuyến nghị khi chỉ có một máy) |
| Chỉ `zerotts_server` | `compose.zerotts.yaml` | Node CPU riêng, gateway chạy ở máy khác |

| File | Nội dung |
| --- | --- |
| `Dockerfile`, `compose.yaml` | Gateway ([GATEWAY.md](GATEWAY.md#chạy-bằng-docker)) |
| `docker/zerotts_server.Dockerfile`, `compose.zerotts.yaml` | `zerotts_server`, CPU ([mục dưới](#zerotts_server)) |
| `compose.zerotts-stack.yaml` | Overlay: gateway gọi `http://zerotts:8080` qua mạng compose, chờ `zerotts_server` healthy (đã nạp model và warmup) rồi mới chạy |
| `scripts/download_zerotts.sh` | Tải weights ZeroTTS theo revision đã pin vào `models/ZeroTTS` |

Mọi lệnh dưới đây chạy từ thư mục gốc repo; `docker compose` đọc `COMPOSE_FILE` trong `.env` nên không cần `-f`.

## zerotts_server

### Yêu cầu

- Docker Compose v2. Không cần GPU, driver NVIDIA hay NVIDIA Container Toolkit.
- Weights trên host, tải một lần (khoảng 860 MB) bằng script; chạy lại không tải lại file đã đúng revision:

  ```bash
  uv run scripts/download_zerotts.sh      # -> models/ZeroTTS, revision pin trong script
  ```

  Script dùng `hf download --local-dir` nên `models/ZeroTTS` chứa file thật, không phải symlink vào HF cache (onnxruntime ≥ 1.25 từ chối external data nằm sau symlink). Máy deploy không có repo uv: chép thư mục `models/ZeroTTS` đã tải sang, hoặc chạy script với `hf` cài bằng `pip install huggingface_hub`.

### Gateway + zerotts_server trên một node CPU

```bash
cp .env.example .env
# .env: COMPOSE_FILE=compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml
#       ZEROTTS_GATEWAY_API_KEY=<key nếu mở ra mạng>
docker compose up -d --build
docker compose ps             # zerotts và gateway đều "healthy"
curl -f localhost:8000/ready  # 200 khi zerotts_server đã nạp model và warmup xong
```

- Gateway gọi `zerotts_server` qua `http://zerotts:8080` (mạng compose); `ZEROTTS_GATEWAY_UPSTREAM_URL` trong `.env` bị bỏ qua. Gateway chỉ khởi động sau khi `zerotts` healthy, nên `/ready` OK ngay khi gateway lên.
- `zerotts_server` vẫn được publish ở `127.0.0.1:8080` của host cho `curl` debug và script nghe. Caller chỉ nên gọi gateway.

Kiểm tra tổng hợp thật qua gateway (thêm Bearer header nếu đã bật auth):

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"zerotts","input":"Xin chào.","voice":"maichi"}' -o /tmp/zerotts.wav
```

### Chỉ zerotts_server (node CPU riêng)

```bash
cp .env.example .env
# .env: COMPOSE_FILE=compose.zerotts.yaml, ZEROTTS_SERVER_BIND=<IP nội bộ hoặc 0.0.0.0>
docker compose up -d --build
curl localhost:8080/health    # 503 model_loading, rồi 200 sau warmup
```

Trên máy chạy gateway: `COMPOSE_FILE=compose.yaml`, `ZEROTTS_GATEWAY_UPSTREAM_URL=http://<node-cpu>:8080`.

`zerotts_server` không có auth. Chỉ mở `ZEROTTS_SERVER_BIND` ra mạng tin cậy và dùng firewall chỉ cho máy gateway vào.

### Image và cấu hình

`docker/zerotts_server.Dockerfile`: `python:3.12-slim`, `uv sync --locked --no-dev --extra server` (`zerotts==0.1.5`, onnxruntime CPU đã pin trong `uv.lock`), chạy bằng user không phải root, build context là gốc repo (dùng chung `.dockerignore` với gateway). Image không chứa weights hay config:

- `configs/` và `models/` mount read-only vào `/srv/configs`, `/srv/models` theo đúng bố cục repo, nên `configs/zerotts_server.toml` (`model_dir = "../models/ZeroTTS"`, tương đối theo vị trí file) dùng chung cho chạy trên host lẫn trong container. Thiếu `configs/` hay `models/` thì `docker compose up` báo lỗi ngay thay vì mount thư mục rỗng.
- Image đặt `ZEROTTS_SERVER_CONFIG=/srv/configs/zerotts_server.toml` và `ZEROTTS_SERVER_HOST=0.0.0.0`; `port` lấy từ config (8080), đổi thì sửa cả `ports` trong `compose.zerotts.yaml` và healthcheck của image. `ZEROTTS_SERVER_MODEL_DIR` (đặt trong `environment`) ghi đè `model_dir` nếu weights mount ở chỗ khác.
- `workers`, `threads`, `busy_timeout`, voice mặc định và sampling: `configs/zerotts_server.toml` ([ZEROTTS_SERVER.md](ZEROTTS_SERVER.md#config)). `workers × threads` không nên vượt số core mà container được dùng.
- Healthcheck gọi `/health`, chỉ 200 sau khi model đã nạp và warmup (khoảng 10 s trên máy dev; `start-period` 120 s cho node chậm).

Biến trong `.env` cho `zerotts_server`:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ZEROTTS_SERVER_BIND`, `ZEROTTS_SERVER_PORT` | `127.0.0.1`, `8080` | Địa chỉ publish trên host |
| `ZEROTTS_SERVER_IMAGE` | `zerotts-server:local` | Tên image; đặt tên registry để push/pull giữa các máy |

Vận hành:

- Sửa `configs/zerotts_server.toml` rồi `docker compose restart zerotts`; không cần build lại.
- Thêm voice: thả voice pack `<tên>/voice.npz` (+ `meta.json`) vào `models/ZeroTTS/voices` trên host rồi restart `zerotts`.
- Đổi revision weights: sửa `REVISION` trong `scripts/download_zerotts.sh`, tải lại, chạy smoke test model thật ([ZEROTTS_SERVER.md](ZEROTTS_SERVER.md#test)) rồi restart.
- Log: `docker compose logs -f zerotts gateway`. Cả hai không ghi text hay audio.
- Khi host khởi động lại, Docker tự chạy lại cả hai (`restart: unless-stopped`) nhưng không theo thứ tự `depends_on`; trong lúc `zerotts_server` chưa warmup xong, `/ready` của gateway trả 503.

## Chỉ gateway

[GATEWAY.md](GATEWAY.md#chạy-bằng-docker).
