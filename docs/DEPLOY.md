# Triển khai bằng Docker

Gateway và upstream luôn là hai container riêng: tách ra thì đổi, build lại hay scale từng bên độc lập. Upstream là `zerotts_server` (ZeroTTS trên onnxruntime CPU, ADR 0005, không cần GPU). audio.cpp chạy OmniVoice vẫn còn cho tới khi #11 gỡ nó; các mục audio.cpp bên dưới chỉ dành cho OmniVoice.

| Cách | `COMPOSE_FILE` trong `.env` | Khi nào dùng |
| --- | --- | --- |
| Chỉ gateway | `compose.yaml` (mặc định) | Upstream đã chạy sẵn trên host hoặc ở máy khác |
| Gateway + `zerotts_server` | `compose.yaml:compose.zerotts.yaml:compose.zerotts-stack.yaml` | Một node CPU chạy cả hai (khuyến nghị khi chỉ có một máy) |
| Chỉ `zerotts_server` | `compose.zerotts.yaml` | Node CPU riêng, gateway chạy ở máy khác |
| Gateway + audio.cpp (CPU) | `compose.yaml:compose.audiocpp.yaml:compose.stack.yaml:compose.cpu.yaml` | OmniVoice, máy không có GPU |
| Chỉ audio.cpp (CPU) | `compose.audiocpp.yaml:compose.cpu.yaml` | OmniVoice, upstream CPU riêng |
| Chỉ audio.cpp | `compose.audiocpp.yaml` | OmniVoice, máy GPU riêng |
| Gateway + audio.cpp | `compose.yaml:compose.audiocpp.yaml:compose.stack.yaml` | OmniVoice, một máy GPU chạy cả hai |

| File | Nội dung |
| --- | --- |
| `Dockerfile`, `compose.yaml` | Gateway ([GATEWAY.md](GATEWAY.md#chạy-bằng-docker)) |
| `docker/zerotts_server.Dockerfile`, `compose.zerotts.yaml` | `zerotts_server`, CPU ([mục dưới](#zerotts_server)) |
| `compose.zerotts-stack.yaml` | Overlay: gateway gọi `http://zerotts:8080` qua mạng compose, chờ `zerotts_server` healthy (đã nạp model và warmup) rồi mới chạy |
| `scripts/download_zerotts.sh` | Tải weights ZeroTTS theo revision đã pin vào `models/ZeroTTS` |
| `docker/audiocpp.Dockerfile`, `compose.audiocpp.yaml` | audio.cpp build từ submodule `references/audio.cpp` (OmniVoice) |
| `compose.cpu.yaml` | Overlay CPU cho audio.cpp: base image Ubuntu, tắt CUDA khi build, bỏ yêu cầu GPU và ghi đè backend lúc chạy |
| `compose.stack.yaml` | Overlay: gateway gọi `http://audiocpp:8080` qua mạng compose, chờ audio.cpp healthy rồi mới chạy |

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

## audio.cpp (OmniVoice, tới #11)

### Yêu cầu

- Docker Compose v2 (CPU overlay cần >= 2.24.0 để hỗ trợ `!reset`).
- Chỉ khi dùng GPU cho audio.cpp: [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit); `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` phải chạy được. Host hiện tại: A100, driver 535. Image CUDA 12.6 chấp nhận driver datacenter 535 (`NVIDIA_REQUIRE_CUDA` của image), và kernel build sẵn cho sm_80 nên không phải JIT.
- Source audio.cpp: `git submodule update --init references/audio.cpp` (không cần submodule lồng; WebUI đã build sẵn trong source).
- Model và giọng trên host, đúng đường dẫn trong `configs/server.json`: `models/g-omnivoice-GGUF/g-omnivoice-f16.gguf` ([USAGE.md](USAGE.md)), `voices/` (`scripts/sample_voices.py`).

### Gateway + audio.cpp trên CPU (không GPU)

```bash
cp .env.example .env
# .env: COMPOSE_FILE=compose.yaml:compose.audiocpp.yaml:compose.stack.yaml:compose.cpu.yaml
#       ZEROTTS_GATEWAY_API_KEY=<key nếu mở ra mạng>
docker compose config        # không còn reservation thiết bị NVIDIA
docker compose up -d --build
docker compose ps
curl -f localhost:8000/ready
```

Overlay dùng cùng Dockerfile nhưng build và chạy trên `ubuntu:24.04`, với CUDA tắt;
không cần driver NVIDIA hay NVIDIA Container Toolkit. `--backend cpu` ghi đè
`backend: cuda` trong `configs/server.json`, không cần sửa config dùng chung với GPU.
Model/giọng, API, WebUI và đường dẫn mount giữ nguyên. Tag image CPU riêng:
`AUDIOCPP_CPU_IMAGE` (mặc định `omnivoice-audiocpp:cpu`). `AUDIOCPP_BUILD_JOBS`
(mặc định 4) giới hạn số tác vụ biên dịch để tránh hết RAM.

CPU chậm hơn GPU đáng kể; `/ready` chỉ kiểm tra upstream, không chạy tổng hợp thử.
Kiểm tra tổng hợp thực sự (thêm Bearer header nếu đã bật auth):

```bash
curl --fail-with-body --max-time 300 http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"g-omnivoice","input":"Xin chào.","voice":"vieneu_v3_Ngoc_Huyen","response_format":"wav"}' \
  -o /tmp/omnivoice-cpu.wav
```

Nếu hết deadline 300 giây, kiểm tra log và cân nhắc tăng `deadline_s` trong
`configs/gateway.toml` theo độ dài input và năng lực CPU; restart gateway sau khi đổi.

### Gateway + audio.cpp trên một máy GPU

```bash
cp .env.example .env
# .env: COMPOSE_FILE=compose.yaml:compose.audiocpp.yaml:compose.stack.yaml
#       AUDIOCPP_GPU=<GPU trống>, ZEROTTS_GATEWAY_API_KEY=<key>
docker compose build          # lần đầu build audio.cpp (CUDA, chỉ OmniVoice)
docker compose up -d
docker compose ps             # audiocpp và gateway đều "healthy"
curl localhost:8000/ready     # 200
```

- Gateway gọi audio.cpp qua `http://audiocpp:8080` (mạng compose); `ZEROTTS_GATEWAY_UPSTREAM_URL` trong `.env` bị bỏ qua.
- audio.cpp vẫn được publish ở `127.0.0.1:8080` của host, để `scripts/render_samples.py` (phía B gọi thẳng upstream), `curl` debug và các lệnh trong [USAGE.md](USAGE.md) chạy như cũ. Caller chỉ nên gọi gateway.

### Chỉ audio.cpp (máy GPU riêng)

```bash
cp .env.example .env
# .env: COMPOSE_FILE=compose.audiocpp.yaml, AUDIOCPP_GPU=..., AUDIOCPP_BIND=<IP nội bộ hoặc 0.0.0.0>
docker compose up -d --build
curl localhost:8080/health
```

Trên máy chạy gateway: `COMPOSE_FILE=compose.yaml`, `ZEROTTS_GATEWAY_UPSTREAM_URL=http://<máy-gpu>:8080`.

audio.cpp không có auth và đọc file theo `voice_ref` (gateway chặn `voice_ref` dạng đường dẫn, gọi thẳng upstream thì không). Chỉ mở `AUDIOCPP_BIND` ra mạng tin cậy và dùng firewall chỉ cho máy gateway vào.

### Image audio.cpp

`docker/audiocpp.Dockerfile` là bản rút gọn của `references/audio.cpp/.devops/cuda.Dockerfile`, build context là `references/audio.cpp` (dùng `.dockerignore` của submodule). Không dùng thẳng image hay Dockerfile của upstream vì:

- Image publish `ghcr.io/0xshug0/audio.cpp:full-cuda12` là tag mutable theo bản mới nhất, và không có tag nào cho commit đang pin (`65df584a`). Gateway được kiểm tra với commit pin, nên build từ submodule.
- Dockerfile upstream build cả bộ model `full` (khoảng 120 family) trong khi gateway chỉ dùng OmniVoice. Ở đây chỉ build `omnivoice` và target `audiocpp_server`, như build trên host ([INSTALL.md](INSTALL.md#3-build)).
- Không cần `python3` hay `model_specs/`: WebUI tải model bằng package manager C++, spec nhúng sẵn (deployment build).
- Danh sách arch CUDA mặc định của upstream chỉ có `80-virtual` cho A100 (PTX, JIT lúc chạy, cần driver hỗ trợ PTX của toolkit 12.9). Ở đây mặc định `80` (SASS + PTX).
- Upstream mặc định CUDA 12.9 + GCC 14. Ở đây dùng CUDA 12.6 (bản đã kiểm tra trên host) và GCC 13 vì nvcc 12.6 không nhận GCC 14.

Image chạy bằng user `ubuntu` (uid 1000), chỉ có `audiocpp_server` (kèm native package manager cho WebUI), thư viện ggml, `ffmpeg` (đọc voice reference không phải WAV) và `curl` (healthcheck). Lệnh mặc định (cờ UI do `AUDIOCPP_UI` quyết định, xem [WebUI](#webui)):

```
audiocpp_server --config /srv/configs/server.json --host 0.0.0.0 --port 8080 --no-ui
```

`configs/`, `models/`, `voices/` được mount read-only vào `/srv/` theo đúng bố cục repo, nên `configs/server.json` (đường dẫn tương đối theo vị trí file: `../models`, `../voices`) dùng chung cho cả chạy trên host lẫn trong container; `host`/`port` trong file bị `--host`/`--port` ghi đè. Thiếu `models/` hay `voices/` thì `docker compose up` báo lỗi ngay thay vì mount thư mục rỗng.

Biến trong `.env` cho audio.cpp:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AUDIOCPP_UI` | `off` | Chế độ WebUI: `off`, `on`, `manage` ([WebUI](#webui)) |
| `HF_TOKEN` | rỗng | Chỉ cho repo Hugging Face gated/private tải từ WebUI (`manage`) |
| `AUDIOCPP_GPU` | `0` | Chỉ số GPU trên host; trong container là device 0, khớp `"device": 0` |
| `AUDIOCPP_BIND`, `AUDIOCPP_PORT` | `127.0.0.1`, `8080` | Địa chỉ publish trên host |
| `AUDIOCPP_CUDA_ARCH` | `80` | Arch build vào image, ví dụ `"80;86;89;90"` cho nhiều loại GPU (build lâu hơn) |
| `AUDIOCPP_CUDA_VERSION` | `12.6.3` | Tag `nvidia/cuda`. Từ 12.8 trở lên GCC 14 cũng được (build arg `GCC_VERSION`) |
| `AUDIOCPP_IMAGE` | `omnivoice-audiocpp:local` | Tên image; đặt tên registry để push/pull giữa các máy |
| `AUDIOCPP_VERSION` | `65df584a` | Chỉ để hiển thị trong `--version` |

### WebUI

audio.cpp có WebUI nhúng sẵn, phục vụ ngay trên cổng của audio.cpp (`AUDIOCPP_PORT`, mặc định 8080), không qua gateway. Bật bằng `AUDIOCPP_UI` trong `.env` rồi `docker compose up -d audiocpp` (không cần build lại):

| `AUDIOCPP_UI` | Cờ server | Cho phép |
| --- | --- | --- |
| `off` (mặc định) | `--no-ui` | Chỉ API |
| `on` | `--ui` | Thử các model trong `configs/server.json`, giọng trong `voices/`, voice clone/design, streaming. Không tải, xoá hay đổi model |
| `manage` | `--ui --ui-management` | Thêm: tải package từ Hugging Face/ModelScope, load/unload model bất kỳ, xoá package, upload file tạm, duyệt thư mục trên server |

Giá trị khác ba giá trị trên thì container thoát với lỗi, không lặng lẽ bật UI.

Mở UI: `AUDIOCPP_BIND` để mặc định `127.0.0.1` và dùng SSH tunnel từ máy mình:

```bash
ssh -L 8080:127.0.0.1:8080 <máy-gpu>   # rồi mở http://127.0.0.1:8080
```

Lưu ý:

- **UI gọi thẳng audio.cpp, không qua text pipeline của gateway** (không chuẩn hoá số/ngày/viết tắt, không có từ điển phát âm, không có `short_text_speed`). Nghe trong UI không giống caller nghe qua gateway. Muốn biết văn bản gateway sẽ gửi lên, gọi `POST /v1/text/normalize` rồi dán kết quả vào UI. So sánh A/B có hệ thống: [LISTENING.md](LISTENING.md).
- `manage` không có auth mà cho phép duyệt thư mục và xoá file trên server: chỉ dùng qua loopback/SSH tunnel, không bao giờ kết hợp với `AUDIOCPP_BIND=0.0.0.0`.
- Danh mục package trong UI liệt kê mọi family của audio.cpp, nhưng image chỉ build OmniVoice (`AUDIOCPP_MODELS`): package family khác vẫn tải được (thường vài GB) mà không load được. Cần family khác thì build lại với `AUDIOCPP_MODELS` trong `args` (danh sách cách nhau bởi `;`, build lâu hơn).
- Package tải từ UI nằm trong volume `audiocpp-ui-models` (`/app/models`), tách khỏi `./models` (mount read-only). Muốn gateway dùng một model đã tải thì vẫn khai báo entry trong `configs/server.json` như mọi model khác ([USAGE.md, mục 4.1.1](USAGE.md#411-thêm-lại-các-model-đã-bỏ)), trỏ tới file trong `./models`.
- Model load từ UI chiếm VRAM và tính vào `max_loaded_models` (2), có thể đẩy `g-omnivoice` ra khỏi bộ nhớ: request kế tiếp qua gateway phải load lại. Trên instance phục vụ caller, dùng `on` thay vì `manage`, hoặc chạy `manage` lúc vắng/ở một instance riêng.
- Image có thể build không kèm package manager (`AUDIOCPP_NATIVE_MODEL_MANAGER=OFF` trong `args` của `compose.audiocpp.yaml`) để bớt BoringSSL; khi đó `manage` báo lỗi lúc khởi động, `on` vẫn chạy.

### Vận hành

- Sửa `configs/server.json`, `configs/gateway.toml` hoặc từ điển rồi `docker compose restart audiocpp` / `docker compose restart gateway`; không cần build lại.
- Thêm model hoặc giọng: đặt file vào `models/`, `voices/`, thêm entry vào `configs/server.json` ([USAGE.md, mục 4.1.1](USAGE.md#411-thêm-lại-các-model-đã-bỏ)), rồi restart audiocpp.
- Đổi commit pin audio.cpp: sửa mặc định `AUDIOCPP_VERSION` trong `compose.audiocpp.yaml`, `docker compose build audiocpp`, chạy lại test của gateway và một lượt nghe ngắn.
- Log: `docker compose logs -f audiocpp gateway`. Log gateway không chứa văn bản hay audio.
- Khi host khởi động lại, Docker tự chạy lại cả hai (`restart: unless-stopped`) nhưng không theo thứ tự `depends_on`; trong lúc audio.cpp chưa lên, `/ready` của gateway trả 503.
