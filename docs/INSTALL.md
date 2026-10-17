# Cài đặt audio.cpp (OmniVoice, CUDA 12.6)

Hướng dẫn build `audio.cpp` có CUDA để chạy OmniVoice trên GPU A100 (compute capability 8.0).
Tham khảo thêm: `references/audio.cpp/docs/build/linux.md`, `references/audio.cpp/docs/models/omnivoice.md`.

## 1. Yêu cầu

| Thành phần | Phiên bản đã kiểm tra |
|---|---|
| OS | Linux x86_64 |
| GCC | 13.3 (cần >= 13) |
| CMake | 3.28.3 |
| Ninja | 1.11.1 (không bắt buộc, nhưng build nhanh hơn) |
| CUDA Toolkit | 12.6 (`/usr/local/cuda-12.6`, nvcc V12.6.85). Bản 12.x hoặc 13.x nào cũng được |
| NVIDIA driver | 535.230.02, chạy được nhờ `libcuda` bản compat 12.8 trong `/usr/local/cuda/compat` |
| Git LFS | để tải weights |

Kiểm tra nhanh:

```bash
gcc --version | head -1
cmake --version | head -1
ninja --version
/usr/local/cuda-12.6/bin/nvcc --version | tail -2
nvidia-smi
```

## 2. Lấy source

`references/audio.cpp` là git submodule, đã pin ở commit `65df584a`:

```bash
git submodule update --init references/audio.cpp
# Chỉ cần khi dùng WebUI (submodule lồng, URL SSH):
git -C references/audio.cpp submodule update --init --recursive
```

Đổi bản audio.cpp: checkout commit mới trong `references/audio.cpp`, chạy lại test của gateway, rồi commit thay đổi submodule. Nếu chạy audio.cpp bằng Docker, sửa luôn `AUDIOCPP_VERSION` trong `compose.audiocpp.yaml` và build lại image ([DEPLOY.md](DEPLOY.md#vận-hành)).

Không muốn build trên host: [DEPLOY.md](DEPLOY.md) build audio.cpp trong Docker với cùng tùy chọn (CUDA 12.6, arch 80, chỉ OmniVoice).

## 3. Build

Phải trỏ rõ tới toolkit 12.6. CMake lấy `nvcc` đầu tiên trên `PATH`, mà đó có thể là một bản CUDA khác.

```bash
cd references/audio.cpp

export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDAToolkit_ROOT=/usr/local/cuda-12.6

./scripts/build_linux.sh \
  --backend cuda \
  --cuda-arch 80 \
  --build-dir build/linux-cuda126-release \
  --build-type Release \
  --deployment-build \
  --model-set custom --models omnivoice \
  --target audiocpp_cli --target audiocpp_server --target audiocpp_gguf \
  2>&1 | tee /tmp/audiocpp_build.log
```

Ý nghĩa các tùy chọn:

- `--cuda-arch 80`: chỉ build cho A100. Nếu bỏ tùy chọn này, script build cả danh sách arch mặc định, chạy được trên nhiều GPU hơn nhưng lâu hơn nhiều. Có thể dùng `native` để build cho GPU đang có trên máy, hoặc liệt kê nhiều arch như `"80;86;89"`.
- `--model-set custom --models omnivoice`: chỉ build family OmniVoice. Nếu cần mọi family, bỏ hai tùy chọn này (mặc định là `full`).
- `--deployment-build`: nhúng package spec vào binary, nên chạy được safetensors mà không cần thư mục `model_specs/`.
- Nếu build lỗi vì CPU/assembler không hỗ trợ lệnh nào đó (ví dụ `vpdpbusd`), thêm `--native-cpu off`.
- Muốn đổi toolkit CUDA trong một build dir có sẵn, phải xóa `CMakeCache.txt` và `CMakeFiles/` trước, hoặc dùng một `--build-dir` mới.

Build xong sẽ có các binary sau trong `build/linux-cuda126-release/bin/`:

| Binary | Công dụng |
|---|---|
| `audiocpp_cli` | chạy inference từ dòng lệnh |
| `audiocpp_server` | HTTP server |
| `audiocpp_gguf` | convert safetensors sang GGUF |

## 4. Kiểm tra binary

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
BIN=build/linux-cuda126-release/bin

$BIN/audiocpp_cli --version        # phải có: backends: cpu,cuda
$BIN/audiocpp_cli --list-devices   # phải thấy NVIDIA A100-PCIE-40GB, compute capability 8.0

# Kiểm tra thư viện CUDA được link đúng bản 12
readelf -d $BIN/audiocpp_server | grep NEEDED | grep cu
ldd $BIN/audiocpp_cli | grep -E 'cudart|cublas|libcuda'
```

Kết quả đúng: `libcudart.so.12` và `libcublas.so.12` trỏ tới `/usr/local/cuda-12.6/lib64`. Nếu thấy `libcudart.so.11`, nghĩa là build đang trộn với một bản CUDA khác. Khi đó xóa build dir rồi build lại theo bước 3.

## 5. Lưu ý

- GPU 4 và 7 đang có tiến trình khác chiếm VRAM. Nên đặt `CUDA_VISIBLE_DEVICES` sang GPU khác.
- Khi chạy binary, luôn export `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64`.
- Driver 535 vẫn chạy được CUDA 12.6 nhờ `libcuda.so.1` bản forward-compat trong `/usr/local/cuda/compat`. Nếu chuyển sang máy khác không có thư mục compat này, nên dùng driver >= 560 (driver 525–555 chỉ chạy được CUDA 12.6 ở chế độ minor-version compatibility).
