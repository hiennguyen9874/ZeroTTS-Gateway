# Issue #19: codec PyTorch cho runtime `torch-cuda`

Môi trường: một A100-PCIE-40GB trống (`CUDA_VISIBLE_DEVICES=2`), driver 535.230.02, torch 2.14.1+cu126, transformers 5.22.1, onnxruntime 1.31.1 (CPU, 8 thread), Python 3.12.14. Weights: `models/ZeroTTS-torch` dựng lại bằng `scripts/build_zerotts_torch.py` sau thay đổi này, `models/ZeroTTS` fp32 revision `c2bfbd67` làm mốc ONNX.

## Thay đổi

- Chép nguyên văn `modeling_moss_audio_tokenizer.py`, `configuration_moss_audio_tokenizer.py` và `__init__.py` của `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` revision `6aa02b01` vào `src/zerotts_server/model/moss/`, kèm `LICENSE` (Apache-2.0) và `VENDORED.md` (nguồn gốc, cách so). Các file này không bị format hay sửa (loại khỏi Ruff), nên vẫn trùng từng byte với bản gốc.
- `TorchCodec` (`torch_runtime.py`) dựng model từ `codec_config.json` và `codec.safetensors`, bỏ encoder, nạp với `strict=True`, chạy fp32 trên cùng GPU với frame generator. Không dùng `from_pretrained` hay `trust_remote_code`, không tải gì lúc chạy. Vẫn còn phụ thuộc `transformers` (pin 5.22.1 trong extra `server-cuda`); bỏ phụ thuộc này để lại cho commit sau.
- Tắt TF32 của cuDNN (`torch.backends.cudnn.allow_tf32 = False`). Mặc định cuDNN chạy convolution fp32 bằng TF32, làm PCM lệch khoảng 1,1e-3 (SNR 68 dB) so với ONNX. Khi tắt, sai khác còn khoảng 1e-5.
- Codec có cả `streaming_decoder()` (giữ trạng thái causal trong `streaming()` qua các chunk), vì `ZeroTTS-torch` đã bỏ codec ONNX nên streaming cũng phải chạy trên codec này. Test parity cho streaming thuộc #20. Thử nhanh với codes ngẫu nhiên theo lịch 1, 2, 4, 8, 16 frame thì sai khác lớn nhất so với ONNX streaming là 1,2e-5.
- Script dựng weights không còn chép `onnx/codec/`. Thay vào đó, script tải thêm `config.json` của MOSS ở cùng revision (SHA-256 được pin) và lưu thành `codec_config.json`. Manifest có thêm nguồn `codec_config`. Codec ONNX của `--model` chỉ còn dùng làm mốc để kiểm `codec_onnx_check`.

## Kết quả

`test_codec_matches_onnx_offline`: lấy codes của golden set do torch-cuda sinh ra (194 ca, cùng cách chọn giọng và seed như parity #18), giải mã offline bằng cả hai codec.

| | Giá trị |
| --- | --- |
| Ca / segment frame | 194 / 4.382 |
| Sai khác tuyệt đối lớn nhất | 1,30e-5 |
| SNR nhỏ nhất (theo segment) | 112,5 dB |
| Ngưỡng của test | ≤ 1e-4 và ≥ 80 dB |

Độ dài PCM trùng nhau (3.840 mẫu mỗi frame), mono, float32.

Tham khảo, chưa tối ưu: giải mã 33 frame mất 17,9 ms (torch, GPU) so với 173 ms (ONNX CPU 8 thread). VRAM cấp phát lớn nhất của riêng codec là khoảng 75 MiB.

Bộ test `-m cuda` đầy đủ: **10 passed** trong 15 phút. Parity frame của #18 được chạy lại sau các bản sửa review: 194 ca, 4.382 frame, codes trùng, 0 near-tie, phần ngưỡng đã dùng 0,31. Kết quả smoke HTTP:

- `zerotts` trả WAV 48 kHz mono.
- Cùng seed cho cùng byte.
- `zerotts-stream` có delta, `done` và `[DONE]`.
- `cuda:99` cho `model_load_failed`.

Trong lúc load, `onnxruntime.InferenceSession` bị thay bằng hàm báo lỗi. Model vẫn load được, nghĩa là runtime không mở phiên ONNX nào. `test_weights_have_no_onnx_and_match_manifest` kiểm tra rằng thư mục không còn file `.onnx`/`.data`, và mọi file đều khớp manifest về tên, SHA-256 và kích thước. Dựng lại không làm đổi `codec.safetensors` (SHA-256 vẫn là `51c7f324…`).

## Tái lập

```bash
uv sync --extra server-cuda --group torch-weights
uv run --group torch-weights python scripts/build_zerotts_torch.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4 --force
CUDA_VISIBLE_DEVICES=2 uv run python -m pytest -m cuda tests/test_torch_cuda.py -s
uv sync
```

Lệnh test in ra dòng `codec parity: ...` với số ca, số frame, sai khác lớn nhất và SNR nhỏ nhất.
