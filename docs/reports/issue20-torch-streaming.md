# Issue #20: streaming `zerotts-stream` trên runtime `torch-cuda`

Môi trường: một A100-PCIE-40GB trống (`CUDA_VISIBLE_DEVICES=2`), torch 2.14.1+cu126, onnxruntime CPU 8 thread. Weights: `models/ZeroTTS-torch` (#17/#19) và `models/ZeroTTS` fp32 `c2bfbd67` làm mốc ONNX.

## Thay đổi

Không đổi code server. #18 không chặn `zerotts-stream` dưới `torch-cuda`, và `TorchCodec.streaming_decoder()` (#19) đã giữ trạng thái causal qua các chunk. Lịch giải mã, sự kiện SSE, `timing.ttft_ms` và cách xử lý lỗi nằm ở engine và app, dùng chung cho cả hai runtime. Slice này chỉ thêm test và tài liệu:

- `tests/test_torch_cuda.py` (marker `cuda`):
  - `test_codec_matches_onnx_streaming`: cùng codes, giải mã bằng streaming decoder của codec PyTorch và codec ONNX theo đúng lịch của engine (segment đầu 1, 2, 4, 8, 16 frame, các segment sau 16 frame; 2 frame im lặng `silence_frame.npy` giữa các segment). Mỗi request dùng một decoder. Nối PCM rồi so với ngưỡng như offline (≤ 1e-4, ≥ 80 dB). Golden set không có ca nhiều segment nên thêm một đoạn 3 segment. Giải mã lại đoạn đó sau `close()` phải ra đúng PCM như lần đầu.
  - Smoke HTTP chạy trên uvicorn thật (TestClient đệm cả body nên không đo được TTFA phía client). Engine thật được truyền vào `create_app` để test lấy audio mốc từ chính engine.
  - `test_stream_deltas_join_to_the_engine_audio`: delta SSE nối lại và PCM thô đều trùng từng byte với audio streaming của engine (cùng seed). Có in TTFA.
  - `test_caller_disconnect_frees_the_worker_and_resets_the_codec`: caller ngắt sau chunk đầu. Request tiếp theo cùng seed vẫn được phục vụ và ra đúng audio mốc, tức worker đã được giải phóng và trạng thái codec đã reset.
- `tests/test_server.py`: `test_cuda_error_mid_stream_ends_the_stream_on_torch_cuda` dùng fake engine, settings `torch-cuda` và lỗi `RuntimeError("CUDA error: out of memory")` sau chunk đầu. SSE kết thúc bằng sự kiện `error`, không có `done`/`[DONE]`. PCM thô bị cắt kết nối, không có chunk kết thúc. Worker rảnh lại cho request sau.

## Kết quả

| | Giá trị |
| --- | --- |
| Ca streaming (golden + 3 segment) / số lần gọi codec | 195 / 980 |
| Sai khác tuyệt đối lớn nhất | 1,42e-5 |
| SNR nhỏ nhất (theo ca) | 111,9 dB |
| Ngưỡng | ≤ 1e-4 và ≥ 80 dB |

Kiểm tra độ nhạy: nếu giải mã từng chunk mà không giữ trạng thái (`decode` offline cho mỗi chunk), sai khác lên 0,99 và SNR còn 2,5 dB. Như vậy test bắt được lỗi mất trạng thái causal.

Smoke HTTP với đoạn 3 segment (24,2 s audio, seed 7, giọng `maichi`): `timing.ttft_ms` = 83,3 ms; delta đầu tới client sau 85 ms (SSE), byte PCM đầu sau 105 ms. Đây là số đo của runtime eager fp32, chưa tối ưu. Tiêu chí ≤ 50 ms thuộc #21 nên test chỉ in ra, không assert.

Lần chạy `-m cuda` (bỏ hai test parity frame của #18 vì không đổi): **7 passed** trong 6 phút 30 giây. Sau khi tách phần chờ `/health` thành `wait_until_ready`, đã chạy lại 6 test HTTP và weights: **6 passed**, TTFA lần này là 87,4 ms (server), 91 ms (SSE) và 83 ms (PCM).

## Tái lập

```bash
uv sync --extra server-cuda --group torch-weights
CUDA_VISIBLE_DEVICES=2 uv run python -m pytest -m cuda tests/test_torch_cuda.py -s
uv sync
```

Lệnh in ra các dòng `streaming codec parity: ...` và `stream: ... ttft_ms=...`.
