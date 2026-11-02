# Issue #21: tối ưu `torch-cuda` (KV cache cố định, CUDA Graph, fp16) và benchmark

Ngày đo: 2026-11-02. Báo cáo này để operator quyết định tiếp hay dừng (#21 gắn `ready-for-human`). Quyết định chưa được ghi. Chất lượng giọng fp16 (tiêu chí 2) thuộc #22 và chưa được đánh giá ở đây.

## Kết luận

| Tiêu chí spec #15 / ADR 0007 | Kết quả (A100, batch 1, server chạy fp16 + CUDA Graph) | Đạt? |
| --- | --- | --- |
| (1) fp32 khớp ONNX ~1e-4, cùng `u` thì codes trùng trên golden set | Có CUDA Graph: 194 ca, 4.382 frame trùng 100%, 0 near-tie, dùng 16% ngưỡng | Đạt |
| (3) RTF ≤ 0,1 | Offline 0,085–0,088; streaming 0,088–0,093 | Đạt |
| (3) TTFA ≤ 50 ms | 18 ms ở mức engine; qua HTTP thật: server `ttft_ms` 19 ms, SSE delta đầu tới client 21 ms | Đạt |
| (2) WER fp16 ≤ CPU fp32 + 0,01, có người nghe | Chưa đo (#22). fp16 bám fp32 trong 0,22% khi teacher-force | Chưa biết |

- So với `onnx-cpu` fp32 trên cùng máy, `torch-cuda` nhanh khoảng **9,4–9,8 lần** theo RTF (0,82–0,83 xuống 0,085–0,088) và TTFA giảm từ 138–158 ms xuống **18 ms**. So với int8: khoảng 5,7–6,3 lần.
- So với bản eager của #18, mỗi frame giảm từ 43 ms xuống **6,5 ms**.
- fp32 có CUDA Graph chỉ chậm hơn fp16 khoảng 8% (7,08 so với 6,51 ms/frame), vì mỗi frame bị giới hạn bởi độ trễ của ~1.800 kernel nhỏ chứ không phải băng thông. fp32 cũng đạt RTF offline 0,092–0,097 và TTFA 25 ms, chỉ streaming câu ngắn chạm 0,102. Nếu #22 cho thấy fp16 làm giảm chất lượng, fp32 là phương án dự phòng với cùng tốc độ gần như vậy.
- N worker trên một GPU: 2 worker cho 16–17,5 giây audio/giây (1,4–1,5 lần một worker), 4 worker cho 26–28 (2,3–2,4 lần). Đổi lại, RTF mỗi request tăng lên 0,14–0,15 và TTFA lên khoảng 30 ms (tối đa 38 ms). VRAM khoảng 1,85–2,0 GiB mỗi worker sau khi load.

## Đã thay đổi

`src/zerotts_server/torch_runtime.py`:

- **Shape cố định.** Text được pad tới 514 token và che bằng key mask. KV cache self-attention của decoder toàn cục cấp phát sẵn cho khối voice, `<soa>` và 1500 frame (cộng một vị trí dự phòng), ghi theo vị trí và che theo vị trí hiện tại. Chỉ số frame nằm trên GPU, và cross K/V của text nằm trong buffer cố định.
- **Hai CUDA Graph mỗi worker:**
  - *bắt đầu segment*: text encoder, cross K/V, khối voice và `<soa>`;
  - *một frame*: 17 bước local, lấy mẫu, cập nhật mask lặp lại, rồi bước decoder toàn cục đưa codes vừa lấy mẫu vào.

  Mỗi frame chép vào 17 số `u` cùng cờ cấm `<eoa>`, và chép ra `<eoa>` cùng 16 codes, qua buffer pinned. Số ngẫu nhiên vẫn sinh trên host bằng numpy RNG theo request, đúng thứ tự như trước.
- **fp16** cho weights và activation. Logits được đổi sang fp32 trước khi phạt lặp và lấy mẫu; RMS norm dùng `F.rms_norm`. Đường fp32 vẫn còn (`dtype="float32"`) để kiểm tra khớp số, và `cuda_graph=False` chạy đúng code đó ở chế độ eager.
- **Giảm số kernel** (đo bằng profiler: 2.368 xuống 1.805 kernel mỗi frame):
  - gộp q/k/v (và cross k/v) thành một phép chiếu, gộp scale attention vào q;
  - RoPE chỉ còn 3 kernel;
  - attention một query dùng matmul trực tiếp thay cho SDPA có mask. SDPA có mask chọn kernel mất khoảng 135 µs cho 1 query trên cả cache.
- **Mỗi worker** có weights, buffer, CUDA stream và graph riêng. Codec chạy trên cùng stream đó. Graph được ghi lúc warmup (`capture_error_mode="thread_local"`).
- **Codec streaming dùng CUDA Graph.** Một bản codec thứ hai giữ chế độ `streaming()` suốt vòng đời worker. Trạng thái causal là tensor GPU cập nhật tại chỗ, nên mỗi cỡ chunk 1–16 frame có một graph. Mỗi stream mới reset trạng thái tại chỗ. Một lần gọi codec giảm từ khoảng 19 ms (gần như toàn bộ là thời gian Python khởi chạy kernel) xuống khoảng 7 ms. PCM trùng tuyệt đối với bản eager (sai khác 0,0). Giải mã offline vẫn chạy eager, một lần mỗi segment.
- Bản ghi log `weights` lúc khởi động có thêm `dtype: float16`.

Đã thử `torch.compile` để gộp thêm kernel, nhưng không dùng: mỗi worker mất hơn 10 giây biên dịch, và lần thử gặp device-side assert (chỉ số ngoài phạm vi) trong code Inductor sinh ra.

## Điều kiện đo

- GPU: A100-PCIE-40GB, `CUDA_VISIBLE_DEVICES=2` (trống, 17 MiB lúc bắt đầu), driver 535.230.02, torch 2.14.1+cu126 (CUDA runtime 12.6). Máy dùng chung: các GPU 1, 4, 5 đang bận việc khác.
- CPU: Xeon Silver 4216, 64 logical CPU, onnxruntime 1.31.1 với 8 thread, Python 3.12.14. Load average 1 phút là 9–15 khi đo torch, 10–23 khi đo ONNX.
- Weights: `models/ZeroTTS-torch` (#17) cho torch; `models/ZeroTTS` fp32 `c2bfbd67` và `models/ZeroTTS-int8` cho ONNX.
- Đo ở mức engine bằng `scripts/bench_zerotts.py`, cùng cách đo như #13: `ZeroTTSEngine` và `split_segments` của server, sampling mặc định, giọng `baotrang`, `seed=42`.
  - Mỗi cấu hình chạy trong một tiến trình riêng, 4 lượt cho mỗi câu × chế độ, bỏ lượt đầu. Số trong bảng là median 3 lượt còn lại.
  - Trong mọi cấu hình, mọi lượt của cùng câu × chế độ cho **cùng SHA-256 audio**.
  - TTFA tính từ lúc gọi engine đến chunk audio đầu tiên, chưa gồm HTTP.
- Câu đo giống #13: ngắn 86 ký tự, vừa 163 ký tự, dài 631 ký tự (4 segment). Cùng seed: torch fp16 ra 51 / 100 / 368 frame (4,08 / 8,0 / 29,9 s); torch fp32 và ONNX fp32 cùng ra 49 / 95 / 367 frame (3,92 / 7,6 / 29,8 s).

## RTF và TTFA (một worker)

| Cấu hình | ms/frame | RTF offline (ngắn / vừa / dài) | RTF streaming | TTFA |
| --- | ---: | --- | --- | --- |
| `onnx-cpu` fp32 | – | 0,823 / 0,830 / 0,831 | 0,841 / 0,838 / 0,884 | 138 / 158 / 146 ms |
| `onnx-cpu` int8 | – | 0,500 / 0,532 / 0,533 | 0,557 / 0,542 / 0,530 | 92 / 112 / 101 ms |
| `torch-cuda` fp32 eager | 43,0 | 0,569 / 0,554 / 0,542 | 0,581 / 0,573 / 0,559 | 83–84 ms |
| `torch-cuda` fp16 eager | 44,0 | 0,564 / 0,574 / 0,555 | 0,592 / 0,591 / 0,574 | 84–86 ms |
| `torch-cuda` fp32 + CUDA Graph | 7,08 | 0,097 / 0,094 / 0,092 | 0,102 / 0,098 / 0,095 | 25 ms |
| **`torch-cuda` fp16 + CUDA Graph (server)** | **6,51** | **0,088 / 0,086 / 0,085** | **0,093 / 0,090 / 0,088** | **18 ms** |

Hai dòng eager chạy cùng code shape cố định nhưng không có graph, cả hai đều bị giới hạn bởi Python. Bản eager của #18 (shape tăng dần) khoảng 40 ms/frame.

Phân bổ thời gian (fp16 + graph, câu dài, offline): bắt đầu segment khoảng 0,3 ms mỗi segment (chỉ tính thời gian khởi chạy; phần GPU nằm trong frame đầu), frame 2,42 s cho 368 frame, codec 85 ms cho 4 lần gọi offline. Ở streaming, codec mất 172 ms cho 32 lần gọi có graph (khoảng 5,4 ms mỗi lần).

## Load, warmup, VRAM (một worker)

| Cấu hình | Load + warmup | Trong đó warmup | VRAM worker (`mem_get_info`) | torch reserved |
| --- | ---: | ---: | ---: | ---: |
| fp16 + graph (server) | 7,8 s | 2,1 s | 1.982 MiB | 710 MiB |
| fp32 + graph | 7,6 s | 1,9 s | 2.520 MiB | 1.250 MiB |
| fp16 eager | 7,2 s | 2,0 s | 774 MiB | 658 MiB |
| fp32 eager | 7,2 s | 1,8 s | 1.320 MiB | 1.210 MiB |

- VRAM worker là mức dùng của thiết bị sau khi load, trừ đi mức trước khi load (437 MiB, gồm CUDA context). Sau khi chạy, allocator tăng thêm khoảng 400 MiB (fp16 + graph: thiết bị dùng tổng 2.813 MiB).
- 16 graph của codec streaming chiếm khoảng 1 GiB ngoài allocator của torch (đo riêng: thêm graph codec làm mức dùng tăng 1.104 MiB, trong khi torch reserved chỉ tăng 114 MiB). Graph frame và bắt đầu segment chiếm khoảng 0,2 GiB. Nếu cần tiết kiệm VRAM, có thể chỉ ghi graph cho các cỡ 1, 2, 4, 8, 16 và giải mã eager chunk cuối mỗi segment. Cách này chưa làm và chưa đo.
- "Load" gồm đọc weights, dựng hai bản codec và ghi graph codec. "Warmup" là warmup và ghi graph của model. Worker đầu tiên mất thêm thời gian khởi tạo CUDA/cuBLAS; các worker sau chỉ mất khoảng 2 s.

## N worker trên một GPU (fp16 + graph)

Mỗi worker chạy cùng request trong thread riêng, cùng lúc, 4 lượt liên tiếp. "Audio s/s" là tổng số giây audio chia cho thời gian thực. Một worker tương ứng 1 / RTF, khoảng 11–11,8.

| Worker | VRAM mỗi worker khi load | Thiết bị dùng tổng sau khi chạy (gồm context) | Audio s/s offline (ngắn / vừa / dài) | Audio s/s streaming | RTF mỗi request | TTFA median (max) |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | 1.982 MiB | 2,7 GiB | 11,4 / 11,7 / 11,8 | 10,7 / 11,1 / 11,4 | 0,085–0,093 | 18 ms |
| 2 | 1.982, 1.844 MiB | 5,0 GiB | 16,6 / 17,2 / 17,5 | 16,1 / 16,6 / 17,1 | 0,115–0,124 | 25 ms (25 ms) |
| 4 | 1.982, 1.844, 1.852, 1.850 MiB | 9,4 GiB | 26,8 / 28,0 / 28,3 | 26,2 / 27,0 / 27,7 | 0,141–0,153 | 30 ms (38 ms) |

Mọi worker cho cùng SHA-256 audio với lần chạy một worker. Test HTTP 2 worker cũng xác nhận điều này. GPU còn dư sức, vì mỗi kernel nhỏ chỉ dùng một phần SM, nên các worker chạy chồng lên nhau được. Chưa đo trên 4 worker.

## Kiểm chứng

`CUDA_VISIBLE_DEVICES=2 uv run python -m pytest -m cuda tests/test_torch_cuda.py -s`: **14 passed** trong 12 phút 19 giây.

- **Parity golden set:** model fp32 có CUDA Graph. `frames()`, đúng hàm engine gọi, chạy bằng graph; phần lockstep chạy cùng code shape cố định ở chế độ eager. 194 ca, 4.382 frame, codes trùng, 0 near-tie. Sai khác lớn nhất: hidden 6,0e-5, logits control 2,0e-4, logits audio 5,0e-5 (dùng 0,16 ngưỡng). Ca sampling không mặc định cũng pass.
- **fp16 so với fp32** (test mới): teacher-force bằng codes fp32 trên 10 ca golden, 245 frame. Sai khác lớn nhất so với độ lớn mỗi tensor: hidden 0,22%, logits control 0,13%, logits audio 0,21%; ngưỡng là 1%. Lỗi kiểu sai vị trí RoPE hay sai mask sẽ lệch cỡ cả độ lớn tensor.
- **Codec PyTorch so với ONNX:** offline 1,30e-5 / 112,5 dB. Streaming, nay dùng graph, 1,42e-5 / 111,9 dB, bằng số của #20.
- **Smoke HTTP** với server như khi triển khai (fp16 + graph):
  - WAV 48 kHz mono, cùng seed cho cùng byte;
  - SSE và PCM thô nối lại trùng audio của engine;
  - caller ngắt giữa chừng thì worker được giải phóng và codec được reset;
  - **2 worker** stream đồng thời ra đúng audio của server một worker (test mới);
  - `cuda:99` cho `model_load_failed`.
- **Không có GPU:** sau `uv sync` (môi trường mặc định, không có torch), `uv run pytest`, `uv run ruff check .` và `uv run ruff format --check .` đều pass.

## Giới hạn

- Chỉ đo trên một A100 của máy dev. GPU khác (ví dụ không có bf16, VRAM nhỏ) chưa đo.
- RTF streaming câu ngắn có fp32 + graph là 0,102, sát ngưỡng. Với fp16 thì còn khoảng 7–15% dư.
- Thời gian mỗi frame giờ chủ yếu là độ trễ của ~400 GEMV nhỏ (khoảng 6 µs mỗi cái) và ~180 layer norm một hàng. Muốn giảm thêm cần kernel tự viết (Triton/CUDA) hoặc gộp batch nhiều request (ngoài phạm vi).
- `run.log` trong thư mục số liệu còn một lần chạy đầu bị hỏng. Script bench khi đó ép dtype qua `functools.partial`, nhưng `load_runtime` truyền `dtype` tường minh nên đè lên, và các cấu hình "fp32" thực ra chạy fp16. Script đã được sửa (đổi `runtime.TORCH_DTYPE` và assert dtype/graph sau khi load). Các file JSON của lần đó đã bị xóa và chạy lại (`run2.log`). Các phép đo 2 và 4 worker trong `run.log` là fp16 + graph đúng như nhãn, nên vẫn dùng.

## Tái lập

Dữ liệu cục bộ, gitignored: `outputs/bench-torch-issue21/`. Gồm các file JSON (mọi lượt, thời gian frame/codec, SHA-256 audio, VRAM, phiên bản), `run.sh`, `run2.sh`, log của các lần chạy, và `cuda-tests.log`.

```bash
uv sync --extra server-cuda --group torch-weights
export CUDA_VISIBLE_DEVICES=2
B="uv run python scripts/bench_zerotts.py --runtime torch-cuda --model-dir models/ZeroTTS-torch"
O=outputs/bench-torch-issue21
$B --out $O/torch-fp16-graph.json
$B --dtype float32 --out $O/torch-fp32-graph.json
$B --dtype float32 --eager --out $O/torch-fp32-eager.json
$B --eager --out $O/torch-fp16-eager.json
$B --workers 2 --out $O/torch-fp16-graph-w2.json
$B --workers 4 --out $O/torch-fp16-graph-w4.json
uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS --out $O/onnx-fp32.json
uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS-int8 --out $O/onnx-int8.json
uv run python -m pytest -m cuda tests/test_torch_cuda.py -s
uv sync
```
