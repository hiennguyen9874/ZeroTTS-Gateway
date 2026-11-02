---
status: proposed
---

# Thêm runtime PyTorch CUDA cho `zerotts_server`, song song với ONNX CPU

`zerotts_server` có thêm runtime `torch-cuda` bên cạnh `onnx-cpu`. Operator chọn runtime trong config; API và hành vi với caller giữ nguyên, và `onnx-cpu` vẫn là mặc định. Runtime mới viết lại model ZeroTTS bằng PyTorch dựa theo `references/ZeroTTS-repo/cpp/src/zerotts.cpp`. Mục tiêu đầu tiên là độ trễ của từng request (TTFA, RTF ở batch 1) trên GPU. Đổi provider của ONNX sang CUDA không đạt mục tiêu này: model sinh tuần tự từng frame, mỗi frame KV cache phải chép qua host, shape của cache tăng dần, và thời gian chủ yếu mất vào việc khởi chạy kernel (ADR 0005 đo được RTF 1,08–1,31 trên A100). Chỉ khi viết lại bằng PyTorch mới có KV cache cấp phát sẵn với shape cố định, nhờ đó ghi được cả một frame thành một CUDA Graph.

Trạng thái *proposed*: chuyển sang *accepted* khi slice tối ưu (CUDA Graph, fp16) đạt cả ba tiêu chí dưới đây. Nếu không đạt thì dừng, và ghi lại số đo làm lý do.

## Considered Options

- **ORT CUDA EP kèm IOBinding.** Ít code nhất, nhưng cache vẫn tăng dần theo frame, shape thay đổi nên không ghi được CUDA Graph, và vẫn bị nghẽn ở việc khởi chạy kernel. Chỉ dùng làm mốc so sánh.
- **TensorRT.** Các phép lấy mẫu (TopK, CumSum, số ngẫu nhiên) và vòng lặp 17 bước của local decoder khó đưa vào TRT; cache vẫn thay đổi shape; engine phải build riêng cho từng GPU. Bị loại.
- **Thay hẳn runtime ONNX CPU.** Sẽ mất kiểu triển khai chỉ cần CPU, và khó lùi lại nếu runtime mới có vấn đề. Bị loại.
- **Gộp nhiều request thành một batch.** Tăng throughput, nhưng phá mô hình mỗi worker một request (spec #1). Hoãn lại; throughput tạm đạt bằng N worker trên cùng một GPU.

## Consequences

- **Weights:** `models/ZeroTTS-torch/` được dựng bằng script chạy trên host, gồm `zerotts.safetensors` (từ `zeroweight-ai/ZeroTTS-GGUF` `zerotts-f32.gguf`, revision `92ca8651`) và `codec.safetensors` (chỉ decoder và quantizer của `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`, revision `6aa02b01`; 178 initializer của codec ONNX khớp safetensors này, 17 trong số đó là weight-norm đã gộp). Thêm tokenizer, voices và `manifest.json` ghi revision nguồn cùng SHA-256 từng file. ONNX vẫn là chuẩn đối chiếu.
- **Codec:** viết bằng PyTorch, code chép từ OpenMOSS theo quy tắc của ADR 0006 (Apache-2.0). Ban đầu còn cần `transformers`; sẽ bỏ nếu chi phí thấp.
- **Số ngẫu nhiên:** vẫn sinh trên host bằng numpy RNG theo từng request như ONNX, rồi chép vào buffer cố định trước mỗi lần chạy lại CUDA Graph.
- **Seed:** chỉ tái lập được trong cùng một runtime và cùng weights; cùng seed, `torch-cuda` và `onnx-cpu` cho audio khác nhau.
- **Phạm vi bản đầu:** chỉ hỗ trợ `cfg_scale = 1` (giá trị khác bị từ chối khi khởi động); chỉ giọng preset, voice encoder nằm ngoài phạm vi.
- **Worker:** mỗi worker giữ bản weights, KV cache, CUDA stream và CUDA Graph riêng trên một `device`; `busy_timeout` và 503 giữ nguyên. Nhiều GPU thì chạy nhiều tiến trình.
- **Config:** `runtime` (`onnx-cpu` mặc định | `torch-cuda`) và `device` đặt ở cấp server, đổi được qua `ZEROTTS_SERVER_RUNTIME` và `ZEROTTS_SERVER_DEVICE`. `threads` chỉ áp dụng cho `onnx-cpu`. Vẫn dùng chung `model_dir`; khi khởi động, server kiểm tra thư mục weights có đúng loại cho runtime không, sai thì báo `model_load_failed`.
- **Không lùi về CPU:** thiếu GPU, thiếu torch hoặc không đủ VRAM khi load thì báo `model_load_failed` (`/health` trả 503). Lỗi CUDA giữa request (kể cả hết bộ nhớ) xử lý như lỗi engine hiện nay. Âm thầm lùi về CPU sẽ khiến tốc độ chậm khoảng 10 lần mà operator không hay biết, và cùng seed cho kết quả khác.
- **Ẩn với caller:** `/v1/models` giữ nguyên. Bản ghi log `weights` lúc khởi động có thêm `runtime`, `device`, danh tính manifest và phiên bản torch.
- **Đóng gói:** extra `server-cuda` và image CUDA riêng kèm compose GPU. Image CPU không chứa torch.
- **Tiêu chí:** (1) ở fp32, khớp ONNX với sai số khoảng 1e-4, và với cùng số ngẫu nhiên `u` thì chuỗi codes trùng trên golden set; (2) ở fp16, WER theo ASR không cao hơn CPU fp32 quá 0,01, và có người nghe xác nhận; (3) trên A100 ở batch 1, RTF ≤ 0,1 và TTFA ≤ 50 ms.
