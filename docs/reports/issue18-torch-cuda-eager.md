# Issue #18: runtime `torch-cuda` eager fp32 so với `onnx-cpu`

Môi trường: một A100-PCIE-40GB trống (`CUDA_VISIBLE_DEVICES=2`), driver 535.230.02, torch 2.14.1+cu126, onnxruntime 1.31.1 (CPU, 8 thread), numpy 2.5.4, Python 3.12.14. Weights: `models/ZeroTTS-torch` (#17) cho torch, `models/ZeroTTS` fp32 revision `c2bfbd67` cho ONNX.

## Cách đo ban đầu

`tests/test_torch_cuda.py::test_golden_set_matches_onnx` chạy trên golden set (194 ca không lỗi của `tests/golden/pipeline.json`, lấy `expected` làm engine text và chia segment như server). Ca thứ `i` dùng giọng `maichi` hoặc `quangminh` (xen kẽ), seed `i`, sampling mặc định, và một RNG cho mọi segment như engine.

- **Lockstep:** ONNX chạy từng frame. Torch cùng frame nhận cùng `u`, được teacher-force bằng codes của ONNX (giống `zerotts_frame_forced` của bản ggml), nên hai bên luôn cùng lịch sử. Mỗi frame so global hidden và logits mà bộ lấy mẫu nhìn thấy (sau repetition penalty và temperature; logits ONNX lấy từ input của các node `TopK`). Codes phải trùng; chỉ chấp nhận lựa chọn khác khi đó là near-tie (hai logits cách nhau ≤ 2e-4, hoặc `u`/top-p nằm sát một bậc CDF).
- **Free-running:** `TorchZeroTTS.frames()` (đúng hàm engine gọi) phải cho codes trùng ONNX, nếu có near-tie thì trùng cho tới near-tie đầu tiên.
- Ngưỡng: |torch − onnx| ≤ 1e-4 + 1e-5·max|onnx| trên từng tensor. Logits lên tới khoảng 200; ở mức đó một ulp fp32 là ~1,5e-5.

## Kết quả

| | Giá trị |
| --- | --- |
| Ca / frame | 194 / 4.382 |
| Codes | trùng 100% ở cả lockstep và free-running; 0 near-tie |
| Sai khác tuyệt đối lớn nhất, global hidden | 1,26e-4 |
| Sai khác tuyệt đối lớn nhất, logits control | 1,83e-4 |
| Sai khác tuyệt đối lớn nhất, logits audio | 1,05e-4 |
| Phần ngưỡng đã dùng | 0,31 |

Sai khác tuyệt đối vượt 1e-4 một chút ở các phần tử lớn (logits ~145), tương đương vài ulp fp32. So theo độ lớn của tensor thì cỡ 1e-6.

Near-tie có thật, dù lần chạy này không gặp. Khi thử với seed khác (`100 + i` trên 1/10 golden set), một frame có hai token mà ONNX cho logits 6,6175122 và 6,6175175, còn torch cho đúng bằng nhau. Sort ổn định nên torch chọn token kia, dù sai khác logits của frame đó chỉ 8,4e-6. Sau điểm đó hai runtime sinh ra hai bản đọc khác nhau. Nghĩa là ở fp32, "codes trùng" chỉ đúng tới near-tie đầu tiên. Đây là lý do test so codes theo kiểu lockstep có teacher-force. Seed vẫn tái lập trong cùng runtime và cùng weights.

Smoke HTTP (`-m cuda`): `zerotts` trả WAV 48 kHz mono 16-bit; cùng seed cho cùng byte, khác seed cho audio khác; `zerotts-stream` có delta, `done`, `[DONE]`; `device = "cuda:99"` cho `model_load_failed`, không lùi về CPU. `onnx-cpu` sau khi tách seam vẫn trùng SHA-256 của #16 (fp32 và int8).

Tốc độ (chỉ để tham khảo, slice này chưa tối ưu): eager fp32 khoảng 40 ms/frame, xấp xỉ ONNX CPU 8 thread (câu 33 frame: torch 1,35 s, ONNX 1,74 s, chưa tính codec). CUDA Graph và fp16 thuộc #21.

## Tái lập

```bash
uv sync --extra server-cuda --group torch-weights
CUDA_VISIBLE_DEVICES=2 uv run python -m pytest -m cuda tests/test_torch_cuda.py -s
uv sync
```

Dòng `golden parity: ...` in ra số ca, số frame, số ca lệch sau near-tie và các sai khác lớn nhất. Mất khoảng 11 phút.

## Sửa sau review #18

- Torch ép top-k dương sang số nguyên trước khi giới hạn, giống feed int64 của ONNX; giữ nguyên hành vi cấu hình hiện có.
- Ngoại lệ near-tie nay phải tái lập đúng cả hai lựa chọn từ logits thực tế của hai runtime, với sai khác trong ngưỡng đã nêu. Không chấp nhận chỉ vì có một ranh giới top-p/CDF gần đó nhưng không giải thích được hai lựa chọn.
- Poll `/health` dùng một response mỗi vòng, tránh lỗi `KeyError` khi model vừa nạp xong. Test CPU tái hiện lỗi với chuỗi 503 → 200 và pass sau sửa.
- Test config/env hợp lệ nay gọi HTTP với engine giả, kiểm tra health, speech, models và log weights. Cấu hình sai bị từ chối trước khi tạo app được phép kiểm tra trực tiếp.
- Thêm test CUDA cho sampling khác mặc định (top-k 1,5; 0,5; 64 và top-p/temperature/repetition penalty khác mặc định), cùng test CPU cho bộ kiểm chứng near-tie.

Kiểm chứng sau sửa:
- `uv run pytest`: **511 passed, 1 skipped, 13 deselected**.
- `uv run pytest -m real_model tests/test_server_smoke.py -rs`: **4 passed, 1 skipped**; ca SHA-256 cần `ZEROTTS_VENDOR_BASELINE` nên chưa xác nhận lại byte-equivalence.
- `uv run ruff check .`, `uv run ruff format --check .`, `git diff --check`: pass.
- Log lần chạy: `/tmp/zerotts-issue18-fixes-pytest.log` và `/tmp/zerotts-issue18-fixes-real-model.log` (tệp tạm trên máy kiểm thử).

Các số đo GPU ở trên thuộc lần chạy ban đầu, không phải kết quả chạy lại sau sửa. Lần sửa này chưa cài extra `server-cuda`, nên cả 8 ca CUDA đều skip; phải chạy lại lệnh GPU ở trên trước khi xác nhận parity sau sửa.
