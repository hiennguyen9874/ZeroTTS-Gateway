# #22 — A/B chất lượng `torch-cuda` fp16 so với `onnx-cpu` fp32 (golden, ASR, nghe)

Ngày đo: 2026-11-02, máy dev dùng chung, commit `a2f6b92`. Phần agent (render, chấm ASR, liệt kê ca lệch) đã xong. **Phần nghe và quyết định cuối cùng còn chờ operator**, xem [Cần người nghe](#cần-người-nghe).

## Kết luận

- **Về ASR, `torch-cuda` fp16 đạt cổng của ADR 0007 (tiêu chí 2):** WER offline `torch-cuda` là **0,0725**, `onnx-cpu` fp32 cùng lượt là **0,0738**. Chênh lệch cuda − cpu là **−0,0012** (khoảng tin cậy 95% bootstrap theo ca: −0,0045 đến +0,0019), thấp hơn xa ngưỡng +0,01; không lần lấy mẫu lại nào vượt +0,01. CER 0,029 cả hai.
- Không nhóm ca lớn nào lệch quá 0,01 WER; nhóm `số` tốt hơn 0,010. Chỉ nhóm nhỏ, nhiễu lệch hơn (`input-errors` 6 hàng: +0,042, một từ). Theo giọng, cuda − cpu nằm trong −0,004 đến +0,002.
- Đổi runtime làm transcript đổi **không nhiều hơn** việc đổi seed trên cùng runtime: trên 450 hàng seed 7, cpu với cuda khác 71 hàng; cpu seed 42 với cpu seed 7 khác 78 hàng.
- Thời lượng audio không lệch: tỉ lệ cuda/cpu trung vị 1,000 (p5 0,875, p95 1,143), tổng 2.664 s và 2.662 s. Cờ clipping 30 (cpu) và 32 (cuda).
- **Streaming (`zerotts-stream`, lượt phụ, không phải cổng):** WER cpu 0,073, cuda 0,077 (+0,0035, khoảng tin cậy −0,0013 đến +0,0086) trên 428 hàng (`maichi`, `quangminh`, seed 42). Mỗi ca chỉ 2 hàng nên nhiễu hơn lượt offline.
- **Đề xuất (chờ operator nghe):** nếu người nghe không thấy fp16 kém đi, coi tiêu chí 2 của ADR 0007 là đạt; không cần chạy fp32 hay giữ khối nào ở fp32.

## Cách đo

- Upstream: hai `zerotts_server` thật chạy song song, cùng code và `configs/zerotts_server.toml` (sampling mặc định), mỗi cái 3 worker. Chỉ khác `runtime` và `model_dir`:
  - `onnx-cpu`, `models/ZeroTTS` (fp32), 8 luồng mỗi worker, onnxruntime 1.31.1;
  - `torch-cuda`, `models/ZeroTTS-torch`, một A100 (`CUDA_VISIBLE_DEVICES=2`), torch 2.14.1+cu126. Log startup: `dtype: float16`, sha256 `zerotts.safetensors` `0fa115d5…`, `codec.safetensors` `51c7f324…`, khớp `manifest.json` (#17).
- Mỗi upstream có một gateway riêng với `configs/gateway.toml` đang ship (toàn bộ pipeline, lexicon chung), chỉ đổi cổng và `upstream_url`.
- Giống #14: golden và bộ nghe `listening`, `--normalize-on`, model `zerotts` (offline). Giọng `maichi`, `baotrang`, `quangminh`, `giahuy` với seed 42; `maichi`, `quangminh` với seed 7. Mỗi runtime 1.380 lần render. Lượt phụ `zerotts-stream`: `maichi`, `quangminh`, seed 42, 438 lần render mỗi runtime (PCM 48 kHz qua SSE). Không có request lỗi.
- Chấm: gipformer (`scripts/asr_server.py`) rồi `scripts/ab_report.py`, giống #9 và #14. Bỏ 30 hàng mỗi bên (ca không phải tiếng Việt), còn 1.350 hàng mỗi runtime (streaming: 428).
- **Kiểm tra mốc:** cả 1.350 WAV `onnx-cpu` hôm nay trùng từng byte với lượt fp32 của #14 (cùng engine text, seed, weights), nên WER cpu 0,0738 đúng bằng #14. Việc chép code (#16) và seam runtime (#18) không làm đổi audio `onnx-cpu`.
- Cùng seed nhưng khác runtime thì audio khác hẳn (fp16 chọn code khác từ những frame đầu, xem `docs/agents/zerotts-server.md`), nên so sánh phải đặt cạnh mức nhiễu giữa hai seed.

## Kết quả theo nhóm ca (offline)

1.350 hàng mỗi runtime. Nhóm giống #9 và #14.

| Nhóm | Hàng | WER cpu | CER cpu | WER cuda | CER cuda |
| --- | ---: | ---: | ---: | ---: | ---: |
| ngày giờ | 60 | 0,000 | 0,000 | 0,002 | 0,000 |
| số | 156 | 0,051 | 0,018 | 0,041 | 0,017 |
| viết tắt | 96 | 0,110 | 0,042 | 0,114 | 0,042 |
| tiếng Anh | 102 | 0,276 | 0,107 | 0,284 | 0,116 |
| markup | 294 | 0,051 | 0,021 | 0,047 | 0,019 |
| reading-rules | 294 | 0,085 | 0,039 | 0,085 | 0,037 |
| tags | 198 | 0,045 | 0,015 | 0,044 | 0,011 |
| hard | 84 | 0,180 | 0,089 | 0,177 | 0,085 |
| long-text | 6 | 0,011 | 0,009 | 0,011 | 0,008 |
| language (phần tiếng Việt) | 18 | 0,364 | 0,021 | 0,364 | 0,036 |
| input-errors (ca không lỗi) | 6 | 0,375 | 0,118 | 0,417 | 0,147 |
| pipeline | 18 | 0,000 | 0,000 | 0,005 | 0,004 |
| short-text, tag-case | 18 | 0,000 | 0,000 | 0,000 | 0,000 |
| **Tất cả** | 1.350 | **0,074** | 0,029 | **0,073** | 0,029 |

Theo giọng và seed (WER cpu / cuda):

| | Hàng | cpu | cuda |
| --- | ---: | ---: | ---: |
| `maichi` | 450 | 0,078 | 0,074 |
| `baotrang` | 225 | 0,079 | 0,076 |
| `quangminh` | 450 | 0,071 | 0,072 |
| `giahuy` | 225 | 0,066 | 0,067 |
| seed 42 | 900 | 0,072 | 0,073 |
| seed 7 | 450 | 0,078 | 0,072 |

Streaming theo nhóm: `pipeline` 0,000 → 0,038 (6 hàng, một hàng thiếu hai từ), `tags` 0,070 → 0,085 (44 hàng), `reading-rules` 0,081 → 0,087; các nhóm khác lệch ≤ 0,01. Đầy đủ: `outputs/ab22/zerotts-stream/report/report.md`.

## Mức nhiễu

Trên 450 hàng `maichi` + `quangminh` seed 7:

| So sánh | Transcript khác | Bên sau nhiều lỗi hơn | Bên sau ít lỗi hơn |
| --- | ---: | ---: | ---: |
| cpu với cuda (cùng seed 7) | 71 | 21 | 31 |
| cpu seed 42 với cpu seed 7 | 78 | 35 | 28 |
| cuda seed 42 với cuda seed 7 | 71 | 26 | 32 |

Trên toàn bộ 1.350 cặp: 197 transcript khác, cuda nhiều lỗi hơn ở 73 hàng và ít lỗi hơn ở 76 hàng (streaming: 64 khác, 27 / 22 trên 428).

## Ca cuda đọc khác cpu

48 ca có ít nhất một hàng mà cuda nhiều lỗi từ hơn cpu (73 hàng: `maichi` 22, `quangminh` 28, `baotrang` 11, `giahuy` 12); 29 ca kém hơn khi cộng qua mọi hàng. Tám ca đầu, transcript gipformer:

| Ca | Input text | cpu nghe được | cuda nghe được | Nhận xét |
| --- | --- | --- | --- | --- |
| `vi-hotline-joined` | Gọi 19001234 nhé. | "gọi một chín không không…" | "rồi", "một viễn chính", mất một "không", "nhá" (4/6 hàng) | **Nghe trước**: mỗi hàng lệch một chỗ khác nhau |
| `vi-otp` | Mã OTP là 123456. | "ô tô" | "OTF" (4/6 hàng, cả streaming) | Engine text "ô tê phê"; cả hai đều không khớp, cuda đọc đều một kiểu khác. **Nghe trước** |
| `table-align` | (bảng giá) | "tái chín" | "tái chính", "tài chính", "tá chín" (3/6) | **Nghe trước**: có thể sai thanh điệu đều |
| `vi-python-version` | Python 3.12 rất hay. | "python" | "fiton", "file thân", "file" | Tên tiếng Anh (int8 ở #14 cũng vậy) |
| `abbr-ai-lower` | tôi dùng ai và api mỗi ngày. | "AI" | "I" (2/6, seed 7) | Chữ cái tiếng Anh |
| `vi-acronyms-comma` | Tôi dùng AI, API và USD. | "USD" | "US", "BUS", "tiêu dùng" (3/6, 1 hàng tốt hơn) | Một âm tiết |
| `vi-dollar-sign` | Vé giá $5. | "vé giá" | "cái giá", "về giá … kia" (`quangminh`) | Một âm tiết đầu câu |
| `vi-version-parts-cardinal` | Cập nhật lên v10.4.1 nhé. | "nhé" | "nhá" | Nhiễu sampling một âm tiết |

Đầy đủ: `outputs/ab22/zerotts/cuda_worse_cases.csv` (theo ca) và `outputs/ab22/zerotts/cuda_worse_rows.csv` (73 hàng, đường dẫn WAV cpu/cuda, cột `review` để điền); streaming: `outputs/ab22/zerotts-stream/cuda_worse_rows.csv` (27 hàng).

## Cần người nghe

Đây là slice `ready-for-human`; agent không nghe được. Trước khi đóng #22:

1. Mở trang so sánh: `python3 -m http.server -d outputs/ab22 9000`, rồi vào `http://localhost:9000/zerotts/listen.html` (offline) và `http://localhost:9000/zerotts-stream/listen.html` (streaming). Mỗi dòng gồm engine text, audio cpu và cuda cùng giọng, cùng seed, kèm transcript. Cuối mỗi trang có 8 hàng ngẫu nhiên không lệch theo ASR cho mỗi giọng.
2. Nghe **từng hàng** trong `cuda_worse_rows.csv`, trước hết ba ca in đậm ở trên, rồi điền cột `review` (`cuda kém hơn`, `như nhau`, `ASR sai`).
3. Nghe phần mẫu ngẫu nhiên để so chất giọng, nhịp, độ sạch, tiếng rè hay "pop" (ASR không bắt được những khác biệt này; fp16 có thể đổi âm sắc dù đọc đúng chữ).
4. Ghi kết quả vào mục dưới và vào #22. Nếu fp16 kém đi: ghi vào ADR 0007 và chọn hướng xử lý (phục vụ fp32 qua `TorchZeroTTS(dtype="float32")`, chậm hơn khoảng 9%: 7,1 so với 6,5 ms/frame theo #21; hoặc giữ một số khối ở fp32).

### Kết quả nghe

_Chưa nghe._

| Lượt | Hàng đã nghe | cuda kém hơn | Ghi chú |
| --- | --- | --- | --- |
| offline (73 hàng lệch + mẫu) | | | |
| streaming (27 hàng lệch + mẫu) | | | |

**`uyết định:** _chờ operator._

## Giới hạn

- Tham chiếu là cách đọc của gateway; ASR viết chữ cái và tên tiếng Anh khác engine text, nên ở nhóm viết tắt/tiếng Anh chỉ nên đọc chênh lệch cpu/cuda, không đọc giá trị tuyệt đối.
- Một model ASR, 4 giọng, 2 seed (streaming: 2 giọng, 1 seed). Mỗi ca chỉ có 6 hàng (streaming 2), nên ca lệch một hai hàng không đủ để kết luận, cần nghe.
- ASR không đo chất giọng, nhịp, tiếng ồn hay độ tự nhiên.
- Một GPU (A100), một bộ weights `models/ZeroTTS-torch`; GPU hoặc phiên bản torch/cuDNN khác có thể cho audio khác.
- Thời gian trong lượt này (RTF, TTFA) đo khi máy đang tải nặng (3 lượt render song song mỗi server, CPU load 20–35), không dùng cho tốc độ (#21).

## Tái lập

Dữ liệu cục bộ, gitignored: `outputs/ab22/` (`conf/`, `logs/`, `services.sh`, `run.sh`, `analyze.py`; offline trong `zerotts/`, streaming trong `zerotts-stream/`).

```bash
uv sync --extra server-cuda --group asr --group dev
GPU=2 outputs/ab22/services.sh   # onnx-cpu :18380 → gateway :18300, torch-cuda :18381 → :18301, ASR :18390
outputs/ab22/run.sh              # render_all.py --normalize-on, 6 lượt song song, ~45 phút
export UV_NO_SYNC=true
O=outputs/ab22/zerotts
uv run python scripts/ab_report.py --variant cpu=$O/cpu-s42 --variant cpu=$O/cpu-s7 \
    --variant cuda=$O/cuda-s42 --variant cuda=$O/cuda-s7 --out $O/report
uv run python outputs/ab22/analyze.py $O                         # nhiễu, CI, ca lệch, listen.html
S=outputs/ab22/zerotts-stream
uv run python scripts/ab_report.py --variant cpu=$S/cpu-s42 --variant cuda=$S/cuda-s42 --out $S/report
uv run python outputs/ab22/analyze.py $S
uv sync                          # về môi trường mặc định
```

Config server/gateway trong `outputs/ab22/conf/` khác bản ship chỉ ở cổng, `runtime`, `model_dir` (đường dẫn tuyệt đối), `workers = 3`, `busy_timeout = 600` và `lexicon_path` tuyệt đối.
