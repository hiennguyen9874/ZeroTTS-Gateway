# #13 — Tốc độ fp32 so với int8 của zerotts_server trên máy dev

Ngày đo: 2026-10-28. **Chỉ áp dụng cho máy dev dùng chung này**; quyết định
có dùng int8 trong triển khai nằm ở #10 (benchmark node thật).

## Kết luận

- Trên máy dev, int8 nhanh hơn fp32 khoảng **1,4–1,5 lần** theo RTF ở cả ba độ dài câu,
  offline lẫn streaming (RTF ~0,80–0,83 xuống ~0,50–0,59).
- TTFA streaming giảm **khoảng 15–25%** (133–157 ms xuống 112–119 ms). Vốn đã thấp nên
  người nghe ít thấy khác biệt hơn so với thời gian hoàn tất.
- Weights nhỏ hơn 2,55 lần (904 MB xuống 354 MB), load + warmup nhanh hơn (10,4 s xuống
  6,3 s), RSS sau khi load mỗi worker giảm từ ~1,2 GiB xuống ~0,55 GiB.
- Lợi ích tốc độ đến từ `local_frame_decode` (nhanh ~1,7 lần mỗi frame) và `prefix_step`
  (~1,4 lần). Codec vẫn fp32 nên trở thành phần lớn hơn trong tổng thời gian.
- int8 **đáng làm tiếp**: đo trên node thật (#10) và nghe/ASR chất lượng. Báo cáo này
  không đánh giá chất lượng giọng; cùng seed, int8 cho audio khác fp32 (số frame khác).

## Điều kiện đo

- Máy: Xeon Silver 4216 (AVX512-VNNI), 64 logical CPU, RAM 503 GiB, máy dùng chung,
  không pin core/NUMA. Python 3.12.14, `onnxruntime==1.31.1`, `zerotts==0.1.5`.
- Weights: `models/ZeroTTS` (fp32, HF revision `c2bfbd67dc648cac455077333f7cf5c18a2e3bb4`)
  và `models/ZeroTTS-int8` (tạo bằng `scripts/quantize_zerotts.py`, xem
  [báo cáo #12](issue12-zerotts-int8.md)).
- Đo ở mức engine bằng `scripts/bench_zerotts.py`: dùng `ZeroTTSEngine` và
  `split_segments` của server, `threads=8` như `configs/zerotts_server.toml`, sampling
  mặc định của config, voice `baotrang`, `seed=42`. Không qua HTTP/gateway nên số không
  gồm overhead HTTP (vài ms theo báo cáo 2026-10-28).
- Thời gian theo graph: bọc `session.run()` của từng graph.
- Mỗi weights chạy trong **tiến trình riêng** (RSS là của đúng một worker), 2 vòng xen kẽ
  fp32 → int8 → fp32 → int8. Mỗi vòng 4 lượt cho mỗi câu × chế độ, bỏ lượt đầu; số dưới
  đây là median 6 lượt còn lại của 2 vòng.
- Tái lập: trong mỗi câu × chế độ × weights, cả 8 lượt cho **cùng SHA-256 audio**.
  fp32 câu ngắn ra 49 frame (3,92 s), khớp báo cáo 2026-10-28.
- TTFA: thời gian từ lúc gọi engine đến chunk audio đầu tiên (không phải client HTTP).

### Load máy

Load average 1 phút khoảng 10,5 lúc bắt đầu, sau đó tăng lên và giữ **~20–23** suốt phần
lớn phép đo (workload khác trên máy, cộng 8 thread của benchmark). Vòng fp32 đầu tiên một
phần chạy lúc load thấp hơn (11–20); toàn bộ int8 chạy ở load ~20–22. Nếu có lệch thì lệch
về phía có lợi cho fp32. Khoảng dao động RTF giữa các lượt lên tới ~±8% ở câu ngắn.

## Câu đo

Engine text (đã là văn bản gateway gửi, không còn số hay viết tắt), định nghĩa trong
`scripts/bench_zerotts.py`:

| Câu | Ký tự | Segment | Frame fp32 / int8 | Audio fp32 / int8 |
| --- | ---: | ---: | ---: | ---: |
| Ngắn (câu của báo cáo 2026-10-28) | 86 | 1 | 49 / 54 | 3,92 / 4,32 s |
| Vừa (2 câu tổng đài) | 163 | 1 | 95 / 91 | 7,60 / 7,28 s |
| Dài (đoạn thông báo 7 câu) | 631 | 4 | 367 / 366 | 29,8 / 29,7 s |

Cùng seed nhưng int8 sinh số frame khác fp32, nên so sánh bằng RTF (thời gian xử lý /
thời lượng audio), không so thời gian tuyệt đối.

## RTF offline

| Câu | fp32 RTF | int8 RTF | Tăng tốc | fp32 tổng | int8 tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ngắn | 0,767 (0,71–0,86) | 0,500 (0,48–0,53) | 1,53× | 3,01 s | 2,16 s |
| Vừa | 0,802 (0,78–0,84) | 0,559 (0,54–0,59) | 1,43× | 6,10 s | 4,07 s |
| Dài | 0,830 (0,78–0,84) | 0,563 (0,53–0,57) | 1,47× | 24,74 s | 16,74 s |

Trong ngoặc: min–max của 6 lượt.

## Streaming (`stream=True`, decode ramp 1, 2, 4, 8, 16 frame)

| Câu | fp32 TTFA | int8 TTFA | fp32 RTF | int8 RTF | Tăng tốc RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ngắn | 133 ms (130–156) | 112 ms (99–127) | 0,823 | 0,579 | 1,42× |
| Vừa | 157 ms (146–186) | 119 ms (115–122) | 0,802 | 0,585 | 1,37× |
| Dài | 156 ms (143–158) | 119 ms (106–134) | 0,832 | 0,561 | 1,48× |

TTFA gần như không phụ thuộc độ dài câu: nó gồm text encoder, prefix và 1 frame đầu
cộng một lần decode codec 1 frame. Câu dài chỉ thêm thời gian text encoder.

## Profile theo graph (offline, median 6 lượt)

Tỷ trọng tính trên median tổng; mỗi cột là median riêng nên tổng các dòng có thể lệch
vài phần nghìn.

Câu ngắn, so với báo cáo 2026-10-28 (fp32, 8 thread, cùng câu, cùng 49 frame):

| Graph | Gọi | 2026-10-28 fp32 | fp32 | int8 |
| --- | ---: | ---: | ---: | ---: |
| Text encoder | 1 | 0,037 s (1,2%) | 0,068 s (2,3%) | 0,029 s (1,4%) |
| Prefix step | 49 / 54 | 0,622 s (20,5%) | 0,636 s (21,2%) | 0,510 s (23,6%) |
| Local frame decode | 49 / 54 | 2,079 s (68,4%) | 1,988 s (66,1%) | 1,270 s (58,8%) |
| Codec decode toàn câu | 1 | 0,286 s (9,4%) | 0,300 s (10,0%) | 0,328 s (15,2%) |
| Tổng | | 3,04 s | 3,01 s | 2,16 s |

fp32 hôm nay khớp báo cáo 2026-10-28 (chênh trong khoảng dao động do load). Text encoder
chỉ một lần gọi ~30–70 ms nên dễ nhiễu.

Câu dài (4 segment, 367 / 366 frame):

| Graph | fp32 | int8 |
| --- | ---: | ---: |
| Text encoder (4 lần) | 0,280 s (1,1%) | 0,175 s (1,0%) |
| Prefix step | 5,597 s (22,6%) | 4,113 s (24,6%) |
| Local frame decode | 15,299 s (61,9%) | 8,817 s (52,7%) |
| Codec decode toàn câu | 3,148 s (12,7%) | 3,394 s (20,3%) |
| Tổng | 24,74 s | 16,74 s |

Thời gian mỗi frame (câu dài): local frame decode 41,7 ms → 24,1 ms (1,73×), prefix step
15,3 ms → 11,2 ms (1,36×). Codec giữ fp32 nên không đổi (~0,1 s mỗi giây audio).
Streaming có cùng phân bố, chỉ thay codec decode toàn câu bằng nhiều lần decode từng
đoạn (câu dài: 3,64 s fp32, 3,69 s int8). Với int8, codec là mục tiêu tối ưu tiếp theo
lớn thứ hai sau local frame decode, nhưng quantize codec chưa được đánh giá.

## Weights, load, RAM

| | fp32 | int8 |
| --- | ---: | ---: |
| Thư mục weights (không tính `.cache` HF) | 903 868 995 B | 354 336 999 B |
| text_encoder.onnx | 323 096 725 B | 100 149 923 B |
| prefix_step.onnx | 348 303 894 B | 125 374 501 B |
| local_frame_decode.onnx | 186 743 011 B | 83 086 162 B |
| codec (thư mục, fp32 cả hai) | 45 246 629 B | 45 246 629 B |
| Load sessions + voices | 10,23–10,27 s | 6,13–6,20 s |
| Warmup | 0,18 s | 0,11 s |
| RSS sau load + warmup | 1201–1237 MiB | 566 MiB |
| RSS đỉnh sau cả benchmark | 2958–2962 MiB (~2,9 GiB) | 2537–2538 MiB (~2,5 GiB) |

RSS của một tiến trình chỉ có Python + import trước khi load là ~74 MiB, nên phần model
mỗi worker khoảng 1,10–1,13 GiB (fp32) và ~0,48 GiB (int8). RSS đỉnh đo sau câu dài
(30 s audio) và gồm bộ nhớ trung gian của onnxruntime (KV cache, arena, codec decode toàn
câu), nên phụ thuộc độ dài request; server thật còn thêm FastAPI. Load đọc từ page cache
(file đã được đọc trước đó), không phải đĩa lạnh.

## Giới hạn

- Một máy dùng chung với load ~20, không pin core; một voice, một seed, ba câu; không đo
  đồng thời nhiều worker hay tranh băng thông RAM giữa worker.
- Số đo ở mức engine; qua gateway/HTTP thêm vài ms.
- Không đánh giá chất lượng (nghe, ASR) của int8.
- Node triển khai có CPU, số core, AVX512-VNNI và băng thông RAM khác; **không dùng số
  này thay cho #10**.

## Tái hiện

Dữ liệu cục bộ, gitignored: `outputs/bench-int8-issue13/` (`fp32-r{1,2}.json`,
`int8-r{1,2}.json`: mọi lượt, thời gian từng graph, SHA-256 audio, load average; `run.log`).

```bash
O=outputs/bench-int8-issue13
for r in 1 2; do
  uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS --out $O/fp32-r$r.json
  uv run python scripts/bench_zerotts.py --model-dir models/ZeroTTS-int8 --out $O/int8-r$r.json
done
```

Mỗi lệnh mất khoảng 4–6 phút ở load này. Script không ghi đè file kết quả đã có.
