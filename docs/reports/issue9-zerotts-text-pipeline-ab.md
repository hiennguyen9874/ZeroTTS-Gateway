# #9 — A/B text pipeline cho ZeroTTS

## Kết luận

- **Bật toàn bộ pipeline cho `zerotts` và `zerotts-stream`** (bỏ `normalize = false` trong `configs/gateway.toml`). WER theo ASR: (a) 0,072, (b) 0,167, (c) 0,072. Kết quả này nhất quán ở cả 4 giọng và 2 seed.
- Đọc từ văn bản thô, ZeroTTS đọc sai hẳn ngày, số điện thoại, mã số, số La Mã, đơn vị và viết tắt tiếng Việt. Không phải do tham chiếu thiên vị (xem [ví dụ](#vì-sao-b-thua)).
- **Giữ lexicon chung như hiện tại.** Bỏ lexicon (c) không đổi WER. ASR viết tên sản phẩm bằng chữ tiếng Anh dù model nhận "phây búc" hay "facebook", nên ASR không phân biệt được (a) với (c). **Người nghe cần chốt** sáu ca `loan-*` (nghe (a) với (c), xem [Cần người nghe](#cần-người-nghe)).
- Ca đọc sai còn lại do pipeline: `vi-pin`, `vi-otp` (viết tắt bị đánh vần thành tên chữ cái) được ghi thành known issue trong golden. `vi-version-thousands` vẫn là known issue cũ.

## Cách đo

| | Biến thể | Gateway |
| --- | --- | --- |
| (a) | normalizer + reading rule + lexicon (`--normalize-on`) | `configs/lexicon.csv` |
| (b) | chỉ guard + làm sạch markup (mặc định lúc đo: `normalize = false`) | như (a), lexicon không dùng tới |
| (c) | như (a) nhưng không lexicon | `lexicon_path` trống; cả sáu mục hiện có đều là tên tiếng Anh, nên "bỏ mục tiếng Anh" là bỏ cả lexicon |

- Upstream: `zerotts_server` thật (`zerotts==0.1.5`, onnxruntime 1.31.1, CPU Xeon Silver 4216 64 luồng dùng chung), 6 worker × 8 luồng, model `zerotts` (offline), sampling mặc định của `configs/zerotts_server.toml`. Commit `09c4726` cộng thay đổi của slice này.
- Ca: golden (190 ca không lỗi, gồm cả các ca `zerotts-*`) và bộ nghe `listening` (43 ca, bỏ 4 ca clone). Giọng `maichi`, `baotrang`, `quangminh`, `giahuy` (2 nữ, 2 nam) với seed 42; `maichi`, `quangminh` với seed 7. Tổng 4.194 lần render, không lần nào lỗi.
- Chấm: gipformer (`g-group-ai-lab/gipformer1.5-68M-rnnt` int8, `scripts/asr_server.py`), rồi `scripts/ab_report.py`. Tham chiếu là input text qua toàn bộ pipeline, cả bản có lẫn không có lexicon; mỗi hàng chấm theo tham chiếu gần hơn. Biến thể vùng miền (`ngàn`/`nghìn`, `tư`/`bốn`, `mốt`/`một`, `lẻ`/`linh`) được gộp. Bỏ 90 hàng: ca `language` không phải tiếng Việt và ca lỗi. WER/CER tính trên cả tập. Chi tiết: [agents/listening.md, mục A/B](../agents/listening.md#text-pipeline-ab).
- RTF trong lượt này (~3) chỉ phản ánh máy quá tải khi chạy 6 luồng song song, không dùng để đánh giá tốc độ (#10).

## Kết quả theo nhóm ca

Mỗi biến thể 1.368 hàng (ca × giọng × seed). Nhóm theo #9: `ngày giờ` = `dates-times`; `số` = `numbers-money`, `measurements`, `codes-phones`; `viết tắt` = `abbrev-english` (trừ tên/từ tiếng Anh) và `abbrev-case`; `tiếng Anh` = các ca tên/từ tiếng Anh của `abbrev-english` và `loanwords`; `markup` = `markup`, `short-sentences`. Các nhóm golden khác giữ tên.

| Nhóm | Hàng | WER a | CER a | WER b | CER b | WER c | CER c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ngày giờ | 60 | 0,000 | 0,000 | 0,176 | 0,166 | 0,000 | 0,000 |
| số | 156 | 0,051 | 0,018 | 0,195 | 0,172 | 0,051 | 0,018 |
| viết tắt | 96 | 0,110 | 0,042 | 0,313 | 0,215 | 0,110 | 0,042 |
| tiếng Anh | 102 | 0,276 | 0,107 | 0,282 | 0,113 | 0,278 | 0,110 |
| markup | 294 | 0,051 | 0,021 | 0,087 | 0,053 | 0,051 | 0,021 |
| reading-rules | 294 | 0,085 | 0,039 | 0,240 | 0,184 | 0,086 | 0,040 |
| tags | 198 | 0,045 | 0,015 | 0,054 | 0,018 | 0,045 | 0,015 |
| long-text | 6 | 0,011 | 0,009 | 0,074 | 0,066 | 0,011 | 0,009 |
| hard | 84 | 0,180 | 0,089 | 0,170 | 0,087 | 0,180 | 0,089 |
| language (phần tiếng Việt) | 18 | 0,364 | 0,021 | 0,394 | 0,214 | 0,341 | 0,263 |
| input-errors (ca không lỗi) | 6 | 0,375 | 0,118 | 0,417 | 0,127 | 0,375 | 0,118 |
| pipeline | 12 | 0,000 | 0,000 | 0,021 | 0,017 | 0,000 | 0,000 |
| speed-ramp | 24 | 0,002 | 0,002 | 0,007 | 0,006 | 0,002 | 0,002 |
| short-text, tag-case | 18 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| **Tất cả** | 1.368 | **0,072** | 0,028 | **0,167** | 0,119 | **0,072** | 0,032 |

Theo giọng, WER của (a) là 0,064–0,077 và của (b) là 0,164–0,169. Theo seed: seed 42 cho (a) 0,070 / (b) 0,169, seed 7 cho (a) 0,076 / (b) 0,163. Không giọng hay seed nào đảo thứ tự.

## Vì sao (b) thua

Transcript của gipformer (gộp nhiều giọng và seed), so với cách đọc (a) nghe được:

| Ca | Input text | (b) nghe được | (a) nghe được |
| --- | --- | --- | --- |
| `vi-hotline` | Tổng đài 1900 1234. | "một ngàn chín trăm, một ngàn hai trăm ba mươi bốn" | "một chín không không, một hai ba bốn" |
| `vi-id-number` | Số CMND 012345678. | "xẻ xào sana mà sáu sáu bảy tám" | "chứng minh nhân dân không một hai…" |
| `vi-weekday-date` | Thứ 2, 01/01/2025. | "không một không một hai không hai mươi lăm" | "ngày một tháng một năm hai nghìn…" |
| `vi-roman` | Thế kỷ XXI. | "mười mười" | "hai mươi mốt" |
| `vi-title-chain` | Mời PGS.TS. Hà và ThS. BS. Lan… | "PGS PS… test BS" | "phó giáo sư tiến sĩ… thạc sĩ bác sĩ" |
| `vi-pin-battery` | Pin 5000mAh. | "fa taoson mile à", "vinfason milia" | "năm nghìn mi li am pe…" |
| `vi-date-time` | Ngày 15/03/2024 lúc 14h30. | bỏ "năm" và "phút" | đọc đủ |

Đây là lỗi đọc thật, không phải do tham chiếu thiên vị: (b) đọc thành số khác, chữ khác, hoặc âm vô nghĩa. Các ca `cases.csv` có độ lệch lớn nhất giữa các biến thể đều thuộc loại này.

## Chỗ (b) hơn (a)

Có 12 ca (b) thấp hơn (a) ≥ 0,05, phần lớn là nhiễu hoặc do cách ASR viết, không phải lỗi pipeline:

- Câu một hai âm tiết (`hard-short-a` "A!", `hard-short-vang`, `hard-that-a-exclaim`): ASR đoán bừa ở cả hai biến thể ("Qatar", "Canada"). Đây là lý do nhóm `hard` nhỉnh hơn ở (b).
- `warnings-once`, `table-pipe-not-table`: (a) đọc "a bê xê", ASR viết "ABC", nên bị tính 1 từ thay vì 3.
- `vi-acronym-tech`, `nfc`, `warnings-order`, `vi-cmu`, `zerotts-cmu`, `fence-unclosed`: engine text gần như giống nhau (chỉ khác chữ hoa/thường); chênh lệch do một lượt sampling lệch.
- `vi-ok` ("ô ca" so với "OK"), `vi-slang-alone` (`k` → "ca"): ASR viết "OK", "K" và cả "coca", nên chỉ nghe mới phân định được.

Ngoài 12 ca trên, có một lỗi thật mà WER tổng không lộ ra, vì phần chữ số của cùng câu đọc đúng hơn ở (a): `vi-pin` và `vi-otp`. sea-g2p đánh vần `PIN` → "phê i nờ", `OTP` → "ô tê phê". ZeroTTS đọc cụm này không rõ (ASR: "fin", "phy n", "ô tô"), trong khi để nguyên viết tắt thì nghe ra "pin", "otp". Hai ca này được ghi thành known issue trong golden. Sửa cần một reading rule hoặc mục lexicon riêng, có golden kèm theo (ngoài phạm vi #9).

Chữ thường hoá toàn bộ của sea-g2p (known issue `vi-lowercase`) không làm tăng lỗi đo được: nhóm markup (a) 0,051 so với (b) 0,087.

## (a) so với (c)

WER chỉ lệch ở 4 ca có tên sản phẩm (`vi-products-normalize-off`, `vi-ipad`, `vi-ios-version`, `vi-ios-version-multipart`), mỗi ca ≤ 0,05 và không nghiêng hẳn về bên nào. ASR viết `FACEBOOK`, `GOOGLE`, `IPHONE`, `IOS` cho cả "phây búc"/"gu gồ"/"ai phôn" lẫn tên gốc, nên WER không phân biệt được hai cách đọc. Giữ lexicon không làm hại gì đo được, và lexicon hiện là cấu hình chung của gateway (chưa có lexicon theo model), nên config giữ nguyên lexicon.

`USD` thì không ổn định ở cả hai bên ("đô la mỹ", "ulabi", "USDT"). Có vẻ ZeroTTS tự đọc `USD` thành "đô la Mỹ" rồi ASR viết lệch; chưa ghi thành known issue khi chưa có người nghe.

## Cần người nghe

Đây là slice `ready-for-human`. Trước khi đóng #9, người nghe chốt:

1. Sáu ca `loan-*` của bộ nghe và `vi-english-words`, `vi-iphone`, `vi-youtube`, `vi-facebook`, `vi-google`, `vi-ios` của golden: so `outputs/ab9/a-s42/<set>/zerotts/<giọng>/index.html` với `outputs/ab9/c-s42/…`. Nếu tên gốc (c) tự nhiên hơn, xoá các mục đó khỏi `configs/lexicon.csv` (OmniVoice sẽ bị gỡ ở #11) và sửa golden tương ứng.
2. `vi-pin`, `vi-otp`, `vi-dollar-sign`, `vi-usd`: xác nhận known issue và cách đọc `USD`.
3. Điền `review_result` trong `results.csv` của những ca trên.

`outputs/` nằm ngoài git; muốn giữ bản nghe thì chép thư mục `outputs/ab9` sang chỗ lưu trữ.

## Giới hạn

- Tham chiếu là cách đọc của chính gateway, nên cách đọc khác mà vẫn đúng bị trừ điểm. Phần lớn được bù bằng việc gộp biến thể vùng miền và lấy tham chiếu gần hơn trong hai bản. Cách ASR viết chữ cái, viết tắt và tên tiếng Anh (`KM`, `MILI`, `AI`, `API`) vẫn làm WER các nhóm `tiếng Anh` và `viết tắt` cao ở mọi biến thể. Ở các nhóm này chỉ nên đọc chênh lệch giữa biến thể, không đọc giá trị tuyệt đối.
- Một model ASR, 4 giọng, 2 seed; ca ngắn một hai âm tiết rất nhiễu.
- Chỉ đo `zerotts` (offline). `zerotts-stream` dùng cùng model và chia đoạn; script đã chạy thử với nó (PCM 48 kHz) nhưng không lặp toàn bộ A/B.
- Đo trước khi đổi config: (b) được render với `normalize = false` lúc đó là mặc định của model ZeroTTS.

## Tái lập

```bash
uv sync --group asr --group dev
uv run hf download g-group-ai-lab/gipformer1.5-68M-rnnt encoder.int8.onnx decoder.int8.onnx \
    joiner.int8.onnx tokens.txt --local-dir models/gipformer1.5-68M-rnnt
uv run zerotts-server --config <zerotts_server.toml với workers = 6>
uv run zerotts-gateway --config <gateway.toml có lexicon, cổng 8000; zerotts: normalize = false để đo (b)>
uv run zerotts-gateway --config <gateway.toml không lexicon_path, cổng 8001>
uv run --group asr python scripts/asr_server.py --port 8090 --num-threads 8
# rồi các lệnh render_all.py / ab_report.py ở docs/agents/listening.md, mục A/B,
# với --voice maichi --voice baotrang --voice quangminh --voice giahuy, --seed 42 và 7
```
