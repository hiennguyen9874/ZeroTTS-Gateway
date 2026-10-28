# #14 — A/B chất lượng fp32 so với int8 (golden, ASR, nghe)

Ngày đo: 2026-10-28, máy dev dùng chung. Phần agent (render, chấm ASR, liệt kê ca
lệch) đã xong. **Phần nghe và quyết định cuối cùng còn chờ người**, xem
[Cần người nghe](#cần-người-nghe).

## Kết luận

- **Về ASR, int8 đạt cổng:** WER int8 là **0,074** (ngưỡng ≤ 0,08), fp32 cùng lượt
  là 0,074. Chênh lệch int8 − fp32 là +0,0004 (khoảng tin cậy 95% bootstrap theo ca:
  −0,003 đến +0,004). CER bằng nhau (0,029).
- Không nhóm ca nào lệch quá 0,01 WER, trừ các nhóm nhỏ và nhiễu (`hard`, `input-errors`;
  xem bảng). Không giọng hay seed nào đổi kết luận: theo giọng int8 − fp32 nằm trong
  −0,003 đến +0,003.
- int8 làm transcript đổi **không nhiều hơn** việc đổi seed trên chính fp32: 74/450 hàng
  khác transcript fp32 so với int8, trong khi fp32 seed 42 so với fp32 seed 7 là 78/450.
- Phần lớn những hàng int8 nghe ra kém hơn là nhiễu sampling một âm tiết ("nhé"/"nhá",
  câu "Thật à." thành "Canada"). Có bốn ca nên nghe trước, vì có thể là đọc sai thật:
  `vi-acronyms-comma`, `vi-dollar-sign`, `inline-code`, `vi-score-untouched`
  (xem [Ca int8 đọc khác fp32](#ca-int8-đọc-khác-fp32)).
- **Đề xuất (chờ người nghe xác nhận):** cho phép int8 trên production dưới dạng opt-in
  như hiện có (#12), fp32 vẫn là mặc định. Đổi int8 thành mặc định cần ADR riêng sau #10.

## Cách đo

- Upstream: hai `zerotts_server` thật chạy song song, cùng code, cùng
  `configs/zerotts_server.toml` (sampling mặc định), mỗi cái 3 worker × 8 luồng.
  Chỉ khác `model_dir`: `models/ZeroTTS` (fp32) và `models/ZeroTTS-int8`
  (log startup `weights: int8`, hash khớp [báo cáo #12](issue12-zerotts-int8.md)).
  `zerotts==0.1.5`, onnxruntime 1.31.1, commit `b4c6e12`.
- Mỗi upstream có một gateway riêng với `configs/gateway.toml` đang ship (toàn bộ
  pipeline, lexicon chung), chỉ đổi cổng và `upstream_url`.
- Giống #9 (a): golden và bộ nghe `listening` (bỏ ca clone), `--normalize-on`, model
  `zerotts` (offline). Giọng `maichi`, `baotrang`, `quangminh`, `giahuy` với seed 42;
  `maichi`, `quangminh` với seed 7. Mỗi weights 1.380 lần render, tổng 2.760, không lỗi.
- Chấm: gipformer (`scripts/asr_server.py`) rồi `scripts/ab_report.py`, giống #9. Bỏ 30
  hàng mỗi bên (ca không phải tiếng Việt), còn 1.350 hàng mỗi weights.
- Kiểm tra lại mốc: trên 1.344 hàng chung với #9, fp32 hôm nay có WER **0,0746, đúng
  bằng #9 (a)**, và WAV fp32 trùng byte với lượt #9 (cùng engine text, seed, weights).
  0,072 của #9 gồm thêm các ca `speed-ramp` đã bị bỏ khỏi bộ nghe khi gỡ OmniVoice; trên
  bộ ca hiện tại fp32 là 0,074.
- Có một lượt chạy phụ không có `--normalize-on` (gateway tự áp `"normalize": false` của
  72 ca golden): fp32 0,080, int8 0,080. Mốc cao hơn vì tham chiếu luôn là bản đã
  normalize, không phải do int8. Lượt này không dùng để so với ngưỡng.

## Kết quả theo nhóm ca

1.350 hàng mỗi weights. Nhóm giống #9.

| Nhóm | Hàng | WER fp32 | CER fp32 | WER int8 | CER int8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ngày giờ | 60 | 0,000 | 0,000 | 0,005 | 0,004 |
| số | 156 | 0,051 | 0,018 | 0,051 | 0,019 |
| viết tắt | 96 | 0,110 | 0,042 | 0,114 | 0,040 |
| tiếng Anh | 102 | 0,276 | 0,107 | 0,284 | 0,116 |
| markup | 294 | 0,051 | 0,021 | 0,048 | 0,019 |
| reading-rules | 294 | 0,085 | 0,039 | 0,086 | 0,038 |
| tags | 198 | 0,045 | 0,015 | 0,041 | 0,011 |
| hard | 84 | 0,180 | 0,089 | 0,170 | 0,087 |
| long-text | 6 | 0,011 | 0,009 | 0,012 | 0,010 |
| language (phần tiếng Việt) | 18 | 0,364 | 0,021 | 0,364 | 0,021 |
| input-errors (ca không lỗi) | 6 | 0,375 | 0,118 | 0,417 | 0,147 |
| pipeline | 18 | 0,000 | 0,000 | 0,000 | 0,000 |
| short-text, tag-case | 18 | 0,000 | 0,000 | 0,000 | 0,000 |
| **Tất cả** | 1.350 | **0,074** | 0,029 | **0,074** | 0,029 |

`input-errors` chỉ 6 hàng (`thinking-word`): một từ lệch ở một hàng. `ngày giờ` int8 có
hai hàng lệch một từ (`vi-colon-time`, `vi-quarter`).

Theo giọng và seed (WER fp32 / int8):

| | Hàng | fp32 | int8 |
| --- | ---: | ---: | ---: |
| `maichi` | 450 | 0,078 | 0,075 |
| `baotrang` | 225 | 0,079 | 0,081 |
| `quangminh` | 450 | 0,071 | 0,073 |
| `giahuy` | 225 | 0,066 | 0,069 |
| seed 42 | 900 | 0,072 | 0,074 |
| seed 7 | 450 | 0,078 | 0,075 |

`baotrang` (WER fp32 / int8 theo nhóm): markup 0,058 / 0,048, số 0,072 / 0,068, viết
tắt 0,101 / 0,116, reading-rules 0,097 / 0,097, hard 0,163 / 0,224 (14 hàng; `Thật à.` bị
ASR nghe thành "Canada"). `maichi`: số 0,052 / 0,059, các nhóm khác bằng hoặc tốt hơn.

## Mức nhiễu

Cùng seed nhưng khác weights thì audio khác hẳn (#12), nên phải so int8 với độ lệch giữa
hai seed trên cùng weights. Trên 450 hàng `maichi` + `quangminh`:

| So sánh | Transcript khác | Bên sau nhiều lỗi hơn | Bên sau ít lỗi hơn |
| --- | ---: | ---: | ---: |
| fp32 với int8 (cùng seed 7) | 74 | 31 | 30 |
| fp32 seed 42 với fp32 seed 7 | 78 | 35 | 28 |
| int8 seed 42 với int8 seed 7 | 70 | 31 | 26 |

Trên toàn bộ 1.350 cặp: 209 transcript khác, int8 nhiều lỗi hơn ở 88 hàng và ít lỗi hơn ở
78 hàng.

## Ca int8 đọc khác fp32

56 ca có ít nhất một hàng mà int8 nhiều lỗi từ hơn fp32 (88 hàng, 42 hàng thuộc `maichi`
hoặc `baotrang`). 37 ca int8 kém hơn khi cộng qua mọi hàng, sắp xếp theo số lỗi thêm:

`vi-acronyms-comma`, `inline-code`, `email-kept`, `vi-email`, `vi-otp`,
`hard-that-a-period`, `vi-python-version`, `vi-score-untouched`, `vi-slang-km`,
`table-align`, `vi-dollar-sign`, `vi-english-sentence`, `vi-hotline-joined`,
`vi-version-parts-cardinal`, `abbr-ai-letters-vi`, `vi-version-thousands`, `stray-strong`,
`vi-euro`, `abbr-ai-lower`, `bullets`, `list-four-syllables`, `table-header`,
`terminal-ellipsis-quote`, `thinking-word`, `url`, `vi-bracket-not-cmu`, `vi-colon-time`,
`vi-fraction`, `vi-ios-version-multipart`, `vi-leading-zero`, `vi-quarter`,
`vi-superscript-cube`, `vi-time-range`, `vi-title-chain`, `vi-usd`, `vi-youtube`,
`long-text`.

Mười ca đầu, transcript gipformer:

| Ca | Input text | fp32 nghe được | int8 nghe được | Nhận xét |
| --- | --- | --- | --- | --- |
| `vi-acronyms-comma` | Tôi dùng AI, API và USD. | "AI API và USD" ("API" ở 5/6 hàng) | "AI IPI và DUSD/USD" (4/6 hàng kém hơn, 0 hàng tốt hơn) | **Nghe trước**: có thể int8 đọc `API` kém đều |
| `inline-code` | Gọi hàm `print()` để in. | "gọi hàm print để in" | "để im", "hầm", "bring" | **Nghe trước**: lệch ở 4/6 hàng |
| `email-kept`, `vi-email` | Gửi về hotro@example.com nhé. | "hotro a còng example…" | "a cộng", "hotracom" | Phát âm "a còng" có thể kém rõ |
| `vi-otp` | Mã OTP là 123456. | "ô tô" | "OTP", "OTF" | Known issue từ #9; int8 không xấu hơn khi nghe theo nghĩa |
| `hard-that-a-period` | Thật à. | "thật à" | "Canada", "Qatar" | Câu hai âm tiết, ASR đoán bừa (giống #9) |
| `vi-python-version` | Python 3.12 rất hay. | "python" | "pyton", "file thân" | Tên tiếng Anh |
| `vi-score-untouched` | Đội A thắng 3-1 hôm qua. | "thắng" | "tháng" (2 hàng) | **Nghe trước**: sai thanh điệu |
| `vi-slang-km` | Đi 5km. | "ki lô mét" | "KM" (1 hàng) | ASR viết tắt |
| `table-align` | (bảng giá) | "tái chín" | "tài chính", "bằng giá" | Một âm tiết |
| `vi-dollar-sign` | Vé giá $5. | "năm USD" | "năm mươi USD", "năm đưa ra" | **Nghe trước** `maichi`: có thể đọc thành "năm mươi" |

Đầy đủ: `outputs/ab14/norm/int8_worse_cases.csv` (theo ca) và
`outputs/ab14/norm/int8_worse_rows.csv` (88 hàng, đường dẫn WAV fp32/int8, cột `review`
để điền).

## Cần người nghe

Đây là slice `ready-for-human`; agent không nghe được. Trước khi đóng #14:

1. Mở trang so sánh: `python3 -m http.server -d outputs/ab14/norm 9000`, rồi vào
   `http://localhost:9000/listen.html`. Mỗi dòng gồm engine text, audio fp32 và int8 cùng
   giọng, cùng seed, kèm transcript.
2. Nghe **từng hàng** trong `int8_worse_rows.csv`, trước hết bốn ca in đậm ở trên, rồi
   mọi hàng `maichi` và `baotrang` (42 hàng). Điền cột `review` (`int8 kém hơn`,
   `như nhau`, `ASR sai`).
3. Nghe thêm một mẫu bình thường (không lệch theo ASR) của `maichi` và `baotrang` để so
   chất giọng, nhịp, độ sạch, vì ASR không bắt được những khác biệt này:
   `outputs/ab14/norm/{fp32,int8}-s42/listening/zerotts/<giọng>/index.html`.
4. Ghi kết quả vào mục dưới đây và chốt kết luận.

### Kết quả nghe

_Chưa nghe._

| Giọng | Hàng đã nghe | int8 kém hơn | Ghi chú |
| --- | --- | --- | --- |
| `maichi` | | | |
| `baotrang` | | | |

**Quyết định:** _chờ người nghe._ Đề xuất của agent: đạt cổng ASR; nếu người nghe không
thấy int8 kém đi trên `maichi` và `baotrang` thì int8 được dùng trên production dưới dạng
opt-in (`model_dir` trỏ tới `models/ZeroTTS-int8`), fp32 vẫn là mặc định. int8 thành mặc
định cần ADR riêng sau #10.

## Giới hạn

- Tham chiếu là cách đọc của gateway; ASR viết chữ cái và tên tiếng Anh khác engine text,
  nên ở nhóm viết tắt/tiếng Anh chỉ nên đọc chênh lệch fp32/int8, không đọc giá trị tuyệt
  đối (như #9).
- Một model ASR, 4 giọng, 2 seed. Mỗi ca chỉ có 6 hàng, nên ca lệch một hai hàng không đủ
  để kết luận, cần nghe.
- ASR không đo chất giọng, nhịp, tiếng ồn hay độ tự nhiên. Đó là phần của người nghe.
- Chỉ đo `zerotts` (offline). `zerotts-stream` dùng cùng weights và chia đoạn nhưng chưa
  được A/B.
- Một bộ int8 (quantize trên máy dev, hash ở báo cáo #12); bộ int8 quantize trên máy khác
  có thể khác hash, khi đó phải đo lại.
- RTF trong lượt này (fp32 ~1,5, int8 ~0,9) chỉ phản ánh máy đang tải nặng, không dùng
  cho tốc độ (#13, #10).

## Tái lập

Dữ liệu cục bộ, gitignored: `outputs/ab14/` (`conf/`, `logs/`, `services.sh`, `run.sh`,
`analyze.py`; lượt chính trong `norm/`, lượt không `--normalize-on` ở gốc `outputs/ab14/`).

```bash
uv sync --group asr --group dev
outputs/ab14/services.sh   # fp32 :18280 → gateway :18200, int8 :18281 → gateway :18201, ASR :18290
outputs/ab14/run.sh        # render_all.py --normalize-on, 4 lượt song song, ~75 phút ở load ~40
O=outputs/ab14/norm
UV_NO_SYNC=true uv run python scripts/ab_report.py --variant fp32=$O/fp32-s42 --variant fp32=$O/fp32-s7 \
    --variant int8=$O/int8-s42 --variant int8=$O/int8-s7 --out $O/report
UV_NO_SYNC=true uv run python outputs/ab14/analyze.py $O   # nhiễu, CI, ca lệch, listen.html
```

Config server/gateway trong `outputs/ab14/conf/` khác bản ship chỉ ở cổng, `model_dir`
(đường dẫn tuyệt đối), `workers = 3`, `busy_timeout = 600` và `lexicon_path` tuyệt đối.
