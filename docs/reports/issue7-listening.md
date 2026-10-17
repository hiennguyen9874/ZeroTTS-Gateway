# Lượt nghe lại #7 sau các sửa lỗi pipeline

Cập nhật tiếp theo: [#17 — từ điển mặc định và render lại hai giọng](issue17-lexicon.md).
Báo cáo dưới đây là lịch sử trước #17; không suy ra sáu ứng viên đã được người nghe chốt.

## Phạm vi và cách đọc kết quả

Lượt ngày 17/10/2026 thay toàn bộ output cũ theo yêu cầu operator. Không còn WAV
của các lượt #15/#18/#19 trong `outputs/`; các báo cáo và comment issue cũ vẫn là
bằng chứng lịch sử, không phải kết quả lượt này.

- Gateway: commit nền `2d54936` cộng thay đổi bộ nghe trong working tree; pipeline
  `5-4f53cda1`, sea-g2p 0.10.0. Không đổi hành vi production trong lượt này.
- Upstream: audio.cpp `65df584a6f798128fc27ff7e2d8aede0df8ae866`, CUDA 12.6,
  checkpoint f16 của cả ba model; bốn entry `g-omnivoice`,
  `g-omnivoice-stream`, `omnivoice-vietnamese`, `omnivoice`.
- Giọng bắt buộc: `vieneu_v3_Doan_Trang`, `vieneu_v3_Ngoc_Linh`.
  Rút thêm ba giọng bằng `random.Random(7).sample` trên danh sách giọng hợp lệ đã
  sort, bỏ hai giọng bắt buộc: `Ngoc_Tran`, `Kim_Thanh`, `Quang_Son`.
  Quang_Son cũng là đối chứng giọng từng mất cụm đầu.
- Seed chính 42. Seed 1 và 2 bổ sung cho câu hỏi 2, 4, 5.
- ASR: gipformer RNN-T int8, sherpa-onnx, `modified_beam_search`, CPU.
- Chạy các model song song trên GPU 0–3, mỗi model có một cặp gateway/upstream
  riêng, cùng config ngoài port/địa chỉ upstream; không dùng GPU 4/7 đang bận.
  Trong mỗi cặp, request tuần tự, mỗi request chỉ gửi một lần, không retry.

Output: `outputs/issue7-20261017/index.html`. `plan.json` lưu mọi lệnh, commit,
SHA-256 của source/config/transcript/voice WAV; `source/` lưu bản source/config
thực dùng (không sao chép checkpoint). Các manifest ghi commit có hậu tố
`-dirty`; không coi hash commit nền là toàn bộ thay đổi đã render.

### Các bộ output

| Thư mục | Nội dung |
| --- | --- |
| `full/` | Toàn bộ golden không lỗi và bộ listening, A/B, seed 42, cả bốn model × năm giọng |
| `production/` | Chạy lại các golden có `normalize: false` với `--normalize-on`, seed 42, A/B |
| `seed-1/`, `seed-2/` | Các ca listening mang câu hỏi 2/4/5, A/B, mọi model/giọng |
| `*/listening/<model>/_clone/` | Clone Mai_Anh, Doan_Trang, Ngoc_Linh; chỉ chạy một lần mỗi model/seed vì tự mang audio tham chiếu |

`--normalize-on` bỏ cờ tắt của ca, **không** ghi đè mặc định model.
Vì vậy `omnivoice` gốc vẫn tắt normalize theo config repo; không dùng các ca đó
để kết luận về luật chuẩn hoá tiếng Việt. Các entry finetune mặc định bật normalize.
Nhãn clone `normalized` nghĩa là đã qua pipeline của gateway theo model đó; với
`omnivoice` gốc, đây vẫn chỉ là các bước guard/làm sạch/kết câu, không phải sea-g2p.

Kế hoạch gồm **112 run / 11.136 request**: `full` 8.232, `production` 1.800,
`seed-1` và `seed-2` mỗi bộ 552. Mọi id nằm trong `plan.json`.

Mỗi bộ có `all_results.csv`, `summary.csv`, `voices.csv`, `cases.csv` và trang
A/B của từng run. `completion.json` chỉ xuất hiện khi đã chạy đủ kế hoạch;
`progress.json` cho biết tiến độ trước đó. Cột review vẫn trống để người nghe điền.

## Sửa phương pháp thí nghiệm

1. Ca clone nay gửi **speed 1.0 ở cả hai phía**, hoặc cùng giá trị `speed` của ca.
   Nếu để gateway tự chọn speed band mà phía direct không có, so sánh không còn
   chỉ khác `reference_text`. Hai ca clone mới dùng chính audio của Doan_Trang và
   Ngoc_Linh; không sửa audio/transcript trong thư viện giọng.
2. Hai ca `tag-case-upper/lower` tắt normalize, cùng dấu chấm cuối và speed 1.0.
   So sánh chúng để tách chữ hoa/thường khỏi khác biệt dấu câu, chuẩn hoá và speed.
   Không bật `--normalize-on` cho đối chứng này. Các ca tag production cũ vẫn giữ.
3. Thêm cách đọc ứng viên cho iPhone/iPad/iOS/YouTube/Facebook/Google ở nhóm
   `loanwords`; đây **không phải** quyết định thêm từ điển mặc định.
4. CSV có `requested_speed`, HTML hiện speed đối chứng. Ô trống nghĩa là script
   không chỉ định, **không** có nghĩa gateway không áp speed band.
5. Giữ các golden cleanup-only, đồng thời render bản theo cấu hình production để
   không nhầm việc gửi nguyên `@`, `<`, `>` với lỗi luật chuẩn hoá.

Trên upstream thật, ở seed 42, hai phía A/B của từng ca `tag-case` có WAV giống
**nguyên byte** ở 40/40 cặp (hai ca × năm giọng × bốn model). Đây là kiểm đối chứng
request, không phải kết luận cách đọc tag đúng.

Test HTTP qua gateway với upstream giả xác nhận speed đối chứng vẫn 1.0 khi
operator bật band 0.6, cùng input/audio ở clone, khác hoa/thường được giữ nguyên
trong cặp tag-case. Toàn bộ test: **371 passed**; Ruff lint/format đạt.

## Hoàn tất toàn bộ kế hoạch

**112/112 run đã kết thúc**, tổng **11.136 request / 11.126 WAV hợp lệ**.
Có 5 run exit 1 do cùng một ca upstream lỗi (10 request), không có lỗi mới ở các
bộ bổ sung. `completion.json` ghi thời điểm kết thúc 05:17:56 UTC ngày 17/10/2026.

| Bộ | Request | WAV | Lỗi tổng hợp | ASR không hỗ trợ ngôn ngữ |
| --- | ---: | ---: | ---: | ---: |
| full | 8.232 | 8.222 | 10 | 120 |
| production | 1.800 | 1.800 | 0 | 0 |
| seed-1 | 552 | 552 | 0 | 0 |
| seed-2 | 552 | 552 | 0 | 0 |

Bộ production (45 ca × năm giọng), recall gateway: g-omnivoice **0,939**,
g-omnivoice-stream **0,939**, omnivoice-vietnamese **0,934**, omnivoice **0,917**.
Model gốc vẫn dùng mặc định normalize tắt như đã nêu; không so trực tiếp điểm này
với bộ golden đầy đủ có nội dung khác.

Đã kiểm đủ manifest/CSV/HTML cho mọi run, file WAV mở được và có frame, cột review
còn trống, hash source/config/voice của kế hoạch khớp file hiện tại.

## Kết quả bộ đầy đủ, seed 42

Bộ `full` đã hoàn tất **8.232 request**, có **8.222 WAV**, 10 lỗi tổng hợp đều
thuộc ca `vi-language-region` ở model gốc (hai phía × năm giọng).

Recall trung bình trên 166 golden/giọng; loại ô không được ASR chấm, không loại
ca có recall thấp. Hai phía đều so với engine text của gateway:

| Model | Gateway | Direct |
| --- | ---: | ---: |
| g-omnivoice | 0,927 | 0,725 |
| g-omnivoice-stream | 0,930 | 0,717 |
| omnivoice-vietnamese | 0,919 | 0,728 |
| omnivoice (normalize tắt theo config) | 0,682 | 0,658 |

Đây không phải điểm tự nhiên/độ giống giọng. Direct đọc dạng viết còn gateway đọc
dạng nói; ASR cũng có thể viết lại tên/số khác chuỗi so sánh.

### Các ca từng bị báo sai, G-OmniVoice offline, năm giọng

- Recall **1,000**: `terminal-ellipsis-quote`, `paragraph-lines`, `heading`,
  `fence-closed`, `thematic-break`, `numbered-list`, `bullets`,
  `vi-quarter`, `vi-time-range`, `vi-superscript`, `vi-hotline`.
  Cả Doan_Trang và Ngoc_Linh đã có đủ từ trong transcript ở các ca này.
- `url`: 0,950; Doan_Trang nghe qua ASR thành “truy cập hoặc nhắc” (0,750),
  không còn mất “truy cập”, nhưng cần nghe lại âm cuối.
- `emoji`: 0,900; Doan_Trang/Ngoc_Linh đủ từ, Quang_Son chỉ còn “cảm ơn”.
- `stray-strong`: 0,950; Quang_Son bị ASR viết “đầm” thay “đậm”.
- `vi-python-version`: 0,857; engine text đã là “ba chấm mười hai”. Ngoc_Linh
  bị viết “Piton”, Quang_Son thành “Peton ba chín mười hai”; không tự kết luận đây
  là lỗi reading rule chỉ từ ASR.
- `vi-short-three`: 0,867; Doan_Trang thiếu “xong rồi”, Quang_Son thiếu “đi thôi”.
  Ngoc_Linh đủ từ. Đây vẫn là giới hạn upstream, không được che bằng cách đổi dấu
  chấm caller tự viết.

### Text dài, phía gateway, năm giọng/model

| Model | Độ dài WAV min–max (s) | Recall TB | TTFB trung vị (ms) |
| --- | ---: | ---: | ---: |
| g-omnivoice | 49,48–62,77 | 0,991 | 6370 |
| g-omnivoice-stream | 50,00–63,48 | 0,990 | 1112 |
| omnivoice-vietnamese | 49,50–63,13 | 0,968 | 6769 |
| omnivoice | 44,06–56,40 | 0,879 | 5427 |

Mọi mẫu dài vượt 30 s. TTFB chỉ là số đo trong lượt có ASR/model khác chạy song
song, không phải benchmark chuẩn. Cần nghe ranh giới chunk để chốt ngữ điệu và
trôi giọng, dù recall cao.

## Lưu ý lỗi và giới hạn của lượt chạy

- Ba golden gửi `language=en` (`english-cleanup`, `table-english`,
  `vi-language-en`) không được gipformer tiếng Việt hỗ trợ: audio vẫn được render,
  nhưng nhận `asr_failed` vì endpoint trả 400. Không coi đây là mất chữ hoặc lỗi tổng
  hợp; không đưa chúng vào trung bình recall. Giữ cờ thật trong CSV, không xoá mẫu.
- Entry `omnivoice` gốc trả 500 cho `vi-language-region` (`language=vi-VN`) ở cả
  direct lẫn gateway. Không đổi ngôn ngữ caller để che lỗi: spec yêu cầu forward
  nguyên giá trị. Các entry có mặc định `language=vi` không thấy lỗi này trong mẫu
  đã chạy. Cần theo dõi phía upstream; không coi là lỗi luật chuẩn hoá của gateway.
- `vi-short-three` vẫn có thể mất “xong rồi” ở Doan_Trang dù engine text đúng và
  speed band đã bật. Đây là ba câu có dấu chấm caller tự viết: không tự đổi chúng
  thành dấu phẩy trái spec. ADR 0004 chỉ là biện pháp giảm lỗi, không bảo đảm mọi
  giọng/seed. Giữ audio và flag trong kết quả để nghe lại.

## Năm câu hỏi: chính sách hiện tại và điều kiện chốt

Đây là lượt render/chấm ASR, **chưa phải đánh giá bằng tai người**.
Không đánh dấu pass/fail cảm nhận, không suy ra tiếng cười, ngữ điệu hoặc độ giống
giọng từ `word_recall`. Recall cũng không phạt chữ thừa; từ tiếng Anh và chữ cái
có thể bị ASR viết khác. Các ca đã biết sai vẫn giữ nguyên trong golden.

| Câu | Quyết định tạm giữ | Lý do / phần cần người nghe |
| --- | --- | --- |
| 1. Viết tắt, từ mượn | Giữ AI/API viết hoa; chưa thêm từ điển production (#17 còn mở) | So sánh `abbr-ai-upper/letters-vi/lexicon` và nhóm `loanwords`; ASR không chọn được cách đọc tự nhiên |
| 2. Chữ hoa/thường | Giữ hành vi sea-g2p chuyển chữ thường; không tự khôi phục hoa | Nghe `case-proper-names`, `hard-all-caps`, nhất là cặp `tag-case-upper/lower` ở ba seed |
| 3. Text dài/chunk | Upstream vẫn sở hữu chunking (ADR 0001); gateway không chia/ghép | Nghe `long-text` offline và streaming, kiểm đủ ≥30 s, chỗ nối, lặp và trôi giọng; ASR chỉ hỗ trợ phát hiện thiếu từ |
| 4. Non-verbal tag | Tiếp tục bảo toàn tag theo API; không hứa model luôn tạo đúng âm thanh | Nghe nhóm tags và tag-case; tag được giữ đúng byte không đồng nghĩa model hỗ trợ tin cậy |
| 5. Reference text | Giữ pipeline cho reference_text theo spec | So sánh clone speed khớp nhau, ba audio tham chiếu × ba seed; cần nghe độ giống giọng và nhiễu |

Theo yêu cầu bàn giao, đóng #7 sau khi commit/push bộ nghe và báo cáo này. Việc
đóng issue **không xác nhận đã nghe bằng tai người** hoặc đã chốt cả năm chính sách;
các quyết định trong bảng vẫn là tạm giữ. #17 còn mở vì chưa triển khai từ điển
production. Nếu nghe dẫn tới đổi chính sách: mở issue/sửa spec và thêm golden
trước khi đổi pipeline.

## Tái lập

Cần khởi động upstream, gateway và ASR theo [LISTENING.md](../LISTENING.md#chấm-bằng-asr-tuỳ-chọn). Chạy tuần tự trên một GPU cũng
cho cùng kế hoạch model/giọng/seed; song song chỉ để giảm thời gian chờ.

```bash
export OMNIVOICE_ASR_URL=http://127.0.0.1:8090
voices=(
  --voice vieneu_v3_Doan_Trang --voice vieneu_v3_Ngoc_Linh
  --voice vieneu_v3_Ngoc_Tran --voice vieneu_v3_Kim_Thanh --voice vieneu_v3_Quang_Son
)
uv run python scripts/render_all.py "${voices[@]}" --out outputs/issue7-repeat/full
# Bản production bổ sung: có thể render toàn bộ golden hoặc chỉ các id trong plan.json.
uv run python scripts/render_all.py "${voices[@]}" --set golden --normalize-on \
  --out outputs/issue7-repeat/production
# Mở trang nghe; chỉ expose trong mạng tin cậy.
python3 -m http.server --bind 127.0.0.1 --directory outputs/issue7-20261017 9000
```

Các lệnh lọc chính xác theo id, seed, model và cặp endpoint của lượt này nằm trong
`plan.json`; supervisor của lượt nằm cạnh đó. Không chạy lại vào cùng thư mục đã
có dữ liệu nếu không chủ ý overwrite.
