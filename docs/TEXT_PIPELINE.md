# Text pipeline

`POST /v1/audio/speech` thay `input` bằng engine text trước khi gửi lên upstream. Bước làm sạch (mọi ngôn ngữ) còn đổi bảng markdown: bỏ dòng phân cách `|---|`, mỗi hàng thành một dòng với các ô nối bằng dấu phẩy, rồi kết dòng như mọi dòng markup (#15). Field riêng `text_processing: {"normalize": bool}` bị gỡ trước khi forward; guard và làm sạch luôn chạy.

- Response thành công có `X-Text-Pipeline-Version` và, khi có cảnh báo, `X-Text-Warnings` (`omitted_code_block`, `omitted_html`, `omitted_url`, `omitted_emoji`, `omitted_nonverbal_tag`).
- Lỗi đầu vào `400`: `text_too_long`, `reasoning_markup`, `model_control`, `empty_text`; trả trước khi gọi upstream.
- Tốc độ text ngắn (tuỳ chọn theo model trong `configs/gateway.toml`, [ADR 0004](adr/0004-short-text-speed.md)): khi caller không gửi `speed` lẫn `options.duration_sec`/`duration`, gateway có thể thêm `speed` để giảm mất cụm đầu (#18, #19). Dạng cũ: `short_text_speed` cùng `short_text_max_syllables` (mặc định 12). Hoặc `short_text_speed_bands`: các cặp `max_syllables`/`speed`, ngưỡng tăng dần, chọn band đầu tiên chứa số âm tiết của engine text (bỏ non-verbal tag); vượt band cuối thì không thêm. Không dùng hai dạng cùng lúc. Config repo dùng 0.7 cho ≤ 8 âm tiết, 0.6 cho 9–12; model khác và mặc định trong code vẫn tắt. Tuỳ chọn `short_text_speed_ramp` (#22, tắt) nội suy tuyến tính speed sau dạng ngắn, ví dụ 0.6 ở 13 âm tiết lên 1.0 ở 24, để bỏ bước nhảy 12→13. Đây là biện pháp giảm lỗi, không bảo đảm model luôn đọc đủ hay không lặp chữ.
- Model không hiểu thẻ (`nonverbal_tags = false` trong `[models.<id>]`; config repo: `zerotts`, `zerotts-stream`): sau bước làm sạch, pipeline gỡ 13 non-verbal tag và phiên âm CMU (`[B EY1 S]`) khỏi input text, với mọi ngôn ngữ và cả khi normalize tắt, rồi thêm một text warning `omitted_nonverbal_tag` (một lần dù gỡ bao nhiêu thẻ). Không để lại khoảng trắng hay dấu câu thừa: dấu câu ngay sau thẻ được giữ khi thẻ đứng sau chữ (`Giá [laughter], 50.000đ` → `Giá, 50.000đ`), bị bỏ khi thẻ đứng đầu text hoặc sau một dấu câu khác (`[sigh]. Thôi được` → `Thôi được.`, `A, [laughter], B` → `A, B`). Ngoặc vuông khác và thẻ viết hoa (`[Laughter]`) không bị đụng tới. Text chỉ còn thẻ trả `400 empty_text`. Model mặc định (`nonverbal_tags = true`) giữ thẻ như cũ.
- Dry-run: `POST /v1/text/normalize` với `{"input", "language"?, "model"?, "text_processing"?}` trả `engine_text`, `warnings`, `language`, `pipeline_version`, không gọi upstream.

Khi effective language là tiếng Việt và normalize bật, pipeline còn: giữ nguyên 13 non-verbal tag (`[laughter]`, `[sigh]`, …) và phiên âm CMU viết hoa (`[B EY1 S]`), bỏ dấu ngoặc của các ngoặc vuông khác; áp từ điển phát âm; áp reading rule ([ADR 0003](adr/0003-gateway-owns-reading-rules.md): phiên bản `v2.0`, `Python 3.12`, `GPT-3.5` đọc "chấm" với mỗi phần là số đếm; danh xưng có dấu chấm như `ThS.` trước tên mở rộng không để lại dấu chấm; thập phân kiểu Anh `3.5` đọc "phẩy"; khoảng giờ/ngày `8h-17h`, `25-26/12` có "đến"; tiền lóng `50k`, `2tr5`; đọc từng chữ số cho hotline 1800/1900, dãy số sau mã/OTP/PIN/số tài khoản và dãy bắt đầu bằng 0; `ngày 31/02` là ngày, `1/3` (a < b ≤ 10, không có từ ngày tháng) là phân số; quý `Q1/2024`; `m²`, `m³` đọc "mét vuông", "mét khối"; số tiền viết liền `1250000` đọc là một số; từ điển vẫn thắng vì chạy trước); chuẩn hoá bằng sea-g2p (`punc_norm=False`); gỡ thẻ `<en>` và ghép chữ cái đánh vần thành viết tắt hoa (`AI`). `reference_text` đi qua cùng pipeline, cùng mã lỗi, không tính vào warnings.

## Từ điển phát âm

`lexicon_path` trong config (tương đối theo file config; bản mặc định `configs/lexicon.csv`). CSV có header `word,replacement,note`:

```csv
word,replacement,note
iPhone,ai phôn,thương hiệu
```

Khớp theo ranh giới từ, mục dài trước, phân biệt hoa thường; chạy trước sea-g2p nên `replacement` phải là dạng nói tiếng Việt. Nạp lúc khởi động (sửa thì restart); file sai định dạng làm gateway không khởi động. Hash của từ điển nằm trong `X-Text-Pipeline-Version` (`<RULES_VERSION>-<hash>`).

Bản mặc định gồm sáu ngoại lệ tên sản phẩm: `iPhone` → `ai phôn`, `iPad` → `ai pát`, `iOS` → `ai ô ét`, `YouTube` → `du túp`, `Facebook` → `phây búc`, `Google` → `gu gồ`; mỗi mục có `note`. Đây là cách đọc ứng viên từ #7 được triển khai giới hạn trong #17, **chưa có xác nhận nghe bằng tai người** ([thiết kế và kết quả](reports/issue17-lexicon.md)). Không đổi cách đọc `AI`/`API` hay Việt hoá mọi từ tiếng Anh. Chỉ áp khi effective language là tiếng Việt và normalize bật; các biến thể như `iphone`, `IPHONE`, `iPhoneX` không khớp. Dùng config repo để nạp từ điển; `Settings()` không tự nạp file, và file riêng của operator thay thế chứ không gộp với bản repo.

Tên sản phẩm trước phiên bản vẫn giữ luật đọc “chấm”: `iOS 18.2` → `ai ô ét mười tám chấm hai`. Gateway dành quyền ưu tiên cho mọi mục từ điển khớp nguyên văn trước, rồi dùng ngữ cảnh tên gốc cho hậu tố phiên bản chưa bị mục khác ghi đè; không đưa reading rule lên trước từ điển.

## Thêm ca regression

Ca golden nằm ở `tests/golden/pipeline.json`, chạy qua route dry-run. Mỗi khi sửa một lỗi đọc sai:

1. Xem engine text hiện tại: `curl -s localhost:8000/v1/text/normalize -d '{"input":"…","language":"vi"}'`.
2. Thêm một dòng: `id`, `group` (`numbers-money`, `dates-times`, `measurements`, `codes-phones`, `abbrev-english`, `reading-rules`, `markup`, `tags`, `input-errors`, `short-sentences`, `language`, …), `input`, tuỳ chọn `language` (mặc định `vi`), `normalize` và `model` (dùng mặc định của model đó trong `configs/gateway.toml`, ví dụ `zerotts` cho ca gỡ thẻ; `render_samples.py` chỉ render ca có `model` khi chạy đúng model ấy), rồi `expected` + `warnings` hoặc `error`, và `why` (vì sao đó là cách đọc đúng).
3. Nếu tạm chấp nhận output sai (hành vi của sea-g2p chưa sửa), ghi output hiện tại vào `expected` và thêm `known_issue` mô tả chỗ sai; không xoá ca.
4. Sửa luật hoặc thêm dòng vào từ điển. Đổi luật thì tăng `RULES_VERSION` trong `src/zerotts_gateway/text.py`.
5. `uv run pytest`.