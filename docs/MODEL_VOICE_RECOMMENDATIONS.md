# Đánh giá model, giọng và khuyến nghị sử dụng

## Kết luận nhanh

**Cấu hình hiện tại phù hợp để dùng thử có kiểm soát cho tiếng Việt, chưa đủ
bằng chứng để cam kết đọc đúng mọi câu trong production.**

Khuyến nghị mặc định:

- **Model:** `g-omnivoice` (offline).
- **Giọng:** `vieneu_v3_Kim_Thanh`; lựa chọn gần tương đương là
  `vieneu_v3_Ngoc_Linh`.
- **Đoạn dài cần phát sớm:** `g-omnivoice-stream` với một trong hai giọng trên.
- **Phương án thay thế để nghe A/B:** `omnivoice-vietnamese` +
  `vieneu_v3_Ngoc_Tran`.
- Chưa chọn Doan_Trang hoặc Quang_Son làm giọng mặc định cho G-OmniVoice.

`configs/server.json` mặc định chỉ khai báo `g-omnivoice` và `g-omnivoice-stream`.
Muốn dùng `omnivoice-vietnamese` hoặc `omnivoice` thì thêm lại entry
theo [USAGE.md, mục 4.1.1](USAGE.md#411-thêm-lại-các-model-đã-bỏ).

Đây là khuyến nghị theo độ đầy đủ của từ qua ASR và các chỉ số audio.
**Chưa có đánh giá bằng tai người cho lượt mới để xếp hạng độ hay, tự nhiên,
ngữ điệu hoặc độ giống giọng.**

## 1. Nguồn dữ liệu và cách hiểu kết quả

Đánh giá dựa trên lượt ngày **17/10/2026**, được ghi nhận trong commit
`b9a984f`. Khi render, gateway ở commit nền `2d54936` cộng thay đổi bộ nghe
trong working tree; pipeline `5-4f53cda1`, sea-g2p 0.10.0. Upstream audio.cpp
ở `65df584a`, dùng checkpoint f16.

- Bốn entry model: `g-omnivoice`, `g-omnivoice-stream`,
  `omnivoice-vietnamese`, `omnivoice`.
- Năm giọng: Doan_Trang, Ngoc_Linh, Ngoc_Tran, Kim_Thanh, Quang_Son.
- Seed chính 42; seed 1 và 2 chỉ bổ sung cho các ca nghe thuộc câu hỏi 2/4/5,
  **không phải chạy lại toàn bộ golden ở ba seed**.
- ASR: gipformer RNN-T int8, sherpa-onnx, CPU.
- **112 run / 11.136 request / 11.126 WAV hợp lệ**, gồm cả gateway và direct.

| Bộ | Phạm vi | Request | WAV |
| --- | --- | ---: | ---: |
| `full` | 166 golden không lỗi + 43 ca listening, seed 42 | 8.232 | 8.222 |
| `production` | 45 golden tắt normalize được chạy lại theo mặc định model, seed 42 | 1.800 | 1.800 |
| `seed-1` | Ca listening thuộc câu hỏi 2/4/5 | 552 | 552 |
| `seed-2` | Ca listening thuộc câu hỏi 2/4/5 | 552 | 552 |

Các bảng so sánh bên dưới dùng **phía gateway**, không gộp điểm direct.

### Giới hạn của phép đo

- `word_recall` đo tỉ lệ từ của engine text xuất hiện đúng thứ tự trong transcript
  ASR. **0,945 không có nghĩa 94,5% câu đọc hoàn toàn đúng.**
- Recall không phạt chữ thừa/lặp từ và không đo ngữ điệu, nhiễu, độ giống giọng.
  ASR có thể viết tên riêng, tiếng Anh, chữ cái hoặc số khác chuỗi mong đợi.
- Cờ `words_missing` là recall dưới 0,8, **không phải xác nhận chắc chắn mất từ**.
- So sánh phản ánh **cấu hình triển khai hiện tại**, không phải benchmark thuần
  model: G-OmniVoice có speed bands; `omnivoice-vietnamese` chưa bật bands;
  `omnivoice` gốc tắt normalize theo config.
- `--normalize-on` của bộ nghe bỏ cờ tắt normalize của ca, không ép bật normalize
  cho model vốn mặc định tắt. Vì vậy bộ `production` của model gốc vẫn tắt.
- TTFB được đo khi nhiều model và ASR chạy song song; không phải benchmark độc lập
  tải và không đồng nghĩa thời gian tới tiếng nói đầu tiên.

## 2. So sánh model

Recall trung bình trên bộ golden đầy đủ: 166 ca × năm giọng, seed 42.
Loại ô ASR không chấm khỏi trung bình, không loại ca có điểm thấp.

| Model | Recall ASR TB | Đánh giá theo cấu hình hiện tại |
| --- | ---: | --- |
| `g-omnivoice` | **0,927** | Cân bằng tốt; ưu tiên mặc định cho tiếng Việt |
| `g-omnivoice-stream` | **0,930** | Chất lượng giữ từ gần offline; ưu tiên khi cần streaming đoạn dài |
| `omnivoice-vietnamese` | **0,919** | Phương án thay thế đáng giữ; cần chọn đúng giọng |
| `omnivoice` | **0,682** | Không ưu tiên cho tiếng Việt với config hiện tại |

Không coi chênh lệch 0,003 giữa hai entry G-OmniVoice là bằng chứng streaming có
chất lượng tốt hơn. Hai entry dùng cùng checkpoint nhưng chế độ chạy khác nhau.

### G-OmniVoice offline

- Khuyến nghị đầu tiên cho câu ngắn/vừa và khi không bắt buộc phát streaming.
- Các ca từng mất từ ở heading, list, xuống dòng đã cải thiện rõ.
- Bộ đoạn dài đạt recall trung bình 0,991.
- Vẫn có mất cụm từ ở một số giọng/câu ngắn; speed bands chỉ giảm lỗi, không bảo đảm
  mọi giọng/seed.

### G-OmniVoice streaming

Có lợi thế rõ về TTFB trong bộ đoạn dài:

| Model | Độ dài WAV min–max (s) | Recall TB | TTFB trung vị (ms) |
| --- | ---: | ---: | ---: |
| `g-omnivoice` | 49,48–62,77 | 0,991 | 6370 |
| `g-omnivoice-stream` | 50,00–63,48 | 0,990 | 1112 |
| `omnivoice-vietnamese` | 49,50–63,13 | 0,968 | 6769 |
| `omnivoice` | 44,06–56,40 | 0,879 | 5427 |

Mỗi hàng có năm mẫu gateway, seed 42. Cần nghe thêm chỗ nối chunk và trôi giọng.

**Không mặc định streaming cho mọi câu ngắn:**

- TTFB trung vị trên golden của hai entry G-OmniVoice gần nhau: 573 ms offline,
  569 ms streaming.
- Khoảng lặng đầu trung vị: 204 ms offline, khoảng 606 ms streaming.
- 248/830 mẫu golden streaming có cờ khoảng lặng đầu trên một giây; offline là
  2/830 mẫu.

Đây là số đo trên bộ thử, không suy ra mọi request production đều có phân bố này.
Cần đo thời gian tới tiếng nói đầu tiên ở client, kể cả buffer phát audio, trước
khi chọn cấu hình cho voice agent.

### OmniVoice Vietnamese

- Kết quả tổng thể thấp hơn G-OmniVoice một ít trong cấu hình hiện tại.
- Ngoc_Tran là tổ hợp nổi bật: recall golden 0,937.
- Với Doan_Trang, model này đạt 0,931 so với 0,920 của G-OmniVoice offline.

Nếu muốn giữ Doan_Trang, nên nghe A/B hai model. Chưa đủ bằng chứng để tự động
chuyển model hoặc kết luận tổ hợp nào nghe tự nhiên hơn.

### OmniVoice gốc

Không ưu tiên cho tác vụ tiếng Việt hiện tại:

- Recall thấp hơn rõ trên golden đầy đủ.
- Có lỗi upstream với `language=vi-VN` ở cả gateway và direct.
- Tuy nhiên, normalize đang tắt nên không quy toàn bộ chênh lệch cho checkpoint.
- Lượt này không đủ để đánh giá khả năng đa ngôn ngữ hoặc kết luận bản gốc kém
  hơn trong mọi tác vụ.

## 3. So sánh giọng

### Với G-OmniVoice offline

Các số dưới đây đều ở seed 42. Hai cột là hai bộ câu khác nhau, không dùng để suy
ra mức cải thiện chung khi bật normalize.

| Giọng | Recall golden đầy đủ (166 ca) | Recall bộ production bổ sung (45 ca) | Khuyến nghị |
| --- | ---: | ---: | --- |
| **Kim_Thanh** | **0,945** | **0,964** | Ưu tiên đầu tiên theo số liệu |
| **Ngoc_Linh** | **0,940** | **0,963** | Lựa chọn gần tương đương |
| Ngoc_Tran | 0,928 | 0,958 | Dự phòng tốt |
| Doan_Trang | 0,920 | 0,932 | Dùng có chọn lọc; kiểm câu ngắn |
| Quang_Son | 0,900 | 0,876 | Chưa chọn làm mặc định |

Kim_Thanh và Ngoc_Linh chênh lệch nhỏ; chưa có kiểm định để khẳng định khác biệt
có ý nghĩa thống kê. Nên chọn giữa hai giọng bằng nghe thực tế theo nhu cầu.

### Giọng phụ thuộc model

Không xếp hạng voice độc lập với model:

- `omnivoice-vietnamese` + Ngoc_Tran đạt recall golden **0,937**.
- Cùng model đó, Doan_Trang đạt **0,931**, Ngoc_Linh **0,921**,
  Kim_Thanh **0,905**, Quang_Son **0,898**.
- Kim_Thanh đứng đầu trên G-OmniVoice không đồng nghĩa cũng đứng đầu ở mọi model.

Chưa có đủ đánh giá cảm nhận để mô tả giọng nào ấm hơn, tự nhiên hơn hoặc hợp
thương hiệu hơn; không suy diễn các thuộc tính này từ điểm ASR.

## 4. Những gì đã cải thiện và rủi ro còn lại

### Đã cải thiện

Trên G-OmniVoice offline, năm giọng, seed 42, các ca sau đạt recall 1,000:

- `terminal-ellipsis-quote`, `paragraph-lines`, `heading`, `fence-closed`,
  `thematic-break`, `numbered-list`, `bullets`.
- `vi-quarter`, `vi-time-range`, `vi-superscript`, `vi-hotline`.

Điều này hỗ trợ kết luận sửa lỗi có ích; không bảo đảm mọi câu cùng dạng đều đúng.

### Chưa giải quyết triệt để

| Rủi ro | Bằng chứng / giới hạn | Hướng sử dụng |
| --- | --- | --- |
| Mất cụm trong câu ngắn | `vi-short-three`: Doan_Trang thiếu “xong rồi”, Quang_Son thiếu “đi thôi” | Không cam kết đọc nguyên văn tuyệt đối; kiểm câu production quan trọng |
| Bỏ cụm sau khi làm sạch | Ca emoji ở Quang_Son chỉ còn “cảm ơn” | Không chọn Quang_Son làm mặc định hiện tại |
| Tên sản phẩm | `iPhone` vẫn có known issue; #17 chưa triển khai từ điển | Cách đọc mới trong bộ nghe chỉ là ứng viên, chưa là chính sách production |
| Non-verbal tag | Giữ tag đúng không bảo đảm model tạo tiếng cười/thở dài | Chưa cam kết tag là chức năng tin cậy |
| Lặp từ, ngữ điệu, giống giọng | Recall không đo được đầy đủ | Cần đánh giá bằng tai, đặc biệt clone và text dài |
| Mã, số tiền, thông tin quan trọng | Điểm trung bình không bảo đảm từng chữ số đúng | Kiểm riêng tập câu thực tế và nghe lại trước khi dùng trong nghiệp vụ |

**Phân biệt lỗi tổng hợp và lỗi chấm:**

- 10 lỗi tổng hợp đều là HTTP 500 ở `omnivoice` gốc với `language=vi-VN`,
  hai phía × năm giọng. Không đổi ngôn ngữ caller trái spec để che lỗi.
- 120 cờ `asr_failed` thuộc ba ca cố ý gửi tiếng Anh: gipformer chỉ hỗ trợ tiếng
  Việt. Audio vẫn được lưu; các ô này không tham gia trung bình recall.
- Ba entry finetune không có lỗi tổng hợp trong kế hoạch này.
- 371 tests pass xác nhận hành vi phần mềm, **không phải chứng nhận chất lượng nghe**.
  Đóng #7 là bàn giao công việc, không xác nhận đã chốt cả năm chính sách nghe.

## 5. Khuyến nghị triển khai

| Nhu cầu | Model + giọng |
| --- | --- |
| Tiếng Việt thông thường, câu ngắn/vừa | **`g-omnivoice` + `vieneu_v3_Kim_Thanh`** |
| Lựa chọn giọng thứ hai | `g-omnivoice` + `vieneu_v3_Ngoc_Linh` |
| Đoạn dài cần phát sớm | `g-omnivoice-stream` + Kim_Thanh hoặc Ngoc_Linh |
| Thử thêm model/giọng | `omnivoice-vietnamese` + `vieneu_v3_Ngoc_Tran` |
| Muốn giữ Doan_Trang | Nghe A/B G-OmniVoice và OmniVoice Vietnamese trước khi chốt |
| Tag bắt buộc đúng hoặc đọc nguyên văn tuyệt đối | Chưa có tổ hợp được xác nhận đáp ứng |

`omnivoice-vietnamese` không có trong config mặc định; thêm
lại theo [USAGE.md, mục 4.1.1](USAGE.md#411-thêm-lại-các-model-đã-bỏ).

### Cấu hình nên giữ

- Gọi qua gateway, giữ normalize tiếng Việt bật cho các entry finetune.
- Giữ speed bands hiện tại cho G-OmniVoice: 0,7 với engine text ≤ 8 âm tiết,
  0,6 với 9–12 âm tiết; trên ngưỡng không thêm speed.
- Không ép speed thấp cho mọi câu: dư thời lượng có thể gây lặp từ.
- Nếu caller chỉ định `speed` hoặc duration thì giá trị đó được ưu tiên; không
  mặc định cho rằng speed bands vẫn còn tác dụng.
- Gateway tiếp tục một request upstream cho mỗi request caller; không thêm
  chia/ghép audio hay retry để che lỗi model.

Chi tiết: [config gateway](../configs/gateway.toml),
[ADR 0004](adr/0004-short-text-speed.md).

### Điều kiện trước khi mở production rộng

1. Nghe Kim_Thanh và Ngoc_Linh trên tập câu thực tế: lời chào, câu rất ngắn, tên
   riêng, số tiền, điện thoại/mã, nhiều câu ngắn và đoạn dài.
2. Chạy nhiều seed trên chính tập đó; kiểm cả mất từ **và chữ thừa/lặp**, không chỉ
   recall trung bình.
3. Nghe chỗ nối chunk, khoảng lặng đầu và độ giống giọng clone; chốt cách đọc
   viết tắt/từ mượn trước khi triển khai từ điển mặc định.
4. Đo thời gian tới tiếng nói đầu tiên, độ trễ p50/p95, timeout và tải đồng thời.
5. Đặt ngưỡng chấp nhận theo nghiệp vụ; chỉ mở rộng khi tập kiểm thực tế đạt ngưỡng.

**Quyết định sử dụng hiện tại:** ưu tiên beta với G-OmniVoice + Kim_Thanh,
giữ Ngoc_Linh làm lựa chọn thứ hai. Không xem HTTP thành công hoặc recall cao là
bảo đảm toàn bộ nội dung đã được đọc đúng.

## 6. Báo cáo và dữ liệu đính kèm

- Báo cáo chi tiết, phương pháp đối chứng, năm câu hỏi còn cần nghe:
  [lượt nghe lại #7](reports/issue7-listening.md).
- Trang A/B local: `outputs/issue7-20261017/index.html`.
- Dưới từng bộ `full/`, `production/`, `seed-1/`, `seed-2/`:
  - `summary.csv`: kết quả theo model/phía.
  - `voices.csv`: kết quả theo model/giọng/phía.
  - `cases.csv`: ca cần xem lại trên nhiều giọng.
  - `all_results.csv`: từng request, transcript ASR và chỉ số audio.
- `plan.json`: lệnh, commit và hash source/config/voice; `source/`: bản source
  dùng trong lượt render; `completion.json`: xác nhận đủ kế hoạch.
- WAV, CSV và HTML nằm trong `outputs/` local, được gitignore; tài liệu này không
  đính kèm audio lên GitHub.

Mở trang nghe từ thư mục gốc repo:

```bash
python3 -m http.server --bind 127.0.0.1 \
  --directory outputs/issue7-20261017 9000
```

Truy cập `http://127.0.0.1:9000`; nếu làm việc trên máy từ xa, dùng port forwarding.
