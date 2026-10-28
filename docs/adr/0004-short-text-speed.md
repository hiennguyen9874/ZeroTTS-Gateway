---
status: superseded by ADR-0005
---

# Operator có thể cho gateway đặt `speed` chậm hơn cho engine text ngắn

Với model có `short_text_speed` trong config của gateway, nếu caller không gửi `speed` lẫn `options.duration_sec` (hay `options.duration`) và engine text có không quá `short_text_max_syllables` âm tiết (mặc định 12; đếm từ có chữ hoặc số, bỏ non-verbal tag), gateway thêm `"speed": short_text_speed` vào request gửi lên upstream. Không cấu hình dạng scalar lẫn band thì gateway không đụng vào tham số sinh. Đây là dạng scalar ban đầu (#18), vẫn tương thích. Sau #19, `configs/gateway.toml` dùng dạng band dưới đây cho `g-omnivoice` và `g-omnivoice-stream`.

Lý do (#18): audio.cpp ước lượng thời lượng audio từ trọng số ký tự của engine text, quy theo tốc độ đọc của giọng mẫu; dưới 2 s thì kéo lên theo căn bậc ba. Model lại chèn một khoảng lặng trước cụm đầu tiên và một khoảng ngắt ở mỗi dấu phẩy, nhưng ước lượng chỉ tính dấu câu bằng nửa chữ cái. Với text ngắn (khoảng 2 s), khi tắt postprocess thấy audio bị lỗi bắt đầu bằng 1,2–1,4 s im lặng, phần còn lại chỉ đủ cho cụm sau. Kết quả là cụm đầu bị bỏ (`Giới thiệu, Nội dung chính.` thành "nội dung chính"). Lỗi xảy ra cả với câu thường (`Xin chào, hôm nay chúng ta học bài mới.` mất "xin chào" ở Quang_Son 3/3 seed), không riêng text do bước làm sạch markup tạo ra. Trong bộ ca #18, text dài (từ khoảng 18 âm tiết) không thấy lỗi này; mở rộng nhiều seed ở #19 vẫn có lỗi nên đây không phải ngưỡng bảo đảm.

Thử nghiệm (6 giọng × 3 seed, gọi thẳng upstream, tỉ lệ từ ASR nghe được):

- Trên 9 ca ngắn và vừa (≤ 12 âm tiết), `speed` 0,7 đưa 8 ca lên 1,000; mặc định chỉ được 0,88–0,98. Ca còn lại, `Xin chào, hôm nay chúng ta học bài mới.`, lên từ 0,951 (14/18) thành 0,975 (16/18), vẫn còn mất "xin chào" ở Quang_Son và Thien_Tam_Duc, mỗi giọng một seed. 0,8 và 0,75 vẫn còn sót (`Cần mang, Áo mưa, Đèn pin, Nước uống.` được 0,944 và 0,986).
- Không giúp: tắt postprocess (giống hệt mặc định), `denoise: false`, 64 bước, chữ thường sau dấu phẩy, sửa `reference_text` của giọng mẫu cho khớp audio (có ca lên, có ca xuống).
- Không phải do giọng mẫu đọc nhanh: Doan_Trang và Quang_Son có tốc độ đọc (trọng số/giây) thấp hơn Mai_Anh, Thuy_Dung.
- `options.duration_sec` cũng chữa được, nhưng gateway không biết thời lượng ước lượng của upstream (cần số frame của audio mẫu sau tiền xử lý), nên đặt một giá trị cố định có thể nén text dài hơn. `speed` chia đều ước lượng của upstream nên an toàn hơn; audio.cpp cũng quy `duration_sec` về `speed`.

## Cập nhật #19: band theo độ dài engine text

Operator có thể dùng `short_text_speed_bands` **thay cho** `short_text_speed`:

```toml
[models.g-omnivoice]
language = "vi"
short_text_speed_bands = [
  { max_syllables = 8, speed = 0.7 },
  { max_syllables = 12, speed = 0.6 },
]
```

- Ngưỡng là số nguyên dương, tăng **nghiêm ngặt**; tốc độ phải hữu hạn và dương. Hai dạng không được bật cùng lúc. Band rỗng tương đương chưa cấu hình band.
- Đếm trên engine text như trước, bỏ non-verbal tag. Chọn band đầu tiên có `max_syllables` ≥ số âm tiết; vượt band cuối thì không thêm `speed`. `short_text_max_syllables` chỉ áp cho dạng scalar.
- Điều kiện bảo toàn `speed` và `options.duration_sec`/`duration` của caller không đổi. Mặc định code không có band hay tốc độ nào; giá trị 8/12 và 0.7/0.6 hoàn toàn do operator chọn, không phải luật theo từ “xin chào”.
- Câu 9 âm tiết còn mất lời chào ở 0.7; 0.6 khôi phục 18/18 mẫu ở seed 42, 1, 2. **Không hạ toàn bộ xuống 0.6**: trong dữ liệu quét, `short-lines` có thể lặp “tạm biệt” khi dư thời lượng, dù recall vẫn 1.000. Giữ nguyên 0.7 cho các ca ≤ 8 âm tiết của #18, chỉ giảm cho 9–12 âm tiết. Không nâng ngưỡng lên 18 hay 24: dữ liệu dài hơn vẫn có lỗi nhưng cũng có thêm chữ khi kéo chậm, không chứng minh được lợi ích chung.
- Giả thuyết “speed bị căn bậc ba làm yếu trên 2 s” là **sai**: `estimate_target_tokens` chia ước lượng **sau** `RuleDurationEstimator`; speed 0.6 kéo dài đầy đủ theo tỉ lệ 1/0.6, chỉ bị làm tròn frame. Tắt postprocess không khôi phục lời chào. Ước lượng thiếu thời gian là yếu tố có thể giảm bằng speed, nhưng chưa chứng minh được nguyên nhân bên trong model; với nhiều seed hơn vẫn có mất/lặp chữ, không coi đây là sửa triệt để upstream.
- Engine text và pipeline version không đổi, nên không thêm golden của text pipeline. Ca nghe `short-greeting` trong `scripts/listening_cases.json` mang `why`; test HTTP khóa lựa chọn band, ngưỡng 8/9/12/13, quyền ưu tiên của caller và mặc định tắt.

Kết quả, giới hạn và cách chạy lại: [báo cáo #19](../reports/issue19-short-greeting.md).

## Cập nhật #22: ramp sau band

Operator có thể thêm `short_text_speed_ramp` (tắt mặc định), dùng **cùng** band hoặc dạng scalar, hoặc đứng riêng:

```toml
short_text_speed_ramp = { from_syllables = 13, to_syllables = 24, from_speed = 0.6, to_speed = 1.0 }
```

- Lý do: band kết thúc ở 12 âm tiết nên 12 → 0.6 rồi 13 → không speed là một bước nhảy lớn. Ước lượng của audio.cpp với Ngoc_Linh cho quỹ thời gian ~4,6 s ở 12 âm tiết nhưng ~4,0 s ở 17 âm tiết. vietvoice-lab gửi mỗi piece một request, nên reply xen câu ngắn và câu dài nghe nhanh chậm không đều (vietvoice-lab `docs/settings/04-tts-speech.md` §3).
- Áp khi dạng ngắn (band hoặc scalar) không chọn speed và engine text có từ `from_syllables` tới `to_syllables` âm tiết (cả hai đầu): speed nội suy tuyến tính từ `from_speed` tới `to_speed`, làm tròn 3 chữ số (18 âm tiết → 0.782). Ngoài khoảng thì không thêm speed. Ở `to_syllables` gateway gửi `to_speed` (1.0 tương đương mặc định upstream).
- Validate chặt như band: ngưỡng là số nguyên dương, `from_syllables < to_syllables`, speed hữu hạn và dương, và ramp phải bắt đầu **trên** ngưỡng của dạng ngắn (band cuối hoặc `short_text_max_syllables`). Quyền ưu tiên của caller (`speed`, `options.duration_sec`/`duration`), cách đếm engine text và log `short_text_speed` giữ như cũ. Engine text và pipeline version không đổi.
- `configs/gateway.toml` chỉ ghi ramp dưới dạng comment, **không bật**. #19 từng bác bỏ việc nâng ngưỡng band lên 18/24 vì kéo chậm text dài hơn đôi khi thêm chữ. Ramp chỉ kéo chậm một phần (0.6 → 1.0), nhưng trước khi bật phải nghe/ASR riêng khoảng 13–24 âm tiết để tìm chữ lặp/thừa, và qua A/B 2×2 của vietvoice-lab (#147: short-piece merge × band/ramp).

## Considered Options

- Luật text ở gateway. Bị loại vì không có cách viết nào làm dài ước lượng mà không đổi nội dung đọc; chữ thường sau dấu phẩy và các dấu câu khác không giúp.
- Gateway luôn đặt `speed` theo model. Bị loại vì đổi nhịp đọc của mọi câu, kể cả text dài vốn không lỗi, và đi ngược user story 30.
- Đổi giọng mẫu trong `voices/`. Bị loại vì lỗi có ở cả giọng khác (Mai_Anh mất `Cần mang` 3/3 seed), không riêng hai giọng.
- Chỉ báo upstream. Vẫn nên báo (ước lượng thời lượng không tính khoảng lặng đầu audio và khoảng ngắt sau dấu câu), nhưng không chờ được.

## Consequences

- Đây là tuỳ chọn của operator, mặc định tắt trong code. `speed` và `options.duration_sec` caller tự gửi luôn được chuyển nguyên (user story 30).
- Ở 0.7, phần tiếng nói của text ngắn dài thêm khoảng 25–35% so với mặc định của upstream, gần tốc độ đọc của giọng mẫu hơn (mặc định text ngắn bị đọc dồn). Band 0.6 cấp thêm thời lượng cho 9–12 âm tiết; đổi lại có thể tăng độ trễ hoặc lặp chữ, cần nghe thử với giọng mới.
- Ngưỡng tính trên engine text, nên số đã chuẩn hoá được đếm theo âm tiết đọc ra (`41` thành "bốn mươi mốt", 3 âm tiết).
- Log của request ghi `short_text_speed` khi gateway đặt `speed`. Pipeline version không đổi vì engine text không đổi.
- Model mới hoặc giọng mới cần nghe thử lại trước khi bật; giá trị chỉ mới đo trên `g-omnivoice` (offline và streaming).
