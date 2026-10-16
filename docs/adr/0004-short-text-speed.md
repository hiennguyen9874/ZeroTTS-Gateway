# Operator có thể cho gateway đặt `speed` chậm hơn cho engine text ngắn

Với model có `short_text_speed` trong config của gateway, nếu caller không gửi `speed` lẫn `options.duration_sec` (hay `options.duration`) và engine text có không quá `short_text_max_syllables` âm tiết (mặc định 12; đếm từ có chữ hoặc số, bỏ non-verbal tag), gateway thêm `"speed": short_text_speed` vào request gửi lên upstream. Không đặt `short_text_speed` thì gateway không đụng vào tham số sinh. `configs/gateway.toml` của repo bật `0.7` cho `g-omnivoice` và `g-omnivoice-stream`.

Lý do (#18): audio.cpp ước lượng thời lượng audio từ trọng số ký tự của engine text, quy theo tốc độ đọc của giọng mẫu; dưới 2 s thì kéo lên theo căn bậc ba. Model lại chèn một khoảng lặng trước cụm đầu tiên và một khoảng ngắt ở mỗi dấu phẩy, nhưng ước lượng chỉ tính dấu câu bằng nửa chữ cái. Với text ngắn (khoảng 2 s), khi tắt postprocess thấy audio bị lỗi bắt đầu bằng 1,2–1,4 s im lặng, phần còn lại chỉ đủ cho cụm sau. Kết quả là cụm đầu bị bỏ (`Giới thiệu, Nội dung chính.` thành "nội dung chính"). Lỗi xảy ra cả với câu thường (`Xin chào, hôm nay chúng ta học bài mới.` mất "xin chào" ở Quang_Son 3/3 seed), không riêng text do bước làm sạch markup tạo ra. Text dài (từ khoảng 18 âm tiết) không thấy lỗi này.

Thử nghiệm (6 giọng × 3 seed, gọi thẳng upstream, tỉ lệ từ ASR nghe được):

- Trên 9 ca ngắn và vừa (≤ 12 âm tiết), `speed` 0,7 đưa 8 ca lên 1,000; mặc định chỉ được 0,88–0,98. Ca còn lại, `Xin chào, hôm nay chúng ta học bài mới.`, lên từ 0,951 (14/18) thành 0,975 (16/18), vẫn còn mất "xin chào" ở Quang_Son và Thien_Tam_Duc, mỗi giọng một seed. 0,8 và 0,75 vẫn còn sót (`Cần mang, Áo mưa, Đèn pin, Nước uống.` được 0,944 và 0,986).
- Không giúp: tắt postprocess (giống hệt mặc định), `denoise: false`, 64 bước, chữ thường sau dấu phẩy, sửa `reference_text` của giọng mẫu cho khớp audio (có ca lên, có ca xuống).
- Không phải do giọng mẫu đọc nhanh: Doan_Trang và Quang_Son có tốc độ đọc (trọng số/giây) thấp hơn Mai_Anh, Thuy_Dung.
- `options.duration_sec` cũng chữa được, nhưng gateway không biết thời lượng ước lượng của upstream (cần số frame của audio mẫu sau tiền xử lý), nên đặt một giá trị cố định có thể nén text dài hơn. `speed` chia đều ước lượng của upstream nên an toàn hơn; audio.cpp cũng quy `duration_sec` về `speed`.

## Considered Options

- Luật text ở gateway. Bị loại vì không có cách viết nào làm dài ước lượng mà không đổi nội dung đọc; chữ thường sau dấu phẩy và các dấu câu khác không giúp.
- Gateway luôn đặt `speed` theo model. Bị loại vì đổi nhịp đọc của mọi câu, kể cả text dài vốn không lỗi, và đi ngược user story 30.
- Đổi giọng mẫu trong `voices/`. Bị loại vì lỗi có ở cả giọng khác (Mai_Anh mất `Cần mang` 3/3 seed), không riêng hai giọng.
- Chỉ báo upstream. Vẫn nên báo (ước lượng thời lượng không tính khoảng lặng đầu audio và khoảng ngắt sau dấu câu), nhưng không chờ được.

## Consequences

- Đây là tuỳ chọn của operator, mặc định tắt trong code. `speed` và `options.duration_sec` caller tự gửi luôn được chuyển nguyên (user story 30).
- Text ngắn được đọc chậm hơn khoảng 25–35% so với mặc định của upstream, gần tốc độ đọc của giọng mẫu hơn (mặc định text ngắn bị đọc dồn).
- Ngưỡng tính trên engine text, nên số đã chuẩn hoá được đếm theo âm tiết đọc ra (`41` thành "bốn mươi mốt", 3 âm tiết).
- Log của request ghi `short_text_speed` khi gateway đặt `speed`. Pipeline version không đổi vì engine text không đổi.
- Model mới hoặc giọng mới cần nghe thử lại trước khi bật; giá trị chỉ mới đo trên `g-omnivoice` (offline và streaming).
