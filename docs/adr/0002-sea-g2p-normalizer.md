# Dùng Normalizer của sea-g2p làm bộ chuẩn hóa tiếng Việt, không dùng G2P

Khi effective language là `vi`, gateway chuẩn hóa dạng viết sang dạng nói bằng `sea_g2p.Normalizer("vi").normalize(text, punc_norm=False)` (pin version) và gửi **chữ** (không phải phoneme) cho OmniVoice. audio.cpp không chuẩn hóa tiếng Việt, còn `normalize_text` của OmniVoice Python chỉ thay `\d+` bằng `num2words`, nên ra kết quả như `năm mươi.khôngđ`.

Chạy thử trên cùng bộ câu, sea-g2p đọc đúng ngày giờ, tiền, phần trăm, khoảng số, tỉ số (`3-1`), đơn vị (`1m75`, `37°C`), `Q1/2024`, số điện thoại. vietnormalizer sai ở chữ số đi kèm đơn vị, tỉ số và ký hiệu. sea-g2p cũng là bộ VieNeu dùng, và có C ABI nếu sau này cần bỏ Python.

## Considered Options

- vietnormalizer: nhẹ, có từ điển phiên âm Anh sang Việt (`iPhone` → `ai-phôn`), nhưng sai nhiều ở các ca trên. Chỉ đáng giữ làm nguồn tham khảo cho lexicon.
- Tự viết luật theo chương 19 của llm-wiki: kiểm soát hoàn toàn, nhưng tốn công và khó phủ đủ ca.

## Consequences

Gateway phải bù những chỗ sea-g2p không làm, hoặc làm sai với OmniVoice:

- Xóa non-verbal tag (`[laughter]` thành `laughter,`). Gateway tách tag ra trước khi normalize rồi chèn lại sau.
- Đọc cú pháp markdown (`**` thành "sao sao", `#` thành "thăng") và đánh vần URL. Gateway làm sạch những thứ này trước khi normalize.
- Bọc tiếng Anh trong `<en>…</en>` và tách viết tắt thành `a i`. Gateway gỡ thẻ; viết tắt bị tách được ghép lại dạng hoa (`AI`). Cách đọc cuối cùng chốt sau khi nghe thử, còn lexicon dùng để override.
- `punc_norm=True` gộp câu ngắn bằng dấu phẩy, làm hỏng ranh giới chunk (xem ADR 0001), nên không dùng.
- Toàn bộ output là chữ thường. Hiện chấp nhận, cần kiểm bằng nghe thử.
- Lexicon chạy **trước** normalize, nên phần thay thế phải là dạng nói tiếng Việt để đi qua normalize mà không bị đổi.
- ZeroTTS (ADR 0005) cũng dùng normalizer này, sau A/B #9 ([báo cáo](../reports/issue9-zerotts-text-pipeline-ab.md)): WER 0,072 so với 0,167 khi gửi văn bản thô. Chữ thường hoá toàn bộ không làm tăng lỗi đo được. Còn một chỗ hại: viết tắt bị đánh vần thành tên chữ cái (`PIN` → `phê i nờ`, `OTP` → `ô tê phê`) thì ZeroTTS đọc kém hơn để nguyên viết tắt. Hai ca này được ghi thành known issue trong golden.
