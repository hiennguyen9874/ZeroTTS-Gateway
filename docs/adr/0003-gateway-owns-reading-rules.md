# Gateway tự giữ reading rule trước sea-g2p, không chờ sửa ở sea-g2p

Khi effective language là `vi`, gateway chạy một bước **reading rule** sau từ điển phát âm và ngay trước Normalizer của sea-g2p. Mỗi reading rule khớp theo mẫu một dạng viết mà sea-g2p đọc sai, rồi viết lại thành dạng nói tiếng Việt mà sea-g2p để nguyên. Gateway giữ các luật này trong code của mình, không gửi bản sửa lên sea-g2p rồi chờ bản phát hành mới.

Lý do: một số lỗi của sea-g2p phá ranh giới câu mà ADR 0001 dựa vào. Ví dụ `v2.0` bị đọc thành "vê hai. không", tức có thêm một dấu chấm giữa câu, nên chunker của audio.cpp cắt sai chỗ. Từ điển phát âm (ADR 0002) không sửa được loại lỗi này vì nó chỉ khớp từ cố định, còn lỗi nằm ở cả một mẫu (`\d+k`, `8h-17h`, `1900 1234`). Gateway đã pin version sea-g2p, nên các luật này ổn định và có golden canh giữ.

## Considered Options

- Gửi PR lên sea-g2p và để lỗi ở dạng known issue cho tới khi có bản mới. Bị loại vì gateway không kiểm soát được thời điểm phát hành, và lỗi ranh giới câu không thể chờ.
- Kết hợp: gateway sửa một danh sách đóng, phần còn lại gửi PR lên sea-g2p. Bị loại để chỉ có một nơi sở hữu cách đọc, và không phải theo dõi hai nơi.
- Đặt luật trước từ điển phát âm. Bị loại vì operator phải luôn ghi đè được cách đọc bằng từ điển.
- Vá output của sea-g2p. Bị loại vì sau khi normalize thì dạng viết gốc (`v2.0`, `50k`) đã mất.

## Consequences

- Thứ tự text pipeline trong spec #1 có thêm một bước: từ điển phát âm, rồi reading rule, rồi sea-g2p.
- Output của reading rule phải đi qua sea-g2p mà không bị đổi, giống ràng buộc với từ điển phát âm. Mỗi luật có golden kèm `why`, và golden kiểm cả câu đi qua sea-g2p.
- Mỗi dạng viết chỉ có một luật sở hữu. Ví dụ `3.5` đọc "phẩy" nhưng `v2.0` và `GPT-3.5` thuộc luật phiên bản và đọc "chấm". Luật nào chồng lên luật khác thì phải có ca golden chứng minh thứ tự.
- Reading rule không phát text warning. Text warning vẫn chỉ báo phần input bị bỏ hoặc gỡ.
- Nâng version sea-g2p thì phải chạy lại golden. Luật nào sea-g2p đã tự sửa đúng thì xoá đi.
- Đổi luật thì phải tăng `RULES_VERSION`, để pipeline version truy được lỗi đọc sai.
