# omnivoice-gateway

Các khái niệm trong luồng chuẩn bị văn bản tiếng Việt trước khi tổng hợp giọng bằng OmniVoice.

## Language

**Gateway**:
Dịch vụ đứng trước upstream, nhận request tổng hợp giọng với cùng hợp đồng API, chuẩn bị văn bản và chuyển tiếp âm thanh; không tự tổng hợp giọng.
_Avoid_: Facade, proxy

**Upstream**:
Máy chủ tổng hợp giọng thật sự đứng sau gateway (audio.cpp chạy OmniVoice).
_Avoid_: Engine, backend

**Caller**:
Ứng dụng gửi request tới gateway, ví dụ worker của voice agent hoặc công cụ đọc văn bản dài.

**Operator**:
Người triển khai và vận hành gateway cùng upstream; sở hữu cấu hình, từ điển phát âm và bí mật.

**Input text**:
Văn bản caller gửi trong request, ở dạng viết.
_Avoid_: Raw text, display text

**Engine text**:
Văn bản sau text pipeline, là thứ gateway thực sự gửi cho upstream.
_Avoid_: Spoken text, normalized text

**Text pipeline**:
Chuỗi bước có thứ tự biến input text thành engine text: làm sạch, áp từ điển phát âm, chuẩn hóa dạng viết sang dạng nói.

**Effective language**:
Ngôn ngữ dùng để quyết định cách chuẩn bị văn bản cho một request: ngôn ngữ caller khai báo, nếu thiếu thì lấy mặc định của model.

**Text warning**:
Mã cố định báo rằng gateway đã bỏ hoặc gỡ một phần input text (ví dụ URL, code block); không chứa nội dung văn bản.

**Non-verbal tag**:
Thẻ trong ngoặc vuông mà model hiểu như một chỉ dẫn, ví dụ `[laughter]` hoặc phiên âm `[B EY1 S]`; phải đi qua text pipeline nguyên vẹn.
