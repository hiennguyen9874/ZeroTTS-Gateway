# Chọn model và voice ZeroTTS

Dùng `zerotts` cho WAV đầy đủ, `zerotts-stream` cho SSE hoặc PCM streaming. Cả hai trả audio mono 48 kHz. Voice là tên preset pack trong `models/ZeroTTS/voices`; xem `GET /v1/audio/voices`. Nếu caller không chọn voice, server dùng `default_voice` theo model trong `configs/zerotts_server.toml`.

Bộ weights có thể gồm `maichi` và `quangminh`; cần nghe thử trên dữ liệu và node thực tế trước khi chọn voice mặc định. ZeroTTS không hỗ trợ clone giọng. Thêm voice pack rồi khởi động lại server như trong [ZEROTTS_SERVER.md](ZEROTTS_SERVER.md#config).

A/B text pipeline #9 và giới hạn đánh giá nằm trong [báo cáo](reports/issue9-zerotts-text-pipeline-ab.md); pipeline đầy đủ hiện được bật cho cả hai model. Các đánh giá OmniVoice trước đây được giữ trong `docs/reports/` như tư liệu lịch sử, không phải khuyến nghị triển khai hiện tại.
