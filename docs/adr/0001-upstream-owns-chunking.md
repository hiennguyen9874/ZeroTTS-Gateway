# audio.cpp sở hữu việc chia chunk, gateway gửi một request upstream cho mỗi request của caller

Gateway chuẩn hóa toàn bộ input text rồi gửi **đúng một** request tới audio.cpp. Gateway không chia text thành nhiều request upstream và không ghép audio. Text dài được audio.cpp tự chia: ở chế độ offline nó tự chia khi audio ước lượng dài hơn `audio_chunk_threshold_sec`, ở model `mode: streaming` nó chia theo `text_chunk_size`. Audio trả về được chuyển tiếp nguyên byte, không resample, không chỉnh loudness, không sửa header WAV.

Lý do: khi request không có `voice_ref` (auto voice hoặc voice design), audio.cpp dùng audio của chunk 0 làm reference cho các chunk sau để giữ giọng nhất quán (`references/audio.cpp/src/models/omnivoice/session.cpp`, khoảng dòng 439–504; OmniVoice Python làm tương tự). Nếu gateway tự tách thành nhiều request độc lập thì cơ chế này mất, mỗi chunk ra một giọng khác, và gateway phải tự ghép WAV, chèn khoảng lặng, sửa header.

## Considered Options

- Gateway chia text thành nhiều request rồi ghép audio. Cách này cho phép tạo pseudo-streaming từ model offline và tiết kiệm VRAM vì không cần entry streaming riêng. Bị loại vì làm vỡ giọng với auto/design voice, và vì phải thêm logic ghép audio.
- Gateway chỉ chia khi request có `voice_ref`/`voice`. Bị loại vì có hai đường xử lý cho cùng một API.

## Consequences

- Muốn streaming thì caller phải gọi một model id có `mode: "streaming"` trong config của audio.cpp (ví dụ `g-omnivoice-stream`). Entry này có session và VRAM riêng.
- Ranh giới chunk do chunker của audio.cpp quyết định, dựa trên dấu câu trong engine text. Vì vậy gateway phải giữ dấu kết câu: dùng `punc_norm=False` của sea-g2p và tự thêm dấu kết câu khi thiếu.
- Voice agent vẫn nên tự chia theo câu ở phía caller và gửi từng câu một. Gateway không quản lý lượt nói hay ngắt lời.
