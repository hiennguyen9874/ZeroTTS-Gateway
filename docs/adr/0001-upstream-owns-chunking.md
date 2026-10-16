# audio.cpp sở hữu việc chia chunk, gateway gửi một request upstream cho mỗi request của caller

Gateway chuẩn hóa toàn bộ input text rồi gửi **đúng một** request tới audio.cpp. Gateway không chia text thành nhiều request upstream và không ghép audio. Text dài được audio.cpp tự chia: ở chế độ offline nó tự chia khi audio ước lượng dài hơn `audio_chunk_threshold_sec`, ở model `mode: streaming` nó chia theo `text_chunk_size`. Audio trả về được chuyển tiếp nguyên byte, không resample, không chỉnh loudness, không sửa header WAV.

Lý do: khi request không có `voice_ref` (auto voice hoặc voice design), audio.cpp dùng audio của chunk 0 làm reference cho các chunk sau để giữ giọng nhất quán (`references/audio.cpp/src/models/omnivoice/session.cpp`, khoảng dòng 439–504; OmniVoice Python làm tương tự). Nếu gateway tự tách thành nhiều request độc lập thì cơ chế này mất, mỗi chunk ra một giọng khác, và gateway phải tự ghép WAV, chèn khoảng lặng, sửa header.

## Considered Options

- Gateway chia text thành nhiều request rồi ghép audio. Cách này cho phép tạo pseudo-streaming từ model offline và tiết kiệm VRAM vì không cần entry streaming riêng. Bị loại vì làm vỡ giọng với auto/design voice, và vì phải thêm logic ghép audio.
- Gateway chỉ chia khi request có `voice_ref`/`voice`. Bị loại vì có hai đường xử lý cho cùng một API.

## Consequences

- Muốn streaming thì caller phải gọi một model id có `mode: "streaming"` trong config của audio.cpp (ví dụ `g-omnivoice-stream`). Entry này có session và VRAM riêng.
- Ranh giới chunk do chunker của audio.cpp quyết định, dựa trên dấu câu trong engine text. Vì vậy gateway phải giữ dấu kết câu: dùng `punc_norm=False` của sea-g2p và tự thêm dấu kết câu khi thiếu.
- Ranh giới do markup tạo ra (mục list, dòng, heading) không phải lúc nào cũng là dấu chấm. Dòng ngắn (≤ 4 âm tiết) chưa có dấu câu riêng được nối với phần sau bằng `, `; câu dẫn ngắn kết bằng `:` (như `Cần mang:` trước list) cũng đổi `:` thành `,`; dòng dài hơn, hoặc đã có dấu câu khác, giữ nguyên hoặc thêm dấu chấm như cũ (#15). Lý do: audio.cpp ước lượng thời lượng audio theo trọng số ký tự (dấu câu chỉ nặng 0,5 so với chữ cái), không chừa thời gian cho khoảng ngắt sau mỗi dấu chấm, nên chuỗi câu cực ngắn như `Mở máy. Đăng nhập. Kiểm tra.` hay bị mất trọn một câu (19/24 giọng ở lượt nghe thử #7). Nối bằng dấu phẩy thì đủ chữ hơn hẳn. Chunker của audio.cpp cũng tách ở `.,;:!?` nên điểm chia chunk vẫn còn. Gateway không tự đặt `speed` hay `duration_sec` cho mọi câu để bù, vì như vậy đổi nhịp đọc của mọi câu và đi ngược việc chuyển nguyên tham số sinh; dấu chấm caller tự viết được giữ nguyên. Ngoại lệ duy nhất là tuỳ chọn `short_text_speed` của operator, chỉ áp cho engine text ngắn khi caller không tự đặt `speed` (ADR 0004).
- Voice agent vẫn nên tự chia theo câu ở phía caller và gửi từng câu một. Gateway không quản lý lượt nói hay ngắt lời.
