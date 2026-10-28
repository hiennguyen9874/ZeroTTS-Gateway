# ZeroTTS thay OmniVoice; upstream là `zerotts_server` (Python + onnxruntime CPU) nói cùng API với audio.cpp

Gateway chuyển hẳn từ OmniVoice sang ZeroTTS (`zeroweight-ai/ZeroTTS`). Upstream mới là `zerotts_server`, một package thứ hai trong repo này. Nó bọc `zerotts==0.1.5` (ONNX, chạy trên CPU) và mô phỏng các route của audio.cpp mà gateway đang dùng (`/health`, `/v1/models`, `/v1/audio/voices`, `/v1/audio/speech` với `response_format` wav/pcm/json và `stream_format` sse/audio). Gateway vẫn mỏng, không tổng hợp giọng, và chỉ đổi tên thành `zerotts_gateway`. Lý do: ZeroTTS được tối ưu cho CPU, đọc tiếng Việt từ văn bản thô với WER thấp hơn OmniVoice khoảng 4 lần (theo ZeroBench-TTS của tác giả), và có streaming thật với TTFA tính bằng trăm ms. Trong khi đó audio.cpp không chạy được ZeroTTS.

## Considered Options

- **Port ZeroTTS vào audio.cpp thành family mới.** audio.cpp (bản pin `65df584a` lẫn main `a5369900`) không có family `zerotts`. File `ZeroTTS-GGUF` dùng metadata `zerotts.*` và tên tensor riêng, chỉ runtime `cpp/` của ZeroTTS đọc được. Codec MOSS-Audio-Tokenizer-Nano đã có sẵn trong audio.cpp (`vieneu_v3_turbo`, `moss_tts_nano`), nhưng phần text encoder, global decoder cross-attention và local decoder 17 bước vẫn phải viết bằng C++ rồi giữ fork. Bị loại vì tốn công, trong khi nhu cầu chỉ 1–2 luồng trên CPU.
- **Bọc runtime ggml `cpp/` của ZeroTTS.** Runtime này chỉ chạy trên CPU, không có CFG, và vẫn cần codec ONNX. Đo trên Xeon Silver 4216, tính riêng phần sinh frame: f32 cho RTF 0,98 ở 4 luồng (local decoder chiếm 61/80 ms mỗi frame), q8_0 cho RTF khoảng 1,5. Chậm hơn ONNX nên bị loại.
- **onnxruntime-gpu trên A100.** RTF 1,08–1,31, TTFA 335–400 ms (KV cache chép qua host mỗi frame), chậm hơn CPU. Bị loại.
- **Nhúng ZeroTTS vào gateway.** Phá nguyên tắc gateway không tổng hợp giọng, và làm mất seam test bằng upstream giả. Bị loại.

Số đo ONNX trên CPU của máy dev, một luồng tổng hợp: RTF 0,80–0,88, TTFA 115–215 ms ở 8 luồng ONNX. Máy này dùng chung (load khoảng 19), nên số đo dao động mạnh. Cần benchmark lại trên node triển khai thật.

## Consequences

- **Giọng:** chỉ có giọng preset (latent 10×768, hiện có 8 giọng). Bản phát hành không có voice encoder nên không clone được. Request có `voice_ref`, `instruct` hoặc `speed` ≠ 1 nhận 400. Thiếu `voice` thì dùng giọng mặc định operator cấu hình theo model, không dùng null voice (giọng không ổn định).
- **Non-verbal tag:** ZeroTTS không hiểu thẻ nên gateway gỡ thẻ và báo text warning.
- **ADR 0004 hết hiệu lực:** model không có `speed`, và lỗi mất cụm đầu là lỗi riêng của OmniVoice.
- **ADR 0001 vẫn giữ, với `zerotts_server` ở vai audio.cpp:** upstream chia text theo câu, mỗi segment ≤ khoảng 15 s (dùng `zerotts.chunking`), vì ZeroTTS cắt cụt ở 512 token và dừng ở 1500 frame. Gateway vẫn gửi đúng một request upstream cho mỗi request của caller.
- **Model id:** `zerotts` (offline) và `zerotts-stream` (streaming), theo kiểu entry của audio.cpp. Caller phải đổi `model` và `voice`.
- **Audio 48 kHz mono gốc, không resample.** Caller đang nhận PCM thô 24 kHz từ OmniVoice phải tự cập nhật. Stream PCM thô không có header nên sai sample rate sẽ hỏng âm thầm.
- **Text pipeline:** ban đầu chỉ chạy guard và làm sạch markup, vì model học trên văn bản thô và tự đọc được tiếng Anh xen kẽ. A/B #9 ([báo cáo](../reports/issue9-zerotts-text-pipeline-ab.md)) đổi quyết định này: ZeroTTS nhận toàn bộ pipeline (normalizer ADR 0002, reading rule, lexicon). WER theo ASR là 0,072 khi có pipeline, 0,167 khi chỉ làm sạch. Đọc từ văn bản thô, model đọc sai ngày, số điện thoại, mã số, số La Mã và viết tắt tiếng Việt (`PGS.TS.`, `CMND`). Bỏ lexicon tiếng Anh không đổi WER (0,072), nên giữ lexicon chung; cách đọc tên sản phẩm còn chờ người nghe chốt.
- **Sampling:** operator cấu hình theo model. Caller chỉ được gửi `seed`.
- **Đồng thời:** mỗi `zerotts_server` có N worker (mặc định 1 × 8 luồng ONNX). Khi mọi worker bận, request chờ tối đa `busy_timeout` (mặc định 5 s) rồi nhận 503. Gateway vẫn không xếp hàng.
- **Weights:** script tải về `models/` theo revision HF đã pin, mount read-only vào container. Khi tải không dùng HF cache symlink: onnxruntime ≥ 1.25 từ chối external data nằm ngoài thư mục model.
- **Phần OmniVoice/audio.cpp** (compose audio.cpp, `configs/server.json`, `short_text_speed`, docs INSTALL/USAGE, các submodule liên quan) chỉ gỡ sau khi ZeroTTS chạy được và qua A/B text pipeline.
