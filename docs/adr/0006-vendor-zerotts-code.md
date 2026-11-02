# Chép code ZeroTTS vào `zerotts_server` thay vì dùng package `zerotts` từ PyPI

`zerotts_server` thôi phụ thuộc `zerotts==0.1.5`. Thay vào đó, phần code cần dùng được chép từ `references/ZeroTTS-repo` (commit `319c3e0`, có `src/zerotts/` giống hệt bản 0.1.5) vào `src/zerotts_server/model/`. Lý do: upstream cần sửa code model, như thêm runtime PyTorch CUDA, bỏ phụ thuộc vào hàm private `_generate_frames` và tự quản bộ giải mã streaming. Package PyPI không cho sửa trực tiếp, nên mỗi chỗ cần đổi lại phải vá từ bên ngoài, dễ hỏng khi nâng phiên bản.

## Considered Options

- **Giữ package, kế thừa hoặc vá từ bên ngoài.** Đang làm vậy với `_generate_frames` và `codec.streaming_decoder()`. Cách này dựa vào API private không được cam kết; runtime CUDA phải vá sâu hơn nữa. Bị loại.
- **Fork repo ZeroTTS rồi pin bằng git dependency.** Sửa được, nhưng phải duy trì thêm một repo, và mỗi lần sửa phải đi qua hai repo. Bị loại.
- **Chép toàn bộ `src/zerotts/`.** Kéo theo `cli`, `hub` (đã có `scripts/download_zerotts.sh`) và `text_norm`, một normalizer tiếng Việt thứ hai bên cạnh sea-g2p (ADR 0002). Bị loại.

## Consequences

- **Phạm vi:** chỉ chép `synthesizer`, `codec`, `tokenizer`, `chunking`, `voices` và hàm ghi WAV trong `audio`. Bỏ các dependency `sounddevice`, `scipy` và `huggingface_hub`. Gateway không được import `zerotts_server.model`.
- **Truy vết bản gốc:** `src/zerotts_server/model/VENDORED.md` ghi commit gốc và danh sách file đã chép, kèm `LICENSE` (MIT) và `NOTICE`. Submodule `references/ZeroTTS-repo` vẫn được giữ làm mốc; khi bản gốc có thay đổi đáng lấy thì so tay rồi áp lại.
- **Sửa code đã chép:** commit đầu tiên chỉ chép nguyên văn, cùng lắm đổi đường import. Mọi thay đổi khác nằm ở các commit sau, để `git diff` thấy rõ phần đã khác bản gốc.
- **Điều kiện chấp nhận:** với cùng weights và cùng seed, audio phải có SHA-256 trùng với `zerotts==0.1.5` (smoke test `real_model`, cả fp32 lẫn int8) trước khi gỡ package khỏi `pyproject.toml`. `scripts/bench_zerotts.py` và `scripts/quantize_zerotts.py` chuyển sang dùng bản đã chép.
- **Cập nhật bản gốc không còn tự động:** không thể chỉ nâng số phiên bản. Bản gốc ra thay đổi nào (ví dụ voice encoder ở `18d66815`) thì phải tự quyết định có lấy hay không.
