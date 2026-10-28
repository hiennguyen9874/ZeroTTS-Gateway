# Rà soát `references/` sau khi chuyển sang ZeroTTS

Mục tiêu: giữ nguồn tham chiếu trực tiếp cho upstream, hợp đồng API, text pipeline và chấm ASR; bỏ engine đã thay thế và clone weights trùng lặp. Không thay đổi runtime, dependency Python hay dữ liệu trong `models/`. Các ADR và báo cáo cũ vẫn được giữ làm lịch sử quyết định.

## Giữ: 6 submodule chỉ đọc

| Thư mục | Lý do giữ | Commit pin |
| --- | --- | --- |
| `ZeroTTS-repo` | Nguồn engine đang dùng: generation, chia câu, streaming, graph và voice pack. Thêm submodule HTTPS từ clone sạch hiện có. | `319c3e07c8575c1e38617f07c773ec76a780a1f1` |
| `audio.cpp` | Nguồn hợp đồng API mà `zerotts_server` mô phỏng, gồm lỗi, SSE và PCM. Giữ được các trích dẫn mã nguồn trong ADR/báo cáo; không dùng để triển khai. | `65df584a6f798128fc27ff7e2d8aede0df8ae866` |
| `sea-g2p` | Normalizer đang dùng; cần đối chiếu khi sửa reading rule và golden. | `e825173f235d08ea19315b2b279fb11153b44cea` |
| `gipformer` | Cấu hình sherpa-onnx và ASR tiếng Việt cho script chấm các lượt nghe, gồm A/B #9. | `0cac5b183d2e1a9e3429ff69080e09cc62ebdf9b` |
| `vietnormalizer` | Nguồn từ điển loanword/acronym để bổ sung lexicon; không dùng làm normalizer runtime (ADR 0002). | `dd38778731d6ca4e9e670a19abb2df1c901a1852` |
| `llm-wiki` | Tài liệu tiếng Việt về ngữ âm, chuẩn hóa và thiết kế text pipeline đã được ADR 0002 tham chiếu. Thêm submodule HTTPS từ clone sạch hiện có. | `4471d7651bc56ff50b0c254c690dda0dfbf5bf9e` |

URL nguồn nằm trong [`.gitmodules`](../.gitmodules). Hai pin mới đã được kiểm tra tồn tại trên GitHub công khai. Không cập nhật bốn pin cũ còn giữ; không chạy theo HEAD mới của wiki. Pin `ZeroTTS-repo` nằm sau tag `v0.1.5` năm commit nhưng không đổi `src/zerotts/`; dependency runtime vẫn là `zerotts==0.1.5`, không cài từ submodule.

```bash
git submodule update --init
```

Không cần `--remote` (sẽ bỏ tính tái lập theo pin), cũng không cần `--recursive`: các dependency C++ lồng trong repo tham chiếu không phục vụ runtime ONNX của dự án. Không cần khởi tạo submodule để chạy gateway, upstream hay test HTTP mặc định. Chỉ đổi pin có chủ đích, trong commit riêng; không sửa nội dung bên trong các repo tham chiếu.

## Bỏ: 9 repo không còn cần clone cục bộ

| Thư mục đã xóa | Nguồn và revision cuối | Lý do |
| --- | --- | --- |
| `OmniVoice` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2) | Engine đã được ZeroTTS thay thế; gỡ cả gitlink và khai báo submodule. |
| `OmniVoice-hf` | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice/tree/c5fdb5ccb189668d56333f77ba2629f4cd7535f4) | Checkpoint của engine cũ. |
| `g-omnivoice-hf` | [g-group-ai-lab/g-omnivoice](https://huggingface.co/g-group-ai-lab/g-omnivoice/tree/f2b73211e57db389d81ab817987f90c02e0898ba) | Finetune của engine cũ; không còn model id tương ứng trong cấu hình gateway. |
| `omnivoice-vietnamese-hf` | [splendor1811/omnivoice-vietnamese](https://huggingface.co/splendor1811/omnivoice-vietnamese/tree/f70093f90d78a5b790ff0f066b3629ebcc33cb5d) | Finetune của engine cũ. |
| `VieNeu-TTS` | [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS/tree/bd0f88eb8de7d560753a6172ba9c868f309a101c) | Engine khác, trước đây chỉ tham khảo cách dùng sea-g2p; đã có nguồn sea-g2p và upstream ZeroTTS trực tiếp. Gỡ cả submodule. |
| `VieNeu-TTS-v3-Turbo` | [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/tree/61b85e3d937fbbacb387714180e8182823512523) | Checkpoint của engine không dùng. |
| `ZeroTTS-GGUF` | [zeroweight-ai/ZeroTTS-GGUF](https://huggingface.co/zeroweight-ai/ZeroTTS-GGUF/tree/92ca8651645d4733df56620b3aadc768a76f7c46) | Runtime ggml đã bị loại trong ADR 0005; upstream dùng ONNX. |
| `ZeroTTS` | [zeroweight-ai/ZeroTTS](https://huggingface.co/zeroweight-ai/ZeroTTS/tree/c2bfbd67dc648cac455077333f7cf5c18a2e3bb4) | Weights vẫn cần, nhưng clone HF trong `references/` chỉ trùng nguồn tải đã pin bằng script. Tải file thật vào `models/ZeroTTS`. |
| `gipformer1.5-68M-rnnt` | [g-group-ai-lab/gipformer1.5-68M-rnnt](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/tree/dd9227dcd8705c13f33bdbe59728d546ab94480f) | Weights ASR vẫn cần khi chấm, nhưng không cần clone HF chứa LFS pointer. Tải các file cần dùng vào `models/`. |

Đã kiểm tra working tree và nhánh remote trước khi xóa. Ngoại lệ duy nhất là ba file LFS trong `OmniVoice-hf` bị thay bằng file 0 byte; không có nội dung chỉnh sửa cần giữ. File ngoài Git trong hai submodule bị gỡ chỉ là cache `__pycache__`. Các clone HF bị gỡ không có file untracked. Lịch sử submodule cũ có thể còn trong `.git/modules/` để khôi phục revision cũ; không xóa kho đối tượng nội bộ của Git.

## Tải weights, không thêm submodule HF

ZeroTTS dùng đúng revision được pin trong [`scripts/download_zerotts.sh`](../scripts/download_zerotts.sh):

```bash
uv run scripts/download_zerotts.sh
```

ASR tùy chọn dùng revision của clone đã rà soát (lệnh tải cũ không pin revision có thể lấy bản mới hơn):

```bash
uv run hf download g-group-ai-lab/gipformer1.5-68M-rnnt \
    --revision dd9227dcd8705c13f33bdbe59728d546ab94480f \
    encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt \
    --local-dir models/gipformer1.5-68M-rnnt
uv run --group asr python scripts/asr_server.py --port 8090
```

Không tải weights qua `git submodule update`: tránh Git LFS tải checkpoint lớn ngoài ý muốn, và tránh nhầm LFS pointer với file ONNX thật. `models/` tiếp tục được gitignore; không xóa hay chuyển weights hiện có khi dọn `references/`.
