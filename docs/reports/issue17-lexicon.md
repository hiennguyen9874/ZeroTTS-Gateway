# #17 — Từ điển phát âm mặc định cho tên sản phẩm

## Kiểm tra, thiết kế và kế hoạch

Nguyên nhân: `configs/lexicon.csv` chỉ có header. Gateway đã hỗ trợ từ điển
phân biệt hoa thường, khớp nguyên từ, mục dài trước, nạp lúc khởi động và hash
nội dung vào pipeline version. Không cần thêm thư viện hay một bảng tên riêng
trong code.

#7 đã đóng theo bàn giao, **không phải đã có người nghe chốt chính sách**.
Triển khai này dùng sáu cách đọc ứng viên đã có trong bộ nghe #7, giới hạn ở
tên sản phẩm; không coi đó là quyết định chung về viết tắt/từ mượn. Operator cần
xác nhận cách đọc trước khi coi tiêu chí nghe của #17 hoàn tất.

Kế hoạch thực hiện:

1. Điền sáu mục có `note` vào file mặc định; giữ nguyên đường dẫn config và cơ chế
   file riêng của operator thay thế bản repo, không tự gộp.
2. Cho golden chạy qua HTTP API với config repo, thay vì `Settings()` có từ điển
   rỗng. Sửa `vi-iphone`, giữ id cũ và bỏ nhãn known issue đã sửa; cập nhật
   `vi-english-words` vì Facebook/Google nay là ngoại lệ rõ ràng. Thêm golden từng
   tên, nhiều tên cùng tag, cờ normalize tắt và language khác tiếng Việt.
3. Giữ luật phiên bản khi tên được thay bằng dạng nói; kiểm ưu tiên từ điển
   nguyên văn, forwarding `input`/`reference_text`, version và ranh giới khớp.
4. Render upstream thật trên Doan_Trang/Ngoc_Linh, ghi ASR và giới hạn; bàn giao
   WAV/trang nghe để người nghe xác nhận.

## Hành vi triển khai

| Input text khớp nguyên từ | Dạng nói trong từ điển |
| --- | --- |
| iPhone | ai phôn |
| iPad | ai pát |
| iOS | ai ô ét |
| YouTube | du túp |
| Facebook | phây búc |
| Google | gu gồ |

Các dạng đọc lấy từ ứng viên của #7, không nhập hàng loạt danh sách vietnormalizer.
Đối chiếu `references/vietnormalizer/vietnormalizer/data/non-vietnamese-words.csv`
thấy Google/Facebook tương ứng `gu-gồ`/`phây-búc`, nhưng YouTube có biến thể
`yu-túp`; chưa thay ứng viên `du túp` khi chưa có người nghe chọn.

Chỉ áp khi effective language là tiếng Việt và normalize bật. Không thay đổi
AI/API, từ tiếng Anh thông thường, tag, guard, warning hay cơ chế truyền audio.
`iphone`, `IPHONE`, `iPhoneX`, `MyGoogleApp` không khớp các mục mới.
Operator sửa file rồi restart; chạy không có config vẫn không tự nạp từ điển.

### Tương tác với reading rule phiên bản

Phát hiện khi thêm từ điển: thay `iOS` thành `ai ô ét` trước reading rule làm
mất dấu hiệu tên gốc, khiến `iOS 18.2` bị đọc “mười tám phẩy hai”. Sửa trong
`Pipeline._apply_lexicon`:

- Dành quyền ưu tiên cho **mọi** mục khớp nguyên văn trước.
- Khi mục khớp chính xác phần tên của mẫu phiên bản hiện có, giao hậu tố chưa bị
  mục khác ghi đè cho cùng luật `_name_version`, dựa trên ngữ cảnh tên gốc.
- Mục đầy đủ `GPT-3.5` vẫn thắng mục `GPT`; mục số riêng `18.2` cũng thắng
  reading rule. Giữ ngoại lệ hàng nghìn và điều kiện ranh giới của luật cũ.
- Không đưa toàn bộ reading rule lên trước từ điển, không thêm luật phát âm
  riêng cho mỗi thương hiệu.

Tăng rules version 5 → 6 vì có sửa code; hash từ điển cũng đổi.
Pipeline dùng trong lượt render: **`6-b1a1812d`** (trước đó `5-4f53cda1`).

## Kiểm chứng tự động

Test chỉ qua HTTP API gateway với upstream giả. Ngoài golden tên sản phẩm,
có ca phiên bản nhiều phần/nhiều tên, ghi đè từ điển đầy đủ hoặc riêng phần số,
ngoại lệ hàng nghìn, không khớp bên trong từ, phân biệt hoa thường, version,
và một request speech chuyển cả input text lẫn reference_text đã chuẩn hoá.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Kết quả cuối: **391 tests passed**; Ruff lint/format và diff check đạt.

## Render thật ngày 17/10/2026

- Gateway nền `c933dfb45e0419aaa09925c2c3386cec25819e87` + thay đổi #17 trong
  working tree; manifest ghi `-dirty`, không coi commit nền là toàn bộ code.
- Upstream audio.cpp `65df584a6f798128fc27ff7e2d8aede0df8ae866`, CUDA 12.6,
  GPU 0, checkpoint `g-omnivoice-f16.gguf`; model `g-omnivoice`, offline.
- sea-g2p 0.10.0; giọng `vieneu_v3_Doan_Trang`, `vieneu_v3_Ngoc_Linh`; seed 42.
- ASR gipformer int8, CPU, sherpa-onnx, modified beam search.
- Bảy ca × hai phía × hai giọng = **28 request, 28 WAV hợp lệ, 0 lỗi tổng hợp,
  0 lỗi ASR**, không retry.
- Cấu hình production: gateway thêm speed 0.7 cho sáu câu tên sản phẩm, 0.6
  cho câu phiên bản; direct không gửi speed (mặc định 1.0). A/B này kiểm hành vi
  production, **không** tách riêng ảnh hưởng từ điển khỏi speed.

Output local (không commit audio):

- `outputs/issue17-lexicon/Doan_Trang/index.html`
- `outputs/issue17-lexicon/Ngoc_Linh/index.html`
- Mỗi thư mục có `manifest.json`, `results.csv` và WAV hai phía.
- `outputs/issue17-lexicon/evidence.json` cùng `source/` lưu hash và bản source/config
  ảnh hưởng engine text, checkpoint và giọng đã dùng.

### Phiên âm ASR phía gateway

| Ca | Doan_Trang | Ngoc_Linh |
| --- | --- | --- |
| vi-iphone | MUA IPHONE MƯỜI LĂM | MUA IPHONE MƯỜI LĂM |
| vi-ipad | MUA IPAD MỚI | MUA IPAD MỚI |
| vi-ios | CẬP NHẬT IOS MƯỜI TÁM | CẬP NHẬT IOS MƯỜI TÁM |
| vi-ios-version | CẬP NHẬT IOS MƯỜI TÁM CHẤM HAI | CẬP NHẬT IOS MƯỜI TÁM CHẤM HAI |
| vi-youtube | MỞ YOUTUBE NHÁ | MỞ YOUTUBE NHÉ |
| vi-facebook | TÔI DÙNG FACEBOOK | TÔI DÙNG FACEBOOK |
| vi-google | TÌM TRÊN GOOGLE | TÌM TRÊN GOOGLE |

`vi-iphone` có engine text `mua ai phôn mười lăm.`, WAV gateway dài 1,22 s
(Doan_Trang) và 1,79 s (Ngoc_Linh). Direct cũng được ASR viết thành IPHONE;
không dùng ASR này để kết luận bên nào phát âm tự nhiên hơn.

Cả 28 hàng vẫn giữ cờ `words_missing`: recall so dạng nói `ai phôn` với chữ ASR
`IPHONE` nên chỉ đạt 0,6 ở `vi-iphone` dù transcript nhận ra tên. Tương tự với
các thương hiệu khác; `nhá`/`nhé` còn là khác biệt thật trong transcript. Không
sửa cách chấm hoặc xoá cờ để báo pass giả. Phía direct của
`vi-ios-version` không có “CHẤM”, trong khi gateway có ở cả hai giọng.

**Chưa có đánh giá bằng tai người**: cột review vẫn trống. Tiêu chí từ điển và
golden đã triển khai; tiêu chí “nghe lại vi-iphone trên ít nhất hai giọng” và
việc chốt cách đọc vẫn chờ người nghe. Theo yêu cầu bàn giao, đóng #17 sau khi
commit/push; việc đóng issue **không xác nhận đã nghe bằng tai người** hay chốt
chính sách. Không coi ASR là bằng chứng hoàn tất tiêu chí nghe.

### Tái lập

Khởi động upstream, gateway với config repo và ASR theo [LISTENING.md](../LISTENING.md#chấm-bằng-asr-tuỳ-chọn), rồi:

```bash
for voice in Doan_Trang Ngoc_Linh; do
  uv run python scripts/render_samples.py \
    --model g-omnivoice --voice "vieneu_v3_${voice}" --seed 42 \
    --id vi-iphone --id vi-ipad --id vi-ios --id vi-ios-version \
    --id vi-youtube --id vi-facebook --id vi-google \
    --asr-url http://127.0.0.1:8090 \
    --out "outputs/issue17-repeat/${voice}"
done
python3 -m http.server --bind 127.0.0.1 --directory outputs/issue17-lexicon 9000
```

Nghe tối thiểu `vi-iphone.gateway.wav` ở cả hai giọng, đối chiếu cách đọc “ai
phôn”, đủ “mua/mười lăm”, không lặp hoặc thêm chữ. Điền reviewer/notes trong
CSV và ghi quyết định vào #7/#17; nếu đổi cách đọc thì sửa từ điển và golden
cùng lúc.
