# #21 — Kiểm chứng tiếng “bụp” và khoảng ngưng trong out.wav

## Kết luận

Comment cuối của [#21](https://github.com/hiennguyen9874/omnivoice-gateway/issues/21#issuecomment-6262139684)
**đúng về hiện tượng chính, nhưng kết luận nguyên nhân bên trong model quá chắc**.

- Tiếng “bụp” là một cụm tín hiệu trầm nằm trong PCM của file, sau đó là khoảng
  gần im lặng. Không phải lỗi cấu trúc WAV hay thiếu byte cuối file.
- Kiểm tra HTTP hiện tại xác nhận gateway chuyển tiếp audio nguyên byte.
- Khoanh vùng về đường sinh/giải mã audio phía upstream là hợp lý. Chưa tách được
  lỗi model, conditioning giọng mẫu và implementation/decoder của audio.cpp.
- Chưa tái hiện được khẳng định seed phụ thuộc request trước đó.
- Không sửa audio trong gateway; không thay pin hay sửa clone trong references/.

Đã đọc hai chương được yêu cầu: chương 2 (PCM, clipping, đổi kiểu, resampling có
state) và chương 3 (RIFF chunks, header bị đọc như PCM, byte lẻ, neural codec).
WAV PCM không dùng MP3/Opus; điều này **không** loại trừ lỗi neural audio decoder
bên trong upstream.

## Mẫu và môi trường kiểm tra

Ngày kiểm tra: 2026-10-19. Gateway đang chạy ở :8000, upstream ở :8080
(không phải :18002 trong nội dung issue). Không restart dịch vụ.

- File gốc: `out.wav`, giữ nguyên.
- SHA-256: `8a5711d41644c602cc7d72a609cda2bb2b8c9c1bf789a199cf208860056fa56e`.
- Source upstream: `65df584a6f798128fc27ff7e2d8aede0df8ae866`, worktree sạch.
- Model: `g-omnivoice`, offline, 32 bước, guidance 2.
- Engine text từ HTTP dry-run, pipeline `6-b1a1812d`:

> hẹn gặp lúc tám giờ ba mươi phút ngày hai mươi lăm tháng mười hai, phí năm mươi nghìn.

Engine text có 19 âm tiết, vượt band 12 của cấu hình hiện tại, nên gateway không
thêm short-text speed cho câu này. Đối chứng upstream phải dùng engine text,
không dùng input text chứa số của caller.

## 1. File gốc thực sự chứa gì?

| Kiểm tra | Kết quả |
|---|---|
| Cấu trúc | Một RIFF/WAVE, fmt dài 16 byte, data bắt đầu tại byte 44 |
| Định dạng | PCM signed 16-bit little-endian, mono, 24.000 Hz |
| Kích thước | 299.564 byte; data 299.520 byte; không byte thừa/thiếu |
| Số mẫu / thời lượng | 149.760 / 6,240 giây |
| Peak / clipping | 0,615112 / không mẫu chạm ±full scale |
| Trung bình toàn file | 0,0000692 |

Không mặc định bỏ 44 byte khi đọc: đã duyệt RIFF chunks và đối chiếu bằng
Python wave cùng ffprobe. Với **file này**, header đúng là 44 byte.
Tìm thấy một RIFF không tự nó đủ chứng minh mọi sample đúng; ở đây còn kiểm
tra chunk size, block alignment và giải mã PCM.

| Vị trí trong file | Quan sát |
|---|---|
| 0,00–0,10 s | Digital silence; khớp padding 100 ms của upstream |
| 0,10–0,20 s | Cụm tín hiệu trầm, peak 0,127686 ≈ −17,9 dBFS, RMS −29,7 dBFS |
| 0,22–0,60 s | RMS −85,8 dBFS, gần im lặng |
| Khoảng 0,62 s | Bắt đầu hoạt động lời nói chính |
| Khoảng 4,60–5,10 s | Khoảng nghỉ giữa hai cụm lời, tương ứng dấu phẩy của câu |

Trên cửa sổ 0,10–0,20 s, FFT có cửa sổ Hann cho **98,6% năng lượng dưới 300 Hz**;
riêng lõi 0,14–0,18 s gần 100%. Vì thế con số “99–100%” trong comment hợp với
lõi xung, không phải mọi cách chọn cửa sổ. Đây không phải một spike chỉ kéo dài
một sample. Tiếng trầm ngắn rồi im lặng giải thích cảm giác “bụp → ngưng → nói”.

Khoảng ngưng nằm sẵn trong timeline của file đã tải xong. Không cần giả thuyết
HTTP buffering/đợi chunk để giải thích khoảng ngưng này. Chưa kiểm tra thiết bị
phát của caller, nên không khẳng định đã loại mọi lỗi playback có thể đi kèm.

## 2. Đối chứng HTTP mới, không chỉ đọc lại comment

1. Gọi thẳng upstream với engine text và seed 1.
2. Gọi gateway với input text gốc, cùng giọng và seed 1.
3. Hai response body giống nhau từng byte, SHA-256:
   `6f2f21e4322eecdbb3c11c7bc223cf5c8b81f5aab1132f11f40afe6844390b24`.
4. Quét seed 1–40 với hậu xử lý mặc định: không gặp lại mẫu
   **xung trầm 0,10–0,20 s rồi khoảng lặng** giống file gốc.
   Đây là kiểm tra mẫu tín hiệu, không phải đánh giá nghe toàn bộ audio.
5. Seed 1–8 chạy cặp `postprocess_output=false/true`: bản false vẫn có khoảng
   lặng đầu dài; true cắt bớt. Không tái hiện xung trầm của file gốc ở các cặp này.
6. Chuỗi A → B → A, với B là câu dài giọng Adam_bua, rồi thêm request không giọng:
   các response A seed 1 vẫn cùng hash. Bản false của A cũng lặp lại cùng hash.

Không thể lấy kết quả này để phủ nhận quan sát lịch sử, nhưng **không đủ cơ sở
khẳng định audio.cpp luôn không deterministic theo seed hoặc do cache bị bẩn**.

Đối chứng phụ, câu dài Adam_bua seed 9, gọi **chỉ upstream**: có một cụm tín
hiệu ngắn trước khoảng lặng. Cụm này xuất hiện cả ở bản false (khoảng 0,70 s)
và true (khoảng 0,20 s, do cắt đầu). Hai đoạn 40 ms đó giống nhau từng sample
PCM16 sau khi căn chỉnh lệch 0,50 s. Đây là tín hiệu cao tần, **không phải chính
xung trầm của out.wav**. Nó minh họa rằng đầu ra trước bước trim đã có cụm âm
rời rạc; không dùng nó để tuyên bố đã tái hiện chính xác lỗi gốc.

## 3. Các điểm cần sửa trong lập luận của comment

### “postprocess_output=false” không phải audio thô ngay sau decoder

Trong `references/audio.cpp/src/models/omnivoice/postprocess.cpp:269–305`:

- Cờ này chỉ bỏ `split_on_silence_pcm16` và `trim_edges_pcm16`.
- Điều chỉnh gain theo reference RMS hoặc peak **vẫn chạy**.
- `fade_and_pad_audio` **vẫn chạy**, gồm fade 100 ms và pad 100 ms mỗi đầu.

Do đó cần gọi đó là **bản tắt split/trim**, không phải “tắt mọi hậu xử lý”.
Nếu xung vẫn còn ở bản này thì loại được split/trim là nguồn tạo xung,
nhưng chưa phân biệt được generator với neural decoder hay các bước còn lại.

### “Chỉ bỏ khoảng lặng từ 500 ms” chưa mô tả đúng code

Lệnh gọi là:

```cpp
split_on_silence_pcm16(pcm, sample_rate, 500, -50.0F, 500, 10);
trim_edges_pcm16(pcm, sample_rate, 100, 100, -50.0F);
```

500 ms đầu là thời gian tối thiểu để **phát hiện** khoảng lặng. Nhưng
`keep_silence_ms=500` giữ lại 500 ms ở mỗi phía của hai cụm âm.
Nếu hai vùng giữ lại chồng nhau, code chia tại trung điểm, **không bỏ mẫu**.
Vì vậy ngay cả một khoảng lặng nội bộ 600–1.000 ms cũng có thể được giữ nguyên;
không phải cứ vượt 500 ms là bị cắt. Khoảng 420 ms trong file gốc đương nhiên còn.

Trim đầu dùng RMS từng cửa sổ 10 ms với ngưỡng −50 dBFS, không dùng peak.
Lõi xung gốc có RMS khoảng −23 dBFS nên đủ mạnh để chặn trim; đây là giải thích
hợp lý cho việc cụm âm đầu và khoảng ngưng phía sau được giữ lại.
Fade chỉ nhân envelope, không phân loại được xung lỗi với lời nói.

### “Offline = một lần sinh” không phải quy tắc chung

`session.cpp:357–414` vẫn chọn chunking cho request offline nếu có
`text_chunk_size` hoặc vượt ngưỡng thời lượng (mặc định 30 s trong types.h).
Với cấu hình và câu ngắn đang kiểm tra, nhánh một lần generate/decode là phù hợp.
Không thể loại lỗi ghép chunk cho mọi request chỉ dựa vào “offline” hay ADR 0001;
ADR quy định một request upstream, không giới hạn upstream chỉ một chunk.

### Chưa thể quy hết cho model hoặc giọng mẫu

- Ratio đạo hàm/RMS “không vượt 10” là heuristic, không phải chứng minh không click.
- DC toàn file nhỏ không loại trừ mọi bất thường DC cục bộ.
- Chưa có A/B tái lập cùng token để phân biệt generator với decoder.
- Chưa có đối chứng giọng gốc và giọng thêm đệm trên nhiều seed trong cùng điều kiện.
  Không kết luận thiếu silence cuối giọng Kim_Thanh là nguyên nhân.

## 4. Hướng xử lý

Không tự động bỏ mọi cụm âm dưới 150 ms theo sau bởi khoảng lặng 200 ms:
có thể cắt mất từ ngắn, phụ âm đầu hoặc non-verbal tag hợp lệ.

Để tìm root cause sâu hơn, chạy trên **một upstream chẩn đoán riêng**, giữ
nguyên pin/config/model/giọng; lưu token ids và waveform ngay sau decoder cho
ca tái hiện, trước finalize. So sánh:

- cùng prompt/reference/seed qua A → B → A;
- token ids khác hay chỉ waveform khác;
- decode lại cùng token ids;
- cùng reference ở triển khai Python chuẩn nếu cần;
- sau đó mới A/B reference có đệm và đánh giá mất từ/giọng.

Chỉ bật trace/dump cho mẫu được phép, không đưa text/audio vào log vận hành.
Cắt xung đầu ở upstream có thể là workaround, không phải đã sửa generator.

## Artifact cục bộ

Không ghi đè out.wav. Tại `/tmp/issue21-verification/` có response WAV,
`results.json` (request/response metadata, hash, số đo), các trích đoạn
`original-head.wav`, `original-leading-bump.wav` và
`original-preview-no-leading-bump.wav`.

Bản preview chỉ dành cho **file này**: bỏ phần trước 0,60 s, thêm 100 ms lặng
và fade 5 ms. Nghe so sánh để xác nhận cảm giác bụp/ngưng ở đầu; không dùng mốc
cắt cố định đó làm policy cho request khác. Artifact /tmp không được version hóa.

Script chẩn đoán của lượt này: `/tmp/verify_issue21*.py`. Không sửa runtime,
config, giọng mẫu, submodule hay issue trên GitHub.
