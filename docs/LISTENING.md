# Nghe thử ZeroTTS

Chạy `zerotts_server` và gateway theo [ZEROTTS_SERVER.md](ZEROTTS_SERVER.md) và [GATEWAY.md](GATEWAY.md). `render_samples.py` so sánh gateway (A) với upstream trực tiếp (B), cùng model, voice và seed. Các ca clone cũ được bỏ qua vì ZeroTTS không hỗ trợ clone.

```bash
uv run python scripts/render_samples.py --model zerotts --voice maichi --out outputs/listen-001
uv run python scripts/render_samples.py --model zerotts-stream --voice maichi --set listening --out outputs/listen-stream
python3 -m http.server -d outputs/listen-001 9000
```

Streaming dùng PCM 48 kHz; mỗi request chỉ gửi một lần. Dùng `--no-direct` để chỉ chạy gateway, `--group` hoặc `--id` để lọc, `--normalize-on` để bật normalize cho mọi ca. Script từ chối thư mục output có dữ liệu trừ khi dùng `--overwrite`. Kết quả gồm WAV, `index.html`, `results.csv` và `manifest.json`; nghe lại trước khi kết luận từ chỉ số.

Chạy mọi voice preset lấy từ `/v1/audio/voices`:

```bash
uv run python scripts/render_all.py --model zerotts --dry-run
uv run python scripts/render_all.py --model zerotts --out outputs/render-all
```

Chấm ASR tuỳ chọn: chạy `scripts/asr_server.py` với dependency group `asr` và cấu hình `OMNIVOICE_ASR_URL` (tên biến lịch sử), `OMNIVOICE_ASR_MODEL`, `OMNIVOICE_ASR_API_KEY`. Xem `--help` của script để biết tham số.

## A/B text pipeline cho ZeroTTS

Lượt #9 so ba cấu hình engine text cho ZeroTTS, kết quả ở [báo cáo](reports/issue9-zerotts-text-pipeline-ab.md):

| Biến thể | Gateway | Cờ render |
| --- | --- | --- |
| (a) toàn bộ pipeline | có lexicon | `--normalize-on` |
| (b) guard + làm sạch markup | có lexicon (không dùng tới) | không cờ; lúc đo, ZeroTTS còn để mặc định `normalize = false` (nay đã bật, muốn đo lại thì đặt lại `normalize = false` cho model ZeroTTS trong config của gateway này) |
| (c) như (a), bỏ lexicon tiếng Anh | `lexicon_path` để trống (mọi mục hiện có đều là tên tiếng Anh) | `--normalize-on` |

Hai gateway trỏ cùng `zerotts_server`; mỗi biến thể × seed là một lần `render_all.py` (`--no-direct`, ASR bật), rồi `scripts/ab_report.py` chấm WER/CER theo nhóm ca:

```bash
export OMNIVOICE_ASR_URL=http://127.0.0.1:8090
V="--model zerotts --voice maichi --voice quangminh --no-direct --seed 42"
uv run python scripts/render_all.py --gateway http://127.0.0.1:8000 $V --normalize-on --out outputs/ab9/a-s42
uv run python scripts/render_all.py --gateway http://127.0.0.1:8000 $V --out outputs/ab9/b-s42
uv run python scripts/render_all.py --gateway http://127.0.0.1:8001 $V --normalize-on --out outputs/ab9/c-s42
uv run python scripts/ab_report.py --variant a=outputs/ab9/a-s42 --variant b=outputs/ab9/b-s42 \
    --variant c=outputs/ab9/c-s42 --out outputs/ab9/report     # lặp --variant a=… để gộp nhiều seed
```

- Tham chiếu của mỗi ca là input text qua toàn bộ pipeline tiếng Việt, một lần có lexicon và một lần không; hàng chấm theo tham chiếu gần hơn, nên `Google` đọc "gu gồ" hay "google" đều tính đúng. So từ chữ thường, bỏ dấu câu, tag và phiên âm CMU, gộp biến thể vùng miền (`ngàn` = `nghìn`, `tư` = `bốn`, `mốt` = `một`, `lẻ` = `linh`). Ca không phải tiếng Việt bị bỏ (ASR chỉ nghe tiếng Việt).
- WER/CER tính trên cả tập: tổng lỗi chia tổng từ (hoặc ký tự, không tính khoảng trắng) của tham chiếu. Hàng không có audio hoặc không có transcript tính sai toàn bộ.
- Tham chiếu là cách đọc của chính gateway, nên biến thể đọc khác mà vẫn đúng ("hai ngàn hai mươi bốn" cho 2024) vẫn bị trừ điểm. `cases.csv` xếp ca theo độ lệch WER giữa các biến thể: nghe những ca đầu bảng trước khi kết luận.
