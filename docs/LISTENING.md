# Nghe thử mẫu

Lượt nghe lại #7 (model/giọng/seed, đối chứng và giới hạn): [báo cáo](reports/issue7-listening.md).

Cần audio.cpp ([USAGE.md](USAGE.md)) và gateway đang chạy. Script đọc các ca không lỗi (mặc định golden; xem [Bộ câu nghe thử](#bộ-câu-nghe-thử)), gọi cả gateway (A) lẫn audio.cpp trực tiếp (B) với cùng model, voice, seed, rồi lưu WAV, chỉ số và trang so sánh:

```bash
uv run python scripts/render_samples.py --out outputs/listen-001   # --model, --voice, --seed, --group, --id, --no-direct, --normalize-on
python -m http.server -d outputs/listen-001 9000                   # mở http://localhost:9000
```

- Model streaming (`mode: streaming` trong `/v1/models`) được gọi bằng `stream_format=audio` + `response_format=pcm` để đo TTFB thật; script tự đóng gói WAV (`--sample-rate`, mặc định 24000). Model offline nhận `wav`.
- Mỗi request gửi đúng một lần, không retry. Stream đứt sau khi đã nhận audio thì kết quả là `stream_aborted` và không lưu file. WAV ghi qua file `.part` rồi rename.
- Thư mục output đã có dữ liệu thì script từ chối; `--overwrite` xoá WAV và báo cáo cũ của script (giữ file khác) rồi chạy lại.
- Console chỉ in id ca và chỉ số, không in văn bản. Exit code 1 khi có request không ra audio.

## Bộ câu nghe thử

Ngoài golden (mặc định, `--set golden`), `--set listening` chạy bộ câu riêng `scripts/listening_cases.json` dùng để trả lời 5 câu hỏi của #7. Bộ này chỉ để nghe; regression vẫn chỉ nằm ở golden. Lọc bằng `--group`, `--id` hoặc `--question` (1–5, lặp được).

| Nhóm | Câu hỏi | Nội dung |
| --- | --- | --- |
| `hard` | – (in hoa: 2) | Ca khó lọc từ vietvoice-tts: `Có không?/./!`, `Thật à?/./!`, NFD, in hoa, không dấu, khoảng trắng lạ, câu rất ngắn, thiếu dấu kết thúc |
| `abbrev-case` | 1, 2 | `AI`/`API` viết hoa, `a i`, dạng từ điển phát âm `ây ai` (viết sẵn, vì từ điển nạp lúc khởi động nên không bật/tắt theo request được), viết tắt tiếng Việt, tên riêng viết hoa |
| `long-text` | 3 | Đoạn ~700 ký tự (≥30 s audio) có ngày, số, phần trăm, tiền, giờ, viết tắt |
| `tags` | 4 | Non-verbal tag ở đầu, giữa, cuối, kề dấu câu, một câu đối chứng không tag, phiên âm CMU |
| `clone` | 5 | Clone từ `voices/vieneu_v3_Mai_Anh.wav` |

```bash
# Câu hỏi 1, 2, 4, 5 và ca khó, model offline
uv run python scripts/render_samples.py --set listening --out outputs/listen-q-offline
# Câu hỏi 3 ở cả hai model: offline và streaming
uv run python scripts/render_samples.py --set listening --question 3 --out outputs/listen-q3-offline
uv run python scripts/render_samples.py --set listening --question 3 --model g-omnivoice-stream --out outputs/listen-q3-stream
```

Ca nghe có thể đặt `speed` để giữ cùng tốc độ ở cả hai phía; CSV ghi `requested_speed` (trống khi không chỉ định, không phải tốc độ hiệu lực do gateway thêm). Hai ca `tag-case-upper/lower` tắt normalize và giữ cùng dấu câu, `speed=1.0`, để tách ảnh hưởng của chữ hoa khỏi các bước khác. Nhóm `loanwords` giữ các dạng nói dùng làm đối chứng cho sáu mục từ điển mặc định #17; golden kiểm tên sản phẩm nguyên văn. Các cách đọc này vẫn cần người nghe xác nhận. Không dùng `--normalize-on` khi so sánh hai ca chữ hoa/thường này.

Ca clone (`"clone": "<voice>"`) gửi `voices/<voice>.wav` dạng base64 trong `voice_ref` (không gửi `voice`), `reference_text` lấy từ dòng `<voice>|…` của `voices/prompt_text`. Phía A (gateway) gửi `reference_text` nguyên văn để gateway chuẩn hoá; phía B (direct) gửi `reference_text` **nguyên văn** và lấy engine text của gateway làm `input`, nên hai phía cùng input, seed, model và chỉ khác `reference_text`. Script đặt `speed=1.0` cho cả hai phía (hoặc `speed` của ca nếu có), không để speed band của gateway làm sai đối chứng. Bộ câu có audio tham chiếu Mai_Anh, Doan_Trang và Ngoc_Linh. Cột `reference_text` trong `results.csv` ghi `normalized`/`verbatim`; trang HTML hiện văn bản tham chiếu của từng phía.

Thêm ca: thêm một dòng vào `scripts/listening_cases.json` với `id` ổn định (không đổi, không dùng lại), `group`, `questions` (danh sách 1–5, có thể rỗng), `input`, `why` (ca này để làm gì), `review_focus` (cần nghe điều gì); tuỳ chọn `language`, `normalize`, `clone`. Chỉ dùng văn bản tự viết, không dùng dữ liệu thật. Muốn so sánh hai biến thể thì thêm hai ca và nhắc id của nhau trong `review_focus`. Chạy `uv run pytest tests/test_render_samples.py` để kiểm tra bộ câu.

## Chấm bằng ASR (tuỳ chọn)

Để bắt lỗi mất chữ không phụ thuộc người nghe, script phiên âm từng WAV qua một endpoint transcription kiểu OpenAI (`POST <url>/v1/audio/transcriptions`, multipart `file`, `model`, `language`, `response_format=json`). Cấu hình bằng biến môi trường (hoặc `--asr-url`, `--asr-model`, `--asr-api-key`):

- `OMNIVOICE_ASR_URL`: base URL của server ASR (không có `/v1`); không đặt thì không chấm, chạy như cũ và các cột ASR để trống.
- `OMNIVOICE_ASR_MODEL`: model gửi kèm (mặc định `gipformer`).
- `OMNIVOICE_ASR_API_KEY`: Bearer key, nếu server yêu cầu.

`scripts/asr_server.py` dựng sẵn endpoint đó cho gipformer (`g-group-ai-lab/gipformer1.5-68M-rnnt`, RNN-T tiếng Việt chạy bằng sherpa-onnx). Phụ thuộc nằm trong dependency group `asr`, không vào image gateway. File ONNX tải vào `models/` (bản clone ở `references/gipformer1.5-68M-rnnt` chỉ có con trỏ Git LFS):

```bash
uv run hf download g-group-ai-lab/gipformer1.5-68M-rnnt encoder.int8.onnx decoder.int8.onnx \
    joiner.int8.onnx tokens.txt --local-dir models/gipformer1.5-68M-rnnt
uv run --group asr python scripts/asr_server.py --port 8090      # --precision fp32, --num-threads 8
export OMNIVOICE_ASR_URL=http://127.0.0.1:8090
uv run python scripts/render_samples.py --set listening --out outputs/listen-asr
```

- `asr_text`: transcript của từng phía. `word_recall`: tỉ lệ từ của engine text của gateway xuất hiện trong transcript theo đúng thứ tự (độ dài dãy con chung dài nhất / số từ), không phân biệt hoa thường, bỏ non-verbal tag, phiên âm CMU và dấu câu. Cả hai phía đều so với engine text của gateway vì đó là cách đọc mong muốn.
- `word_recall` < 0,8 thêm flag `words_missing`; request ASR lỗi thêm `asr_failed` (`word_recall` để trống, run không vì thế mà thất bại). Trang HTML hiện transcript và `word_recall` dưới player.
- Đây là gợi ý, không phải phán quyết: ASR viết chữ cái, từ tiếng Anh, ký hiệu khác engine text (`bê` → `B`, `hotro` → `hot cho`) nên các ca này có thể bị flag dù đọc đúng; nghe lại trước khi kết luận.

## Ca `normalize: false`

Nhiều ca golden (nhóm `markup`, `short-sentences`, …) đặt `normalize: false` để kiểm tra riêng bước làm sạch, nên gửi nguyên `@`, `<`, `>` tới model, khác production (normalize bật). Trang HTML gắn nhãn `normalize: false` cho các ca này và cột `normalize` của `results.csv` ghi `false`. Muốn nghe chúng như production thì thêm `--normalize-on`: bỏ qua `normalize: false` của ca, `manifest.json` ghi `normalize_on: true`, nhãn HTML đổi thành "rendered with normalize on".

## Chạy tất cả model × giọng

`scripts/render_all.py` gọi `render_samples.py` cho mọi (bộ câu, model, giọng): mặc định cả `golden` lẫn `listening`, mọi model TTS trong `/v1/models` (config mặc định chỉ có `g-omnivoice` và `g-omnivoice-stream`; muốn so nhiều model thì thêm entry theo [USAGE.md, mục 4.1.1](USAGE.md#411-thêm-lại-các-model-đã-bỏ)), mọi `voices/*.wav` có dòng trong `voices/prompt_text`.

```bash
uv run python scripts/render_all.py --dry-run                      # xem kế hoạch
uv run python scripts/render_all.py --out outputs/render-all-001   # chạy lại cùng --out thì bỏ qua run đã xong
uv run python scripts/render_all.py --set listening --model g-omnivoice --voice vieneu_v3_Mai_Anh --no-direct
python -m http.server -d outputs/render-all-001 9000
```

- Mỗi run nằm ở `<set>/<model>/<voice>/`; nhóm chỉ gồm ca clone (tự mang audio tham chiếu) chạy một lần mỗi model ở `<set>/<model>/_clone/`.
- Vòng ngoài là model để audio.cpp chỉ nạp mỗi model một lần. Run lỗi được ghi vào `runs.json` và không chặn run sau; exit code 1 khi có run lỗi.
- Gộp kết quả: `all_results.csv` (mọi hàng `results.csv`, thêm `set`, `model`, `voice`, `run`), `summary.csv` (theo set, model, phía: số request, lỗi, cờ, số ca `words_missing`, `word_recall` trung bình, trung vị rtf/ttfb/thời lượng), `index.html` (link tới trang của từng run).
- Tóm tắt nhiều giọng (cần ASR; biến `OMNIVOICE_ASR_*` được truyền xuống từng run): `voices.csv` theo (set, model, giọng, phía): số ca `words_missing` và `word_recall` trung bình, để chọn giọng nghe chính; `cases.csv` theo (set, model, ca, phía): số giọng mất chữ, `all_voices_missing` khi ca hỏng ở mọi giọng đã chấm. Cuối lượt chạy console in hai bảng này (giọng ít mất chữ nhất trước, run `_clone` sau cùng; chỉ ca mất chữ ở ít nhất một giọng, `ALL` = mọi giọng) và `index.html` có thêm hai bảng.
- `--normalize-on` được truyền xuống `render_samples.py`.

## Output

Output trong `--out` (mặc định `outputs/samples`):

- `<id>.gateway.wav`, `<id>.direct.wav`: PCM s16le mono.
- `index.html`: input text, engine text, player A/B, chỉ số và cờ cạnh từng player.
- `manifest.json`: `case_set` (`golden`/`listening`), `git_commit` (thêm `-dirty` nếu có thay đổi chưa commit), `pipeline_version`, `model`, `mode`, `voice`, `seed`, `normalize_on`, `asr` (`url`, `model` hoặc `null`), `started_utc`/`finished_utc`, `cases`, `groups` và `questions` (nhóm / câu hỏi → danh sách id ca), `counts` theo kết quả.
- `results.csv`: mỗi hàng là một cặp (ca, phía `gateway`/`direct`):
  - `group`, `questions` (câu hỏi của #7, ví dụ `1,2`; lọc cột này để nghe theo câu hỏi), `normalize` (`false` khi ca gửi `normalize: false`), `reference_text` (ca clone: `normalized`/`verbatim`).
  - `result`: `audio` hoặc mã lỗi (`stream_aborted`, `type` lỗi của upstream/gateway, `transport_*`, `incomplete_pcm`, `bad_wav`); `http_status`, `file`.
  - Chỉ số: `duration_s`, `ttfb_ms`, `total_ms`, `rtf` (= total / duration), `peak_dbfs`, `rms_dbfs`, `lead_silence_ms`, `trail_silence_ms` (ngưỡng ~-40 dBFS), `clipped_samples`.
  - `flags`: gợi ý chỗ cần nghe kỹ, không phải đánh giá chất lượng: `silent`, `very_short` (< 0,3 s), `clipping`, `long_lead_silence` (> 1 s), `long_trail_silence` (> 1,5 s); khi có ASR thêm `words_missing` (`word_recall` < 0,8) và `asr_failed`.
  - `word_recall`, `asr_text`: chỉ khi có ASR (xem [Chấm bằng ASR](#chấm-bằng-asr-tuỳ-chọn)).
  - `text_warnings` (`X-Text-Warnings` của gateway; phía direct luôn trống), `known_issue`, `review_focus`, `input`, `engine_text` (engine text của gateway, ghi cho cả hai phía).
  - `requested_speed`: tốc độ chỉ định trong thí nghiệm ở cả hai phía; ca clone mặc định `1.0`. Trống khi script không gửi speed; gateway vẫn có thể thêm speed theo config.
  - `sent_text`: text request của phía đó đưa tới model: phía gateway là engine text, phía direct là input text thô (ca clone: engine text của gateway).
  - Người nghe điền: `review_result` (`pass` / `fail` / `ambiguous` / `not-listened`), `reviewer`, `notes`.