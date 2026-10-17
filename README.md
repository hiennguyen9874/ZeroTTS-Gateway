# OmniVoice Gateway

Gateway đứng trước audio.cpp, giữ nguyên API `/v1/audio/speech` của upstream. Spec: issue #1.

## Chạy

```bash
uv run omnivoice-gateway --config configs/gateway.toml
```

Biến môi trường:

- `OMNIVOICE_GATEWAY_CONFIG`: đường dẫn config, dùng khi không truyền `--config`
- `OMNIVOICE_GATEWAY_UPSTREAM_URL`: ghi đè `upstream_url`
- `OMNIVOICE_GATEWAY_API_KEY`: đặt thì mọi route trừ `/health`, `/ready` yêu cầu `Authorization: Bearer <key>`
- `OMNIVOICE_GATEWAY_HOST`: ghi đè `host` (địa chỉ lắng nghe)

## Chạy bằng Docker

Container chỉ chứa gateway; audio.cpp chạy ở nơi khác (trên host hoặc máy riêng).

```bash
cp .env.example .env        # sửa OMNIVOICE_GATEWAY_UPSTREAM_URL, OMNIVOICE_GATEWAY_API_KEY
docker compose up -d --build
curl localhost:8000/health  # 200
curl localhost:8000/ready   # 200 khi tới được upstream, 503 khi không
```

- `configs/` được mount read-only vào `/config`; gateway đọc `/config/gateway.toml` và từ điển theo `lexicon_path`. Sửa config hoặc từ điển rồi `docker compose restart`, không cần build lại.
- Trong container gateway lắng nghe trên `0.0.0.0:8000` (image đặt `OMNIVOICE_GATEWAY_HOST=0.0.0.0`; `port` vẫn lấy từ config, đổi thì sửa cả `ports` trong `compose.yaml`). Cổng trên host: `GATEWAY_PORT`.
- `127.0.0.1` trong container là chính container. audio.cpp trên host: dùng `http://host.docker.internal:8080` (mặc định, compose đã thêm `host-gateway`) và cho audio.cpp lắng nghe trên `0.0.0.0` thay vì `127.0.0.1`. audio.cpp ở máy khác: dùng host/IP của máy đó.
- Image: `python:3.12-slim`, `uv sync --locked --no-dev`, chạy bằng user không phải root; `.dockerignore` chỉ cho vào `pyproject.toml`, `uv.lock`, `README.md`, `src/`.

## Text pipeline

`POST /v1/audio/speech` thay `input` bằng engine text trước khi gửi lên upstream. Field riêng `text_processing: {"normalize": bool}` bị gỡ trước khi forward; guard và làm sạch luôn chạy.

- Response thành công có `X-Text-Pipeline-Version` và, khi có cảnh báo, `X-Text-Warnings` (`omitted_code_block`, `omitted_html`, `omitted_url`, `omitted_emoji`).
- Lỗi đầu vào `400`: `text_too_long`, `reasoning_markup`, `model_control`, `empty_text`; trả trước khi gọi upstream.
- Tốc độ text ngắn (tuỳ chọn theo model trong `configs/gateway.toml`, [ADR 0004](docs/adr/0004-short-text-speed.md)): khi caller không gửi `speed` lẫn `options.duration_sec`/`duration`, gateway có thể thêm `speed` để giảm mất cụm đầu (#18, #19). Dạng cũ: `short_text_speed` cùng `short_text_max_syllables` (mặc định 12). Hoặc `short_text_speed_bands`: các cặp `max_syllables`/`speed`, ngưỡng tăng dần, chọn band đầu tiên chứa số âm tiết của engine text (bỏ non-verbal tag); vượt band cuối thì không thêm. Không dùng hai dạng cùng lúc. Config repo dùng 0.7 cho ≤ 8 âm tiết, 0.6 cho 9–12; model khác và mặc định trong code vẫn tắt. Đây là biện pháp giảm lỗi, không bảo đảm model luôn đọc đủ hay không lặp chữ.
- Dry-run: `POST /v1/text/normalize` với `{"input", "language"?, "model"?, "text_processing"?}` trả `engine_text`, `warnings`, `language`, `pipeline_version`, không gọi upstream.

Khi effective language là tiếng Việt và normalize bật, pipeline còn: giữ nguyên 13 non-verbal tag (`[laughter]`, `[sigh]`, …) và phiên âm CMU viết hoa (`[B EY1 S]`), bỏ dấu ngoặc của các ngoặc vuông khác; áp từ điển phát âm; chuẩn hoá bằng sea-g2p (`punc_norm=False`); gỡ thẻ `<en>` và ghép chữ cái đánh vần thành viết tắt hoa (`AI`). `reference_text` đi qua cùng pipeline, cùng mã lỗi, không tính vào warnings.

### Từ điển phát âm

`lexicon_path` trong config (tương đối theo file config; bản mặc định `configs/lexicon.csv`). CSV có header `word,replacement,note`:

```csv
word,replacement,note
iPhone,ai phôn,thương hiệu
```

Khớp theo ranh giới từ, mục dài trước, phân biệt hoa thường; chạy trước sea-g2p nên `replacement` phải là dạng nói tiếng Việt. Nạp lúc khởi động (sửa thì restart); file sai định dạng làm gateway không khởi động. Hash của từ điển nằm trong `X-Text-Pipeline-Version` (`<RULES_VERSION>-<hash>`).

### Thêm ca regression

Ca golden nằm ở `tests/golden/pipeline.json`, chạy qua route dry-run. Mỗi khi sửa một lỗi đọc sai:

1. Xem engine text hiện tại: `curl -s localhost:8000/v1/text/normalize -d '{"input":"…","language":"vi"}'`.
2. Thêm một dòng: `id`, `group` (`numbers-money`, `dates-times`, `measurements`, `codes-phones`, `abbrev-english`, `markup`, `tags`, `input-errors`, `short-sentences`, `language`, …), `input`, tuỳ chọn `language` (mặc định `vi`) và `normalize`, rồi `expected` + `warnings` hoặc `error`, và `why` (vì sao đó là cách đọc đúng).
3. Nếu tạm chấp nhận output sai (hành vi của sea-g2p chưa sửa), ghi output hiện tại vào `expected` và thêm `known_issue` mô tả chỗ sai; không xoá ca.
4. Sửa luật hoặc thêm dòng vào từ điển. Đổi luật thì tăng `RULES_VERSION` trong `src/omnivoice_gateway/text.py`.
5. `uv run pytest`.

## Nghe thử mẫu

Cần audio.cpp (`docs/USAGE.md`) và gateway đang chạy. Script đọc các ca không lỗi (mặc định golden; xem Bộ câu nghe thử), gọi cả gateway (A) lẫn audio.cpp trực tiếp (B) với cùng model, voice, seed, rồi lưu WAV, chỉ số và trang so sánh:

```bash
uv run python scripts/render_samples.py --out outputs/listen-001   # --model, --voice, --seed, --group, --id, --no-direct, --normalize-on
python -m http.server -d outputs/listen-001 9000                   # mở http://localhost:9000
```

- Model streaming (`mode: streaming` trong `/v1/models`) được gọi bằng `stream_format=audio` + `response_format=pcm` để đo TTFB thật; script tự đóng gói WAV (`--sample-rate`, mặc định 24000). Model offline nhận `wav`.
- Mỗi request gửi đúng một lần, không retry. Stream đứt sau khi đã nhận audio thì kết quả là `stream_aborted` và không lưu file. WAV ghi qua file `.part` rồi rename.
- Thư mục output đã có dữ liệu thì script từ chối; `--overwrite` xoá WAV và báo cáo cũ của script (giữ file khác) rồi chạy lại.
- Console chỉ in id ca và chỉ số, không in văn bản. Exit code 1 khi có request không ra audio.

### Bộ câu nghe thử

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

Ca clone (`"clone": "<voice>"`) gửi `voices/<voice>.wav` dạng base64 trong `voice_ref` (không gửi `voice`), `reference_text` lấy từ dòng `<voice>|…` của `voices/prompt_text`. Phía A (gateway) gửi `reference_text` nguyên văn để gateway chuẩn hoá; phía B (direct) gửi `reference_text` **nguyên văn** và lấy engine text của gateway làm `input`, nên hai phía cùng input, seed, model và chỉ khác `reference_text`. Cột `reference_text` trong `results.csv` ghi `normalized`/`verbatim`; trang HTML hiện văn bản tham chiếu của từng phía.

Thêm ca: thêm một dòng vào `scripts/listening_cases.json` với `id` ổn định (không đổi, không dùng lại), `group`, `questions` (danh sách 1–5, có thể rỗng), `input`, `why` (ca này để làm gì), `review_focus` (cần nghe điều gì); tuỳ chọn `language`, `normalize`, `clone`. Chỉ dùng văn bản tự viết, không dùng dữ liệu thật. Muốn so sánh hai biến thể thì thêm hai ca và nhắc id của nhau trong `review_focus`. Chạy `uv run pytest tests/test_render_samples.py` để kiểm tra bộ câu.

### Chấm bằng ASR (tuỳ chọn)

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

### Ca `normalize: false`

Nhiều ca golden (nhóm `markup`, `short-sentences`, …) đặt `normalize: false` để kiểm tra riêng bước làm sạch, nên gửi nguyên `@`, `<`, `>` tới model, khác production (normalize bật). Trang HTML gắn nhãn `normalize: false` cho các ca này và cột `normalize` của `results.csv` ghi `false`. Muốn nghe chúng như production thì thêm `--normalize-on`: bỏ qua `normalize: false` của ca, `manifest.json` ghi `normalize_on: true`, nhãn HTML đổi thành "rendered with normalize on".

### Chạy tất cả model × giọng

`scripts/render_all.py` gọi `render_samples.py` cho mọi (bộ câu, model, giọng): mặc định cả `golden` lẫn `listening`, mọi model TTS trong `/v1/models`, mọi `voices/*.wav` có dòng trong `voices/prompt_text`.

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

### Output

Output trong `--out` (mặc định `outputs/samples`):

- `<id>.gateway.wav`, `<id>.direct.wav`: PCM s16le mono.
- `index.html`: input text, engine text, player A/B, chỉ số và cờ cạnh từng player.
- `manifest.json`: `case_set` (`golden`/`listening`), `git_commit` (thêm `-dirty` nếu có thay đổi chưa commit), `pipeline_version`, `model`, `mode`, `voice`, `seed`, `normalize_on`, `asr` (`url`, `model` hoặc `null`), `started_utc`/`finished_utc`, `cases`, `groups` và `questions` (nhóm / câu hỏi → danh sách id ca), `counts` theo kết quả.
- `results.csv`: mỗi hàng là một cặp (ca, phía `gateway`/`direct`):
  - `group`, `questions` (câu hỏi của #7, ví dụ `1,2`; lọc cột này để nghe theo câu hỏi), `normalize` (`false` khi ca gửi `normalize: false`), `reference_text` (ca clone: `normalized`/`verbatim`).
  - `result`: `audio` hoặc mã lỗi (`stream_aborted`, `type` lỗi của upstream/gateway, `transport_*`, `incomplete_pcm`, `bad_wav`); `http_status`, `file`.
  - Chỉ số: `duration_s`, `ttfb_ms`, `total_ms`, `rtf` (= total / duration), `peak_dbfs`, `rms_dbfs`, `lead_silence_ms`, `trail_silence_ms` (ngưỡng ~-40 dBFS), `clipped_samples`.
  - `flags`: gợi ý chỗ cần nghe kỹ, không phải đánh giá chất lượng: `silent`, `very_short` (< 0,3 s), `clipping`, `long_lead_silence` (> 1 s), `long_trail_silence` (> 1,5 s); khi có ASR thêm `words_missing` (`word_recall` < 0,8) và `asr_failed`.
  - `word_recall`, `asr_text`: chỉ khi có ASR (xem Chấm bằng ASR).
  - `text_warnings` (`X-Text-Warnings` của gateway; phía direct luôn trống), `known_issue`, `review_focus`, `input`, `engine_text` (engine text của gateway, ghi cho cả hai phía).
  - `sent_text`: text request của phía đó đưa tới model: phía gateway là engine text, phía direct là input text thô (ca clone: engine text của gateway).
  - Người nghe điền: `review_result` (`pass` / `fail` / `ambiguous` / `not-listened`), `reviewer`, `notes`.

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
