# Hướng dẫn sử dụng OmniVoice với audio.cpp

Tài liệu này hướng dẫn dùng OmniVoice (bản gốc và hai bản finetune tiếng Việt) với `audio.cpp`: tạo file GGUF, chạy bằng CLI, clone giọng, design giọng, chạy server và gọi API tương thích OpenAI.
Phần cài đặt và build có trong [INSTALL.md](INSTALL.md).

Tài liệu gốc để tham khảo thêm:

- `references/audio.cpp/docs/models/omnivoice.md`: các route và option của OmniVoice
- `references/audio.cpp/docs/usage.md`: option chung của CLI, batch, request sequence
- `references/audio.cpp/docs/gguf.md`: converter GGUF
- `references/audio.cpp/app/server/README.md` và `app/server/api.md`: server và API
- `references/audio.cpp/docs/reports/omnivoice_weight_type_benchmark.md`: benchmark weight type
- `references/OmniVoice/docs/voice-design.md`, `generation-parameters.md`: tài liệu upstream

Mọi lệnh dưới đây chạy từ thư mục gốc của project (`omnivoice-gateway/`). Các lệnh đã được chạy thử trên A100 với build `linux-cuda126-release`.

---

## 0. Chuẩn bị shell

```bash
cd /home/jovyan/workspace/speech/tts/omnivoice-gateway

export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0          # GPU 4 và 7 đang bị tiến trình khác chiếm

BIN=references/audio.cpp/build/linux-cuda126-release/bin
CLI=$BIN/audiocpp_cli
SERVER=$BIN/audiocpp_server
GGUF=$BIN/audiocpp_gguf

# Model mặc định cho các ví dụ: bản finetune tiếng Việt G-OmniVoice, f16
M=models/g-omnivoice-GGUF/g-omnivoice-f16.gguf
```

Các model đang có:

| Model | Safetensors | GGUF |
|---|---|---|
| OmniVoice gốc (k2-fsa) | `references/OmniVoice-hf/` | `models/audio.cpp-gguf/OmniVoice-GGUF/omnivoice-{f16,bf16,q8_0}.gguf` |
| G-OmniVoice (tiếng Việt) | `models/g-omnivoice/` | `models/g-omnivoice-GGUF/g-omnivoice-{f16,q8_0}.gguf` |
| omnivoice-vietnamese | `models/omnivoice-vietnamese/` | `models/omnivoice-vietnamese-GGUF/omnivoice-vietnamese-{f16,q8_0}.gguf` |

Ba bản trên dùng chung kiến trúc và audio tokenizer, nên đều chạy qua `--family omnivoice` và dùng được mọi option trong tài liệu này.

**Nên dùng bản f16.** Trên A100, q8_0 chậm hơn f16 khoảng 20%. Benchmark upstream cũng cho thấy f16 là kiểu nhanh nhất cho generator.

---

## 1. Ba chế độ sinh giọng

| Chế độ | Input cần có | `--task` |
|---|---|---|
| Auto voice: model tự chọn giọng | `--text` | `tts` |
| Voice clone: nhân bản giọng từ audio mẫu | `--text`, `--voice-ref`, `--reference-text` | `tts` hoặc `clon` |
| Voice design: tạo giọng từ mô tả | `--text`, `--instruct` | `tts` hoặc `vdes` |

`clon` và `vdes` chỉ là tên gọi khác của cùng một đường tổng hợp. Chế độ thực tế được quyết định bởi input bạn truyền vào: có `--voice-ref` thì clone, có `--instruct` thì design. Nên luôn dùng `--task tts`.

---

## 2. Tạo file GGUF

GGUF của audio.cpp là file **standalone**: nó chứa weights của generator, weights của audio tokenizer, các file phụ (config, tokenizer, ...) và package spec. Nhờ vậy chỉ cần truyền đúng một file `.gguf` vào `--model`.

### 2.1. Yêu cầu thư mục nguồn

Model root phải có cấu trúc giống package HF của OmniVoice:

```text
<model_root>/
  config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
  model.safetensors
  audio_tokenizer/
    config.json, preprocessor_config.json, model.safetensors
```

Bản finetune nào không kèm `audio_tokenizer/`, ví dụ `omnivoice-vietnamese`, thì copy hoặc hardlink thư mục đó từ `k2-fsa/OmniVoice`. Audio tokenizer của các bản finetune giống hệt bản gốc.

### 2.2. Convert

```bash
SRC=models/g-omnivoice
OUT=models/g-omnivoice-GGUF
mkdir -p $OUT

for T in f16 q8_0; do
  $GGUF \
    --input weights=$SRC/model.safetensors \
    --input audio_tokenizer_weights=$SRC/audio_tokenizer/model.safetensors \
    --root $SRC \
    --output $OUT/g-omnivoice-$T.gguf \
    --family omnivoice --type $T --overwrite
done
```

- `--input weights=...` và `--input audio_tokenizer_weights=...`: hai namespace bắt buộc của OmniVoice. Tên namespace phải khớp chính xác.
- `--root`: thư mục chứa các file phụ cần nhúng vào GGUF.
- `--type`: `orig`, `f16`, `bf16`, `q8_0` (các kiểu `q2_k`…`q6_k` là thử nghiệm, chưa được kiểm chứng với OmniVoice).
- Với `q8_0`, converter chỉ lượng tử hóa các tensor phù hợp. Các tensor nhạy cảm vẫn giữ kiểu an toàn, nên chất lượng khác với việc ép `audio_tokenizer_weight_type=q8_0` lúc runtime (xem mục 3.6).

### 2.3. Kiểm tra GGUF

```bash
$GGUF --inspect models/g-omnivoice-GGUF/g-omnivoice-f16.gguf
```

Kết quả đúng phải có `embedded_sidecars=true`, `embedded_model_spec=true`, `model_spec_family=omnivoice`, cùng hai namespace `weights` và `audio_tokenizer_weights`.

### 2.4. Lưu ý về đường dẫn `--model`

- Truyền trực tiếp file `.gguf` là cách chắc chắn nhất.
- Có thể truyền một thư mục. audio.cpp sẽ chọn `model.gguf` hoặc file `.gguf` duy nhất trong thư mục đó. Thư mục chứa **nhiều** GGUF (như `models/g-omnivoice-GGUF/`) sẽ bị từ chối vì không rõ chọn file nào.
- Vẫn chạy được trực tiếp từ safetensors bằng cách truyền thư mục model root (xem mục 3.7).

---

## 3. Chạy bằng CLI

Dạng chung:

```bash
$CLI --task tts --family omnivoice --backend cuda --model <gguf|thư mục> [input] --out out.wav
```

Luôn truyền `--family omnivoice`. Nếu bỏ, CLI phải tự dò loader và khởi động chậm hơn nhiều.

Option hay dùng: `--log` in log tiến độ và thời gian, `--metrics` in `wall_ms`, `audio_duration_ms`, `rtf`, `x_realtime`. Audio đầu ra là WAV mono 24 kHz. Thêm `--out-format pcm24|float32` nếu cần định dạng khác `pcm16`.

Xem toàn bộ option mà model hỗ trợ:

```bash
$CLI --task tts --family omnivoice --model $M --help
```

### 3.1. Auto voice

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M \
  --language vi --seed 42 \
  --text "Xin chào, đây là giọng do mô hình tự chọn." \
  --out outputs/auto.wav --metrics
```

Mỗi lần chạy với seed khác sẽ ra một giọng khác. Upstream khuyến cáo auto voice và voice design có thể không ổn định với câu rất ngắn (1–2 giây). Gặp trường hợp này thì nên dùng voice clone.

### 3.2. Voice clone

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M \
  --language vi --seed 42 \
  --voice-ref voices/vieneu_v3_Mai_Anh.wav \
  --reference-text "Chào mừng đến với website của chúng tôi. Đây là trang web cung cấp một giải pháp truyền văn bản thành giọng nói." \
  --text "Đây là giọng được nhân bản từ file mẫu." \
  --out outputs/clone.wav --metrics
```

Lưu ý:

- **`--reference-text` là bắt buộc trong audio.cpp.** Bản Python upstream tự chạy ASR khi thiếu transcript, còn audio.cpp sẽ báo lỗi `requires reference_text when reference audio is provided`. Transcript phải khớp chính xác nội dung file mẫu.
- File mẫu nên là WAV sạch, chỉ một người nói, dài 3–10 giây. Khoảng lặng dài và tạp âm làm giảm chất lượng clone. Mặc định `preprocess_prompt=true` sẽ cắt bớt khoảng lặng dài trong audio mẫu và thêm dấu câu vào cuối transcript.
- Muốn clone tốt tiếng Việt thì dùng audio mẫu tiếng Việt. `assets/resources/b.wav` trong repo audio.cpp là giọng tiếng Anh.
- Giọng trong `voices/` (`vieneu_v3_*`) được giải mã từ preset của VieNeu-TTS v3 Turbo bằng `scripts/decode_vieneu_voices.py`; transcript trong `voices/prompt_text` do ASR sinh ra nên cần nghe và sửa tay. Khi dùng thật, hãy thay bằng bản ghi giọng người thật.
- Nếu truyền cả `--voice-ref` và `--instruct` mà hai thứ mâu thuẫn nhau, model thường theo audio mẫu. Khi chúng nhất quán, `--instruct` có thể giúp clone ổn định hơn.

### 3.3. Voice design

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M \
  --language vi --seed 42 \
  --instruct "female, young adult, moderate pitch" \
  --text "Đây là giọng được tạo từ mô tả thuộc tính." \
  --out outputs/design.wav --metrics
```

`--instruct` là chuỗi thuộc tính, ngăn cách bằng dấu phẩy. Mỗi nhóm chỉ chọn **một** giá trị, và có thể kết hợp nhiều nhóm với nhau:

| Nhóm | Giá trị (tiếng Anh / tiếng Trung) |
|---|---|
| Giới tính | `male` / 男, `female` / 女 |
| Tuổi | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Cao độ | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| Phong cách | `whisper` |
| Accent tiếng Anh (chỉ có tác dụng khi text là tiếng Anh) | `american accent`, `british accent`, `australian accent`, `canadian accent`, `indian accent`, `chinese accent`, `korean accent`, `japanese accent`, `portuguese accent`, `russian accent` |
| Phương ngữ tiếng Trung (chỉ có tác dụng khi text là tiếng Trung) | 河南话, 四川话, 东北话, ... (xem `references/OmniVoice/docs/voice-design.md`) |

Ví dụ: `"male, elderly, low pitch, whisper"`, `"female"`. Có thể bỏ nhóm nào không cần, model sẽ tự điền. Thuộc tính không phân biệt hoa thường.

Lưu ý: voice design của OmniVoice gốc chỉ được train trên dữ liệu tiếng Trung và tiếng Anh. Với tiếng Việt, model vẫn sinh được giọng nhưng đôi khi bỏ qua một số thuộc tính. README của G-OmniVoice có ví dụ `northern Vietnamese accent`, nhưng thuộc tính này không nằm trong danh sách upstream và mình chưa kiểm chứng được nó có tác dụng.

### 3.4. Ngôn ngữ, tag và phát âm

- `--language` nhận mã ISO (`vi`, `en`, `zh`, ...) hoặc tên ngôn ngữ của OmniVoice (`Vietnamese`). Nếu bỏ trống, model tự xác định ngôn ngữ. Mã không hợp lệ sẽ báo lỗi `unsupported OmniVoice language`. Bản finetune tiếng Việt nên luôn chạy với `--language vi`.
- **Tag phi ngôn ngữ** được viết trực tiếp vào text: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.
  ```bash
  --text "[laughter] Thật bất ngờ quá, mình không nghĩ là bạn đến."
  ```
- **Sửa phát âm**: tiếng Trung dùng pinyin kèm số thanh (`ZHE2`), tiếng Anh dùng phiên âm CMU viết hoa trong ngoặc vuông (`[B EY1 S]`). Tiếng Việt không có cơ chế tương tự, nên cần **chuẩn hóa text trước** (số, ngày tháng, ký hiệu, viết tắt) theo khuyến nghị của G-OmniVoice.

### 3.5. Tham số sinh

Truyền trực tiếp trên dòng lệnh:

| Option | Mặc định | Ý nghĩa |
|---|---:|---|
| `--num-inference-steps` | `32` | Số bước diffusion. Thời gian chạy gần như tỉ lệ tuyến tính với số bước: 16 bước nhanh gấp khoảng 1.9 lần nhưng chất lượng có thể giảm |
| `--guidance-scale` | `2.0` | Độ mạnh CFG |
| `--seed` | ngẫu nhiên | Cố định seed để kết quả tái lập được. Nếu bỏ, mỗi request tiếp tục từ trạng thái RNG của session |
| `--language` | auto | Gợi ý ngôn ngữ |
| `--text-chunk-size` | tắt | Chia text dài theo số ký tự (đồng thời là kích thước chunk khi streaming) |
| `--text-chunk-mode` | `tag_aware` | `default`, `tag_aware`, `japanese`, `endline` |

Truyền bằng `--request-option key=value`:

| Key | Mặc định | Ý nghĩa |
|---|---:|---|
| `speed` | `1.0` | Tốc độ nói. Lớn hơn 1 là nhanh hơn, audio ngắn hơn |
| `duration_sec` | ước lượng | Ép độ dài audio (giây). Được ưu tiên hơn `speed` |
| `shift` | `0.1` | Shift của lịch timestep |
| `denoise` | `true` | Thêm token denoise khi có audio mẫu |
| `preprocess_prompt` | `true` | Tiền xử lý audio mẫu và transcript |
| `postprocess_output` | `true` | Cắt khoảng lặng dài ở audio đầu ra. Tắt option này nếu cần `duration_sec` khớp chính xác |
| `layer_penalty_factor` | `5.0` | Phạt các codebook sâu |
| `position_temperature` | `5.0` | Nhiệt độ Gumbel khi chọn vị trí unmask |
| `class_temperature` | `0.0` | Nhiệt độ khi sample token. `0` là greedy |
| `audio_chunk_duration_sec` | `15.0` | Độ dài mỗi chunk khi model tự chia text dài |
| `audio_chunk_threshold_sec` | `30.0` | Model chỉ tự chia khi audio ước lượng dài hơn ngưỡng này |

Ví dụ:

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M --language vi \
  --voice-ref voices/vieneu_v3_Mai_Anh.wav --reference-text "<transcript>" \
  --text "Nói nhanh hơn một chút." \
  --num-inference-steps 16 \
  --request-option speed=1.2 \
  --out outputs/fast.wav
```

### 3.6. Session option (`--session-option`)

Session option được áp dụng khi tạo session, nên không đổi được theo từng request.

| Key | Giá trị | Ghi chú |
|---|---|---|
| `omnivoice.generator_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Lượng tử hóa weights của generator lúc load. Chủ yếu dùng khi chạy từ safetensors |
| `omnivoice.audio_tokenizer_weight_type` | như trên | **Không dùng `q8_0`**: benchmark cho thấy audio đầu ra không nghe được và cũng không nhanh hơn |
| `omnivoice.weight_type` | như trên | Giá trị dự phòng chung cho generator |
| `omnivoice.perf_mode` | `off`, `flash_attention` | Bật flash attention. Không dùng chung được với `mem_saver=true` |
| `omnivoice.mem_saver` | `true`/`false` | Giải phóng graph giữa các phase để tiết kiệm VRAM, đổi lại chạy chậm hơn |
| `omnivoice.*_graph_arena_mb`, `omnivoice.*_weight_context_mb` | MiB | Thường không cần chỉnh |

### 3.7. Chạy trực tiếp từ safetensors

```bash
$CLI --task tts --family omnivoice --backend cuda \
  --model models/g-omnivoice \
  --session-option omnivoice.generator_weight_type=f16 \
  --session-option omnivoice.audio_tokenizer_weight_type=f16 \
  --language vi --text "Chạy từ safetensors." --out outputs/st.wav
```

Cách này cần binary build với `--deployment-build` (bản hiện tại có) hoặc phải thấy được thư mục `model_specs/`. Cặp `f16`/`f16` là cấu hình upstream khuyến nghị.

### 3.8. Text dài và streaming trên CLI

Với text dài, model tự chia thành các chunk khoảng 15 giây khi audio ước lượng dài hơn 30 giây. Muốn tự kiểm soát cách chia thì dùng `--text-chunk-size`.

OmniVoice không có streaming gốc. audio.cpp làm **pseudo-streaming**: chia text thành chunk, sinh xong chunk nào thì trả chunk đó, cuối cùng ghép lại thành một file WAV.

```bash
$CLI --task tts --mode streaming --family omnivoice --backend cuda --model $M \
  --language vi \
  --voice-ref voices/vieneu_v3_Mai_Anh.wav --reference-text "<transcript>" \
  --text "Câu thứ nhất. Câu thứ hai dài hơn một chút. Câu thứ ba kết thúc." \
  --text-chunk-size 160 \
  --out outputs/stream.wav --out-dir outputs/stream_chunks
```

`--out` là file đã ghép, còn `--out-dir` chứa `chunk_0.wav`, `chunk_1.wav`, ...

### 3.9. Chạy nhiều request trong một session

Load model một lần rồi chạy nhiều request sẽ tránh được chi phí load lại. Các request sau cũng nhanh hơn request đầu nhờ dùng lại graph.

**Mỗi dòng là một request** (option chung như `--instruct`, `--voice-ref` áp dụng cho mọi dòng):

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M --language vi \
  --voice-ref voices/vieneu_v3_Mai_Anh.wav --reference-text "<transcript>" \
  --batch-text-file lines.txt --out-dir outputs/batch --metrics
# Thay --out-dir bằng: --batch-merge-audio concat --out outputs/merged.wav  để ghép thành 1 file
```

Ngoài `--batch-text-file` còn có `--batch-text-dir <dir>` (mỗi file `.txt`/`.md`/`.json` là một request).

**Mỗi request có tham số riêng** (`--request-sequence`):

```json
{
  "requests": [
    {
      "id": "clone_01",
      "text": "Yêu cầu thứ nhất, nhân bản giọng.",
      "voice_ref": "voices/vieneu_v3_Mai_Anh.wav",
      "reference_text": "Chào mừng đến với website của chúng tôi. Đây là trang web cung cấp một giải pháp truyền văn bản thành giọng nói.",
      "language": "vi",
      "seed": 1
    },
    {
      "id": "design_01",
      "text": "Yêu cầu thứ hai, thiết kế giọng.",
      "instruct": "male, elderly, low pitch",
      "language": "vi",
      "seed": 1
    }
  ]
}
```

```bash
$CLI --task tts --family omnivoice --backend cuda --model $M \
  --request-sequence requests.json --out-dir outputs/seq --metrics
# -> outputs/seq/clone_01.wav, outputs/seq/design_01.wav
```

Đường dẫn trong file JSON là tương đối so với thư mục đang chạy lệnh, nên dùng đường dẫn tuyệt đối cho chắc. Các request được chạy **tuần tự** trong cùng session, không gộp thành batch trên GPU.

---

## 4. Chạy server (`audiocpp_server`)

Server giữ model và session sống liên tục giữa các request, nên đây là cách nên dùng khi deploy. Mỗi model chỉ chạy một request tại một thời điểm, các request khác phải xếp hàng chờ.

### 4.1. Config

File mẫu nằm ở [`configs/server.json`](../configs/server.json) (đã chạy thử). Mặc định chỉ khai báo G-OmniVoice ở hai chế độ: `g-omnivoice` (offline) và `g-omnivoice-stream` (streaming). Các model khác (`omnivoice-vietnamese`, `omnivoice`) đã bỏ khỏi config; muốn dùng lại thì xem [mục 4.1.1](#411-thêm-lại-các-model-đã-bỏ).

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 4,
  "lazy_load": true,
  "max_loaded_models": 2,
  "busy_timeout_ms": 120000,
  "voice_dir": "../voices",
  "models": [
    {
      "id": "g-omnivoice",
      "family": "omnivoice",
      "path": "../models/g-omnivoice-GGUF/g-omnivoice-f16.gguf",
      "task": "tts",
      "mode": "offline",
      "default_request_options": { "language": "vi", "num_inference_steps": 32, "guidance_scale": 2.0 }
    },
    {
      "id": "g-omnivoice-stream",
      "family": "omnivoice",
      "path": "../models/g-omnivoice-GGUF/g-omnivoice-f16.gguf",
      "task": "tts",
      "mode": "streaming",
      "default_request_options": { "language": "vi", "text_chunk_size": 160, "text_chunk_mode": "tag_aware" }
    }
  ]
}
```

Giải thích các trường:

- `id`: tên model mà client gửi trong trường `model` của request. Đây không phải family hay tên file.
- `path` và `voice_dir` tương đối: được tính so với **thư mục chứa file config**.
- `mode`: `offline` hoặc `streaming`. Streaming chỉ chạy được trên entry có `mode: "streaming"`. Muốn dùng cả hai chế độ thì phải khai báo hai entry, như ví dụ trên. Mỗi entry có session riêng và tốn VRAM riêng.
- `lazy_load: true`: chỉ load model khi có request đầu tiên tới model đó. Model đã load sẽ ở lại trong bộ nhớ, trừ khi bị evict vì vượt `max_loaded_models`, bị unload do `idle_unload_ms`, hoặc được gọi unload thủ công.
- `max_loaded_models`: số model được nằm trong bộ nhớ cùng lúc. Khi vượt quá, server unload model idle ít dùng gần đây nhất. Mỗi model OmniVoice f16 cần khoảng vài GB VRAM.
- `busy_timeout_ms`: thời gian tối đa một request chờ model đang bận. Quá thời gian này server trả HTTP 503 `server_busy`.

Cách ánh xạ option của CLI sang config:

| CLI | Trong config |
|---|---|
| `--session-option omnivoice.perf_mode=flash_attention` | `"session_options": {"omnivoice.perf_mode": "flash_attention"}` (giữ nguyên tiền tố `omnivoice.`) |
| `--request-option speed=1.1`, `--seed`, `--language`, `--instruct` ... | `"default_request_options": {...}`. Request gửi lên có thể ghi đè từng key |
| `--voice-ref` + `--reference-text` | `default_voice_preset` / `voice_presets` (mục 4.3) |

Nếu chạy từ safetensors, đặt `path` là thư mục model và thêm `session_options` `omnivoice.generator_weight_type=f16`, `omnivoice.audio_tokenizer_weight_type=f16`.

#### 4.1.1. Thêm lại các model đã bỏ

Config mặc định chỉ có `g-omnivoice` và `g-omnivoice-stream`. Các entry dưới đây từng có trong `configs/server.json` và đã chạy thử (lượt nghe #7, [báo cáo](reports/issue7-listening.md)). Muốn dùng entry nào thì thêm object tương ứng vào mảng `"models"` của `configs/server.json`, rồi khởi động lại audio.cpp:

```json
    {
      "id": "omnivoice-vietnamese",
      "family": "omnivoice",
      "path": "../models/omnivoice-vietnamese-GGUF/omnivoice-vietnamese-f16.gguf",
      "task": "tts",
      "mode": "offline",
      "default_request_options": { "language": "vi" }
    },
    {
      "id": "omnivoice",
      "family": "omnivoice",
      "path": "../models/audio.cpp-gguf/OmniVoice-GGUF/omnivoice-f16.gguf",
      "task": "tts",
      "mode": "offline"
    }
```

| Entry | Dùng khi | Cần file |
|---|---|---|
| `omnivoice-vietnamese` | Nghe A/B với một bản finetune tiếng Việt khác | `models/omnivoice-vietnamese-GGUF/omnivoice-vietnamese-f16.gguf` (convert ở mục 2; cần copy `audio_tokenizer/` từ bản gốc) |
| `omnivoice` | OmniVoice gốc đa ngôn ngữ, đối chứng | `models/audio.cpp-gguf/OmniVoice-GGUF/omnivoice-f16.gguf` |

Lưu ý khi thêm lại:

- Mỗi entry là một object trong mảng `"models"`, ngăn cách bằng dấu phẩy; JSON không cho dấu phẩy thừa sau phần tử cuối.
- Với `lazy_load: true`, file GGUF chỉ được mở ở request đầu tiên tới model đó. Thiếu file thì request đó lỗi, nên kiểm tra đường dẫn trước (`ls models/...`) hoặc gọi thử một request.
- `max_loaded_models: 2` vừa đủ cho `g-omnivoice` + `g-omnivoice-stream`. Thêm entry thì tăng giá trị này nếu GPU đủ VRAM, nếu không server sẽ unload model idle lâu nhất và request sau phải load lại.
- Muốn so model khác ở chế độ streaming thì thêm entry riêng có `"mode": "streaming"` (giống `g-omnivoice-stream`, đổi `id` và `path`).
- Gateway không cần sửa: [`configs/gateway.toml`](../configs/gateway.toml) vẫn giữ mặc định cho các id này (`language` cho `omnivoice-vietnamese`, `normalize = false` cho `omnivoice`). Gateway forward mọi model id; id không có trên upstream sẽ nhận lỗi của audio.cpp.
- Kiểm tra sau khi khởi động lại: `curl -s localhost:8080/v1/models` phải liệt kê id vừa thêm.

### 4.2. Khởi động

```bash
CUDA_VISIBLE_DEVICES=0 $SERVER --config configs/server.json --no-ui --log --log-file server.log
```

- Bỏ `--no-ui` (hoặc thêm `--ui`) để dùng WebUI tại `http://127.0.0.1:8080`.
- Có thể ghi đè config bằng `--host 0.0.0.0`, `--port 9000`, `--max-loaded-models 1`, `--idle-unload-ms 600000`, `--busy-timeout-ms ...`, `--voice-dir ...`.
- Server không có xác thực. Chỉ bind `0.0.0.0` trong mạng tin cậy, hoặc đặt sau gateway/reverse proxy có auth.

Kiểm tra:

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -s 'localhost:8080/v1/models?include_params=true'      # liệt kê request option model hỗ trợ
curl -s 'localhost:8080/v1/audio/voices?model=g-omnivoice'  # preset + giọng trong voice_dir
```

`/health` trả OK chỉ cho biết server đang chạy. Khi `lazy_load` bật, model chưa chắc đã được load. Request đầu tiên tới mỗi model sẽ chậm hơn vì phải load model.

Giải phóng VRAM:

```bash
curl -s -X POST localhost:8080/v1/tasks/unload_all_models
curl -s -X POST localhost:8080/v1/tasks/unload_models -H 'Content-Type: application/json' -d '{"model_ids":["g-omnivoice"]}'
```

### 4.3. Quản lý giọng clone trên server

Có ba cách để client không phải gửi kèm audio mẫu trong mỗi request.

**a) Thư viện giọng `voice_dir`** (dùng chung cho mọi model). Cấu trúc thư mục:

```text
voices/
  vieneu_v3_Mai_Anh.wav
  prompt_text          # đúng tên này, không có đuôi .txt
```

Mỗi dòng trong `prompt_text` có dạng `<tên file không đuôi>|<transcript>`:

```text
vieneu_v3_Mai_Anh|Chào mừng đến với website của chúng tôi. Đây là trang web cung cấp một giải pháp truyền văn bản thành giọng nói.
```

Khi request gửi `"voice": "vieneu_v3_Mai_Anh"`, server tự đọc `voices/vieneu_v3_Mai_Anh.wav` và lấy transcript tương ứng. Thêm giọng mới chỉ cần thêm file WAV và một dòng vào `prompt_text`, không phải restart server.

**b) Preset riêng cho từng model**, khai báo trong entry của model:

```json
"voice_presets": {
  "tong_dai_vien": { "voice_ref": "/abs/path/tong_dai_vien.wav", "reference_text": "..." },
  "doc_tin":       { "voice_ref": "/abs/path/doc_tin.wav",       "reference_text": "..." }
},
"default_voice_preset": "tong_dai_vien"
```

**c) Giọng mặc định**: `default_voice_preset` được dùng khi request không gửi `voice` và cũng không gửi `voice_ref`. Có thể khai báo inline thay vì tham chiếu tên preset: `"default_voice_preset": {"voice_ref": "...", "reference_text": "..."}`.

Thứ tự server resolve giọng:

1. Request có `voice_ref`: dùng audio này, bỏ qua tra cứu theo tên.
2. `voice` trùng tên một preset của model: dùng preset đó.
3. `voice` trùng tên một file `<voice_dir>/<voice>.wav`: clone từ thư viện.
4. Không khớp gì: giá trị `voice` được truyền thẳng vào model như một voice ID. OmniVoice không có giọng dựng sẵn, nên khi đó model sinh giọng auto hoặc giọng design.

Nếu request có `reference_text`, nó sẽ ghi đè transcript của preset hoặc của thư viện.

Server cache token của audio mẫu theo nội dung audio. Gửi cùng một giọng liên tục sẽ không phải encode lại audio mẫu.

---

## 5. API tương thích OpenAI

Endpoint: `POST /v1/audio/speech`. Mặc định trả về `audio/wav` (mono 24 kHz, PCM16).

### 5.1. Ánh xạ trường

| Trường JSON | Ý nghĩa với OmniVoice |
|---|---|
| `model` | `id` của entry trong config |
| `input` | Text cần đọc (bắt buộc) |
| `voice` | Tên preset hoặc tên giọng trong `voice_dir` (mục 4.3) |
| `instructions` | Chuẩn OpenAI. Server chuyển trường này thành `instruction`, mà OmniVoice coi là `instruct`, tức là **voice design** |
| `voice_ref` | Audio mẫu: đường dẫn trên **máy chủ** (string hoặc `{"type":"path","path":...}`), hoặc `{"type":"base64","data":"..."}` (tối đa 5 MiB, chấp nhận cả data URI) |
| `reference_text` | Transcript của audio mẫu |
| `language` | `vi`, `en`, ... |
| `speed` | Tốc độ nói, phải lớn hơn 0 |
| `seed`, `num_inference_steps`, `guidance_scale` | Có thể đặt ở top-level |
| `options` | Mọi request option khác: `instruct`, `duration_sec`, `class_temperature`, `text_chunk_size`, `denoise`, ... |
| `response_format` | `wav` (mặc định), `json` (WAV dạng base64 trong JSON), `pcm` (dùng khi streaming). Bản build hiện tại không hỗ trợ `mp3` |
| `stream_format` | `sse` hoặc `audio`. Chỉ dùng với entry có `mode: "streaming"` |

### 5.2. curl

```bash
U=http://127.0.0.1:8080/v1/audio/speech

# Auto voice
curl -s $U -H 'Content-Type: application/json' -o auto.wav \
  -d '{"model":"g-omnivoice","input":"Xin chào, đây là giọng tự động.","seed":42}'

# Voice design (cách OpenAI: instructions)
curl -s $U -H 'Content-Type: application/json' -o design.wav \
  -d '{"model":"g-omnivoice","input":"Chào mừng bạn đến với hệ thống.","instructions":"male, middle-aged, low pitch","seed":1}'

# Voice design (cách audio.cpp: options.instruct)
curl -s $U -H 'Content-Type: application/json' -o design2.wav \
  -d '{"model":"g-omnivoice","input":"Chào mừng bạn.","options":{"instruct":"female, child, high pitch"}}'

# Clone từ thư viện giọng
curl -s $U -H 'Content-Type: application/json' -o clone.wav \
  -d '{"model":"g-omnivoice","input":"Đây là giọng được nhân bản.","voice":"vieneu_v3_Mai_Anh"}'

# Clone từ file trên máy chủ
curl -s $U -H 'Content-Type: application/json' -o clone2.wav \
  -d '{"model":"g-omnivoice","input":"Nhân bản qua đường dẫn.",
       "voice_ref":"/abs/path/ref.wav","reference_text":"<transcript>"}'

# Clone bằng audio base64 gửi từ client (khi client và server ở hai máy khác nhau)
B64=$(base64 -w0 ref.wav)
printf '{"model":"g-omnivoice","input":"Nhân bản bằng base64.","voice_ref":{"type":"base64","data":"%s"},"reference_text":"<transcript>"}' "$B64" > req.json
curl -s $U -H 'Content-Type: application/json' --data-binary @req.json -o clone3.wav

# Tham số sinh
curl -s $U -H 'Content-Type: application/json' -o tuned.wav \
  -d '{"model":"g-omnivoice","input":"Thử ít bước.","voice":"vieneu_v3_Mai_Anh","speed":1.2,
       "num_inference_steps":16,"guidance_scale":2.0,"options":{"class_temperature":0.0}}'
```

Luôn kiểm tra HTTP status. Khi lỗi, body trả về là JSON `{"error":{...}}` chứ không phải WAV. Thêm `--fail-with-body` vào lệnh curl để curl báo lỗi trong trường hợp này.

### 5.3. Streaming

Request phải gửi tới entry có `mode: "streaming"` (ở đây là `g-omnivoice-stream`). Mỗi chunk text được sinh xong sẽ trả về ngay. `text_chunk_size` nhỏ thì nhận được audio đầu tiên sớm hơn, nhưng ngữ điệu giữa các chunk có thể kém liền mạch hơn.

SSE:

```bash
curl -N $U -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{"model":"g-omnivoice-stream","input":"Câu thứ nhất. Câu thứ hai dài hơn. Câu thứ ba.",
       "response_format":"pcm","stream_format":"sse","voice":"vieneu_v3_Mai_Anh",
       "options":{"text_chunk_size":60}}'
```

Luồng SSE trả về lần lượt:

```text
data: {"type":"speech.audio.delta","audio":"<base64 PCM s16le 24kHz mono>"}
...
data: {"type":"speech.audio.done","timing":{"ttft_ms":677.5}}
data: [DONE]
```

HTTP 200 lúc bắt đầu stream chưa có nghĩa là request đã thành công. Client phải đọc tới `speech.audio.done` và xử lý cả event lỗi. Nếu đặt sau nginx, nhớ tắt proxy buffering.

PCM thô qua chunked transfer:

```bash
curl -sN $U -H 'Content-Type: application/json' -o out.pcm \
  -d '{"model":"g-omnivoice-stream","input":"Thử stream PCM thô.","response_format":"pcm","stream_format":"audio","voice":"vieneu_v3_Mai_Anh"}'
ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
```

### 5.4. Python, OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

# Clone từ voice_dir
r = client.audio.speech.create(
    model="g-omnivoice", voice="vieneu_v3_Mai_Anh",
    input="Xin chào từ OpenAI SDK.", response_format="wav",
)
r.write_to_file("clone.wav")

# Voice design: SDK bắt buộc có voice, truyền chuỗi rỗng
r = client.audio.speech.create(
    model="g-omnivoice", voice="",
    input="Giọng thiết kế qua SDK.",
    instructions="female, young adult, high pitch",
    response_format="wav",
    extra_body={"seed": 3, "num_inference_steps": 24},   # các trường ngoài chuẩn OpenAI
)
r.write_to_file("design.wav")

# Streaming PCM thô
with client.audio.speech.with_streaming_response.create(
    model="g-omnivoice-stream", voice="vieneu_v3_Mai_Anh",
    input="Thử streaming bằng SDK. Câu thứ hai.",
    response_format="pcm",
    extra_body={"stream_format": "audio"},
) as resp:
    with open("out.pcm", "wb") as f:
        for chunk in resp.iter_bytes():
            f.write(chunk)      # s16le, 24 kHz, mono
```

Các trường riêng của audio.cpp như `voice_ref`, `reference_text`, `options`, `seed`, `language` được truyền qua `extra_body`.

### 5.5. Python, chỉ dùng thư viện chuẩn và SSE

```python
import base64, json, urllib.request, wave

payload = {
    "model": "g-omnivoice-stream", "input": "Câu một. Câu hai. Câu ba.",
    "voice": "vieneu_v3_Mai_Anh", "response_format": "pcm", "stream_format": "sse",
}
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/audio/speech",
    data=json.dumps(payload, ensure_ascii=False).encode(),
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
)
pcm = bytearray()
with urllib.request.urlopen(req, timeout=600) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "speech.audio.delta":
            pcm += base64.b64decode(ev["audio"])     # phát ngay nếu cần
        elif ev.get("type") == "speech.audio.done":
            print("timing:", ev.get("timing"))
        elif "error" in ev:
            raise RuntimeError(ev)

with wave.open("stream.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
```

---

## 6. Mẹo và xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `requires reference_text when reference audio is provided` | Clone bắt buộc phải có transcript: thêm `--reference-text` hoặc `reference_text`, hoặc thêm dòng vào `voice_dir/prompt_text` |
| `unsupported OmniVoice language` | Dùng mã ISO (`vi`) hoặc tên ngôn ngữ (`Vietnamese`) |
| `could not open WAV input` | Đường dẫn `voice_ref` phải tồn tại trên máy chủ. Client ở máy khác thì gửi audio dạng base64 |
| Lỗi chọn model khi `--model` là thư mục | Thư mục có nhiều `.gguf`. Truyền thẳng đường dẫn tới file |
| `merged batch audio requires --out` | `--batch-merge-audio concat` cần có `--out <file>` |
| Streaming không ra từng đoạn | Request phải gửi tới entry `mode: "streaming"` và có `stream_format` |
| HTTP 503 `server_busy` | Model đang xử lý request khác quá `busy_timeout_ms`. Tăng timeout, hoặc chạy thêm process server trên GPU khác |
| Hết VRAM | Giảm `max_loaded_models`, đặt `idle_unload_ms`, hoặc bật `omnivoice.mem_saver=true` |
| Audio clone có khoảng lặng dài ở đầu, hoặc Whisper đọc ra câu bịa | Thường do audio mẫu. Hãy cắt khoảng lặng của audio mẫu, giữ `preprocess_prompt=true` và `postprocess_output=true` |
| Audio rè hoặc không nghe được | Kiểm tra lại có lỡ đặt `omnivoice.audio_tokenizer_weight_type=q8_0` không |
| Câu rất ngắn bị lỗi khi dùng auto hoặc design | Dùng voice clone, hoặc nối thêm vào câu |

Một số khuyến nghị:

- Nên dùng **f16 GGUF**. Nếu cần giảm độ trễ, hạ `num_inference_steps` xuống 16–24 rồi nghe thử chất lượng.
- Có thể thử `omnivoice.perf_mode=flash_attention`. Mình đã chạy thử được nhưng chưa đo tốc độ trên A100.
- Cố định `seed` để kết quả tái lập được. Khi đã cố định seed thì mỗi request cùng tham số cho cùng một kết quả.
- Đo tốc độ bằng request warm (chạy lại sau request đầu tiên). Trong response timing của server, `rtf` là `wall_ms` chia cho độ dài audio, không tính thời gian load model.
- Text dài nên được chia theo câu trước khi gửi, hoặc đặt `text_chunk_size`. Chuẩn hóa số, ngày tháng và viết tắt sang chữ trước khi sinh.
