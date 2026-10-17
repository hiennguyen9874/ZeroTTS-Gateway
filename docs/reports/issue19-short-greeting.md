# #19 — Giảm mất lời chào ở đầu câu ngắn

## Quyết định

Giữ tốc độ 0.7 cho engine text ≤ 8 âm tiết; dùng 0.6 cho 9–12 âm tiết ở `g-omnivoice` và `g-omnivoice-stream`. Operator chọn các giá trị bằng `short_text_speed_bands` trong `configs/gateway.toml` ([ADR 0004](../adr/0004-short-text-speed.md)). Không sửa engine text, không thêm request, không đổi mặc định tham số sinh trong code, không bật cho model chưa đo. Caller gửi `speed` hoặc `options.duration_sec`/`duration` thì gateway không thêm tốc độ.

Đây là **biện pháp giảm lỗi**, không bảo đảm model luôn đọc đủ. Không hạ đồng loạt mọi text ngắn xuống 0.6: có thể làm các ca cực ngắn lặp chữ. Các ca regression của #18 (5–8 âm tiết) giữ đúng tốc độ cũ; chỉ câu dài hơn trong phạm vi text ngắn được cấp thêm thời lượng.

## Bằng chứng và nguyên nhân

Upstream audio.cpp `65df584a`, GGUF G-OmniVoice f16, A100; giọng Doan_Trang, Quang_Son, Mai_Anh, Thuy_Dung, Ngoc_Huyen, Thien_Tam_Duc. Chấm bằng gipformer (LCS / số từ mong đợi, như `render_samples.py`); recall không phạt chữ thừa nên phải xem cả transcript. Đây là kết quả ASR, **chưa phải chốt chất lượng bằng tai người**.

### Dữ liệu thí nghiệm đã có trong workspace

`outputs/listen-issue19/c1/rows.csv`, gọi thẳng upstream, câu `Xin chào, hôm nay chúng ta học bài mới.`, 6 giọng × seed 42, 1, 2:

| Điều kiện | Recall trung bình | Đủ mọi từ |
|---|---:|---:|
| Mặc định | 0.9507 | 14/18 |
| Mặc định, tắt postprocess | 0.9507 | 14/18 |
| speed 0.7 | 0.9753 | 16/18 |
| speed 0.7, tắt postprocess | 0.9753 | 16/18 |
| speed 0.6 | 1.0000 | 18/18 |
| speed 0.6, tắt postprocess | 1.0000 | 18/18 |

Ở Quang_Son, seed 42, tắt postprocess: speed 0.7 cho audio 3.24 s, im lặng đầu 1.743 s, ASR chỉ có “HÔM NAY CHÚNG TA HỌC BÀI MỚI”; speed 0.6 cho 3.76 s, im lặng đầu 1.931 s nhưng **đủ lời chào**. Thien_Tam_Duc, seed 1: 0.7 cho 3.96 s, im lặng đầu 2.017 s và mất lời chào; 0.6 cho 4.60 s, im lặng đầu 0.666 s và đủ lời chào. Vì vậy không phải postprocess cắt mất lời chào đã được sinh. Khoảng lặng đầu dài cũng không tự nó chứng minh mất chữ.

Đã chạy lại riêng hai tổ hợp giọng/seed này trong `diagnose-fresh/results.csv` ngày 2026-10-17: các mức mặc định / 0.7 / 0.6 với postprocess tắt tái lập đúng transcript và số đo trên. Gửi `options.duration_sec: 4` hoặc `5` thay cho speed cũng khôi phục đủ lời chào ở cả hai tổ hợp (4/4 mẫu); WAV dài 4.20 / 5.20 s do upstream thêm padding. Do đó cấp thêm thời lượng giúp được, nhưng không dùng duration cố định ở gateway vì mỗi giọng và độ dài text có ước lượng khác nhau.

Giả thuyết speed chỉ có tác dụng yếu do căn bậc ba là sai theo code upstream: `src/models/omnivoice/prompt_builder.cpp`, `RuleDurationEstimator::estimate_duration` kéo dài ước lượng dưới **50 frame** (2 s), rồi `estimate_target_tokens` mới **chia cho speed** và làm tròn xuống frame. Speed tác động đầy đủ cả trên lẫn dưới ngưỡng; thời lượng chưa postprocess ở trên xác nhận điều đó. Việc bỏ cụm đầu có liên quan tới mức thời lượng cấp cho model, nhưng không thể quy hoàn toàn cho công thức ngưỡng 2 s.

`c2/rows.csv` còn có các cụm mở đầu khác: `Chào bạn,`, `Lưu ý,`, `Sáng nay,` cũng có lần mất từ ở tốc độ mặc định; `Vâng,` không lỗi trong 18 mẫu. Bỏ dấu phẩy không chữa ổn định (0.9753 cả ở 1.0 lẫn 0.7); đổi thành dấu chấm còn kém hơn (0.9137 / 0.9260). Không có bằng chứng để thêm luật dành riêng cho “xin chào” hay xóa dấu câu của caller.

### Vì sao không đổi toàn bộ short_text_speed thành 0.6

`c3/rows.csv` (6 giọng × 3 seed), speed 0.6: recall lần lượt là greeting 1.000, heading 1.000, paragraph-lines 1.000, short-lines 1.000, bullets 0.9792, numbered-list 1.000. Nhưng `short-lines` có trường hợp thành “XIN CHÀO TẠM BIỆT **TẠM BIỆT** HẸN GẶP LẠI”: recall 1.000 che mất chất lượng bị giảm. Chọn band theo độ dài để không kéo chậm thêm 5 ca cũ chỉ vì câu 9 âm tiết này.

Đối chứng nhiều seed hơn, `c4-10seeds/rows.csv` (42, 1–9; 60 mẫu/cấu hình):

| Câu | speed | Recall | Đủ mọi từ |
|---|---:|---:|---:|
| Greeting | 0.7 | 0.9741 | 53/60 |
| Greeting | 0.65 | 0.9926 | 58/60 |
| Greeting | 0.6 | 0.9852 | 56/60 |

0.65 có chữ thừa trong dữ liệu quét; 0.6 không có chữ thừa ở greeting trong 60 mẫu này, và khôi phục cả 18 mẫu của bộ acceptance ban đầu. Hiệu quả **không đơn điệu** theo speed, không nên khẳng định đã tìm ra tốc độ tối ưu hay nguyên nhân bên trong model. Ở 0.6 vẫn mất lời chào với Thuy_Dung seed 7, Ngoc_Huyen seed 4 và 7, Thien_Tam_Duc seed 5. Đây vẫn là giới hạn upstream; không thêm retry, queue hay ghép audio để che lỗi.

## Chạy lại qua gateway với config mới

Bộ xác nhận ngày 2026-10-17 dùng code nền `bbbc848` cùng thay đổi #19, cùng sáu giọng và ba seed trên, normalize tắt **và** bật; cả model offline lẫn streaming (PCM được `render_samples.py` ghi thành WAV để chấm). WAV, transcript, engine text và manifest ở `outputs/listen-issue19/verify-bands/<model>/<off|on>/<voice>/<seed>/{golden,listening}`.

Hoàn tất **432/432 request** thành công, không lỗi ASR. Mỗi ô là recall trung bình của 18 mẫu; “tắt / bật” là normalize. `short-greeting` đủ từ **72/72 mẫu** qua hai model và hai chế độ normalize.

| Ca | Offline tắt | Offline bật | Streaming tắt | Streaming bật | #18 tắt / bật |
|---|---:|---:|---:|---:|---|
| short-greeting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9753 gọi thẳng, speed 0.7 |
| heading | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 |
| paragraph-lines | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 |
| short-lines | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 |
| bullets | 1.0000 | 0.9792 | 1.0000 | 0.9861 | 1.000 / 0.9792 |
| numbered-list | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 |

Tất cả ca ≥ 0.95 và không kém baseline #18. `summary.csv` trong thư mục xác nhận chứa số liệu tổng hợp. `bullets` vẫn có lỗi upstream ở normalize bật; không che lỗi bằng cách loại mẫu khỏi thống kê.

So sánh với #18: yêu cầu từng ca ≥ 0.95 và không thấp hơn baseline tương ứng. Các ca 5–8 âm tiết không đổi engine text hay tham số sinh so với #18. Không dùng recall như bảo đảm không có chữ thừa; cần nghe WAV trước khi chốt chất lượng cảm nhận.

### Tái lập

Khởi động upstream, gateway với `--config configs/gateway.toml` và ASR theo README. Nếu môi trường đặt `UV_NO_SYNC=true`, cài nhóm ASR bằng `UV_NO_SYNC=false uv sync --group asr --group dev` trước. Sau đó chạy ở thư mục gốc repo (chọn thư mục output mới cho mỗi lượt):

```bash
export OMNIVOICE_ASR_URL=http://127.0.0.1:8090
out=outputs/listen-issue19/replay
for model in g-omnivoice g-omnivoice-stream; do
  for norm in off on; do
    for voice in Doan_Trang Quang_Son Mai_Anh Thuy_Dung Ngoc_Huyen Thien_Tam_Duc; do
      for seed in 42 1 2; do
        args=(--model "$model" --voice "vieneu_v3_$voice" --seed "$seed" --no-direct)
        if [ "$norm" = on ]; then args+=(--normalize-on); fi
        base="$out/$model/$norm/$voice/$seed"
        uv run python scripts/render_samples.py "${args[@]}" --set golden \
          --id heading --id paragraph-lines --id short-lines --id bullets \
          --id numbered-list --out "$base/golden"
        uv run python scripts/render_samples.py "${args[@]}" --set listening \
          --id short-greeting --out "$base/listening"
      done
    done
  done
done
```

Để kiểm nguyên nhân riêng, gọi thẳng upstream với engine text của ca `short-greeting`, cùng giọng/seed, `response_format: wav`, lần lượt bỏ `speed` / gửi 0.7 / 0.6, mỗi cấu hình bật và tắt `options.postprocess_output`. Đo độ dài và khoảng lặng trên WAV **chưa postprocess**, không suy từ audio đã trim. Không sửa file trong `references/` hay giọng mẫu.
