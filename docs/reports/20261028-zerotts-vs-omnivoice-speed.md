# Đo tốc độ ZeroTTS và OmniVoice trên máy đang chạy

Ngày đo: 2026-10-28. Không đổi cấu hình, restart hay sửa source của hai dịch vụ.

## Kết luận

Với câu mẫu của operator, **OmniVoice hoàn tất WAV nhanh hơn khoảng 3,68 lần**. Đây là so sánh hai cấu hình triển khai hiện tại: **ZeroTTS ONNX CPU FP32** với **OmniVoice audio.cpp CUDA FP16**, không phải benchmark hai model trên cùng backend.

ZeroTTS không bị chậm ở gateway hay normalizer. Nút thắt là vòng sinh frame autoregressive trên CPU, đặc biệt `local_frame_decode.onnx`. Tuy nhiên, nếu cần phát ngay khi có âm thanh đầu tiên, **ZeroTTS streaming có TTFA thấp hơn OmniVoice streaming** trong phép đo này.

## Điều kiện đo

- Máy: Xeon Silver 4216 @ 2,10 GHz, 32 core vật lý / 64 logical CPU, RAM 503 GiB; máy dùng chung, load khoảng 8–9 tại lần kiểm tra đầu. Không có CPU quota/throttling ở cgroup hiện tại.
- ZeroTTS: gateway `:48000`, upstream `:38080`, `zerotts==0.1.5`, ONNX Runtime `1.31.1`, `workers=1`, `threads=8`, `cfg_scale=1.0`, voice `baotrang`. Runtime mặc định dùng `CPUExecutionProvider`; môi trường hiện tại chỉ có CPU/Azure providers, không có CUDA provider.
- OmniVoice: gateway `:8000`, upstream `:8080`, audio.cpp CUDA, device 0 (A100 40 GB), `g-omnivoice-f16.gguf`, 32 inference steps, guidance 2; voice `vieneu_v3_Ngoc_Linh`. Tiến trình upstream đang mở `/dev/nvidia0`.
- Input: `Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.`
- Cả hai gateway tạo **cùng engine text**: `hẹn gặp lúc tám giờ ba mươi phút ngày hai mươi lăm tháng mười hai, phí năm mươi nghìn.`
- Request tuần tự, không chạy hai TTS đồng thời, `seed=42`; đo bằng Python stdlib HTTP client và `time.perf_counter()`.
- Offline: 4 lượt mỗi model, bỏ lượt đầu, báo median 3 lượt còn lại. Đây là đo warm service, không đo cold startup.
- Streaming: 3 lượt mỗi model, bỏ lượt đầu, median 2 lượt còn lại. TTFA tính tại client khi đọc được event chứa PCM, không tính lúc chỉ nhận HTTP headers. Mẫu nhỏ, chưa đại diện p95 hay tải đồng thời.

## Kết quả offline qua gateway

| Chỉ số | ZeroTTS | OmniVoice |
| --- | ---: | ---: |
| Median thời gian nhận đủ WAV | 2,9818 s | 0,8102 s |
| Khoảng thời gian 3 lượt đo | 2,9140–3,0066 s | 0,8049–0,8193 s |
| Thời lượng audio | 3,92 s | 4,81 s |
| Sample rate | 48 kHz | 24 kHz |
| RTF = thời gian xử lý / thời lượng audio | 0,7607 | 0,1684 |

Cả hai đều nhanh hơn realtime (RTF < 1). OmniVoice nhanh hơn 3,68 lần theo thời gian chờ toàn bộ câu, hoặc khoảng 4,52 lần theo RTF. Hai voice/model có tốc độ đọc và thời lượng đầu ra khác nhau; kết quả này không đánh giá chất lượng hay mức đọc đủ nội dung.

## Tách gateway khỏi upstream

- `POST /v1/text/normalize`, median 10 HTTP request: ZeroTTS **1,98 ms**, OmniVoice **1,94 ms**. Số này gồm cả HTTP overhead, không chỉ riêng thời gian xử lý text.
- Gửi thẳng cùng engine text vào ZeroTTS upstream: 2,9392 / 2,8938 / 3,0269 s. Header `X-AudioCPP-Wall-Ms` tương ứng 2937,1 / 2891,9 / 3025,1 ms: hầu hết thời gian nằm trong synthesis.
- Gửi thẳng OmniVoice upstream: 0,8346 / 0,7298 / 0,7263 s. Dao động so với lần đo gateway phản ánh thời điểm/cache/backend, không nên lấy hiệu median của hai nhóm làm overhead chính xác.
- Gateway không split, synthesize hay resample. WAV offline chỉ sẵn sàng sau khi upstream hoàn thành câu; thay đổi cách chuyển bytes HTTP không loại bỏ chi phí synthesis.

## Profile ZeroTTS và thử số thread

Tạo engine riêng trong tiến trình benchmark, không thay engine của service. Dùng cùng engine text, voice, sampling và seed; bọc `session.run()` để đo thời gian theo graph. Mỗi cấu hình chạy 4 lượt, bỏ lượt đầu.

| ONNX threads | Median thời gian engine | Audio |
| ---: | ---: | ---: |
| 2 | 6,0787 s | 3,92 s |
| 4 | 3,8957 s | 3,92 s |
| 8 (hiện tại) | 3,0397 s | 3,92 s |
| 16 | 3,1300 s | 3,92 s |

Không có bằng chứng nên tăng lên 16 thread; 8 tốt nhất trong bốn cấu hình đã thử, chưa phải tối ưu toàn cục. Máy không được pin core/NUMA và vẫn có workload khác.

Ở lượt có tổng thời gian median của cấu hình 8 thread:

| Thành phần | Số lần gọi | Thời gian | Tỷ trọng tổng |
| --- | ---: | ---: | ---: |
| Text encoder | 1 | 0,0367 s | 1,2% |
| Global/prefix decoder | 49 | 0,6223 s | 20,5% |
| Local frame decoder | 49 | 2,0794 s | 68,4% |
| Codec decode toàn câu | 1 | 0,2860 s | 9,4% |

49 frame × 80 ms = 3,92 s audio. Runtime sinh frame tuần tự, với global và local ONNX calls mỗi frame; đây là phần chi phối thời gian. Overhead ngoài các graph khoảng 0,5%. `cfg_scale` đã là 1,0 nên không có chi phí nhân đôi batch do CFG > 1.

## Streaming: nhanh bắt đầu không đồng nghĩa nhanh hoàn tất

Dùng `zerotts-stream` / `g-omnivoice-stream`, `stream=true`, `stream_format=sse`, `response_format=pcm`.

| Chỉ số warm, median 2 lượt | ZeroTTS | OmniVoice |
| --- | ---: | ---: |
| Client nhận audio đầu tiên | **0,1620 s** | 0,7213 s |
| Nhận hết response | 3,3752 s | **0,7273 s** |

ZeroTTS gửi nhiều audio delta nhờ decode ramp 1, 2, 4, 8, 16 frame. Với câu ngắn này, OmniVoice chỉ gửi một audio delta, gần như sau khi xong cả câu. Lần đầu gọi entry OmniVoice streaming mất 6,71 s ở client, trong khi timing nội bộ báo khoảng 771 ms; phù hợp với overhead khởi tạo/load ngoài phần timing synthesis nhưng chưa profile riêng nguyên nhân. Lượt này bị loại khỏi số warm, không được diễn giải là tốc độ ổn định.

PCM streaming phải phát đúng sample rate: ZeroTTS 48 kHz, OmniVoice 24 kHz. Request offline của operator không bật streaming nên chưa tận dụng lợi thế TTFA của ZeroTTS.

## Vì sao khác con số quảng bá?

`references/ZeroTTS-repo/README.md`, mục “Speed — CPU”, so sánh **tất cả model trên CPU**, pin 8 thread vào core pool riêng. Bảng đó không phải OmniVoice CUDA/audio.cpp trên A100. Không có mâu thuẫn khi ZeroTTS nhanh hơn OmniVoice CPU trong bảng của tác giả nhưng chậm hơn cấu hình GPU hiện tại.

## Hướng xử lý

1. **Ưu tiên nhận cả WAV nhanh nhất:** cấu hình OmniVoice CUDA hiện tại thắng ở mẫu này.
2. **Ưu tiên audio đầu tiên sớm:** thử client phát `zerotts-stream` ngay khi nhận delta; không chờ gom hết response.
3. **Tối ưu ZeroTTS CPU tiếp:** benchmark một bản ONNX INT8 riêng và pin core/NUMA. Upstream có `tools/quantize_onnx_int8.py`; README nói INT8 giảm RTF trên DDR4, nhưng chưa đo ở lần kiểm tra này. Cần kiểm tra ASR/nghe lại trước khi thay weights, không mặc định coi tốc độ hay chất lượng được bảo đảm.
4. **Không chỉ tăng workers để giảm latency một request:** workers tăng năng lực xử lý đồng thời, không rút ngắn vòng sinh frame của một câu; còn có thể tranh CPU/băng thông RAM.
5. **Không mặc định chuyển sang ONNX CUDA:** [ADR 0005](../adr/0005-zerotts-replaces-omnivoice.md) ghi nhận lần thử trước trên A100 đạt RTF 1,08–1,31, chậm hơn CPU vì KV cache chuyển qua host mỗi frame. Đây là bằng chứng lịch sử, không phải phép đo GPU mới trong báo cáo này. Cần tối ưu runtime/device-resident cache rồi đo lại, không chỉ cài GPU provider.

## Dữ liệu và cách đo lại nhanh

Dữ liệu cục bộ, gitignored: `outputs/tts-speed-comparison-20261028/`:

- `offline.json`: thời gian client, thời lượng audio và response headers.
- `direct-and-streaming.json`: direct upstream và streaming.
- `tts-profile-{2,4,8,16}.json`: thời gian load và profile từng lượt theo số thread.

Đo lại WAV với cùng seed; chạy mỗi lệnh vài lần, bỏ lượt đầu. `time_starttransfer` của WAV offline gần bằng thời gian hoàn tất câu, **không phải TTFA của streaming**. File output đặt trong `/tmp` để không ghi đè audio của operator.

```bash
curl -sS --fail http://localhost:48000/v1/audio/speech \
  -H 'Content-Type: application/json' -o /tmp/bench_zerotts.wav \
  -w 'ZeroTTS headers=%{time_starttransfer}s total=%{time_total}s\n' \
  -d '{"model":"zerotts","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"baotrang","seed":42}'

# Đặt OMNIVOICE_GATEWAY_API_KEY bằng API key của service trước khi chạy.
curl -sS --fail http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OMNIVOICE_GATEWAY_API_KEY" \
  -o /tmp/bench_omnivoice.wav \
  -w 'OmniVoice headers=%{time_starttransfer}s total=%{time_total}s\n' \
  -d '{"model":"g-omnivoice","input":"Hẹn gặp lúc 8h30 ngày 25/12, phí 50k.","voice":"vieneu_v3_Ngoc_Linh","seed":42}'
```

Giới hạn: một câu và một seed, các voice khác nhau, không benchmark concurrency hay chấm chất lượng. Không dùng số này để kết luận mọi văn bản hoặc mọi cấu hình ZeroTTS đều chậm hơn OmniVoice.
