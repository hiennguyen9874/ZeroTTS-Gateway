# #17 — Thư mục weights `models/ZeroTTS-torch`

## Môi trường và cách làm

Kiểm chứng trên máy dev, Linux x86_64, Python 3.12.14, numpy 2.5.4, gguf 0.19.0,
safetensors 0.8.0, onnx 1.20.1, huggingface_hub 2.2.0. Thư mục nguồn `models/ZeroTTS`,
HF revision `c2bfbd67dc648cac455077333f7cf5c18a2e3bb4`.

```bash
uv run --group torch-weights python scripts/build_zerotts_torch.py \
  --revision c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
```

Nguồn tải từ Hub (file thật, không qua symlink cache HF), SHA-256 khớp giá trị pin trong script:

| Nguồn | Revision | File | SHA-256 |
| --- | --- | --- | --- |
| `zeroweight-ai/ZeroTTS-GGUF` | `92ca8651645d4733df56620b3aadc768a76f7c46` | `gguf/zerotts-f32.gguf` | `9b0490c317b369e9a77a17eb1f77f5d5fb98d6e3fcb82a3002e8ea62ebbab531` |
| `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` | `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68` | `model-00001-of-00001.safetensors` | `34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67` |

## Kết quả

| File | Byte | Tensor | SHA-256 |
| --- | ---: | ---: | --- |
| `zerotts.safetensors` | 809312832 | 377 | `0fa115d51839b02ba02d29dd56933eef116692e820599fc745850d3f38720862` |
| `codec.safetensors` | 46104040 | 246 | `51c7f3243183984b8c39007bc70bfec4d01d96985d908941da121cd3105355c9` |
| `manifest.json` | | | `2243077a6aceae90773bcaf2202be77ed2e46f296038baa127a14baec4f1e9b2` |

- Mọi tensor trong GGUF f32 (377, đều F32) có trong `zerotts.safetensors`, cùng tên và giá trị bằng nhau tuyệt đối (đọc lại từ đĩa).
- `codec.safetensors` chỉ gồm `decoder.*` (128) và `quantizer.*` (118); bỏ 128 tensor encoder.
- Cả 178 initializer của `moss_audio_tokenizer_decode_full.onnx` và `moss_audio_tokenizer_decode_step.onnx` đều khớp một tensor codec (giữ nguyên, chuyển vị, hoặc weight-norm đã gộp). 17 tensor là weight-norm đã gộp; sai khác lớn nhất 8,9e-8 (ngưỡng 1e-6). Thử nhân một tensor codec với (1 + 2e-5) thì kiểm tra báo lỗi.
- Thư mục có 27 file (kể cả manifest), không có symlink hay `.cache`; kích thước và SHA-256 mọi file khớp manifest. Thư mục nguồn giữ nguyên byte.

## Tái lập

Hai lần dựng độc lập vào hai thư mục (một lần tải mới vào thư mục tạm, một lần dùng `--download-dir`), rồi một lần `--force` vào `models/ZeroTTS-torch`: `manifest.json` giống hệt từng byte, nên SHA-256 mọi file giống nhau.

Lần thử đầu, `zerotts.safetensors` khác hash giữa hai lần chạy: safetensors ghi `__metadata__` từ hash map nên thứ tự các key thay đổi. Script hiện chỉ ghi một key `gguf` (JSON các key-value của GGUF), nên header ổn định.

Đã kiểm tra CLI từ chối: đích đã tồn tại mà thiếu `--force`, revision không đủ 40 ký tự hex, nguồn và đích chồng nhau, nguồn đã int8; các lỗi này không tạo hay sửa đích.

Không suy rộng sang phiên bản thư viện khác. Nhiều node nên copy một thư mục hoàn chỉnh, gồm manifest.
