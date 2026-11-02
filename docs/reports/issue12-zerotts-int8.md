# #12 — Weights int8 tuỳ chọn cho zerotts_server

## Môi trường và cách làm

Kiểm chứng ngày 2026-10-28 trên máy dev dùng chung, Linux x86_64, Intel Xeon Silver 4216, Python 3.12, `onnx==1.20.1`, `onnxruntime==1.31.1`. Nguồn: thư mục local `models/ZeroTTS`, HF revision `c2bfbd67dc648cac455077333f7cf5c18a2e3bb4`.

Theo `references/ZeroTTS-repo/tools/quantize_onnx_int8.py`: fold graph bằng ORT_ENABLE_BASIC để lộ weights sau Identity, rồi quantize động QInt8 cho MatMul của ba graph nóng. Codec giữ fp32. Không sửa engine, server extra hay image.

## Kích thước và tái lập

| Graph | fp32 (byte) | int8 (byte) | MatMulInteger |
| --- | ---: | ---: | ---: |
| text_encoder.onnx | 323096725 | 100149923 | 72 |
| prefix_step.onnx | 348303894 | 125374501 | 72 |
| local_frame_decode.onnx | 186743011 | 83086162 | 425 |

Hai lần quantize độc lập trên cùng máy, cùng nguồn và phiên bản thư viện cho **SHA-256 giống nhau ở cả ba graph**, manifest giống nhau. Chạy lại với `--force` vào thư mục kiểm chứng cũng cho cùng kết quả.

| Graph | SHA-256 int8 (cả hai lần) |
| --- | --- |
| text_encoder.onnx | `663f66993dbf1da88ced3b73692cf8efa9d19c7bbd3f0bdf9dc4a37814408ea0` |
| prefix_step.onnx | `764ce324a33ee34783235eb406dd77ef8554a2c609122f418cea06cfc3c5d82c` |
| local_frame_decode.onnx | `2564547923658c69f5bef2e1102b2efc82169cada6e86eb366a8e11be7dc2f26` |

Đã đối chiếu SHA-256 nguồn với manifest và kiểm tra mọi file runtime khác (config, tokenizer, voices, codec, kể cả external data) giữ nguyên byte. Thư mục đích không có symlink hay metadata cache HF. Kiểm tra CLI từ chối: đích đã tồn tại mà thiếu `--force`, nguồn/đích chồng nhau, nguồn đã int8, nguồn thiếu file, revision không khớp metadata HF; các lỗi không sửa đích.

Không suy rộng kết quả này sang máy hay phiên bản khác. Nhiều node nên copy **một thư mục int8 hoàn chỉnh, gồm manifest**, mount read-only. Nếu các lần quantize cho hash khác nhau thì bắt buộc dùng chung một bản như vậy để tái lập seed.

## Smoke test và seed

`ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 uv run python -m pytest -m real_model tests/test_server_smoke.py -q -s`: **4 passed** (51,77 s). Kiểm tra offline WAV mono 48 kHz, nhiều đoạn dài, seed không phụ thuộc RNG toàn cục, SSE nhiều đoạn. TTFA đo được 151,2 ms; đây không phải benchmark node triển khai hay đánh giá chất lượng giọng.

Cùng seed 1234, voice maichi, cùng request offline qua HTTP: SHA-256 WAV fp32 là `90ab15e2f3f7e44f8ddae67a1366ed8283466a581e0ae5f8c907a83ebf2876ba`, int8 là `e0f467c14f6dc6023c207aac7348d52ef14ef15d7793aad2704ab40180735acd`. Audio **khác nhau**. Seed chỉ tái lập trên cùng bộ weights/runtime, không phải giữa fp32 và int8.

Test HTTP với fake engine/thư mục model giả: `uv run python -m pytest tests/test_server.py -q` — **119 passed**, gồm hai ca log startup fp32/int8, giữ nguyên model id. Hai ca log đã fail trước khi thêm hành vi mới và pass sau sửa.

`uv run pytest -q` — **488 passed, 1 failed, 4 deselected**. Lỗi còn lại là `tests/test_gateway.py::test_shipped_config_loads`: test mong upstream cổng 8080 nhưng config local do operator sửa sẵn dùng 38080. Giữ nguyên các sửa đổi config đó, không đổi test để che lỗi. `uv run ruff check .`, `uv run ruff format --check .` và `git diff --check` pass. Chưa build Docker, chưa A/B nghe int8 hay benchmark node thật.

## Tái hiện

Chạy từ gốc repo. Chọn hai thư mục đích chưa tồn tại (hoặc chủ động dùng `--force` để thay thế đúng thư mục đích); không cần tải lại weights:

```bash
uv sync --group quantize
REV=c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
uv run --group quantize python scripts/quantize_zerotts.py --revision "$REV"
uv run --group quantize python scripts/quantize_zerotts.py --revision "$REV" --dst models/ZeroTTS-int8-repro
diff models/ZeroTTS-int8/quantization.json models/ZeroTTS-int8-repro/quantization.json
sha256sum models/ZeroTTS-int8/onnx/*.onnx
sha256sum models/ZeroTTS-int8-repro/onnx/*.onnx
ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 uv run python -m pytest -m real_model tests/test_server_smoke.py -q -s
```

Artifact local: `models/ZeroTTS-int8/quantization.json`, `models/ZeroTTS-int8-repro/quantization.json` (gitignored, cùng weights). Log của lượt kiểm chứng: `/tmp/issue12-quantize-1.log`, `/tmp/issue12-quantize-2.log`, `/tmp/issue12-quantize-force.log`, `/tmp/issue12-validation.log`, `/tmp/issue12-smoke-int8.log`, `/tmp/issue12-cross-weights.log`, `/tmp/issue12-pytest.log`. Bảng hash và kết quả ở trên là bản ghi bền vững trong repo.
