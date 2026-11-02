# Issue #16: đối chiếu code ZeroTTS đã chép

Trên Linux x86_64, Python 3.12.14, numpy 2.5.4, onnxruntime 1.31.1 và tokenizers 0.23.3, chạy HTTP qua `TestClient` trước và sau khi chép code. Trước khi chép, package `zerotts==0.1.5` cài trong môi trường có cây `src/zerotts` giống từng file với submodule commit `319c3e0`. Hai lần dùng cùng weights, `maichi`, seed `1234`, input `Xin chào các bạn. Hôm nay trời rất đẹp.`, mặc định sampling và 8 thread. Offline tính SHA-256 toàn bộ WAV; streaming tính SHA-256 PCM nối từ các SSE delta, sau khi kiểm tra `done` và `[DONE]`.

| Weights | Offline WAV | Streaming PCM |
| --- | --- | --- |
| fp32 | `cfa1a2aa11845b5d518094ca14e2b840bf5ef63cc8925b4a649ee4a8216368ff` | `1cc72857629becb3ec3955e99e1e2597fdffbf99c23a12addbc0a7de5664141f` |
| int8 | `51206719bd450ca93ff8d563a5c8072a9820377be5b891c939718910bd81c9fa` | `6a1c2dd7d0c74c30b7f35cbd5620adc5345c9f056947fdc6ba43d16aaf62e420` |

Cả bốn hash trước và sau đều trùng. fp32 dùng revision weights `c2bfbd67dc648cac455077333f7cf5c18a2e3bb4`; int8 tạo từ revision đó bằng onnx 1.20.1 / onnxruntime 1.31.1. SHA-256 ba graph theo thứ tự text encoder, prefix step, local frame decode:

- fp32: `d37557a4abe07953a02686a176664486e73838389a14ff4a3f36b47ac2bccd52`, `b7544c9fdd3535b21fe8dea855407180ad2084f9da298f8a153893522496565b`, `3c4540ef4e69dcf604dc6f60f6f99f119189694ff5f31113def20406e274f025`.
- int8: `663f66993dbf1da88ced3b73692cf8efa9d19c7bbd3f0bdf9dc4a37814408ea0`, `764ce324a33ee34783235eb406dd77ef8554a2c609122f418cea06cfc3c5d82c`, `2564547923658c69f5bef2e1102b2efc82169cada6e86eb366a8e11be7dc2f26`.

Chạy lại test sau khi chuẩn bị cùng weights và môi trường:

```bash
ZEROTTS_VENDOR_BASELINE=fp32 ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS \
  uv run pytest -m real_model tests/test_server_smoke.py -k vendored
ZEROTTS_VENDOR_BASELINE=int8 ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-int8 \
  uv run pytest -m real_model tests/test_server_smoke.py -k vendored
```

Hash là mốc của môi trường này, không cam kết bit-identical giữa CPU, phiên bản thư viện hoặc weights khác nhau. Khi khác môi trường cần tự tạo baseline với package 0.1.5 và so cùng máy, không kết luận thay đổi hành vi chỉ từ hash khác.
