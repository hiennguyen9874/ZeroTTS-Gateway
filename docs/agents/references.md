# References and weights

`references/` holds six read-only pinned submodules: development references, never runtime dependencies. Pins and URLs are in `.gitmodules` and `git submodule status`.

```bash
git submodule update --init   # no --remote (breaks pin reproducibility), no --recursive (nested C++ deps unused)
```

Nothing needs them to run the gateway, the upstream or the default tests. Leave their contents unedited; bump a pin only on purpose, in its own commit.

## What to read where

- `ZeroTTS-repo`: source of the vendored `src/zerotts_server/model/` code (pin is five commits past `v0.1.5` with `src/zerotts/` unchanged). See `src/zerotts_server/model/VENDORED.md` for provenance and a file comparison command. `src/zerotts/synthesizer.py` generation and streaming, `src/zerotts/chunking.py` sentence splitting, `webui/engine.py` multi-segment streaming, `docs/RUNTIME.md` / `docs/VOICES.md` graph and voice-pack contracts.
- `audio.cpp`: the API contract `zerotts_server` mirrors, not something to deploy (it cannot load ZeroTTS, ADR 0005). `app/server/api.md`, `app/server/README.md` for route schemas, errors, SSE and PCM streaming. ADR/report citations into `src/models/omnivoice/` stay valid at this pin.
- `sea-g2p`: the chosen normalizer (ADR 0002); use `Normalizer("vi")`, not the G2P/phonemizer path. Read `python/` and `src/` when debugging mis-readings or known issues.
- `vietnormalizer`: not a runtime normalizer; a source of loanword and acronym dictionaries for the lexicon (`vietnormalizer/data/*.csv`, larger list in `public/`).
- `gipformer`: `infer_onnx.py` is the sherpa-onnx transducer setup mirrored by `scripts/asr_server.py`.
- `llm-wiki`: Vietnamese speech knowledge base for pipeline design. Under `outputs/`: `chuong-19-cau-text-to-speech-chunker-normalizer-lexicon.md`, `chuong-05-ngu-am-chu-viet-va-van-ban-tieng-viet.md`, `chuong-13-tts-tu-van-ban-toi-waveform.md`, `survey-cau-text-to-speech-tien-hau-xu-ly-sau-llm.md`, `lua-chon-va-thiet-ke-tts-tieng-viet-realtime.md`.

Keep retired engines and HF model clones out of `references/`; their last submodule revisions are in git history and historical evaluations remain in ADRs and reports. The `torch-cuda` weights builder does download a pinned ZeroTTS-GGUF **file** as a conversion source; it does not deploy the rejected ggml runtime (ADR 0007). See `docs/agents/zerotts-cuda.md`.

## Weights

Weights go into gitignored `models/` via `hf download --local-dir` at a pinned revision, never through submodules (Git LFS would pull large checkpoints or leave pointer files that look like ONNX).

- ZeroTTS: `uv run scripts/download_zerotts.sh` (revision pinned in the script).
- ASR (optional, for listening runs):

  ```bash
  uv run hf download g-group-ai-lab/gipformer1.5-68M-rnnt \
      --revision dd9227dcd8705c13f33bdbe59728d546ab94480f \
      encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt \
      --local-dir models/gipformer1.5-68M-rnnt
  ```
