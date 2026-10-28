# ZeroTTS Gateway

A thin **gateway** in front of a TTS **upstream**. The upstream is migrating from audio.cpp running OmniVoice to `zerotts_server` (ZeroTTS on onnxruntime CPU, a second package in this repo; ADR 0005); OmniVoice code, config and docs stay until #11 removes them. The gateway keeps audio.cpp's API (`POST /v1/audio/speech`, `/v1/models`, `/v1/audio/voices`) byte-compatible so callers only change the base URL, and adds: a **text pipeline** (guard → markup cleanup → Vietnamese lexicon + sea-g2p normalization), `voice_ref` path blocking, optional Bearer auth, `/ready` that checks the upstream, and content-free logs. It never synthesizes, splits, stitches, resamples, retries, or queues.

## Spec and work slices

- Spec: issue #1, the ZeroTTS migration (`gh issue view 1 --json title,body,comments,labels,state`). Source of truth for the API contract, `zerotts_server` behaviour, error rules, and out-of-scope list. Read it before changing behaviour.
- Slices are sub-issues #2–#11 with native GitHub blocked-by links; take one whose blockers are closed and read it with `gh issue view <n>`. `ready-for-human` slices (#9 A/B, #10 node benchmark) need a person for the final call.
- Decisions already made: `docs/adr/0001-upstream-owns-chunking.md` (one upstream request per caller request; keep sentence punctuation), `docs/adr/0002-sea-g2p-normalizer.md` (sea-g2p `Normalizer`, `punc_norm=False`, and what the gateway must patch around it), `docs/adr/0004-short-text-speed.md` (superseded: OmniVoice-only `short_text_speed`, removed in #11), `docs/adr/0005-zerotts-replaces-omnivoice.md` (ZeroTTS replaces OmniVoice; why `zerotts_server` and not an audio.cpp port, ggml or GPU; voices, 48 kHz, rejected options, workers; the #9 A/B turned the full text pipeline on for ZeroTTS, see `docs/reports/issue9-zerotts-text-pipeline-ab.md`).

## Conventions

- Use the terms in `GLOSSARY.md` (gateway, upstream, caller, operator, input text, engine text, effective language, text warning, non-verbal tag, voice) in code, docs, and issues.
- Docs, ADRs, and issues are written in Vietnamese; code and code comments in English.
- Tests go through HTTP APIs only. Gateway: fake upstream injected via `httpx` transport (`create_app(settings, transport=...)`). `zerotts_server`: fake engine injected (`create_app(settings, engine=...)`), plus one opt-in smoke test on the real model, skipped unless weights are in `models/`. Test only observable behaviour (status, body, headers, what the fake upstream received). Pipeline goldens go through `POST /v1/text/normalize`; each case carries a `why`, and known-wrong outputs stay as marked known issues. Every mis-reading fix adds a golden.
- Error bodies use audio.cpp's shape `{"error":{"message","type"}}`; `message` never contains request text. Logs never contain text or audio.
- `references/` holds read-only clones: never edit them. `audio.cpp`, `sea-g2p`, `OmniVoice`, `VieNeu-TTS`, `gipformer`, `vietnormalizer` are git submodules pinned to the commits the gateway was built against (`git submodule update --init`); bump a pin only on purpose, in its own commit. Other entries (HF model repos, `llm-wiki`, …) stay gitignored local clones. `models/` and `voices_candidates/` are gitignored local artifacts.

## Local runtime docs

- `docs/INSTALL.md`: building audio.cpp with CUDA 12.6 on A100.
- `docs/USAGE.md`: converting OmniVoice checkpoints to GGUF, CLI/server usage, voice clone/design, model table.
- `docs/GATEWAY.md`: running the gateway, env vars, routes, config, Docker.
- `docs/ZEROTTS_SERVER.md`: running `zerotts_server`, weights download, config (`configs/zerotts_server.toml`), routes, rejected options.
- `docs/DEPLOY.md`: Docker deployment shapes (gateway only, upstream only, both) via `COMPOSE_FILE`; `docker/zerotts_server.Dockerfile` + `compose.zerotts.yaml` (+ `compose.zerotts-stack.yaml`) run `zerotts_server` on CPU with weights from `scripts/download_zerotts.sh` mounted read-only; `docker/audiocpp.Dockerfile` builds audio.cpp from the pinned submodule.
- `docs/TEXT_PIPELINE.md`: text pipeline behaviour, pronunciation lexicon, adding golden cases.
- `docs/LISTENING.md`: `render_samples.py`/`render_all.py` listening runs, ASR scoring, output format, and the ZeroTTS text pipeline A/B (`scripts/ab_report.py`).
- `docs/MODEL_VOICE_RECOMMENDATIONS.md`: model/voice evaluation and recommended defaults.
- OmniVoice-only until #11: `docs/INSTALL.md`, `docs/USAGE.md`, `configs/server.json`, `compose.audiocpp.yaml`, `compose.cpu.yaml`, `compose.stack.yaml`, `docker/audiocpp.Dockerfile`.
- `configs/server.json` is the audio.cpp server config (model ids, `mode: streaming` entry); `configs/gateway.toml` is the gateway's per-model defaults.
- `scripts/sample_voices.py`: samples Vietnamese HF datasets into `voices_candidates/` and installs picked voices into `voices/`.

## References

All under `references/` unless a URL.

### Upstream runtime

- `audio.cpp`: the OmniVoice upstream, and the API contract `zerotts_server` mirrors. C++/ggml multi-model audio engine with an OpenAI-style HTTP server; pinned at commit `65df584a`. It cannot load ZeroTTS (ADR 0005). Reach for it when the gateway contract or upstream behaviour is in question:
  - `app/server/api.md`, `app/server/README.md`: route schemas, error format, streaming formats, `voice_ref`. The contract the gateway mirrors.
  - `docs/models/omnivoice.md`: OmniVoice request/session options, streaming, non-verbal tags.
  - `docs/reports/omnivoice_weight_type_benchmark.md`: f16 vs bf16 vs q8_0 speed (why f16 is the default).
  - `src/models/omnivoice/session.cpp`: long-text chunking and chunk-0 voice reuse (the basis of ADR 0001).
- vLLM-Omni OmniVoice serving: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/text_to_speech/#omnivoice. An alternative upstream; read only when comparing serving options.

### OmniVoice model

- `OmniVoice`: k2-fsa's Python reference implementation. `docs/generation-parameters.md`, `docs/voice-design.md` (design attributes), `docs/tips.md`; `omnivoice/` for the original text handling, tags, and chunking.
- `OmniVoice-hf`: base checkpoint `k2-fsa/OmniVoice` (Qwen3-0.6B backbone + audio tokenizer). Source for the audio.cpp GGUF in `models/audio.cpp-gguf/`.
- `g-omnivoice-hf`: `g-group-ai-lab/g-omnivoice`, Vietnamese finetune with lowest WER at base-level speaker similarity; its card says to normalize text before synthesis. Served as `g-omnivoice` / `g-omnivoice-stream`.
- `omnivoice-vietnamese-hf`: `splendor1811/omnivoice-vietnamese`, Vietnamese finetune on 1,000 h of dolly-vn data; no bundled audio tokenizer (reuses the base one). Served as `omnivoice-vietnamese`.

### ASR for listening runs

- `gipformer`: g-group-ai-lab's Vietnamese ASR code; `infer_onnx.py` shows the sherpa-onnx transducer setup that `scripts/asr_server.py` mirrors.
- `gipformer1.5-68M-rnnt`: the model card and config only (ONNX files are Git LFS pointers); download the files into `models/gipformer1.5-68M-rnnt` (`docs/LISTENING.md`, "Chấm bằng ASR").

### Vietnamese text processing

- `sea-g2p`: the chosen normalizer (ADR 0002). Rust core with Python bindings and a C ABI; use `Normalizer("vi")`, not the G2P/phonemizer path. Read `python/` and `src/` when debugging a mis-reading or a known issue.
- `vietnormalizer`: rejected normalizer, still useful as a source of English→Vietnamese loanword and acronym dictionaries (`vietnormalizer/data/*.csv`, larger 20k list in `public/`) for the pronunciation lexicon.
- `VieNeu-TTS`: Vietnamese TTS that uses sea-g2p; reference for how sea-g2p output is fed to a model and for streaming design.
- `llm-wiki`: personal speech knowledge base (Vietnamese). Read when designing a pipeline rule:
  - `outputs/chuong-19-cau-text-to-speech-chunker-normalizer-lexicon.md`: chunker, normalizer, lexicon design.
  - `outputs/chuong-05-ngu-am-chu-viet-va-van-ban-tieng-viet.md`: Vietnamese phonology and orthography.
  - `outputs/chuong-13-tts-tu-van-ban-toi-waveform.md`: TTS from text to waveform.
  - `outputs/survey-cau-text-to-speech-tien-hau-xu-ly-sau-llm.md`: survey of pre/post-processing between LLM output and TTS.
  - `outputs/lua-chon-va-thiet-ke-tts-tieng-viet-realtime.md`: Vietnamese realtime TTS model selection (incl. OmniVoice, G-OmniVoice).


### ZeroTTS (the new upstream model)

Gitignored local clones, not submodules. ZeroTTS is a 202M Vietnamese TTS (text encoder + cross-attention global decoder + 17-step local depth decoder, MOSS-Audio-Tokenizer-Nano codec, 48 kHz, 12.5 frames/s). Voices are preset latents (10×768); the release has no voice encoder, so no cloning.

- `ZeroTTS`: HF repo `zeroweight-ai/ZeroTTS` with the ONNX graphs, `config.json`, `tokenizer.json`, and `voices/<name>/` packs. These are the weights `zerotts_server` loads; the ONNX files here are LFS pointers. Download with `hf download --local-dir`, because onnxruntime ≥ 1.25 rejects external data behind HF-cache symlinks.
- `ZeroTTS-repo`: source of the `zerotts` pip package (pinned `0.1.5`). Read `src/zerotts/synthesizer.py` for the generation loop and `synthesize_stream`, `src/zerotts/chunking.py` for sentence splitting, `webui/engine.py` for multi-segment streaming, `docs/RUNTIME.md` for the graph contract, and `docs/VOICES.md` for voice packs.
- `ZeroTTS-GGUF`: GGUF repack for ZeroTTS's own ggml/WASM runtime (`ZeroTTS-repo/cpp/`); audio.cpp cannot read it. It was rejected as an engine (ADR 0005) and is kept for comparison only.