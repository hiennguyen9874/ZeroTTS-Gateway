# OmniVoice Gateway

A thin **gateway** in front of audio.cpp (the **upstream**) running OmniVoice. It keeps audio.cpp's API (`POST /v1/audio/speech`, `/v1/models`, `/v1/audio/voices`) byte-compatible so callers only change the base URL, and adds: a **text pipeline** (guard → markup cleanup → Vietnamese lexicon + sea-g2p normalization), `voice_ref` path blocking, optional Bearer auth, `/ready` that checks the upstream, and content-free logs. It never synthesizes, splits, stitches, resamples, retries, or queues.

## Spec and work slices

- Spec: issue #1 (`gh issue view 1 -R hiennguyen9874/omnivoice-gateway`). Source of truth for the API contract, pipeline step order, error codes, and out-of-scope list. Read it before changing behaviour.
- Slices are sub-issues #2–#7 (skeleton, streaming, generic pipeline, Vietnamese normalization, golden set, listening run). Check `gh issue view <n>` for the slice you are working on.
- Decisions already made: `docs/adr/0001-upstream-owns-chunking.md` (one upstream request per caller request; keep sentence punctuation), `docs/adr/0002-sea-g2p-normalizer.md` (sea-g2p `Normalizer`, `punc_norm=False`, and what the gateway must patch around it).

## Conventions

- Use the terms in `GLOSSARY.md` (gateway, upstream, caller, operator, input text, engine text, effective language, text warning, non-verbal tag) in code, docs, and issues.
- Docs, ADRs, and issues are written in Vietnamese; code and code comments in English.
- Tests have one seam: the gateway's HTTP API, with a fake upstream injected via `httpx` transport (`create_app(settings, transport=...)`). Test only observable behaviour (status, body, headers, what the fake upstream received). Pipeline goldens go through `POST /v1/text/normalize`; each case carries a `why`, and known-wrong outputs stay as marked known issues. Every mis-reading fix adds a golden.
- Error bodies use audio.cpp's shape `{"error":{"message","type"}}`; `message` never contains request text. Logs never contain text or audio.
- `references/`, `models/`, `voices_candidates/` are gitignored local clones/artifacts: read them, never edit them.

## Local runtime docs

- `docs/INSTALL.md`: building audio.cpp with CUDA 12.6 on A100.
- `docs/USAGE.md`: converting OmniVoice checkpoints to GGUF, CLI/server usage, voice clone/design, model table.
- `configs/server.json` is the audio.cpp server config (model ids, `mode: streaming` entry); `configs/gateway.toml` is the gateway's per-model defaults.
- `scripts/sample_voices.py`: samples Vietnamese HF datasets into `voices_candidates/` and installs picked voices into `voices/`.

## References

All under `references/` unless a URL.

### Upstream runtime

- `audio.cpp`: the upstream. C++/ggml multi-model audio engine with an OpenAI-style HTTP server; pinned at commit `65df584a`. Reach for it when the gateway contract or upstream behaviour is in question:
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
- `OmniVoice-GGUF-hf` (Serveurperso, for `omnivoice.cpp`) and `omnivoice-GGUF-hf` (cstr, for `CrispASR`): third-party GGUFs in other runtimes' layouts that audio.cpp cannot load. audio.cpp GGUFs live in `models/`.
- `omnivoice.cpp`: alternative C++/GGML OmniVoice port with its own OpenAI-style `tts-server`; prior art for punctuation-aware chunking and cross-fade.
- `CrispASR`: whisper.cpp fork bundling many ASR/TTS backends, including OmniVoice. Comparison only.

### Vietnamese text processing

- `sea-g2p`: the chosen normalizer (ADR 0002). Rust core with Python bindings and a C ABI; use `Normalizer("vi")`, not the G2P/phonemizer path. Read `python/` and `src/` when debugging a mis-reading or a known issue.
- `vietnormalizer`: rejected normalizer, still useful as a source of English→Vietnamese loanword and acronym dictionaries (`vietnormalizer/data/*.csv`, larger 20k list in `public/`) for the pronunciation lexicon.
- `VieNeu-TTS`: Vietnamese TTS that uses sea-g2p; reference for how sea-g2p output is fed to a model and for streaming design.
- `vietvoice-tts`: sister project, a FastAPI gateway in front of VieNeu. Prior art for test style (pytest, golden JSON for text, fake upstream for streaming and errors), text policy (`docs/guides/text-policy.md`), and the listening run (`docs/guides/listening-run.md`). Copy the style, not the policy.
- `llm-wiki`: personal speech knowledge base (Vietnamese). Read when designing a pipeline rule:
  - `outputs/chuong-19-cau-text-to-speech-chunker-normalizer-lexicon.md`: chunker, normalizer, lexicon design.
  - `outputs/chuong-05-ngu-am-chu-viet-va-van-ban-tieng-viet.md`: Vietnamese phonology and orthography.
  - `outputs/chuong-13-tts-tu-van-ban-toi-waveform.md`: TTS from text to waveform.
  - `outputs/survey-cau-text-to-speech-tien-hau-xu-ly-sau-llm.md`: survey of pre/post-processing between LLM output and TTS.
  - `outputs/lua-chon-va-thiet-ke-tts-tieng-viet-realtime.md`: Vietnamese realtime TTS model selection (incl. OmniVoice, G-OmniVoice).
