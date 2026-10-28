# ZeroTTS Gateway

A thin **gateway** in front of a TTS **upstream**. The upstream is `zerotts_server` (ZeroTTS on onnxruntime CPU, a second package in this repo; ADR 0005). The gateway keeps audio.cpp's API (`POST /v1/audio/speech`, `/v1/models`, `/v1/audio/voices`) compatible; migration also changes model, voice and audio rate, and adds: a **text pipeline** (guard → markup cleanup → Vietnamese lexicon + sea-g2p normalization), `voice_ref` path blocking, optional Bearer auth, `/ready` that checks the upstream, and content-free logs. It never synthesizes, splits, stitches, resamples, retries, or queues.

## Spec and work slices

- Spec: issue #1, the ZeroTTS migration (`gh issue view 1 --json title,body,comments,labels,state`). Source of truth for the API contract, `zerotts_server` behaviour, error rules, and out-of-scope list. Read it before changing behaviour.
- Slices are sub-issues #2–#11 with native GitHub blocked-by links; take one whose blockers are closed and read it with `gh issue view <n>`. `ready-for-human` slices (#9 A/B, #10 node benchmark) need a person for the final call.
- Decisions already made: `docs/adr/0001-upstream-owns-chunking.md` (one upstream request per caller request; keep sentence punctuation), `docs/adr/0002-sea-g2p-normalizer.md` (sea-g2p `Normalizer`, `punc_norm=False`, and what the gateway must patch around it), `docs/adr/0004-short-text-speed.md` (superseded: OmniVoice-only `short_text_speed`, removed in #11), `docs/adr/0005-zerotts-replaces-omnivoice.md` (ZeroTTS replaces OmniVoice; why `zerotts_server` and not an audio.cpp port, ggml or GPU; voices, 48 kHz, rejected options, workers; the #9 A/B turned the full text pipeline on for ZeroTTS, see `docs/reports/issue9-zerotts-text-pipeline-ab.md`).

## Conventions

- Use the terms in `GLOSSARY.md` (gateway, upstream, caller, operator, input text, engine text, effective language, text warning, non-verbal tag, voice) in code, docs, and issues.
- Docs, ADRs, and issues are written in Vietnamese; code and code comments in English.
- Tests go through HTTP APIs only. Gateway: fake upstream injected via `httpx` transport (`create_app(settings, transport=...)`). `zerotts_server`: fake engine injected (`create_app(settings, engine=...)`), plus one opt-in smoke test on the real model, skipped unless weights are in `models/`. Test only observable behaviour (status, body, headers, what the fake upstream received). Pipeline goldens go through `POST /v1/text/normalize`; each case carries a `why`, and known-wrong outputs stay as marked known issues. Every mis-reading fix adds a golden.
- Error bodies use audio.cpp's shape `{"error":{"message","type"}}`; `message` never contains request text. Logs never contain text or audio.
- `references/` holds read-only git submodules: `ZeroTTS-repo`, `audio.cpp`, `sea-g2p`, `gipformer`, `vietnormalizer`, and `llm-wiki`. Initialize with `git submodule update --init` (no `--remote` or `--recursive` needed); bump a pin only on purpose, in its own commit. They are development references, not runtime dependencies. HF model clones and retired engines have been removed; download weights into gitignored `models/`, not `references/`. See `docs/REFERENCES.md` for the keep/remove audit and revisions. `voices_candidates/` also remains gitignored.

## Local runtime docs

- `docs/GATEWAY.md`: running the gateway, env vars, routes, config, Docker.
- `docs/ZEROTTS_SERVER.md`: running `zerotts_server`, weights download, config (`configs/zerotts_server.toml`), routes, rejected options.
- `docs/DEPLOY.md`: Docker deployment shapes (gateway only, upstream only, both) via `COMPOSE_FILE`; `docker/zerotts_server.Dockerfile` + `compose.zerotts.yaml` (+ `compose.zerotts-stack.yaml`) run `zerotts_server` on CPU with weights from `scripts/download_zerotts.sh` mounted read-only.
- `docs/TEXT_PIPELINE.md`: text pipeline behaviour, pronunciation lexicon, adding golden cases.
- `docs/LISTENING.md`: `render_samples.py`/`render_all.py` listening runs, ASR scoring, output format, and the ZeroTTS text pipeline A/B (`scripts/ab_report.py`).
- `docs/MODEL_VOICE_RECOMMENDATIONS.md`: model/voice evaluation and recommended defaults.

## References

All six directories under `references/` are pinned submodules. The full inventory and removal rationale are in [docs/REFERENCES.md](docs/REFERENCES.md).

### Upstream and API contract

- `ZeroTTS-repo`: source of the `zerotts==0.1.5` dependency. The reference pin is five commits after tag `v0.1.5`, with no changes to `src/zerotts/`; it does not change the installed package. Read `src/zerotts/synthesizer.py` for generation and streaming, `src/zerotts/chunking.py` for sentence splitting, `webui/engine.py` for multi-segment streaming, and `docs/RUNTIME.md` / `docs/VOICES.md` for graph and voice-pack contracts.
- `audio.cpp`: retained only as the API contract that `zerotts_server` mirrors, pinned at `65df584a`; not an upstream to deploy. Read `app/server/api.md` and `app/server/README.md` for route schemas, errors, SSE and PCM streaming. Historical ADR/report citations into `src/models/omnivoice/` remain valid. It cannot load ZeroTTS (ADR 0005).

### ASR for listening runs

- `gipformer`: `infer_onnx.py` shows the sherpa-onnx transducer setup mirrored by `scripts/asr_server.py`. Download ASR weights into `models/gipformer1.5-68M-rnnt`; no HF clone is needed (see `docs/REFERENCES.md`).

### Vietnamese text processing

- `sea-g2p`: the chosen normalizer (ADR 0002). Use `Normalizer("vi")`, not the G2P/phonemizer path. Read `python/` and `src/` when debugging mis-readings or known issues.
- `vietnormalizer`: not a runtime normalizer; retained as a source of English→Vietnamese loanword and acronym dictionaries (`vietnormalizer/data/*.csv`, larger list in `public/`) for the pronunciation lexicon.
- `llm-wiki`: Vietnamese speech knowledge base, retained for pipeline design. Relevant files under `outputs/`:
  - `chuong-19-cau-text-to-speech-chunker-normalizer-lexicon.md`
  - `chuong-05-ngu-am-chu-viet-va-van-ban-tieng-viet.md`
  - `chuong-13-tts-tu-van-ban-toi-waveform.md`
  - `survey-cau-text-to-speech-tien-hau-xu-ly-sau-llm.md`
  - `lua-chon-va-thiet-ke-tts-tieng-viet-realtime.md`

### Weights

ZeroTTS ONNX graphs and preset voice packs come from `zeroweight-ai/ZeroTTS` on Hugging Face. Use `scripts/download_zerotts.sh` to download the pinned revision into `models/ZeroTTS` with real files: onnxruntime ≥ 1.25 rejects external data behind HF-cache symlinks. HF model clones, OmniVoice, VieNeu-TTS and ZeroTTS-GGUF are no longer kept in `references/`; historical evaluation results remain in ADRs and reports.
