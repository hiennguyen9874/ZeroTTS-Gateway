# ZeroTTS Gateway

A thin **gateway** in front of the TTS **upstream**, `zerotts_server` (the second package in this repo). The upstream runs ZeroTTS on ONNX CPU by default or PyTorch CUDA. Both speak audio.cpp's API (`POST /v1/audio/speech`, `/v1/models`, `/v1/audio/voices`). The gateway adds a **text pipeline**, `voice_ref` path blocking, optional Bearer auth, upstream readiness checks, and content-free logs. It does not synthesize, split, stitch, resample, retry, or queue: at most one upstream request per speech request.

## Spec and slices

- Spec: read issue #1 before changing the API or upstream behaviour; read issue #15 for the CUDA runtime (`gh issue view <n> --json title,body,comments,labels,state`). Later decisions and slices amend the original spec, including vendoring and the pipeline default.
- Slices: take an issue only after its blockers are closed; check its current status first. Human-review slices require operator sign-off; a score or a closed issue alone is not evidence of listening.
- Decisions: read the matching ADR in `docs/adr/` before reopening one; evaluation evidence is in `docs/reports/`.

## Conventions

- Use the terms in `GLOSSARY.md` in code, docs and issues.
- Code, comments and agent docs (`AGENTS.md`, `docs/agents/`) in English; human docs (`README.md`, `docs/MIGRATION.md`), ADRs, reports and issues in Vietnamese.
- Test gateway and server behaviour through HTTP, asserting status, body, headers and calls to the fake upstream or engine. Inject them via `create_app(settings, transport=...)` and `create_app(settings, engine=...)` respectively; use a real server for streaming timing and disconnects because TestClient buffers bodies. Startup validation, script helpers and CUDA runtime-seam parity/acceptance checks can be tested directly.
- Error bodies use audio.cpp's shape `{"error":{"message","type"}}`. Messages and logs carry no request text and no audio.
- `references/` is read-only pinned submodules; weights go in gitignored `models/`.
- Check: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.

## Agent docs

Read the matching file before working in its area:

- Gateway: `docs/agents/gateway.md` for routes, auth, forwarding, timeouts or gateway config.
- Text pipeline: `docs/agents/text-pipeline.md` for mis-readings, reading rules, lexicon or golden cases.
- Upstream: `docs/agents/zerotts-server.md` for `src/zerotts_server/`, server config, ONNX weights, voices, workers or streaming.
- CUDA: `docs/agents/zerotts-cuda.md` for the PyTorch runtime, CUDA weights, GPU tests or GPU benchmarks; also read the upstream doc for shared behaviour.
- Deployment: `docs/agents/deploy.md` for Docker, Compose, environment or deployment shapes.
- Listening: `docs/agents/listening.md` for rendering, ASR scoring or text-pipeline A/B runs.
- References: `docs/agents/references.md` for pinned submodules or weight downloads.
