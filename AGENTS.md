# ZeroTTS Gateway

A thin **gateway** in front of a TTS **upstream**, `zerotts_server` (ZeroTTS on onnxruntime CPU, the second package in this repo). Both speak audio.cpp's API (`POST /v1/audio/speech`, `/v1/models`, `/v1/audio/voices`). The gateway adds a **text pipeline**, `voice_ref` path blocking, optional Bearer auth, `/ready` that checks the upstream, and content-free logs. It never synthesizes, splits, stitches, resamples, retries, or queues: one upstream request per caller request.

## Spec and slices

- Spec: issue #1 (`gh issue view 1 --json title,body,comments,labels,state`), the source of truth for the API contract, `zerotts_server` behaviour, error rules and out-of-scope list. Read it before changing behaviour.
- Slices are sub-issues #2–#11 with GitHub blocked-by links; take one whose blockers are closed (`gh issue view <n>`). `ready-for-human` slices (#9 A/B, #10 node benchmark) need a person for the final call.
- Decisions live in `docs/adr/` (0001 upstream owns chunking, 0002 sea-g2p normalizer, 0003 gateway owns reading rules, 0004 superseded, 0005 ZeroTTS replaces OmniVoice). Read the matching ADR before reopening a decision. Evaluation evidence lives in `docs/reports/`.

## Conventions

- Use the terms in `GLOSSARY.md` (gateway, upstream, caller, operator, input text, engine text, effective language, text warning, non-verbal tag, voice) in code, docs and issues.
- Code, comments and agent docs (`AGENTS.md`, `docs/agents/`) in English; human docs (`README.md`, `docs/MIGRATION.md`), ADRs, reports and issues in Vietnamese.
- Tests go through HTTP only and assert observable behaviour (status, body, headers, what the fake upstream received). Gateway: fake upstream via `create_app(settings, transport=...)`. `zerotts_server`: fake engine via `create_app(settings, engine=...)`.
- Error bodies use audio.cpp's shape `{"error":{"message","type"}}`. Messages and logs carry no request text and no audio.
- `references/` is read-only pinned submodules; weights go in gitignored `models/`.
- Check: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.

## Agent docs

Read the matching file before working in its area:

- `docs/agents/gateway.md`: changing `src/zerotts_gateway/` routes, auth, upstream forwarding, timeouts or `configs/gateway.toml`.
- `docs/agents/text-pipeline.md`: fixing a mis-reading, editing reading rules, `configs/lexicon.csv`, or golden cases in `tests/golden/pipeline.json`.
- `docs/agents/zerotts-server.md`: changing `src/zerotts_server/`, weights, voices, workers, streaming, or `configs/zerotts_server.toml`.
- `docs/agents/deploy.md`: Dockerfiles, `compose*.yaml`, `.env`, or a deployment shape.
- `docs/agents/listening.md`: listening runs, ASR scoring, or a text pipeline A/B (`scripts/render_*.py`, `scripts/ab_report.py`, `scripts/asr_server.py`).
- `docs/agents/references.md`: reading upstream source in `references/`, bumping a submodule pin, or downloading weights.
