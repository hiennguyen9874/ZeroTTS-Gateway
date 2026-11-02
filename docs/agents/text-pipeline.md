# Text pipeline

Code: `src/zerotts_gateway/text.py`. `POST /v1/audio/speech` replaces `input` with **engine text** before forwarding; `POST /v1/text/normalize` (`{"input", "language"?, "model"?, "text_processing"?}`) returns `engine_text`, `warnings`, `language`, `pipeline_version` without calling the upstream. Use the dry-run to see what any input becomes.

## Stages

1. **Guard** (always): NFC-normalize, then reject `text_too_long` (`max_input_chars`), `reasoning_markup` and `model_control` with `400`, before cleanup or any upstream call.
2. **Cleanup** (always, every language): drops code blocks, HTML tags, URLs and emoji, each with a text warning; flattens markdown tables (separator row dropped, cells joined by commas). Unpunctuated non-final lines of at most four words join with a comma, longer ones with a period; existing sentence punctuation stays. This preserves short markup/list items more reliably.
3. **Non-verbal tag removal** (models with `nonverbal_tags = false`, i.e. `zerotts` and `zerotts-stream`; every language, even with normalize off): removes the 13 lowercase tags (`[laughter]`, `[sigh]`, …) and CMU phones (`[B EY1 S]`), adds one `omitted_nonverbal_tag` warning, leaves no stray spaces or punctuation (`Giá [laughter], 50.000đ` → `Giá, 50.000đ`; `[sigh]. Thôi được` → `Thôi được.`; `A, [laughter], B` → `A, B`). Tag removal leaves other brackets and `[Laughter]` untouched; the Vietnamese stage may still remove their brackets. Only tags left → `400 empty_text`.
4. **Vietnamese normalization** (effective language `vi` and normalize on), in order: lexicon → reading rules → sea-g2p `Normalizer("vi")` with `punc_norm=False` (ADR 0002) → strip `<en>` tags and rejoin spelled letters into uppercase acronyms (`AI`). Brackets other than tags lose their bracket characters; supported tags and phones are protected.
5. **Finish** (always): collapse whitespace, ensure terminal punctuation and reject `empty_text` with `400` if nothing speakable remains.

`reference_text` goes through the same pipeline and error codes but adds no warnings.

Successful speech responses carry `X-Text-Pipeline-Version` (`<RULES_VERSION>-<lexicon hash>`) and, when anything was omitted, `X-Text-Warnings`. The dry-run returns version and warnings in JSON.

## Reading rules

Gateway-owned pattern rewrites for forms sea-g2p misreads (ADR 0003). See `src/zerotts_gateway/text.py` and `tests/golden/pipeline.json` for the current rules and examples. The lexicon runs first and wins; a product name before a version still gets the "chấm" rule (`iOS 18.2` → `ai ô ét mười tám chấm hai`). Keep that precedence when changing either stage.

## Lexicon

`lexicon_path` (relative to the config file; repo default `configs/lexicon.csv`), CSV header `word,replacement,note`. Case-sensitive, word-boundary match, longest entry first; `replacement` must be spoken Vietnamese because it runs before sea-g2p. Loaded at startup: edits need a restart, and a malformed file stops the gateway from starting. `Settings()` does not load it by itself; an operator file replaces the repo file rather than merging.

The six product-name entries (`iPhone`, `iPad`, `iOS`, `YouTube`, `Facebook`, `Google`) are #7 candidates shipped in #17 with **no human listening sign-off** (`docs/reports/issue17-lexicon.md`). Keep `AI`/`API` readings as they are and keep the lexicon to named exceptions, not a general Vietnamisation of English.

## Fixing a mis-reading

Every fix adds a golden in `tests/golden/pipeline.json`, run through the dry-run route.

1. Get the current engine text: `curl -s localhost:8000/v1/text/normalize -d '{"input":"…","language":"vi"}'`.
2. Add a case: `id`, `group` (reuse an existing group), `input`, optional `language` (default `vi`), `normalize`, `model` (takes that model's defaults from `configs/gateway.toml`; `render_samples.py` renders such cases only for that model), then `expected` + `warnings` or `error`, and `why` (why this reading is right).
3. If a wrong output is accepted for now (unfixed sea-g2p behaviour), put the current output in `expected` and describe the error in `known_issue`; keep the case.
4. Fix the rule or add a lexicon row. A rule change bumps `RULES_VERSION` in `src/zerotts_gateway/text.py`.
5. `uv run pytest` is green.
