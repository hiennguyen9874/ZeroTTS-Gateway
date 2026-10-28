# Text pipeline

Code: `src/zerotts_gateway/text.py`. `POST /v1/audio/speech` replaces `input` with **engine text** before forwarding; `POST /v1/text/normalize` (`{"input", "language"?, "model"?, "text_processing"?}`) returns `engine_text`, `warnings`, `language`, `pipeline_version` without calling the upstream. Use the dry-run to see what any input becomes.

## Stages

1. **Guard** (always): `400` before any upstream call for `text_too_long` (`max_input_chars`), `reasoning_markup`, `model_control`, `empty_text`.
2. **Cleanup** (always, every language): drops code blocks, HTML, URLs, emoji, each with a text warning; flattens markdown tables (separator row dropped, each row one line with cells joined by commas).
3. **Non-verbal tag removal** (models with `nonverbal_tags = false`, i.e. `zerotts` and `zerotts-stream`; every language, even with normalize off): removes the 13 lowercase tags (`[laughter]`, `[sigh]`, …) and CMU phones (`[B EY1 S]`), adds one `omitted_nonverbal_tag` warning, leaves no stray spaces or punctuation (`Giá [laughter], 50.000đ` → `Giá, 50.000đ`; `[sigh]. Thôi được` → `Thôi được.`; `A, [laughter], B` → `A, B`). Other brackets and `[Laughter]` are untouched. Only tags left → `400 empty_text`.
4. **Vietnamese normalization** (effective language `vi` and normalize on), in order: lexicon → reading rules → sea-g2p `Normalizer("vi")` with `punc_norm=False` (ADR 0002) → strip `<en>` tags and rejoin spelled letters into uppercase acronyms (`AI`). Brackets other than tags lose their bracket characters.

`reference_text` goes through the same pipeline and error codes but adds no warnings.

Success responses carry `X-Text-Pipeline-Version` (`<RULES_VERSION>-<lexicon hash>`) and, when anything was omitted, `X-Text-Warnings`.

## Reading rules

Gateway-owned pattern rewrites for forms sea-g2p misreads (ADR 0003). Covered today: versions `v2.0`/`Python 3.12`/`GPT-3.5` read "chấm" with cardinal parts; titles like `ThS.` lose the dot before a name; English decimals `3.5` read "phẩy"; ranges `8h-17h`, `25-26/12` get "đến"; slang money `50k`, `2tr5`; digit-by-digit for 1800/1900 hotlines, digits after code/OTP/PIN/account words, and runs starting with 0; `ngày 31/02` is a date while bare `1/3` (a < b ≤ 10) is a fraction; quarters `Q1/2024`; `m²`/`m³`; `1250000` as one number. The lexicon runs first and wins; a product name before a version still gets the "chấm" rule (`iOS 18.2` → `ai ô ét mười tám chấm hai`). Never move reading rules ahead of the lexicon.

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
