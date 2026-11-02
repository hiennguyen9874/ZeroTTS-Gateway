# Listening runs

Metrics guide, ears decide: listen to the WAVs before drawing a conclusion from a score. Each script's `--help` and module docstring cover flags and output layout.

- `scripts/render_samples.py`: one model × voice; renders each case through the gateway (A) and the upstream direct (B) with the same model, voice and seed. Point both at the same upstream to isolate the text pipeline. Writes WAVs, `index.html`, `results.csv`, `manifest.json`; refuses a non-empty `--out` without `--overwrite`. Streaming models render 48 kHz PCM, one request per case and side.
- `scripts/render_all.py`: runs `render_samples.py` for each case set × model × voice. Reruns skip any run with a manifest, even a failed run; inspect `runs.json` and use `--overwrite` to retry.
- Listen with `python3 -m http.server -d <out> 9000`.

Both need `zerotts_server` and the gateway running (`docs/agents/zerotts-server.md`, `docs/agents/gateway.md`).

## ASR scoring

Download ASR weights (gipformer) using the pinned command in `docs/agents/references.md#weights`. With `UV_NO_SYNC=true`, first run `UV_NO_SYNC=false uv sync --group asr --group dev`; then start `uv run --group asr python scripts/asr_server.py --port 8090`. The render scripts read `OMNIVOICE_ASR_URL`, `OMNIVOICE_ASR_MODEL`, `OMNIVOICE_ASR_API_KEY`; the `OMNIVOICE_` prefix is historical, keep it.

## Text pipeline A/B

Issue #9 compared three engine-text variants for ZeroTTS (`docs/reports/issue9-zerotts-text-pipeline-ab.md`): (a) full pipeline, (b) guard + cleanup + tag removal, (c) full pipeline without the lexicon. ASR favored the full pipeline, now enabled in `configs/gateway.toml`; the lexicon still needs human listening sign-off. To re-run, use two gateways on the same `zerotts_server`: `:8000` with `normalize = false` on the ZeroTTS models (variant b; variant a overrides it with `--normalize-on`), and `:8001` with an empty `lexicon_path`. Variant b still honors per-case `normalize` overrides, so check both case sets for `normalize: true` before interpreting it as cleanup-only.

```bash
export OMNIVOICE_ASR_URL=http://127.0.0.1:8090
V="--model zerotts --voice maichi --voice quangminh --no-direct --seed 42"
uv run python scripts/render_all.py --gateway http://127.0.0.1:8000 $V --normalize-on --out outputs/ab9/a-s42
uv run python scripts/render_all.py --gateway http://127.0.0.1:8000 $V --out outputs/ab9/b-s42
uv run python scripts/render_all.py --gateway http://127.0.0.1:8001 $V --normalize-on --out outputs/ab9/c-s42
uv run python scripts/ab_report.py --variant a=outputs/ab9/a-s42 --variant b=outputs/ab9/b-s42 \
    --variant c=outputs/ab9/c-s42 --out outputs/ab9/report   # repeat --variant a=… to pool seeds
```

`uv run python scripts/ab_report.py --help` explains scoring and output. Its reference is the gateway's own reading, so a correct alternative can still lose points. Missing audio or transcript counts fully wrong; non-Vietnamese cases are dropped. `cases.csv` ranks cases by WER spread across variants: listen to the top ones first.
