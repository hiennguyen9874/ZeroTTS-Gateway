#!/usr/bin/env bash
# Downloads the pinned ZeroTTS weights into models/ZeroTTS (or the directory given as $1).
#
# --local-dir writes real files, not symlinks into the HF cache: onnxruntime >= 1.25 rejects
# external data behind symlinks (ADR 0005). Re-running skips files already at this revision
# (metadata in <dir>/.cache/huggingface). Needs the `hf` CLI: `uv run scripts/download_zerotts.sh`.
# Bump REVISION only on purpose, then rerun the real-model smoke test (docs/agents/zerotts-server.md).
set -euo pipefail

REPO=zeroweight-ai/ZeroTTS
REVISION=c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
DEST=${1:-"$(dirname "$0")/../models/ZeroTTS"}

hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" \
  --include 'config.json' --include 'tokenizer.json' --include 'null_voice_emb.npy' \
  --include 'silence_frame.npy' --include 'onnx/*' --include 'onnx/codec/*' \
  --include 'voices/index.json' --include 'voices/*/voice.npz' --include 'voices/*/meta.json'
