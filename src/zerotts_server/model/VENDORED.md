# Vendored ZeroTTS

Source: https://github.com/ZeroWeight-AI/ZeroTTS, commit
`319c3e07c8575c1e38617f07c773ec76a780a1f1` (the `src/zerotts` tree is
identical to PyPI `zerotts==0.1.5`). License: `LICENSE`; upstream `NOTICE`
and its referenced MOSS license are included unchanged.

Copied from `src/zerotts/`: `synthesizer.py`, `codec.py`, `tokenizer.py`,
`chunking.py`, `voices.py`, and `save_wav` from `audio.py`. The package
initializer is local. `hub.py`, `cli.py`, and `text_norm/` are not included.
The local-only config validation formerly in `hub.py` is in `synthesizer.py`;
Hub downloading and `from_pretrained` were removed. The copied files are
formatted and linted for this repository. Compare against the pinned submodule
with `git diff --no-index references/ZeroTTS-repo/src/zerotts/synthesizer.py src/zerotts_server/model/synthesizer.py` (and likewise for the other files).
