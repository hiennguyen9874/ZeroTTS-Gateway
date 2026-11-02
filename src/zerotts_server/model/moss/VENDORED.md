# Vendored MOSS-Audio-Tokenizer-Nano (PyTorch)

Source: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano, revision
`6aa02b01e445cc585582cf0ba480bc3ea6c8dd68` (the revision whose weights
`scripts/build_zerotts_torch.py` converts into `codec.safetensors`).
License: Apache-2.0, full text in `LICENSE`. Copyright the OpenMOSS Team and
the HuggingFace Inc. team (file headers).

Copied unmodified: `modeling_moss_audio_tokenizer.py`,
`configuration_moss_audio_tokenizer.py`, `__init__.py`. The files are not
formatted or linted here (excluded from Ruff in `pyproject.toml`), so they stay
byte-identical; check with

```bash
S=$(uv run python -c "from huggingface_hub import snapshot_download as d; print(d('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano', revision='6aa02b01e445cc585582cf0ba480bc3ea6c8dd68', allow_patterns=['*.py']))")
for f in __init__.py configuration_moss_audio_tokenizer.py modeling_moss_audio_tokenizer.py; do
  cmp "$S/$f" "src/zerotts_server/model/moss/$f"
done
```

Only the `torch-cuda` runtime imports it (`TorchCodec` in `torch_runtime.py`):
the model is built from `codec_config.json` and `codec.safetensors` in the weights
directory, encoder removed, decode only. It is never loaded through
`transformers` `from_pretrained`/`trust_remote_code` and downloads nothing.
It still imports `transformers` (base classes, config, logging), pinned in the
`server-cuda` extra.
