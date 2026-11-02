# torch-cuda runtime

Read this for `src/zerotts_server/torch_runtime.py`, CUDA weight building, GPU parity tests or benchmarks. Shared API, workers, streaming and CPU weights are in `docs/agents/zerotts-server.md`. ADR 0007 is proposed; the ASR comparison does not replace the operator's pending listening and deployment decision (`docs/reports/issue22-torch-cuda-quality-ab.md`). The provided Docker image and Compose files are CPU-only; CUDA image/Compose packaging is tracked in issue #23.

## Run on a host

First download the fp32 ONNX weights with `uv run scripts/download_zerotts.sh`. Set `SOURCE_HF_COMMIT` to their actual 40-character HF revision (the default download's revision is in that script). If `UV_NO_SYNC=true`, first install the required groups with `UV_NO_SYNC=false uv sync --extra server-cuda --group torch-weights`. Then build and run:

```bash
uv run --group torch-weights python scripts/build_zerotts_torch.py --revision "$SOURCE_HF_COMMIT"
ZEROTTS_SERVER_RUNTIME=torch-cuda ZEROTTS_SERVER_MODEL_DIR=models/ZeroTTS-torch \
  uv run --extra server-cuda zerotts-server --config configs/zerotts_server.toml
```

Select a GPU with `ZEROTTS_SERVER_DEVICE=cuda:0` (the default) or `cuda:<index>`; `CUDA_VISIBLE_DEVICES` remaps those indices. Each worker has its own weights, buffers, CUDA stream and graphs on that one device, so allow VRAM per worker. The server's `threads` setting applies only to ONNX. One process per GPU; callers use the same model ids and routes.

The normal CPU image and default dev sync do not install torch.

The builder downloads pinned GGUF and MOSS sources, verifies their hashes and builds a self-contained safetensors directory; it uses the ONNX codec only as a reference and copies no ONNX files. `--help` covers source, output and download paths. Use the source directory's real revision, not a new label; available HF metadata must agree. Keep downloads outside the source and output directories, including symlink aliases. Existing output requires `--force`. Keep `manifest.json` with the weights: startup logs identify their source and hashes from it, but do not perform an integrity scan. Provenance and conversion checks: `docs/reports/issue17-zerotts-torch-weights.md`.

`torch-cuda` requires its own `zerotts-torch` manifest and `cfg_scale = 1` for every model. A mismatched weights directory, missing torch/GPU or insufficient VRAM produces `/health` 503 `model_load_failed`; there is no CPU fallback. Unknown runtimes and invalid device syntax fail config validation. A CUDA failure during synthesis follows the normal engine-error contract.

## Runtime invariants

- `runtime.FrameGenerator` yields one `(1, K)` frame of codes; the engine joins the generator with an offline or streaming codec. Both runtimes draw from the per-request numpy RNG in the same order. Preserve the ONNX sampler's ordering and clipping; fp32 codes can diverge at verified near-ties.
- Served CUDA uses fp16 model computation with fp32 sampling and fixed-shape CUDA Graphs for segment start and frame generation. Each worker owns its state. The fp32 and eager paths remain available for parity and benchmarks.
- `TorchCodec` uses the vendored PyTorch MOSS decoder in fp32, on the worker's CUDA stream; it opens no ONNX session or remote model code. Streaming maintains causal state across segments, replays per-chunk graphs and resets state for each request. Preserve stream ordering and reset on disconnect. Provenance and license: `src/zerotts_server/model/moss/VENDORED.md`.

## Tests and measurements

Default tests exclude marker `cuda`. On a free GPU with both `models/ZeroTTS-torch` and fp32 `models/ZeroTTS` present, run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra server-cuda --group torch-weights \
  python -m pytest -m cuda tests/test_torch_cuda.py -s
uv run pytest tests/test_torch_cuda_validation.py
```

CUDA tests skip when their dependencies or weights are missing; ONNX parity also needs the `torch-weights` group. Check the result for skips before claiming GPU coverage. The parity tests compare fp32 hidden states, logits and codes against ONNX on the golden set. In teacher-forced lockstep, a different pick is allowed only when bounded logit changes reproduce both actual picks; free-running codes must match up to the first such near-tie. Offline and streaming codec PCM must be finite, within `1e-4` absolute error and at least 80 dB SNR. Other tests cover fp16, seeded HTTP output, streaming, disconnect/reset and concurrent workers. The CPU validation tests exercise the acceptance helpers themselves. For changes to the weights builder, run `uv run --group torch-weights pytest tests/test_build_zerotts_torch.py`; the module skips when its optional dependencies are absent.

Benchmark with `scripts/bench_zerotts.py --help`; run each configuration in its own process and include `--extra server-cuda` in `uv run`. Compare the observed hardware and load, not just a historical number: `docs/reports/issue21-torch-cuda-speed.md` records the A100 measurement and its post-review limitations. Quality and listening cases are in the issue #22 report above.
