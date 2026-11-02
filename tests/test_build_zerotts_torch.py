import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("gguf")
safetensors = pytest.importorskip("safetensors.numpy")

spec = importlib.util.spec_from_file_location(
    "build_zerotts_torch", Path(__file__).resolve().parents[1] / "scripts/build_zerotts_torch.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize("target", ["source", "output"])
@pytest.mark.parametrize("location", ["same", "child", "alias", "sibling"])
def test_download_directory_cannot_write_into_weights(
    tmp_path, monkeypatch, capsys, target, location
):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    protected = source if target == "source" else output
    downloads = protected
    if location == "child":
        downloads = protected / "downloads"
    elif location == "alias":
        downloads = tmp_path / "alias"
        downloads.symlink_to(protected, target_is_directory=True)
    elif location == "sibling":
        downloads = tmp_path / "downloads"
    for name in ("config.json", "tokenizer.json", "silence_frame.npy"):
        (source / name).touch()
    (source / "voices").mkdir()
    codec = source / "onnx" / "codec"
    codec.mkdir(parents=True)
    for name in builder.CODEC_ONNX:
        (codec / name).touch()

    def download(*args):
        if location == "sibling":
            raise RuntimeError("safe download reached")
        pytest.fail("download must not run for an unsafe directory")

    monkeypatch.setattr(builder, "download", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--model",
            str(source),
            "--dst",
            str(output),
            "--force",
            "--revision",
            "a" * 40,
            "--download-dir",
            str(downloads),
        ],
    )
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    if location == "sibling":
        with pytest.raises(RuntimeError, match="safe download reached"):
            builder.main()
    else:
        with pytest.raises(SystemExit) as error:
            builder.main()
        assert error.value.code == 2
        assert "download directory" in capsys.readouterr().err
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1.001, 1.0])
@pytest.mark.parametrize("name", ["model.decoder.test", "unnamed"])
def test_codec_verification_rejects_invalid_differences(tmp_path, value, name):
    codec = tmp_path / "codec.safetensors"
    safetensors.save_file({"decoder.test": np.ones(2, dtype=np.float32)}, codec)
    for graph in builder.CODEC_ONNX:
        initializer = onnx.numpy_helper.from_array(
            np.array([value, 1], dtype=np.float32), name=name
        )
        model = onnx.helper.make_model(onnx.helper.make_graph([], "test", [], [], [initializer]))
        onnx.save(model, tmp_path / graph)
    if value == 1.0:
        report = builder.verify_codec_onnx(codec, tmp_path)
        assert all(stats["max_abs_diff"] == 0 for stats in report.values())
    else:
        with pytest.raises(ValueError, match="matches no codec tensor"):
            builder.verify_codec_onnx(codec, tmp_path)
