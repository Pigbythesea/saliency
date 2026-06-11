import csv
import builtins
import importlib.util
import sys
import urllib.error
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def _load_export_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_deepgaze_maps.py"
    spec = importlib.util.spec_from_file_location("export_deepgaze_maps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exporter = _load_export_module()


def test_load_manifest_rows_requires_image_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path"])
        writer.writeheader()
        writer.writerow({"image_id": "a", "image_path": "images/a.jpg"})

    rows = exporter.load_manifest_rows(manifest)

    assert rows == [{"image_id": "a", "image_path": "images/a.jpg"}]


def test_load_manifest_rows_rejects_missing_columns(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("image_id\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image_path"):
        exporter.load_manifest_rows(manifest)


def test_centerbias_resize_is_log_normalized():
    centerbias = np.zeros((4, 4), dtype=np.float32)

    resized = exporter.resize_and_normalize_centerbias(centerbias, (2, 3))

    assert resized.shape == (2, 3)
    assert np.isclose(np.exp(resized).sum(), 1.0)


def test_output_path_rejects_separator(tmp_path):
    with pytest.raises(ValueError, match="path separators"):
        exporter.output_path_for_image_id(tmp_path, "nested/id")


def test_output_path_for_row_supports_collision_safe_map_key(tmp_path):
    image_root = tmp_path / "images"
    row_a = {"image_id": "001", "image_path": "Action/001.jpg"}
    row_b = {"image_id": "001", "image_path": "Art/001.jpg"}

    path_a = exporter.output_path_for_row(
        tmp_path,
        row_a,
        image_root=image_root,
        filename_template="{map_key}.npy",
    )
    path_b = exporter.output_path_for_row(
        tmp_path,
        row_b,
        image_root=image_root,
        filename_template="{map_key}.npy",
    )

    assert path_a != path_b
    assert path_a.suffix == ".npy"
    assert path_b.suffix == ".npy"


def test_resolve_deepgaze_model_class_defaults_and_msdb():
    class FakeDeepGazeIIE:
        pass

    class FakeDeepGazeMSDB:
        pass

    fake_module = SimpleNamespace(
        DeepGazeIIE=FakeDeepGazeIIE,
        DeepGazeMSDB=FakeDeepGazeMSDB,
    )

    assert exporter.resolve_deepgaze_model_class(fake_module, "deepgaze_iie") is FakeDeepGazeIIE
    assert exporter.resolve_deepgaze_model_class(fake_module, "deepgaze_msdb") is FakeDeepGazeMSDB


def test_dry_run_does_not_build_deepgaze_model(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path"])
        writer.writeheader()
        writer.writerow({"image_id": "a", "image_path": "images/a.jpg"})
    centerbias = tmp_path / "centerbias.npy"
    np.save(centerbias, np.zeros((2, 2), dtype=np.float32))

    def fail_build(*_args, **_kwargs):
        raise AssertionError("dry-run should not instantiate a DeepGaze model")

    monkeypatch.setattr(exporter, "build_deepgaze_model", fail_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_deepgaze_maps.py",
            "--model",
            "deepgaze_msdb",
            "--manifest",
            str(manifest),
            "--image-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "maps"),
            "--centerbias",
            str(centerbias),
            "--dry-run",
        ],
    )

    exporter.main()

    assert "Manifest rows: 1" in capsys.readouterr().out


def test_msdb_download_failure_gets_actionable_error(monkeypatch):
    class FakeTorch:
        pass

    class FakeMSDB:
        def __init__(self, pretrained=True):
            raise urllib.error.URLError("network failed")

    fake_deepgaze = SimpleNamespace(DeepGazeMSDB=FakeMSDB)

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return FakeTorch
        if name == "deepgaze_pytorch":
            return fake_deepgaze
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(exporter, "resolve_device", lambda _torch, value: "cpu")

    with pytest.raises(RuntimeError, match="DINOv2"):
        exporter.build_deepgaze_model("cpu", model_name="deepgaze_msdb")


def test_predict_deepgaze_map_with_fake_model():
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __call__(self, image_tensor, centerbias_tensor):
            assert image_tensor.shape == (1, 3, 4, 5)
            assert centerbias_tensor.shape == (1, 4, 5)
            values = torch.zeros((1, 4, 5), dtype=torch.float32)
            return values - torch.log(torch.as_tensor(20.0))

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    centerbias = np.zeros((8, 8), dtype=np.float32)

    prediction = exporter.predict_deepgaze_map(
        FakeModel(),
        torch,
        image,
        centerbias,
        device="cpu",
    )

    assert prediction.shape == (4, 5)
    assert prediction.dtype == np.float32
    assert np.isclose(prediction.sum(), 1.0)


def test_predict_deepgaze_msdb_passes_pixel_per_dva_and_dataset():
    torch = pytest.importorskip("torch")

    class FakeMSDBModel:
        def __call__(self, image_tensor, centerbias_tensor, *, pixel_per_dva, dataset):
            assert image_tensor.shape == (1, 3, 4, 5)
            assert centerbias_tensor.shape == (1, 4, 5)
            assert pixel_per_dva == 21.7
            assert dataset is None
            values = torch.zeros((1, 4, 5), dtype=torch.float32)
            return values - torch.log(torch.as_tensor(20.0))

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    centerbias = np.zeros((8, 8), dtype=np.float32)

    prediction = exporter.predict_deepgaze_map(
        FakeMSDBModel(),
        torch,
        image,
        centerbias,
        device="cpu",
        model_name="deepgaze_msdb",
        pixel_per_dva=21.7,
        msdb_dataset=None,
    )

    assert prediction.shape == (4, 5)
    assert np.isclose(prediction.sum(), 1.0)


def test_resolve_msdb_dataset_index():
    assert exporter.resolve_msdb_dataset_index("averaged") is None
    assert exporter.resolve_msdb_dataset_index("cat2000") == 1
    with pytest.raises(ValueError, match="Unsupported"):
        exporter.resolve_msdb_dataset_index("unknown")
