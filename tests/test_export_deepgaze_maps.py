import csv
import importlib.util
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
