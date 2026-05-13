import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hma.datasets import SALICONDataset
from scripts.prepare_dataset import MANIFEST_COLUMNS, build_salicon_manifest


EXPECTED_ITEM_KEYS = {
    "image",
    "image_id",
    "image_path",
    "fixation_map",
    "fixation_points",
    "metadata",
}


def _write_image(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(
        np.full((size[1], size[0], 3), value, dtype=np.uint8),
        mode="RGB",
    )
    image.save(path)


def _write_map(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8), mode="L")
    image.save(path)


def _make_fake_salicon_root(tmp_path):
    root = tmp_path / "salicon"
    _write_image(root / "images" / "val" / "image_001.jpg", (12, 8), 64)
    _write_image(root / "images" / "val" / "image_002.jpg", (10, 6), 128)
    _write_map(root / "maps" / "val" / "image_001.png", (12, 8), 255)
    _write_map(root / "maps" / "val" / "image_002.png", (10, 6), 96)
    return root


def test_build_salicon_manifest_writes_required_columns(tmp_path):
    root = _make_fake_salicon_root(tmp_path)
    output = tmp_path / "manifests" / "salicon_manifest.csv"

    summary = build_salicon_manifest(root, output)

    assert summary["rows_written"] == 2
    assert output.is_file()
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MANIFEST_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["image_id"] == "image_001"
    assert rows[0]["split"] == "val"
    assert rows[0]["image_path"] == "images/val/image_001.jpg"
    assert rows[0]["fixation_map_path"] == "maps/val/image_001.png"
    assert rows[0]["width"] == "12"
    assert rows[0]["height"] == "8"


def test_salicon_dataset_loads_expected_fields(tmp_path):
    root = _make_fake_salicon_root(tmp_path)
    manifest = tmp_path / "salicon_manifest.csv"
    build_salicon_manifest(root, manifest)

    dataset = SALICONDataset(
        root=root,
        manifest_path=manifest,
        split="val",
        image_size=(4, 6),
        validate_files=True,
    )
    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == EXPECTED_ITEM_KEYS
    assert item["image"].mode == "RGB"
    assert item["image"].size == (6, 4)
    assert item["image_id"] == "image_001"
    assert item["image_path"].endswith("images\\val\\image_001.jpg") or item[
        "image_path"
    ].endswith("images/val/image_001.jpg")
    assert item["fixation_map"].shape == (4, 6)
    assert item["fixation_map"].dtype == np.float32
    assert np.isfinite(item["fixation_map"]).all()
    assert 0.0 <= item["fixation_map"].min() <= item["fixation_map"].max() <= 1.0
    assert item["fixation_points"] is None
    assert item["metadata"]["dataset"] == "salicon"
    assert item["metadata"]["split"] == "val"
    assert item["metadata"]["width"] == 12
    assert item["metadata"]["height"] == 8
    assert item["metadata"]["fixation_map_path"].endswith(
        "maps\\val\\image_001.png"
    ) or item["metadata"]["fixation_map_path"].endswith("maps/val/image_001.png")


def test_salicon_dataset_respects_max_items(tmp_path):
    root = _make_fake_salicon_root(tmp_path)
    manifest = tmp_path / "salicon_manifest.csv"
    build_salicon_manifest(root, manifest)

    dataset = SALICONDataset(
        root=root,
        manifest_path=manifest,
        split="val",
        max_items=1,
    )

    assert len(dataset) == 1


def test_salicon_dataset_validate_files_raises_for_missing_file(tmp_path):
    root = _make_fake_salicon_root(tmp_path)
    manifest = tmp_path / "salicon_manifest.csv"
    build_salicon_manifest(root, manifest)
    (root / "maps" / "val" / "image_001.png").unlink()

    with pytest.raises(FileNotFoundError):
        SALICONDataset(
            root=root,
            manifest_path=manifest,
            split="val",
            validate_files=True,
        )
