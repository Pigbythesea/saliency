import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hma.datasets import CAT2000Dataset
from scripts.prepare_dataset import CAT2000_MANIFEST_COLUMNS, build_cat2000_manifest


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


def _make_fake_cat2000_root(tmp_path):
    root = tmp_path / "cat2000"
    _write_image(root / "images" / "val" / "Action" / "action_001.jpg", (12, 8), 64)
    _write_image(root / "images" / "val" / "Art" / "art_001.jpg", (10, 6), 128)
    _write_map(root / "fixations" / "val" / "Action" / "action_001.png", (12, 8), 255)
    _write_map(root / "fixations" / "val" / "Art" / "art_001.png", (10, 6), 96)
    return root


def test_build_cat2000_manifest_writes_categories_and_required_columns(tmp_path):
    root = _make_fake_cat2000_root(tmp_path)
    output = tmp_path / "manifests" / "cat2000_manifest.csv"

    summary = build_cat2000_manifest(root, output)

    assert summary["rows_written"] == 2
    assert output.is_file()
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CAT2000_MANIFEST_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["image_id"] == "action_001"
    assert rows[0]["category"] == "Action"
    assert rows[0]["split"] == "val"
    assert rows[0]["image_path"] == "images/val/Action/action_001.jpg"
    assert rows[0]["fixation_map_path"] == "fixations/val/Action/action_001.png"
    assert rows[0]["width"] == "12"
    assert rows[0]["height"] == "8"
    assert {row["category"] for row in rows} == {"Action", "Art"}


def test_cat2000_dataset_loads_expected_fields(tmp_path):
    root = _make_fake_cat2000_root(tmp_path)
    manifest = tmp_path / "cat2000_manifest.csv"
    build_cat2000_manifest(root, manifest)

    dataset = CAT2000Dataset(
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
    assert item["image_id"] == "action_001"
    assert item["fixation_map"].shape == (4, 6)
    assert item["fixation_map"].dtype == np.float32
    assert np.isfinite(item["fixation_map"]).all()
    assert 0.0 <= item["fixation_map"].min() <= item["fixation_map"].max() <= 1.0
    assert item["fixation_points"] is None
    assert item["metadata"]["dataset"] == "cat2000"
    assert item["metadata"]["category"] == "Action"
    assert item["metadata"]["split"] == "val"
    assert item["metadata"]["width"] == 12
    assert item["metadata"]["height"] == 8


def test_cat2000_dataset_filters_by_category(tmp_path):
    root = _make_fake_cat2000_root(tmp_path)
    manifest = tmp_path / "cat2000_manifest.csv"
    build_cat2000_manifest(root, manifest)

    dataset = CAT2000Dataset(
        root=root,
        manifest_path=manifest,
        split="val",
        categories=["Art"],
    )

    assert len(dataset) == 1
    assert dataset[0]["metadata"]["category"] == "Art"


def test_cat2000_dataset_respects_max_items(tmp_path):
    root = _make_fake_cat2000_root(tmp_path)
    manifest = tmp_path / "cat2000_manifest.csv"
    build_cat2000_manifest(root, manifest)

    dataset = CAT2000Dataset(
        root=root,
        manifest_path=manifest,
        split="val",
        max_items=1,
    )

    assert len(dataset) == 1


def test_cat2000_dataset_validate_files_raises_for_missing_file(tmp_path):
    root = _make_fake_cat2000_root(tmp_path)
    manifest = tmp_path / "cat2000_manifest.csv"
    build_cat2000_manifest(root, manifest)
    (root / "fixations" / "val" / "Action" / "action_001.png").unlink()

    with pytest.raises(FileNotFoundError):
        CAT2000Dataset(
            root=root,
            manifest_path=manifest,
            split="val",
            validate_files=True,
        )
