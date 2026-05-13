import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hma.datasets import COCOSearch18Dataset
from hma.datasets.fixation_utils import points_to_fixation_map
from scripts.prepare_dataset import (
    COCO_SEARCH18_MANIFEST_COLUMNS,
    build_coco_search18_manifest,
)


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


def _make_fake_coco_search18_root(tmp_path):
    root = tmp_path / "coco_search18"
    _write_image(root / "images" / "train" / "000001.jpg", (20, 10), 64)
    _write_image(root / "images" / "train" / "000002.jpg", (16, 8), 128)
    annotations = [
        {
            "image_id": "000001",
            "image_path": "images/train/000001.jpg",
            "split": "train",
            "target_category": "bottle",
            "task": "present",
            "subject_id": "S1",
            "trial_id": "T1",
            "fixation_points": [[10, 5], [12, 6]],
        },
        {
            "image_id": "000002",
            "image_path": "images/train/000002.jpg",
            "split": "train",
            "target": "chair",
            "task": "absent",
            "subject": "S2",
            "trial": "T2",
            "X": [4, 8],
            "Y": [2, 4],
        },
    ]
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps({"annotations": annotations}), encoding="utf-8"
    )
    return root, annotations_path


def test_build_coco_search18_manifest_preserves_task_fields(tmp_path):
    root, annotations = _make_fake_coco_search18_root(tmp_path)
    output = tmp_path / "manifests" / "coco_search18_manifest.csv"

    summary = build_coco_search18_manifest(root, annotations, output)

    assert summary["rows_written"] == 2
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COCO_SEARCH18_MANIFEST_COLUMNS
        rows = list(reader)

    assert rows[0]["image_id"] == "000001"
    assert rows[0]["image_path"] == "images/train/000001.jpg"
    assert rows[0]["target_category"] == "bottle"
    assert rows[0]["task"] == "present"
    assert rows[0]["subject_id"] == "S1"
    assert rows[0]["trial_id"] == "T1"
    assert json.loads(rows[0]["fixation_points"]) == [[10.0, 5.0], [12.0, 6.0]]


def test_coco_search18_dataset_loads_points_map_and_metadata(tmp_path):
    root, annotations = _make_fake_coco_search18_root(tmp_path)
    manifest = tmp_path / "coco_search18_manifest.csv"
    build_coco_search18_manifest(root, annotations, manifest)

    dataset = COCOSearch18Dataset(
        root=root,
        manifest_path=manifest,
        split="train",
        image_size=(5, 10),
        fixation_sigma=1.5,
        validate_files=True,
    )
    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == EXPECTED_ITEM_KEYS
    assert item["image"].mode == "RGB"
    assert item["image"].size == (10, 5)
    assert item["image_id"] == "000001"
    assert item["fixation_points"].shape == (2, 2)
    assert np.allclose(item["fixation_points"][0], [5.0, 2.5])
    assert item["fixation_map"].shape == (5, 10)
    assert item["fixation_map"].dtype == np.float32
    assert np.isfinite(item["fixation_map"]).all()
    assert item["fixation_map"].max() > 0.0
    assert item["metadata"]["dataset"] == "coco_search18"
    assert item["metadata"]["target_category"] == "bottle"
    assert item["metadata"]["task"] == "present"
    assert item["metadata"]["subject_id"] == "S1"
    assert item["metadata"]["trial_id"] == "T1"


def test_points_to_fixation_map_handles_valid_and_invalid_points():
    valid_map = points_to_fixation_map(
        np.array([[5.0, 5.0], [100.0, 100.0]], dtype=np.float32),
        height=10,
        width=10,
        sigma=1.0,
    )
    empty_map = points_to_fixation_map(
        np.array([[100.0, 100.0]], dtype=np.float32),
        height=10,
        width=10,
        sigma=1.0,
    )

    assert valid_map.shape == (10, 10)
    assert np.isfinite(valid_map).all()
    assert valid_map.max() > 0.0
    assert np.allclose(empty_map, 0.0)


def test_coco_search18_dataset_respects_max_items(tmp_path):
    root, annotations = _make_fake_coco_search18_root(tmp_path)
    manifest = tmp_path / "coco_search18_manifest.csv"
    build_coco_search18_manifest(root, annotations, manifest)

    dataset = COCOSearch18Dataset(
        root=root,
        manifest_path=manifest,
        split="train",
        max_items=1,
    )

    assert len(dataset) == 1


def test_coco_search18_dataset_validate_files_raises_for_missing_image(tmp_path):
    root, annotations = _make_fake_coco_search18_root(tmp_path)
    manifest = tmp_path / "coco_search18_manifest.csv"
    build_coco_search18_manifest(root, annotations, manifest)
    (root / "images" / "train" / "000001.jpg").unlink()

    with pytest.raises(FileNotFoundError):
        COCOSearch18Dataset(
            root=root,
            manifest_path=manifest,
            split="train",
            validate_files=True,
        )
