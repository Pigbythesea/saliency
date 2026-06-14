import csv
from pathlib import Path

from PIL import Image

from hma.saliency.precomputed import precomputed_map_key, precomputed_row_key
from hma.external.registry import load_external_registry
from scripts.run_external_model import _load_image_rows
from scripts.preflight_paper1_matrix_v2_cluster import _validate_image_manifest
from scripts.restore_matched_dinov2_provenance import CELL_NAMES, restore


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(127, 127, 127)).save(path)


def test_cat2000_duplicate_ids_with_distinct_paths_do_not_collide(tmp_path):
    root = tmp_path / "images"
    paths = ["Action/001.jpg", "Art/001.jpg"]
    for value in paths:
        _write_image(root / value)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"image_id": "001", "image_path": value, "split": "train"}
            for value in paths
        ],
    )

    rows = _load_image_rows(
        manifest,
        image_root=root,
        split="train",
        subject_id=None,
        roi=None,
        max_items=None,
        artifact_key="map_key",
    )

    assert len(rows) == 2
    assert len({row["map_key"] for row in rows}) == 2


def test_coco_repeated_path_reuses_map_but_rows_remain_distinct(tmp_path):
    root = tmp_path / "images"
    paths = ["chair/001.jpg", "chair/001.jpg", "clock/001.jpg"]
    for value in set(paths):
        _write_image(root / value)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"image_id": "001", "image_path": value, "split": "val"}
            for value in paths
        ],
    )

    routed = _load_image_rows(
        manifest,
        image_root=root,
        split="val",
        subject_id=None,
        roi=None,
        max_items=None,
        artifact_key="map_key",
    )
    map_keys = [precomputed_map_key(value) for value in paths]
    row_keys = [
        precomputed_row_key(map_key, index)
        for index, map_key in enumerate(map_keys)
    ]

    assert len(routed) == 2
    assert map_keys[0] == map_keys[1]
    assert map_keys[0] != map_keys[2]
    assert len(set(row_keys)) == 3


def test_neural_artifacts_keep_dataset_image_ids(tmp_path):
    root = tmp_path / "images"
    paths = ["shared/001.jpg", "shared/002.jpg"]
    for value in paths:
        _write_image(root / value)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"image_id": "nsd-001", "image_path": value, "split": "train"}
            for value in paths
        ],
    )

    rows = _load_image_rows(
        manifest,
        image_root=root,
        split="train",
        subject_id=None,
        roi=None,
        max_items=None,
    )

    assert len(rows) == 1
    assert rows[0]["image_id"] == "nsd-001"


def test_cluster_preflight_counts_unique_paths_not_repeated_ids(tmp_path):
    root = tmp_path / "images"
    paths = ["Action/001.jpg", "Art/001.jpg"]
    for value in paths:
        _write_image(root / value)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"image_id": "001", "image_path": value, "split": "train"}
            for value in paths
        ],
    )

    result = _validate_image_manifest(
        manifest,
        root,
        expected_unique=2,
        verify_all_images=True,
    )

    assert result["passed"] is True
    assert result["detail"].startswith("2 unique images")


def test_matrix_v2_deit_family_uses_matched_crop():
    registry = load_external_registry()

    assert {
        float(registry.model(model)["preprocessing"]["crop_pct"])
        for model in (
            "deit_small_static",
            "dynamicvit_deit_small_keep_0_7",
            "tome_deit_small_r13",
        )
    } == {0.875}


def test_dinov2_provenance_restore_verifies_every_file(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    for cell in CELL_NAMES:
        cell_dir = source / cell
        cell_dir.mkdir(parents=True)
        (cell_dir / "metadata.json").write_text(cell, encoding="utf-8")

    manifest = restore(
        source,
        destination,
        tmp_path / "hashes.csv",
    )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(CELL_NAMES)
    assert all(row["verified"] == "True" for row in rows)
