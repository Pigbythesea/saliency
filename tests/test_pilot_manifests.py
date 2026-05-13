import csv
from collections import Counter

from hma.experiments.pilot_manifests import create_pilot_manifest


def _write_manifest(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pilot_manifest_generation_is_deterministic(tmp_path):
    source = tmp_path / "source.csv"
    fieldnames = ["image_id", "split", "category"]
    rows = [
        {"image_id": f"img_{index}", "split": "val", "category": "all"}
        for index in range(20)
    ]
    _write_manifest(source, rows, fieldnames)

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    create_pilot_manifest(source, first, max_rows=5, split="val", seed=123)
    create_pilot_manifest(source, second, max_rows=5, split="val", seed=123)

    assert _read_rows(first) == _read_rows(second)


def test_pilot_manifest_stratifies_by_category(tmp_path):
    source = tmp_path / "cat.csv"
    fieldnames = ["image_id", "split", "category"]
    rows = []
    for category in ("Action", "Art", "Social"):
        rows.extend(
            {
                "image_id": f"{category}_{index}",
                "split": "train",
                "category": category,
            }
            for index in range(10)
        )
    _write_manifest(source, rows, fieldnames)

    output = tmp_path / "pilot.csv"
    summary = create_pilot_manifest(
        source,
        output,
        max_rows=9,
        split="train",
        stratify_column="category",
        seed=123,
    )
    selected = _read_rows(output)

    assert summary["rows_written"] == 9
    assert Counter(row["category"] for row in selected) == {
        "Action": 3,
        "Art": 3,
        "Social": 3,
    }


def test_pilot_manifest_preserves_coco_metadata_columns(tmp_path):
    source = tmp_path / "coco.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "split",
        "width",
        "height",
        "target_category",
        "task",
        "fixation_points",
        "subject_id",
        "trial_id",
    ]
    rows = [
        {
            "image_id": f"trial_{index}",
            "image_path": f"images/cat_{index % 2}/trial_{index}.jpg",
            "split": "val",
            "width": "640",
            "height": "480",
            "target_category": f"cat_{index % 2}",
            "task": "search",
            "fixation_points": "[[1.0, 2.0]]",
            "subject_id": str(index % 3),
            "trial_id": str(index),
        }
        for index in range(12)
    ]
    _write_manifest(source, rows, fieldnames)

    output = tmp_path / "pilot.csv"
    create_pilot_manifest(
        source,
        output,
        max_rows=6,
        split="val",
        stratify_column="target_category",
        seed=123,
    )

    selected = _read_rows(output)
    assert list(selected[0]) == fieldnames
    assert all(row["fixation_points"] == "[[1.0, 2.0]]" for row in selected)
    assert set(row["target_category"] for row in selected) == {"cat_0", "cat_1"}
