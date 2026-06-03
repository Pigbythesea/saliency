import csv
import json

from scripts.create_salicon_observer_manifest import create_salicon_observer_manifest


def test_create_salicon_observer_manifest_joins_json_annotations(tmp_path):
    base_manifest = tmp_path / "salicon_manifest.csv"
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "fixation_map_path",
                "fixation_points_path",
                "split",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "COCO_val2014_000000000123",
                "image_path": "images/val/COCO_val2014_000000000123.jpg",
                "fixation_map_path": "maps/val/COCO_val2014_000000000123.png",
                "fixation_points_path": "fixations/val/COCO_val2014_000000000123.mat",
                "split": "val",
                "width": "640",
                "height": "480",
            }
        )

    annotations = tmp_path / "fixations_val2014.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 123,
                        "file_name": "COCO_val2014_000000000123.jpg",
                    }
                ],
                "annotations": [
                    {
                        "id": 7,
                        "image_id": 123,
                        "worker_id": 42,
                        "fixations": [[1, 2], [3.5, 4.5]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "observer_manifest.csv"

    summary = create_salicon_observer_manifest(
        base_manifest=base_manifest,
        annotation_jsons=[annotations],
        output_manifest=output,
    )

    assert summary["rows_written"] == 1
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["subject_id"] == "42"
    assert rows[0]["trial_id"] == "7"
    assert json.loads(rows[0]["fixation_points"]) == [[1.0, 2.0], [3.5, 4.5]]
