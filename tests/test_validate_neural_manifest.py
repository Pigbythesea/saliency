import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.validate_neural_manifest import validate_neural_manifest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), mode="RGB").save(path)


def test_validate_neural_manifest_checks_files_and_response_shapes(tmp_path):
    root = tmp_path / "root"
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "split",
                "subject_id",
                "roi",
                "roi_response_path",
                "roi_responses",
            ],
        )
        writer.writeheader()
        for index in range(3):
            image_id = f"image_{index:03d}"
            image_path = root / "images" / f"{image_id}.png"
            response_path = root / "responses" / f"{image_id}.npy"
            _write_image(image_path)
            response_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(response_path, np.array([index, index + 1], dtype=np.float32))
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_path": f"images/{image_id}.png",
                    "split": "train",
                    "subject_id": "subj01",
                    "roi": "V1",
                    "roi_response_path": f"responses/{image_id}.npy",
                    "roi_responses": "",
                }
            )

    summary = validate_neural_manifest(
        manifest,
        root=root,
        split="train",
        subject_id="subj01",
        roi="V1",
        max_items=2,
    )

    assert summary["selected_rows"] == 2
    assert summary["response_shape"] == [2]
    assert summary["selected_subject_counts"] == {"subj01": 2}
    assert summary["selected_roi_counts"] == {"V1": 2}
