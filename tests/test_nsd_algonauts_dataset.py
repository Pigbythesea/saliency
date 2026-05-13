import csv
from pathlib import Path

import numpy as np
from PIL import Image

from hma.datasets import NSDAlgonautsDataset


EXPECTED_ITEM_KEYS = {
    "image",
    "image_id",
    "image_path",
    "fixation_map",
    "fixation_points",
    "metadata",
}


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((8, 12, 3), 128, dtype=np.uint8), mode="RGB")
    image.save(path)


def _make_manifest(tmp_path):
    root = tmp_path / "nsd"
    image_path = root / "images" / "image_001.png"
    response_path = root / "responses" / "image_001.npy"
    _write_image(image_path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(response_path, np.array([0.1, 0.2, 0.3], dtype=np.float32))

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
        writer.writerow(
            {
                "image_id": "image_001",
                "image_path": "images/image_001.png",
                "split": "train",
                "subject_id": "subj01",
                "roi": "V1",
                "roi_response_path": "responses/image_001.npy",
                "roi_responses": "",
            }
        )
    return root, manifest


def test_nsd_algonauts_dataset_loads_image_and_roi_responses(tmp_path):
    root, manifest = _make_manifest(tmp_path)

    dataset = NSDAlgonautsDataset(
        root=root,
        manifest_path=manifest,
        split="train",
        max_items=1,
        validate_files=True,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert set(item) == EXPECTED_ITEM_KEYS
    assert item["image"].mode == "RGB"
    assert item["image_id"] == "image_001"
    assert item["fixation_map"] is None
    assert item["fixation_points"] is None
    assert item["metadata"]["subject_id"] == "subj01"
    assert item["metadata"]["roi"] == "V1"
    assert np.allclose(item["metadata"]["roi_responses"], [0.1, 0.2, 0.3])
