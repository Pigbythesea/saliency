import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.create_algonauts_manifest import create_algonauts_manifest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), mode="RGB").save(path)


def test_create_algonauts_manifest_writes_manifest_and_responses(tmp_path):
    root = tmp_path / "algonauts"
    subject_dir = root / "subj01"
    image_dir = subject_dir / "training_split" / "training_images"
    fmri_dir = subject_dir / "training_split" / "training_fmri"
    fmri_dir.mkdir(parents=True)
    for index in range(3):
        _write_image(image_dir / f"train-{index + 1:04d}_nsd-{index:05d}.png")
    np.save(fmri_dir / "lh_training_fmri.npy", np.arange(18, dtype=np.float32).reshape(3, 6))

    manifest = tmp_path / "manifest.csv"
    summary = create_algonauts_manifest(
        root=root,
        subject="subj01",
        hemisphere="lh",
        num_vertices=4,
        roi="all_lh_4",
        output_manifest=manifest,
        max_items=2,
    )

    assert summary["rows"] == 2
    assert summary["response_dim"] == 4
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["image_id"] == "train-0001_nsd-00000"
    assert rows[0]["roi"] == "all_lh_4"
    response_path = root / rows[0]["roi_response_path"]
    assert np.allclose(np.load(response_path), [0, 1, 2, 3])
