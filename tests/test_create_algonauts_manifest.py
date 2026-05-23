import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.create_algonauts_manifest import (
    create_algonauts_manifest,
    resolve_prf_visual_roi_labels,
)


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


def test_resolve_prf_visual_roi_labels_combines_dorsal_and_ventral_labels():
    assert resolve_prf_visual_roi_labels("V1") == {"V1v", "V1d"}
    assert resolve_prf_visual_roi_labels("V2") == {"V2v", "V2d"}
    assert resolve_prf_visual_roi_labels("V3") == {"V3v", "V3d"}
    assert resolve_prf_visual_roi_labels("hV4") == {"hV4"}


def test_create_algonauts_manifest_writes_bilateral_roi_responses(tmp_path):
    root = tmp_path / "algonauts"
    subject_dir = root / "subj01"
    image_dir = subject_dir / "training_split" / "training_images"
    fmri_dir = subject_dir / "training_split" / "training_fmri"
    mask_dir = subject_dir / "roi_masks"
    fmri_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(2):
        _write_image(image_dir / f"train-{index + 1:04d}_nsd-{index:05d}.png")
    np.save(fmri_dir / "lh_training_fmri.npy", np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
    np.save(fmri_dir / "rh_training_fmri.npy", np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32))
    np.save(mask_dir / "lh.prf-visualrois_challenge_space.npy", np.array([1, 2, 3, 7]))
    np.save(mask_dir / "rh.prf-visualrois_challenge_space.npy", np.array([0, 1, 2]))
    np.save(mask_dir / "mapping_prf-visualrois.npy", {0: "Unknown", 1: "V1v", 2: "V1d", 3: "V2v", 7: "hV4"})

    manifest = tmp_path / "roi_manifest.csv"
    summary = create_algonauts_manifest(
        root=root,
        subject="subj01",
        hemispheres=["lh", "rh"],
        roi_class="prf-visualrois",
        roi_names=["V1", "hV4"],
        combine_hemispheres=True,
        output_manifest=manifest,
    )

    assert summary["rows"] == 4
    assert summary["response_dim"] == "V1:4,hV4:1"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["roi"] == "V1"
    assert rows[2]["roi"] == "hV4"
    assert np.allclose(np.load(root / rows[0]["roi_response_path"]), [1, 2, 20, 30])
    assert np.allclose(np.load(root / rows[2]["roi_response_path"]), [4])


def test_create_algonauts_manifest_attaches_noise_ceilings_for_roi_rows(tmp_path):
    root = tmp_path / "algonauts"
    subject_dir = root / "subj01"
    image_dir = subject_dir / "training_split" / "training_images"
    fmri_dir = subject_dir / "training_split" / "training_fmri"
    mask_dir = subject_dir / "roi_masks"
    ceiling_dir = subject_dir / "noise_ceilings"
    fmri_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    ceiling_dir.mkdir(parents=True)
    for index in range(3):
        _write_image(image_dir / f"train-{index + 1:04d}_nsd-{index:05d}.png")
    np.save(fmri_dir / "lh_training_fmri.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    np.save(fmri_dir / "rh_training_fmri.npy", np.arange(30, 39, dtype=np.float32).reshape(3, 3))
    np.save(mask_dir / "lh.prf-visualrois_challenge_space.npy", np.array([1, 2, 3, 7]))
    np.save(mask_dir / "rh.prf-visualrois_challenge_space.npy", np.array([0, 1, 2]))
    np.save(mask_dir / "mapping_prf-visualrois.npy", {0: "Unknown", 1: "V1v", 2: "V1d", 3: "V2v", 7: "hV4"})
    np.save(ceiling_dir / "V1.npy", np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32))
    np.save(ceiling_dir / "hV4.npy", np.array([0.2], dtype=np.float32))

    manifest = tmp_path / "full_roi_manifest.csv"
    summary = create_algonauts_manifest(
        root=root,
        subject="subj01",
        hemispheres=["lh", "rh"],
        roi_class="prf-visualrois",
        roi_names=["V1", "hV4"],
        combine_hemispheres=True,
        output_manifest=manifest,
        attach_noise_ceilings=True,
        max_items=3,
    )

    assert summary["rows"] == 6
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert "noise_ceiling_path" in fieldnames
    assert "noise_ceiling_values" in fieldnames
    assert "noise_ceiling_source" in fieldnames
    assert len(rows) == 6
    assert rows[0]["noise_ceiling_path"] == "subj01/noise_ceilings/V1.npy"
    assert rows[0]["noise_ceiling_values"] == ""
    assert rows[0]["noise_ceiling_source"] == "nsd_ncsnr_mgh_n_trials_3"
    assert rows[3]["noise_ceiling_path"] == "subj01/noise_ceilings/hV4.npy"
    assert np.load(root / rows[0]["roi_response_path"]).shape == (4,)
    assert np.load(root / rows[3]["roi_response_path"]).shape == (1,)
