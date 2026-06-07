import csv
from pathlib import Path

import numpy as np
from PIL import Image

from hma.utils.config import load_yaml, save_yaml
from scripts.create_paper1_v1_roi_expanded_configs import (
    audit_paper1_v1_roi_expanded_readiness,
    generate_paper1_v1_roi_expanded_readiness,
)


MODELS = [
    "resnet50",
    "vit_base_patch16_224",
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
]
PRF_ROIS = ["V1", "V2", "V3", "hV4"]
STREAM_ROIS = [
    "midventral",
    "midlateral",
    "midparietal",
    "ventral",
    "lateral",
    "parietal",
]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), mode="RGB").save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                "noise_ceiling_path",
                "noise_ceiling_values",
                "noise_ceiling_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_config(tmp_path: Path) -> Path:
    config = {
        "discovery_matrix": {
            "subject_id": "subj01",
            "output_root": "outputs/paper1_experiment_v1/neural_subj01_roi_expanded",
            "config_root": "configs/experiments/paper1_experiment_v1/neural_subj01_roi_expanded",
            "summary_output_dir": "outputs/paper1_experiment_v1/summary",
            "models": MODELS,
            "roi_groups": {
                "prf_visualrois": {
                    "roi_class": "prf-visualrois",
                    "manifest_path": "data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv",
                    "rois": PRF_ROIS,
                },
                "streams": {
                    "roi_class": "streams",
                    "manifest_path": "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv",
                    "mapping_path": "data/raw/nsd_algonauts/subj01/roi_masks/mapping_streams.npy",
                    "rois": STREAM_ROIS,
                },
            },
            "expected_cells": {"models": 4, "rois": 10, "model_roi_cells": 40},
            "max_items": 2,
        },
        "encoding": {
            "method": "flatten_pca",
            "response_key": "roi_responses",
            "full_image_count": True,
            "validation_selected_layer": True,
            "pca_components": 512,
            "pca_solver": "randomized",
            "pca_whiten": False,
            "ridge_alphas": [0.001, 0.01, 0.1, 1.0],
            "validation_fraction": 0.2,
            "selection_primary_score": "mean_noise_normalized_score",
        },
    }
    path = tmp_path / "configs" / "paper1_experiment_v1.yaml"
    save_yaml(config, path)
    return path


def _make_fake_algonauts_data() -> None:
    root = Path("data/raw/nsd_algonauts")
    subject_dir = root / "subj01"
    image_dir = subject_dir / "training_split" / "training_images"
    fmri_dir = subject_dir / "training_split" / "training_fmri"
    mask_dir = subject_dir / "roi_masks"
    fmri_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(2):
        _write_image(image_dir / f"train-{index + 1:04d}_nsd-{index:05d}.png")
    np.save(fmri_dir / "lh_training_fmri.npy", np.arange(20, dtype=np.float32).reshape(2, 10))
    np.save(fmri_dir / "rh_training_fmri.npy", np.arange(40, 60, dtype=np.float32).reshape(2, 10))
    np.save(mask_dir / "lh.streams_challenge_space.npy", np.array([2, 3, 4, 5, 6, 7, 2, 3, 4, 5]))
    np.save(mask_dir / "rh.streams_challenge_space.npy", np.array([2, 3, 4, 5, 6, 7, 6, 7, 0, 0]))
    np.save(
        mask_dir / "mapping_streams.npy",
        {
            0: "Unknown",
            1: "early",
            2: "midventral",
            3: "midlateral",
            4: "midparietal",
            5: "ventral",
            6: "lateral",
            7: "parietal",
        },
    )

    rows = []
    for roi in PRF_ROIS:
        for index in range(2):
            image_id = f"train-{index + 1:04d}_nsd-{index:05d}"
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": f"subj01/training_split/training_images/{image_id}.png",
                    "split": "train",
                    "subject_id": "subj01",
                    "roi": roi,
                    "roi_response_path": "",
                    "roi_responses": "",
                    "noise_ceiling_path": "",
                    "noise_ceiling_values": "",
                    "noise_ceiling_source": "",
                }
            )
    _write_manifest(Path("data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv"), rows)


def test_generate_paper1_v1_roi_expanded_readiness_writes_config_driven_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    _make_fake_algonauts_data()

    result = generate_paper1_v1_roi_expanded_readiness(config_path=config_path)

    assert Path("data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv").is_file()
    assert len(result["config_paths"]) == 40
    assert all(row["status"] == "pass" for row in result["audit_rows"])
    assert Path("outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv").is_file()

    configs = {path.name: load_yaml(path) for path in result["config_paths"]}
    prf = configs["resnet50_v1_flatten_pca_validation_selection_full.yaml"]
    stream = configs["resnet50_midventral_flatten_pca_validation_selection_full.yaml"]
    dino = configs["vit_small_patch14_dinov2_midventral_flatten_pca_validation_selection_full.yaml"]

    assert prf["dataset"]["manifest_path"] == (
        "data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv"
    )
    assert stream["dataset"]["manifest_path"] == (
        "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv"
    )
    assert stream["dataset"]["max_items"] == 2
    assert stream["neural"]["feature_reduction"] == "flatten_pca"
    assert stream["neural"]["pca_components"] == 512
    assert stream["neural"]["selection"]["enabled"] is True
    assert stream["neural"]["rsa"]["enabled"] is False
    assert dino["preprocessing"]["input_size"] == [518, 518]


def test_paper1_v1_audit_fails_when_a_required_config_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    config = load_yaml(config_path)
    _make_fake_algonauts_data()

    result = generate_paper1_v1_roi_expanded_readiness(config_path=config_path)
    audit_rows = audit_paper1_v1_roi_expanded_readiness(
        config=config,
        config_paths=result["config_paths"][:-1],
    )

    by_check = {row["check"]: row for row in audit_rows}
    assert by_check["expected_config_count"]["status"] == "fail"
