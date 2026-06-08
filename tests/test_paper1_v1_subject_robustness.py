import csv
from pathlib import Path

from hma.utils.config import load_yaml, save_yaml
from scripts.compute_paper1_v1_subject_robustness_geometry import (
    write_subject_geometry_scope_configs,
)
from scripts.create_paper1_v1_subject_robustness_configs import (
    audit_paper1_v1_subject_robustness_configs,
    generate_paper1_v1_subject_robustness_configs,
)
from scripts.summarize_paper1_v1_subject_robustness_results import (
    subject_robustness_decision_rows,
)


MODELS = [
    "resnet50",
    "vit_base_patch16_224",
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
]
SUBJECTS = ["subj02", "subj03", "subj04"]
ROIS = ["V1", "V2", "V3", "hV4"]


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
                    "rois": ROIS,
                }
            },
            "expected_cells": {"models": 4, "rois": 4, "model_roi_cells": 16},
            "max_items": 2,
        },
        "confirmatory_matrix": {
            "subjects": SUBJECTS,
            "trigger": "test",
            "config_root": "configs/experiments/paper1_experiment_v1/neural_subject_robustness",
            "output_root_template": "outputs/paper1_experiment_v1/neural_{subject}_subject_robustness",
            "summary_output_dir": "outputs/paper1_experiment_v1/summary",
            "manifest_path_template": (
                "data/manifests/nsd_algonauts_{subject}_prf_visualrois_full_manifest.csv"
            ),
            "reduced_models": MODELS,
            "rois": ROIS,
            "roi_class": "prf-visualrois",
            "expected_cells": {
                "subjects": 3,
                "models": 4,
                "rois": 4,
                "subject_model_roi_cells": 48,
                "cells_per_subject": 16,
            },
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
        "geometry": {
            "methods": ["linear_cka_full9841", "subset_rsa"],
            "subset_sizes": [512],
            "subset_seeds": [123],
        },
    }
    path = tmp_path / "configs" / "paper1_experiment_v1.yaml"
    save_yaml(config, path)
    return path


def _write_manifests() -> None:
    for subject in SUBJECTS:
        path = Path(f"data/manifests/nsd_algonauts_{subject}_prf_visualrois_full_manifest.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "image_id,image_path,split,subject_id,roi,roi_response_path,roi_responses\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_subject_robustness_configs_writes_subject_scoped_prf_configs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    _write_manifests()

    result = generate_paper1_v1_subject_robustness_configs(config_path=config_path)

    assert len(result["config_paths"]) == 48
    assert all(row["status"] == "pass" for row in result["audit_rows"])
    assert Path(
        "outputs/paper1_experiment_v1/summary/subject_robustness_artifact_audit.csv"
    ).is_file()

    configs = [load_yaml(path) for path in result["config_paths"]]
    assert {cfg["dataset"]["subject_id"] for cfg in configs} == set(SUBJECTS)
    assert {cfg["dataset"]["roi"] for cfg in configs} == set(ROIS)
    assert {
        cfg["dataset"]["manifest_path"] for cfg in configs
    } == {
        f"data/manifests/nsd_algonauts_{subject}_prf_visualrois_full_manifest.csv"
        for subject in SUBJECTS
    }
    assert all("streams" not in cfg["dataset"]["manifest_path"] for cfg in configs)
    assert all(
        cfg["output"]["dir"].startswith(
            f"outputs/paper1_experiment_v1/neural_{cfg['dataset']['subject_id']}_subject_robustness/"
        )
        for cfg in configs
    )


def test_subject_robustness_audit_fails_when_required_config_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    config = load_yaml(config_path)
    _write_manifests()
    result = generate_paper1_v1_subject_robustness_configs(config_path=config_path)

    rows = audit_paper1_v1_subject_robustness_configs(
        config=config,
        config_paths=result["config_paths"][:-1],
    )

    by_check = {row["check"]: row for row in rows}
    assert by_check["subject_robustness_expected_config_count"]["status"] == "fail"


def test_subject_robustness_geometry_scopes_are_prf_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    config = load_yaml(config_path)

    scope_paths = write_subject_geometry_scope_configs(config)

    assert len(scope_paths) == 3
    for path in scope_paths:
        scope = load_yaml(path)
        roi_groups = scope["discovery_matrix"]["roi_groups"]
        assert list(roi_groups) == ["prf_visualrois"]
        assert roi_groups["prf_visualrois"]["rois"] == ROIS


def test_subject_robustness_decisions_use_subject_specific_leaders(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)
    config = load_yaml(config_path)
    summary_dir = Path("outputs/paper1_experiment_v1/summary")
    _write_csv(
        summary_dir / "roi_expanded_encoding_model_rankings.csv",
        [
            {"model": "vit_small_patch14_dinov2", "rank_mean_noise_normalized": "1"},
            {"model": "resnet50", "rank_mean_noise_normalized": "2"},
        ],
    )
    _write_csv(
        summary_dir / "roi_expanded_geometry_model_rankings.csv",
        [
            {
                "model": "vit_small_patch14_dinov2",
                "geometry_method": "linear_cka_full9841",
                "rank_mean_geometry": "1",
            },
            {
                "model": "resnet50",
                "geometry_method": "linear_cka_full9841",
                "rank_mean_geometry": "2",
            },
        ],
    )
    encoding_rows = _ranking_rows(
        "subj02",
        "vit_small_patch14_dinov2",
        "rank_mean_noise_normalized",
    ) + _ranking_rows(
        "subj03",
        "vit_small_patch14_dinov2",
        "rank_mean_noise_normalized",
    ) + _ranking_rows(
        "subj04",
        "resnet50",
        "rank_mean_noise_normalized",
    )
    geometry_rows = _ranking_rows(
        "subj02",
        "vit_small_patch14_dinov2",
        "rank_mean_geometry",
        geometry_method="linear_cka_full9841",
    ) + _ranking_rows(
        "subj03",
        "resnet50",
        "rank_mean_geometry",
        geometry_method="linear_cka_full9841",
    ) + _ranking_rows(
        "subj04",
        "resnet50",
        "rank_mean_geometry",
        geometry_method="linear_cka_full9841",
    )

    decisions = subject_robustness_decision_rows(
        config=config,
        encoding_rows=encoding_rows,
        geometry_rows=geometry_rows,
    )

    by_subject = {row["subject_id"]: row for row in decisions}
    assert by_subject["subj02"]["decision_label"] == "replicated"
    assert by_subject["subj03"]["decision_label"] == "partial"
    assert by_subject["subj04"]["decision_label"] == "failed"
    assert by_subject["all_confirmatory_subjects"]["decision_label"] == "partial"
    assert {row["subject_id"] for row in decisions}.issuperset(set(SUBJECTS))


def _ranking_rows(
    subject: str,
    leader: str,
    rank_key: str,
    *,
    geometry_method: str | None = None,
) -> list[dict[str, str]]:
    rows = []
    for model in MODELS:
        row = {
            "subject_id": subject,
            "model": model,
            rank_key: "1" if model == leader else "2",
        }
        if geometry_method is not None:
            row["geometry_method"] = geometry_method
        rows.append(row)
    return rows
