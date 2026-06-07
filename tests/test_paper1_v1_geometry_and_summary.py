import csv
import json
from pathlib import Path

import numpy as np

from hma.experiments.summarize_neural_roi_results import summarize_neural_roi_results
from hma.utils.config import save_yaml
from scripts import compute_matched_geometry as geometry_script


MODELS = [
    "resnet50",
    "vit_base_patch16_224",
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
]
ROIS = [
    "V1",
    "V2",
    "V3",
    "hV4",
    "midventral",
    "midlateral",
    "midparietal",
    "ventral",
    "lateral",
    "parietal",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _v1_config(tmp_path: Path, *, max_items: int = 9841) -> Path:
    path = tmp_path / "configs" / "paper1_experiment_v1.yaml"
    save_yaml(
        {
            "baseline_inputs": {
                "behavioral_csv": str(tmp_path / "outputs" / "behavior.csv"),
            },
            "discovery_matrix": {
                "subject_id": "subj01",
                "output_root": str(tmp_path / "outputs" / "paper1_experiment_v1" / "neural_subj01_roi_expanded"),
                "config_root": str(tmp_path / "configs" / "experiments" / "paper1_experiment_v1" / "neural_subj01_roi_expanded"),
                "summary_output_dir": str(tmp_path / "outputs" / "paper1_experiment_v1" / "summary"),
                "models": MODELS,
                "roi_groups": {
                    "prf_visualrois": {"rois": ROIS[:4]},
                    "streams": {"rois": ROIS[4:]},
                },
                "expected_cells": {"models": 4, "rois": 10, "model_roi_cells": 40},
                "max_items": max_items,
            },
            "encoding": {"method": "flatten_pca"},
            "geometry": {
                "methods": ["linear_cka_full9841", "subset_rsa"],
                "subset_sizes": [512, 1024, 2048],
                "subset_seeds": [123, 456, 789],
                "feature_rdm_metric": "correlation",
                "response_rdm_metric": "correlation",
                "rdm_compare_method": "spearman",
            },
        },
        path,
    )
    return path


def _cell_stem(model: str, roi: str) -> str:
    slug = roi.lower().replace("hv4", "hv4")
    return f"{model}_{slug}_flatten_pca_validation_selection_full"


def _write_v1_config_files(config_path: Path) -> list[Path]:
    # Re-read with the project loader to avoid assuming YAML formatting.
    from hma.utils.config import load_yaml

    config = load_yaml(config_path)
    config_root = Path(config["discovery_matrix"]["config_root"])
    written = []
    for model in MODELS:
        for roi in ROIS:
            path = config_root / f"{_cell_stem(model, roi)}.yaml"
            save_yaml({"model": {"name": model}, "dataset": {"roi": roi}}, path)
            written.append(path)
    return written


def _write_geometry_discovery_outputs(config_path: Path) -> None:
    from hma.utils.config import load_yaml

    config = load_yaml(config_path)
    output_root = Path(config["discovery_matrix"]["output_root"])
    for model in MODELS:
        for roi in ROIS:
            output_dir = output_root / _cell_stem(model, roi)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_name": model,
                        "model": model,
                        "subjects": ["subj01"],
                        "rois": [roi],
                        "feature_reduction": "flatten_pca",
                        "num_items": 9841,
                        "config_path": str(config_path),
                        "activations": str(output_dir / "activations.npz"),
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(output_dir / "encoding_scores.csv", [{"layer": "layer3"}], ["layer"])
            np.savez(output_dir / "activations.npz", image_ids=np.array(["a", "b"]))


def test_compute_matched_geometry_discovers_v1_outputs_and_preserves_filters(
    tmp_path,
    monkeypatch,
):
    config_path = _v1_config(tmp_path)
    _write_v1_config_files(config_path)
    _write_geometry_discovery_outputs(config_path)

    calls: list[Path] = []

    def fake_compute_output_geometry(output_dir, **_kwargs):
        calls.append(output_dir)
        return [
            {
                "dataset": "nsd_algonauts",
                "model": "resnet50",
                "subject_id": "subj01",
                "roi": "midventral",
                "layer": "layer3",
                "geometry_method": "linear_cka_full9841",
                "score": "0.1",
                "valid": "true",
                "status": "ok",
                "num_images_total": "9841",
                "num_images_used": "9841",
            }
        ]

    monkeypatch.setattr(geometry_script, "_compute_output_geometry", fake_compute_output_geometry)

    written = geometry_script.compute_matched_geometry(
        config_path,
        models=["resnet50"],
        rois=["midventral"],
    )

    assert len(calls) == 1
    assert len(written) == 1
    assert written[0].name == "geometry_scores.csv"


def test_compute_matched_geometry_rejects_missing_v1_activation(tmp_path):
    config_path = _v1_config(tmp_path)
    _write_v1_config_files(config_path)
    _write_geometry_discovery_outputs(config_path)
    missing = (
        tmp_path
        / "outputs"
        / "paper1_experiment_v1"
        / "neural_subj01_roi_expanded"
        / _cell_stem("resnet50", "midventral")
        / "activations.npz"
    )
    missing.unlink()

    try:
        geometry_script.compute_matched_geometry(
            config_path,
            models=["resnet50"],
            rois=["midventral"],
        )
    except FileNotFoundError as exc:
        assert "activations.npz" in str(exc)
    else:
        raise AssertionError("Expected missing activation to fail")


def test_compute_matched_geometry_preserves_legacy_scope(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs" / "legacy"
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "resnet50",
                "subjects": ["subj01"],
                "rois": ["V1"],
                "feature_reduction": "flatten_pca",
                "num_items": 9841,
                "config_path": str(tmp_path / "cell.yaml"),
            }
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "encoding_scores.csv", [{"layer": "layer3"}], ["layer"])
    np.savez(output_dir / "activations.npz", image_ids=np.array(["a", "b"]))
    config_path = tmp_path / "paper1_config.yaml"
    save_yaml(
        {
            "paper1_scope": {
                "models": ["resnet50"],
                "rois": ["V1"],
                "feature_reduction": "flatten_pca",
                "expected_num_items": 9841,
                "geometry": {"methods": ["linear_cka_full9841"]},
                "neural_output_dirs": [str(output_dir)],
            }
        },
        config_path,
    )
    monkeypatch.setattr(
        geometry_script,
        "_compute_output_geometry",
        lambda *_args, **_kwargs: [{"geometry_method": "linear_cka_full9841"}],
    )

    written = geometry_script.compute_matched_geometry(config_path)

    assert written == [output_dir / "geometry_scores.csv"]


def _write_summary_cell(
    output_root: Path,
    model: str,
    roi: str,
    model_index: int,
    *,
    subset_score_sign: float = 1.0,
) -> Path:
    output_dir = output_root / _cell_stem(model, roi)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": model,
                "model": model,
                "subjects": ["subj01"],
                "rois": [roi],
                "num_items": 9841,
                "config_path": "cell.yaml",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "encoding_scores.csv",
        [
            {
                "dataset": "nsd_algonauts",
                "model": model,
                "subject_id": "subj01",
                "roi": roi,
                "layer": "layer3",
                "metric": "correlation",
                "mean_score": 0.1 * model_index,
                "mean_noise_normalized_score": 0.2 * model_index,
                "alpha_selection_mode": "selection_validation",
                "feature_reduction": "flatten_pca",
            }
        ],
        [
            "dataset",
            "model",
            "subject_id",
            "roi",
            "layer",
            "metric",
            "mean_score",
            "mean_noise_normalized_score",
            "alpha_selection_mode",
            "feature_reduction",
        ],
    )
    geometry_rows = [
        {
            "dataset": "nsd_algonauts",
            "model": model,
            "subject_id": "subj01",
            "roi": roi,
            "layer": "layer3",
            "geometry_method": "linear_cka_full9841",
            "score": 0.3 * model_index,
            "valid": "true",
            "status": "ok",
            "num_images_total": 9841,
            "num_images_used": 9841,
            "model_feature_reduction": "flatten_pca",
        }
    ]
    for size in [512, 1024, 2048]:
        for seed in [123, 456, 789]:
            geometry_rows.append(
                {
                    "dataset": "nsd_algonauts",
                    "model": model,
                    "subject_id": "subj01",
                    "roi": roi,
                    "layer": "layer3",
                    "geometry_method": f"subset_rsa_corr_rdm_spearman_size{size}_seed{seed}",
                    "score": subset_score_sign * 0.25 * model_index + size / 100000,
                    "valid": "true",
                    "status": "ok",
                    "num_images_total": 9841,
                    "num_images_used": size,
                    "subset_seed": seed,
                    "subset_size": size,
                    "model_feature_reduction": "flatten_pca",
                }
            )
    _write_csv(
        output_dir / "geometry_scores.csv",
        geometry_rows,
        [
            "dataset",
            "model",
            "subject_id",
            "roi",
            "layer",
            "geometry_method",
            "score",
            "valid",
            "status",
            "num_images_total",
            "num_images_used",
            "subset_seed",
            "subset_size",
            "model_feature_reduction",
        ],
    )
    return output_dir


def test_v1_summary_writes_roi_expanded_aliases_and_audit(tmp_path):
    config_path = _v1_config(tmp_path)
    _write_v1_config_files(config_path)
    output_root = (
        tmp_path
        / "outputs"
        / "paper1_experiment_v1"
        / "neural_subj01_roi_expanded"
    )
    input_dirs = []
    behavior_rows = []
    for model_index, model in enumerate(MODELS, start=1):
        for roi in ROIS:
            input_dirs.append(_write_summary_cell(output_root, model, roi, model_index))
        behavior_rows.append(
            {
                "dataset": "salicon_static2000",
                "model": model,
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 2000,
                "mean": model_index,
                "fixation_protocol": "points",
            }
        )
        behavior_rows.append(
            {
                "dataset": "coco_search18_static2000",
                "model": model,
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 2000,
                "mean": model_index,
                "fixation_protocol": "task_points",
            }
        )
    behavior_csv = tmp_path / "outputs" / "behavior.csv"
    _write_csv(
        behavior_csv,
        behavior_rows,
        [
            "dataset",
            "model",
            "saliency_method",
            "saliency_family",
            "metric",
            "n",
            "mean",
            "fixation_protocol",
        ],
    )

    outputs = summarize_neural_roi_results(
        input_dirs,
        tmp_path / "outputs" / "paper1_experiment_v1" / "summary",
        behavioral_csv=behavior_csv,
        scope_config=config_path,
    )

    alias_rows = _read_csv(outputs["roi_expanded_geometry_model_rankings"])
    agreement_rows = _read_csv(outputs["roi_expanded_geometry_method_agreement"])
    method_sensitivity_rows = _read_csv(
        outputs["roi_expanded_geometry_method_sensitivity_decisions"]
    )
    failure_gate_rows = _read_csv(outputs["roi_expanded_failure_gate_summary"])
    sensitivity_rows = _read_csv(outputs["matched_cross_axis_sensitivity"])
    audit_rows = _read_csv(outputs["experiment_artifact_audit"])
    expected_subset_methods = {
        f"subset_rsa_corr_rdm_spearman_size{size}_seed{seed}"
        for size in [512, 1024, 2048]
        for seed in [123, 456, 789]
    }

    assert {row["model"] for row in alias_rows} == set(MODELS)
    assert any("midventral" in row["rois"] for row in alias_rows)
    assert {
        row["sensitivity_geometry_method"]
        for row in agreement_rows
        if row["sensitivity_geometry_method"].startswith("subset_rsa_")
    } == expected_subset_methods
    assert method_sensitivity_rows
    assert {
        row["primary_geometry_method"] for row in method_sensitivity_rows
    } == {"linear_cka_full9841"}
    assert expected_subset_methods <= set(
        ";".join(row["subset_rsa_methods"] for row in method_sensitivity_rows).split(";")
    )
    assert any(
        row["geometry_sensitivity_label"] == "stable_across_geometry_methods"
        and row["relationship"] == "encoding_vs_geometry"
        for row in method_sensitivity_rows
    )
    assert any(
        row["geometry_sensitivity_label"] == "not_tested"
        and row["relationship"] == "behavior_vs_noise_normalized"
        for row in method_sensitivity_rows
    )
    assert any(
        set(row["models"].split(";")) == set(MODELS)
        for row in method_sensitivity_rows
        if row["relationship"] == "encoding_vs_geometry"
    )
    assert {
        row["value"]
        for row in failure_gate_rows
        if row["summary_item"] == "failure_gate_next_step"
    } <= {
        "subject_robustness_subj02_subj04",
        "geometry_uncertainty_repair",
        "downgraded_paper1_framing",
    }
    assert any(row["sensitivity_type"] == "leave_one_model" for row in sensitivity_rows)
    assert any(row["sensitivity_type"] == "leave_one_roi" for row in sensitivity_rows)
    assert {row["behavior_dataset"] for row in _read_csv(outputs["roi_expanded_cross_level_correlations"])} == {
        "salicon_static2000",
        "coco_search18_static2000",
    }
    assert all(row["status"] == "pass" for row in audit_rows)


def test_v1_geometry_method_sensitivity_flags_direction_conflict(tmp_path):
    config_path = _v1_config(tmp_path)
    _write_v1_config_files(config_path)
    output_root = (
        tmp_path
        / "outputs"
        / "paper1_experiment_v1"
        / "neural_subj01_roi_expanded"
    )
    input_dirs = []
    behavior_rows = []
    for model_index, model in enumerate(MODELS, start=1):
        for roi in ROIS:
            input_dirs.append(
                _write_summary_cell(
                    output_root,
                    model,
                    roi,
                    model_index,
                    subset_score_sign=-1.0,
                )
            )
        behavior_rows.append(
            {
                "dataset": "salicon_static2000",
                "model": model,
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 2000,
                "mean": model_index,
                "fixation_protocol": "points",
            }
        )
    behavior_csv = tmp_path / "outputs" / "behavior.csv"
    _write_csv(
        behavior_csv,
        behavior_rows,
        [
            "dataset",
            "model",
            "saliency_method",
            "saliency_family",
            "metric",
            "n",
            "mean",
            "fixation_protocol",
        ],
    )

    outputs = summarize_neural_roi_results(
        input_dirs,
        tmp_path / "outputs" / "paper1_experiment_v1" / "summary",
        behavioral_csv=behavior_csv,
        scope_config=config_path,
    )

    method_sensitivity_rows = _read_csv(
        outputs["roi_expanded_geometry_method_sensitivity_decisions"]
    )

    assert any(
        row["geometry_sensitivity_label"] == "direction_conflict"
        and row["relationship"] == "encoding_vs_geometry"
        for row in method_sensitivity_rows
    )
