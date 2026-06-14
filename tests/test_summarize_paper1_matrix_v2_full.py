import csv
import json
from pathlib import Path

import scripts.summarize_paper1_matrix_v2_full as summary_module
from hma.saliency.precomputed import precomputed_map_key, precomputed_row_key
from scripts.summarize_paper1_matrix_v2_full import (
    BEHAVIOR_METRICS,
    MODELS,
    ROIS,
    summarize_full_results,
)


DATASETS = [
    ("cat2000_static2000", "points"),
    ("coco_search18_static2000", "task_points"),
    ("salicon_static2000", "points"),
]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_neural_outputs(root):
    for model_index, model in enumerate(MODELS):
        for roi_index, roi in enumerate(ROIS):
            output = root / "neural" / "full" / model / roi.lower()
            layer = f"blocks.{model_index + roi_index}"
            alpha = float(10 ** (model_index + 1))
            score = 0.4 + 0.02 * model_index + 0.01 * roi_index
            _write_csv(
                output / "encoding_scores.csv",
                [
                    {
                        "model": model,
                        "roi": roi,
                        "subject_id": "subj01",
                        "layer": layer,
                        "metric": "correlation",
                        "metric_scope": "benchmark_style_noise_normalized",
                        "n_train": 7873,
                        "n_test": 1968,
                        "num_targets": 10,
                        "mean_score": score,
                        "median_score": score,
                        "std_score": 0.1,
                        "mean_noise_normalized_score": score + 0.05,
                        "median_noise_normalized_score": score + 0.05,
                        "selected_ridge_alpha": alpha,
                    }
                ],
            )
            _write_json(
                output / "metadata.json",
                {
                    "num_items": 9841,
                    "selected_layer": layer,
                    "selected_ridge_alpha": alpha,
                    "selection_score": score,
                },
            )
            _write_json(
                output / "selection_artifact.json",
                {"primary_score": "mean_noise_normalized_score"},
            )
            geometry_rows = [
                {
                    "model": model,
                    "roi": roi,
                    "subject_id": "subj01",
                    "layer": layer,
                    "geometry_method": "linear_cka",
                    "score": score,
                    "valid": "true",
                    "status": "ok",
                    "num_images_total": 9841,
                    "num_images_used": 9841,
                    "subset_seed": "",
                    "subset_size": "",
                    "control_type": "observed",
                    "control_seed": "",
                    "paired_observed_score": "",
                    "observed_minus_null": "",
                }
            ]
            for seed in (123, 456, 789):
                geometry_rows.append(
                    {
                        "model": model,
                        "roi": roi,
                        "subject_id": "subj01",
                        "layer": layer,
                        "geometry_method": "linear_cka",
                        "score": score - 0.2,
                        "valid": "true",
                        "status": "ok",
                        "num_images_total": 9841,
                        "num_images_used": 9841,
                        "subset_seed": "",
                        "subset_size": "",
                        "control_type": "response_permutation",
                        "control_seed": seed,
                        "paired_observed_score": score,
                        "observed_minus_null": 0.2,
                    }
                )
            for subset_size in (512, 1024, 2048):
                for seed in (123, 456, 789):
                    observed_score = score + 0.01
                    common = {
                        "model": model,
                        "roi": roi,
                        "subject_id": "subj01",
                        "layer": layer,
                        "geometry_method": "subset_rsa",
                        "valid": "true",
                        "status": "ok",
                        "num_images_total": 9841,
                        "num_images_used": subset_size,
                        "subset_seed": seed,
                        "subset_size": subset_size,
                    }
                    geometry_rows.append(
                        {
                            **common,
                            "score": observed_score,
                            "control_type": "observed",
                            "control_seed": "",
                            "paired_observed_score": "",
                            "observed_minus_null": "",
                        }
                    )
                    geometry_rows.append(
                        {
                            **common,
                            "score": observed_score - 0.2,
                            "control_type": "response_permutation",
                            "control_seed": seed,
                            "paired_observed_score": observed_score,
                            "observed_minus_null": 0.2,
                        }
                    )
            _write_csv(output / "geometry_scores.csv", geometry_rows)


def _build_behavior_outputs(root, manifests):
    for dataset, protocol in DATASETS:
        manifest_rows = _read_csv(manifests[dataset])
        map_keys = [precomputed_map_key(row["image_path"]) for row in manifest_rows]
        unique_map_keys = sorted(set(map_keys))
        for model_index, model in enumerate(MODELS):
            output = (
                root
                / "behavior"
                / "full"
                / dataset
                / f"{model}_operational_routing"
            )
            if model == "deit_small_static":
                metrics = {
                    "nss": 0.0,
                    "shuffled_auc": 0.5,
                    "auc_borji": 0.5,
                    "auc_judd": 0.5,
                    "cc": 0.0,
                    "similarity": 0.3,
                    "kl": 1.5,
                }
            else:
                metrics = {
                    "nss": 0.1 * model_index,
                    "shuffled_auc": 0.55 + 0.02 * model_index,
                    "auc_borji": 0.56 + 0.02 * model_index,
                    "auc_judd": 0.57 + 0.02 * model_index,
                    "cc": 0.05 * model_index,
                    "similarity": 0.3 + 0.02 * model_index,
                    "kl": 2.0 - 0.1 * model_index,
                }
            _write_json(
                output / "aggregate_metrics.json",
                {
                    "dataset": dataset,
                    "model": model,
                    "num_items": len(manifest_rows),
                    "fixation_protocol": protocol,
                    "saliency_method": "precomputed_map",
                    "metrics": metrics,
                    "raw_metrics": {
                        f"raw_{metric}": (
                            1.0
                            if model == "deit_small_static"
                            and metric in {"shuffled_auc", "auc_borji"}
                            else value
                        )
                        for metric, value in metrics.items()
                    },
                    "map_lookup_count": len(manifest_rows),
                    "unique_map_lookup_count": len(unique_map_keys),
                    "unique_row_key_count": len(manifest_rows),
                    "constant_map_count": (
                        len(manifest_rows)
                        if model == "deit_small_static"
                        else 0
                    ),
                },
            )
            _write_csv(
                output / "per_image_metrics.csv",
                [
                    {
                        "row_key": precomputed_row_key(map_key, index),
                        "map_key": map_key,
                        "image_id": f"image-{index}",
                        **metrics,
                    }
                    for index, map_key in enumerate(map_keys)
                ],
            )
            artifact_dataset = summary_module.BEHAVIOR_ARTIFACT_DATASETS[dataset]
            artifact = (
                root
                / "external_artifacts"
                / "behavior_full"
                / artifact_dataset
                / model
            )
            _write_json(
                artifact / "manifest.json",
                {"image_ids_file": "image_ids.json"},
            )
            _write_json(artifact / "image_ids.json", unique_map_keys)
            map_dir = root / "routing_maps" / "full" / dataset / model
            map_dir.mkdir(parents=True, exist_ok=True)
            for map_key in unique_map_keys:
                (map_dir / f"{map_key}.npy").write_bytes(b"map")


def _build_efficiency_outputs(root):
    for model_index, model in enumerate(MODELS):
        output = root / "external_artifacts" / "full" / model
        _write_json(
            output / "efficiency.json",
            {
                "parameters": 20_000_000 + model_index,
                "theoretical_flops": 4_600_000_000,
                "realized_flops": 4_600_000_000 - model_index * 900_000_000,
                "latency_ms_per_image": 2.5 + model_index * 0.5,
                "latency_ms_per_image_std": 0.1,
                "latency_ms_per_image_cv": 0.03,
                "latency_ms_per_image_min": 2.4 + model_index * 0.5,
                "latency_ms_per_image_max": 2.6 + model_index * 0.5,
                "latency_repeats_ms_per_image": [2.5] * 5,
                "batch_size": 16,
                "warmup_batches": 20,
                "measured_batches_per_repeat": 100,
                "timing_repeats": 5,
                "cuda_synchronized": True,
                "hardware": {"device_name": "test-gpu"},
                "fvcore_unsupported_ops": {},
                "fvcore_uncalled_modules": [],
                "peak_memory_bytes": 110_000_000 - model_index * 1_000_000,
                "resource_summary": {
                    "realized_token_counts.final": {
                        "minimum": 50,
                        "mean": 100 - model_index * 20,
                        "maximum": 197,
                    }
                },
            },
        )


def test_summarize_full_results_audits_and_preserves_axis_detail(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "outputs" / "paper1_matrix_v2"
    output = root / "summary" / "full"
    manifests = {}
    for dataset, _protocol in DATASETS:
        path = tmp_path / f"{dataset}.csv"
        paths = (
            ["images/a.jpg", "images/a.jpg", "images/b.jpg"]
            if dataset.startswith("coco_search18")
            else ["images/a.jpg", "images/b.jpg", "images/c.jpg"]
        )
        _write_csv(
            path,
            [
                {"image_id": f"id-{index}", "image_path": value}
                for index, value in enumerate(paths)
            ],
        )
        manifests[dataset] = path
    monkeypatch.setattr(summary_module, "BEHAVIOR_MANIFESTS", manifests)
    monkeypatch.setattr(summary_module, "EXPECTED_BEHAVIOR_ITEMS", 3)
    monkeypatch.setattr(
        summary_module,
        "_dinov2_provenance_valid",
        lambda: True,
    )
    monkeypatch.setattr(
        summary_module,
        "_artifact_preprocessing_valid",
        lambda _root: True,
    )
    _build_neural_outputs(root)
    _build_behavior_outputs(root, manifests)
    _build_efficiency_outputs(root)

    paths = summarize_full_results(root, output)

    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    behavior = _read_csv(paths["behavior"])
    quadrants = _read_csv(paths["cross_axis_quadrants"])
    static_rows = [row for row in behavior if row["model"] == "deit_small_static"]

    assert audit["passed"] is True
    assert len(behavior) == 9
    assert len(quadrants) == 72
    assert {row["behavior_scope"] for row in quadrants} == {
        "free_viewing",
        "task_search",
    }
    assert {row["neural_axis"] for row in quadrants} == {
        "encoding",
        "linear_cka",
        "subset_rsa_2048",
    }
    assert all(float(row["shuffled_auc"]) == 0.5 for row in static_rows)
    assert all(float(row["raw_shuffled_auc"]) == 1.0 for row in static_rows)
    assert all(row["constant_map_policy"] for row in static_rows)
    assert all(row["evidence_status"] == "diagnostic_only" for row in quadrants)
    key_audit = _read_csv(paths["behavior_map_key_audit"])
    assert len(key_audit) == 9
    assert all(row["passed"] == "True" for row in key_audit)
    assert all(row["missing_map_file_count"] == "0" for row in key_audit)
    assert all(row["unexpected_map_file_count"] == "0" for row in key_audit)
    assert set(BEHAVIOR_METRICS) <= set(behavior[0])
