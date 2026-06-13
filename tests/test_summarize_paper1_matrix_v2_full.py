import csv
import json

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
            geometry_rows = []
            for method, subset_size in [
                ("linear_cka", ""),
                ("subset_rsa", 512),
                ("subset_rsa", 1024),
                ("subset_rsa", 2048),
            ]:
                geometry_rows.append(
                    {
                        "model": model,
                        "roi": roi,
                        "subject_id": "subj01",
                        "layer": layer,
                        "geometry_method": method,
                        "score": score + (0.01 if method == "subset_rsa" else 0.0),
                        "valid": "true",
                        "status": "ok",
                        "num_images_total": 9841,
                        "num_images_used": 9841 if method == "linear_cka" else subset_size,
                        "subset_seed": "" if method == "linear_cka" else 123,
                        "subset_size": subset_size,
                    }
                )
            _write_csv(output / "geometry_scores.csv", geometry_rows)


def _build_behavior_outputs(root):
    for dataset, protocol in DATASETS:
        for model_index, model in enumerate(MODELS):
            output = root / "behavior" / "full" / dataset / f"{model}_routing"
            if model == "deit_small_static":
                metrics = {
                    "nss": 0.0,
                    "shuffled_auc": 1.0,
                    "auc_borji": 1.0,
                    "auc_judd": 0.51,
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
                    "num_items": 2000,
                    "fixation_protocol": protocol,
                    "saliency_method": "precomputed_map",
                    "metrics": metrics,
                },
            )
            _write_csv(
                output / "per_image_metrics.csv",
                [{"image_id": "a", **metrics}],
            )


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


def test_summarize_full_results_audits_and_preserves_axis_detail(tmp_path):
    root = tmp_path / "outputs" / "paper1_matrix_v2"
    output = root / "summary" / "full"
    _build_neural_outputs(root)
    _build_behavior_outputs(root)
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
    assert all(
        row["auc_tie_correction_applied"] == "True" for row in static_rows
    )
    assert set(BEHAVIOR_METRICS) <= set(behavior[0])
