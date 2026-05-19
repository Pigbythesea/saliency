import csv

from hma.experiments.summarize_results import (
    metric_higher_is_better,
    summarize_aggregate_results,
)


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_results_writes_expected_tables(tmp_path):
    aggregate_csv = tmp_path / "results.csv"
    rows = [
        {
            "dataset": "salicon",
            "model": "center_bias_baseline",
            "saliency_method": "center_bias",
            "saliency_family": "baseline",
            "metric": "nss",
            "n": 2,
            "mean": 0.5,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.5,
            "ci95_high": 0.5,
        },
        {
            "dataset": "salicon",
            "model": "resnet50",
            "saliency_method": "vanilla_gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "nss",
            "n": 2,
            "mean": 0.7,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.7,
            "ci95_high": 0.7,
        },
        {
            "dataset": "salicon",
            "model": "resnet50",
            "saliency_method": "occlusion",
            "saliency_family": "perturbation",
            "metric": "nss",
            "n": 2,
            "mean": 0.6,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.6,
            "ci95_high": 0.6,
        },
        {
            "dataset": "salicon",
            "model": "deepgaze_reference",
            "saliency_method": "deepgaze_precomputed",
            "saliency_family": "reference",
            "metric": "nss",
            "n": 2,
            "mean": 0.8,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.8,
            "ci95_high": 0.8,
        },
    ]
    _write_csv(aggregate_csv, rows, list(rows[0]))

    efficiency_csv = tmp_path / "efficiency.csv"
    _write_csv(
        efficiency_csv,
        [{"model_name": "resnet50", "latency_mean_ms": 10.0, "parameter_count": 100}],
        ["model_name", "latency_mean_ms", "parameter_count"],
    )

    outputs = summarize_aggregate_results(
        aggregate_csv,
        tmp_path / "summary",
        efficiency_csv=efficiency_csv,
    )

    assert outputs["top_rows"].is_file()
    assert outputs["best_non_baseline"].is_file()
    assert outputs["center_bias_deltas"].is_file()
    assert outputs["family_rankings"].is_file()
    assert outputs["key_comparisons"].is_file()
    assert outputs["pilot_static_stability"].is_file()
    assert outputs["interpretation_note"].is_file()
    assert outputs["alignment_per_efficiency"].is_file()

    with outputs["family_rankings"].open("r", encoding="utf-8", newline="") as handle:
        family_rows = list(csv.DictReader(handle))
    assert {row["saliency_family"] for row in family_rows} >= {
        "baseline",
        "evidence_sensitivity",
        "perturbation",
        "reference",
    }


def test_summarize_results_writes_pilot_static_stability(tmp_path):
    aggregate_csv = tmp_path / "results.csv"
    rows = []
    for dataset, mean in [
        ("salicon_pilot500", 0.4),
        ("salicon_static2000", 0.5),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "model": "resnet50",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 10,
                "mean": mean,
                "std": 0.1,
                "stderr": 0.01,
                "ci95_low": mean - 0.02,
                "ci95_high": mean + 0.02,
            }
        )
    _write_csv(aggregate_csv, rows, list(rows[0]))

    outputs = summarize_aggregate_results(aggregate_csv, tmp_path / "summary")

    with outputs["pilot_static_stability"].open("r", encoding="utf-8", newline="") as handle:
        stability_rows = list(csv.DictReader(handle))
    assert stability_rows[0]["dataset_base"] == "salicon"
    assert stability_rows[0]["saliency_method"] == "gradcam"
    assert float(stability_rows[0]["mean_delta_static_minus_pilot"]) > 0


def test_summary_metric_direction_handles_lower_is_better():
    assert metric_higher_is_better("auc_borji")
    assert not metric_higher_is_better("kl")
    assert not metric_higher_is_better("emd")
