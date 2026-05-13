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
    assert outputs["alignment_per_efficiency"].is_file()


def test_summary_metric_direction_handles_lower_is_better():
    assert metric_higher_is_better("auc_borji")
    assert not metric_higher_is_better("kl")
    assert not metric_higher_is_better("emd")
