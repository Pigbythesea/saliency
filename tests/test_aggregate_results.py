import csv
import json
import math

import pytest

from hma.experiments.aggregate_results import (
    aggregate_result_files,
    find_per_image_csvs,
    save_aggregate_table,
)


def _write_result_dir(path, model, nss_values, cc_values):
    path.mkdir()
    with (path / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "nss", "cc"])
        writer.writeheader()
        for index, (nss, cc) in enumerate(zip(nss_values, cc_values)):
            writer.writerow(
                {
                    "image_id": f"img_{index}",
                    "image_path": f"images/img_{index}.png",
                    "nss": nss,
                    "cc": cc,
                }
            )
    (path / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "experiment": path.name,
                "dataset": "dummy_static_saliency",
                "model": model,
                "saliency_method": "dummy_gradient_free",
                "saliency_family": "unknown",
            }
        ),
        encoding="utf-8",
    )


def test_find_per_image_csvs_accepts_direct_files_and_directories(tmp_path):
    result_dir = tmp_path / "run_a"
    _write_result_dir(result_dir, "model_a", [1.0], [0.1])

    found = find_per_image_csvs([tmp_path, result_dir / "per_image_metrics.csv"])

    assert found == [(result_dir / "per_image_metrics.csv").resolve()]


def test_aggregate_result_files_computes_expected_columns_and_stats(tmp_path):
    _write_result_dir(tmp_path / "run_a", "model_a", [1.0, 3.0], [0.2, 0.4])
    _write_result_dir(tmp_path / "run_b", "model_b", [2.0, 4.0], [0.5, 0.7])

    rows = aggregate_result_files([tmp_path])
    required_columns = {
        "dataset",
        "model",
        "saliency_method",
        "saliency_family",
        "metric",
        "n",
        "mean",
        "std",
        "stderr",
        "ci95_low",
        "ci95_high",
    }

    assert rows
    assert required_columns.issubset(rows[0])

    model_a_nss = next(row for row in rows if row["model"] == "model_a" and row["metric"] == "nss")
    assert model_a_nss["dataset"] == "dummy_static_saliency"
    assert model_a_nss["saliency_method"] == "dummy_gradient_free"
    assert model_a_nss["saliency_family"] == "unknown"
    assert model_a_nss["n"] == 2
    assert model_a_nss["mean"] == pytest.approx(2.0)
    assert model_a_nss["std"] == pytest.approx(math.sqrt(2.0))
    assert model_a_nss["stderr"] == pytest.approx(1.0)
    assert model_a_nss["ci95_low"] == pytest.approx(0.04)
    assert model_a_nss["ci95_high"] == pytest.approx(3.96)


def test_save_aggregate_table_writes_csv(tmp_path):
    rows = [
        {
            "dataset": "d",
            "model": "m",
            "saliency_method": "s",
            "saliency_family": "baseline",
            "metric": "nss",
            "n": 1,
            "mean": 1.0,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 1.0,
            "ci95_high": 1.0,
        }
    ]

    output = save_aggregate_table(rows, tmp_path / "nested" / "results.csv")

    assert output.is_file()
    with output.open("r", encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["metric"] == "nss"
