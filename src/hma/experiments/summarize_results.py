"""Summary tables for aggregated benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from hma.experiments.aggregate_results import load_aggregate_table
from hma.utils.paths import ensure_dir


LOWER_IS_BETTER_METRICS = {"kl", "kl_divergence", "emd", "emd_2d", "mae"}
EFFICIENCY_FIELDS = [
    "latency_mean_ms",
    "parameter_count",
    "model_size_mb",
    "flops",
]


def metric_higher_is_better(metric: str) -> bool:
    """Return whether larger values should rank higher for a metric."""
    return str(metric) not in LOWER_IS_BETTER_METRICS


def summarize_aggregate_results(
    aggregate_csv: str | Path,
    output_dir: str | Path,
    *,
    efficiency_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Write compact V2 result summaries and return output paths."""
    rows = load_aggregate_table(aggregate_csv)
    output = ensure_dir(output_dir)
    efficiency_rows = _load_csv_rows(efficiency_csv) if efficiency_csv else []

    outputs = {
        "top_rows": output / "top_rows_by_dataset_metric_family.csv",
        "best_non_baseline": output / "best_non_baseline_by_dataset_metric.csv",
        "center_bias_deltas": output / "center_bias_deltas.csv",
        "family_rankings": output / "family_rankings.csv",
    }
    _write_rows(outputs["top_rows"], _top_rows(rows, ["dataset", "metric", "saliency_family"]))
    _write_rows(
        outputs["best_non_baseline"],
        _top_rows(
            [
                row
                for row in rows
                if str(row.get("saliency_family", "")) != "baseline"
            ],
            ["dataset", "metric"],
        ),
    )
    _write_rows(outputs["center_bias_deltas"], _center_bias_deltas(rows))
    _write_rows(outputs["family_rankings"], _family_rankings(rows))

    if efficiency_rows:
        outputs["alignment_per_efficiency"] = output / "alignment_per_efficiency.csv"
        _write_rows(
            outputs["alignment_per_efficiency"],
            _alignment_per_efficiency(rows, efficiency_rows),
        )
    return outputs


def _top_rows(rows: Iterable[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(key, "unknown")) for key in group_keys), []).append(row)

    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        metric = str(candidates[0].get("metric", ""))
        best = sorted(
            candidates,
            key=lambda row: _float(row.get("mean")),
            reverse=metric_higher_is_better(metric),
        )[0]
        selected.append({**{group_keys[i]: key[i] for i in range(len(group_keys))}, **best})
    return selected


def _center_bias_deltas(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    row_list = list(rows)
    center_by_key = {}
    for row in row_list:
        if str(row.get("saliency_method")) == "center_bias":
            center_by_key[(str(row.get("dataset")), str(row.get("metric")))] = _float(row.get("mean"))

    deltas = []
    for row in row_list:
        key = (str(row.get("dataset")), str(row.get("metric")))
        if key not in center_by_key or str(row.get("saliency_method")) == "center_bias":
            continue
        metric = str(row.get("metric"))
        mean = _float(row.get("mean"))
        center = center_by_key[key]
        delta = mean - center if metric_higher_is_better(metric) else center - mean
        deltas.append(
            {
                **row,
                "center_bias_mean": center,
                "delta_vs_center_bias": delta,
            }
        )
    return deltas


def _family_rankings(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "unknown")),
            str(row.get("metric", "unknown")),
            str(row.get("saliency_family", "unknown")),
        )
        grouped.setdefault(key, []).append(_float(row.get("mean")))
        counts[key] = counts.get(key, 0) + int(float(row.get("n", 0) or 0))

    output = []
    for dataset, metric, family in sorted(grouped):
        values = grouped[(dataset, metric, family)]
        output.append(
            {
                "dataset": dataset,
                "metric": metric,
                "saliency_family": family,
                "num_rows": len(values),
                "total_n": counts[(dataset, metric, family)],
                "family_mean": sum(values) / len(values),
            }
        )
    output.sort(
        key=lambda row: (
            row["dataset"],
            row["metric"],
            -row["family_mean"] if metric_higher_is_better(str(row["metric"])) else row["family_mean"],
        )
    )
    return output


def _alignment_per_efficiency(
    aggregate_rows: Iterable[dict[str, Any]],
    efficiency_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    efficiency_by_model = {
        str(row.get("model") or row.get("model_name")): row
        for row in efficiency_rows
        if row.get("model") or row.get("model_name")
    }
    output = []
    for row in aggregate_rows:
        efficiency = efficiency_by_model.get(str(row.get("model", "")))
        if efficiency is None:
            continue
        summary = dict(row)
        metric = str(row.get("metric", ""))
        mean = _float(row.get("mean"))
        for field in EFFICIENCY_FIELDS:
            value = _optional_float(efficiency.get(field))
            if value is None or value <= 0:
                continue
            if metric_higher_is_better(metric):
                summary[f"{metric}_per_{field}"] = mean / value
            else:
                summary[f"inverse_{metric}_per_{field}"] = 1.0 / ((mean + 1e-8) * value)
        output.append(summary)
    return output


def _load_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    fieldnames = sorted({key for row in row_list for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
