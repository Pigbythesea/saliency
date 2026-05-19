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
        "key_comparisons": output / "key_comparisons.csv",
        "pilot_static_stability": output / "pilot_static_stability.csv",
        "interpretation_note": output / "v2_interpretation_note.md",
    }
    top_rows = _top_rows(rows, ["dataset", "metric", "saliency_family"])
    best_non_baseline = _top_rows(
        [
            row
            for row in rows
            if str(row.get("saliency_family", "")) != "baseline"
        ],
        ["dataset", "metric"],
    )
    center_bias_deltas = _center_bias_deltas(rows)
    family_rankings = _family_rankings(rows)
    key_comparisons = _key_comparisons(rows)
    pilot_static_stability = _pilot_static_stability(rows)

    _write_rows(outputs["top_rows"], top_rows)
    _write_rows(
        outputs["best_non_baseline"],
        best_non_baseline,
    )
    _write_rows(outputs["center_bias_deltas"], center_bias_deltas)
    _write_rows(outputs["family_rankings"], family_rankings)
    _write_rows(outputs["key_comparisons"], key_comparisons)
    _write_rows(outputs["pilot_static_stability"], pilot_static_stability)
    _write_interpretation_note(
        outputs["interpretation_note"],
        rows=rows,
        best_non_baseline=best_non_baseline,
        center_bias_deltas=center_bias_deltas,
        family_rankings=family_rankings,
        key_comparisons=key_comparisons,
        pilot_static_stability=pilot_static_stability,
        has_efficiency=bool(efficiency_rows),
    )

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


def _key_comparisons(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    row_list = list(rows)
    output: list[dict[str, Any]] = []
    by_dataset_metric: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in row_list:
        by_dataset_metric.setdefault(
            (str(row.get("dataset", "unknown")), str(row.get("metric", "unknown"))),
            [],
        ).append(row)

    for (dataset, metric), candidates in sorted(by_dataset_metric.items()):
        center = _find_method(candidates, "center_bias")
        non_baseline = [
            row for row in candidates if str(row.get("saliency_family")) != "baseline"
        ]
        if center is not None and non_baseline:
            best = _best_row(non_baseline, metric)
            output.append(_comparison_row("center_bias_vs_best_non_baseline", center, best))

        for model in sorted({str(row.get("model")) for row in candidates}):
            model_rows = [row for row in candidates if str(row.get("model")) == model]
            gradcam = _find_method(model_rows, "gradcam")
            vanilla = _find_method(model_rows, "vanilla_gradient")
            if gradcam is not None and vanilla is not None:
                output.append(_comparison_row("gradcam_vs_vanilla_gradient", gradcam, vanilla))

        internal = [
            row for row in candidates if str(row.get("saliency_family")) == "internal_routing"
        ]
        evidence = [
            row for row in candidates if str(row.get("saliency_family")) == "evidence_sensitivity"
        ]
        if internal and evidence:
            output.append(
                _comparison_row(
                    "best_internal_routing_vs_best_evidence_sensitivity",
                    _best_row(internal, metric),
                    _best_row(evidence, metric),
                )
            )
    return output


def _pilot_static_stability(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    row_list = list(rows)
    ranked: dict[tuple[str, str, str], dict[tuple[str, str, str], dict[str, Any]]] = {}
    for dataset, metric, scale in sorted(
        {
            (_dataset_base(str(row.get("dataset", ""))), str(row.get("metric", "")), _dataset_scale(str(row.get("dataset", ""))))
            for row in row_list
            if _dataset_scale(str(row.get("dataset", ""))) in {"pilot500", "static2000"}
        }
    ):
        candidates = [
            row
            for row in row_list
            if _dataset_base(str(row.get("dataset", ""))) == dataset
            and str(row.get("metric", "")) == metric
            and _dataset_scale(str(row.get("dataset", ""))) == scale
        ]
        sorted_rows = sorted(
            candidates,
            key=lambda row: _float(row.get("mean")),
            reverse=metric_higher_is_better(metric),
        )
        for rank, row in enumerate(sorted_rows, start=1):
            key = (
                str(row.get("model", "")),
                str(row.get("saliency_method", "")),
                str(row.get("saliency_family", "")),
            )
            ranked.setdefault((dataset, metric, scale), {})[key] = {**row, "rank": rank}

    output: list[dict[str, Any]] = []
    bases = sorted({key[:2] for key in ranked})
    for dataset, metric in bases:
        pilot = ranked.get((dataset, metric, "pilot500"), {})
        static = ranked.get((dataset, metric, "static2000"), {})
        for key in sorted(set(pilot) & set(static)):
            pilot_row = pilot[key]
            static_row = static[key]
            output.append(
                {
                    "dataset_base": dataset,
                    "metric": metric,
                    "model": key[0],
                    "saliency_method": key[1],
                    "saliency_family": key[2],
                    "pilot_rank": pilot_row["rank"],
                    "static_rank": static_row["rank"],
                    "rank_delta_static_minus_pilot": int(static_row["rank"]) - int(pilot_row["rank"]),
                    "pilot_mean": pilot_row.get("mean"),
                    "static_mean": static_row.get("mean"),
                    "mean_delta_static_minus_pilot": _float(static_row.get("mean")) - _float(pilot_row.get("mean")),
                }
            )
    return output


def _comparison_row(label: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    metric = str(left.get("metric", ""))
    left_mean = _float(left.get("mean"))
    right_mean = _float(right.get("mean"))
    delta = (
        left_mean - right_mean
        if metric_higher_is_better(metric)
        else right_mean - left_mean
    )
    return {
        "comparison": label,
        "dataset": left.get("dataset"),
        "metric": metric,
        "left_model": left.get("model"),
        "left_saliency_method": left.get("saliency_method"),
        "left_saliency_family": left.get("saliency_family"),
        "left_mean": left_mean,
        "right_model": right.get("model"),
        "right_saliency_method": right.get("saliency_method"),
        "right_saliency_family": right.get("saliency_family"),
        "right_mean": right_mean,
        "delta_positive_favors_left": delta,
        "left_ci95_low": left.get("ci95_low"),
        "left_ci95_high": left.get("ci95_high"),
        "right_ci95_low": right.get("ci95_low"),
        "right_ci95_high": right.get("ci95_high"),
        "ci95_overlap": _ci95_overlap(left, right),
    }


def _find_method(rows: Iterable[dict[str, Any]], method: str) -> dict[str, Any] | None:
    return next((row for row in rows if str(row.get("saliency_method")) == method), None)


def _best_row(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: _float(row.get("mean")),
        reverse=metric_higher_is_better(metric),
    )[0]


def _ci95_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_low = _optional_float(left.get("ci95_low"))
    left_high = _optional_float(left.get("ci95_high"))
    right_low = _optional_float(right.get("ci95_low"))
    right_high = _optional_float(right.get("ci95_high"))
    if None in {left_low, left_high, right_low, right_high}:
        return False
    return bool(max(left_low, right_low) <= min(left_high, right_high))


def _dataset_base(dataset: str) -> str:
    for suffix in ("_pilot500", "_static2000"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def _dataset_scale(dataset: str) -> str:
    if dataset.endswith("_pilot500"):
        return "pilot500"
    if dataset.endswith("_static2000"):
        return "static2000"
    return "unknown"


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


def _write_interpretation_note(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    best_non_baseline: list[dict[str, Any]],
    center_bias_deltas: list[dict[str, Any]],
    family_rankings: list[dict[str, Any]],
    key_comparisons: list[dict[str, Any]],
    pilot_static_stability: list[dict[str, Any]],
    has_efficiency: bool,
) -> None:
    datasets = sorted({str(row.get("dataset", "unknown")) for row in rows})
    nss_winners = [
        row for row in best_non_baseline if str(row.get("metric")) == "nss"
    ]
    positive_center_deltas = [
        row
        for row in center_bias_deltas
        if str(row.get("metric")) == "nss" and _float(row.get("delta_vs_center_bias")) > 0
    ]
    family_nss = [
        row for row in family_rankings if str(row.get("metric")) == "nss"
    ]

    lines = [
        "# V2 Static Benchmark Interpretation Note",
        "",
        "This note is generated from the aggregate V2 saliency benchmark tables.",
        "Treat pilot-scale rows as reliability checks, not final scientific claims.",
        "",
        "## Scope",
        "",
        f"- Datasets summarized: {', '.join(datasets) if datasets else 'none'}.",
        "- Metrics use controlled directions: KL, EMD, and MAE are lower-is-better.",
        "- Saliency families remain separated to avoid mixing explanation methods.",
        "",
        "## Center Bias",
        "",
    ]
    if positive_center_deltas:
        lines.append(
            "- At least one non-baseline NSS row exceeds its center-bias baseline; inspect "
            "`center_bias_deltas.csv` before making architecture-level claims."
        )
    else:
        lines.append(
            "- No non-baseline NSS row exceeds the center-bias baseline in the summarized "
            "rows, or no center-bias comparison is available."
        )

    lines.extend(["", "## Saliency Families", ""])
    if family_nss:
        preview = sorted(
            family_nss,
            key=lambda row: (
                str(row.get("dataset", "")),
                -_float(row.get("family_mean")),
            ),
        )[:6]
        for row in preview:
            lines.append(
                "- "
                f"{row.get('dataset')}: {row.get('saliency_family')} "
                f"NSS family mean={_float(row.get('family_mean')):.4g}."
            )
    else:
        lines.append("- No NSS family-ranking rows are available.")

    lines.extend(["", "## Best Non-Baseline NSS Rows", ""])
    if nss_winners:
        for row in sorted(nss_winners, key=lambda item: str(item.get("dataset", ""))):
            lines.append(
                "- "
                f"{row.get('dataset')}: {row.get('model')} + "
                f"{row.get('saliency_method')} mean={_float(row.get('mean')):.4g}."
            )
    else:
        lines.append("- No non-baseline NSS rows are available.")

    lines.extend(["", "## Stability And Comparisons", ""])
    if key_comparisons:
        lines.append(
            "- Key comparison rows were generated for center-bias, Grad-CAM, "
            "gradient, attention-rollout, and evidence-sensitivity contrasts."
        )
    else:
        lines.append("- No key comparison rows are available.")
    if pilot_static_stability:
        lines.append(
            "- Pilot-to-static ranking stability rows were generated; use them "
            "to separate smoke-run artifacts from scaled benchmark patterns."
        )
    else:
        lines.append("- No pilot/static ranking stability rows are available.")

    lines.extend(["", "## Efficiency", ""])
    if has_efficiency:
        lines.append(
            "- Alignment-per-efficiency rows were generated; use them to compare alignment "
            "per latency, parameter count, and model size."
        )
    else:
        lines.append("- No efficiency CSV was provided for this summary run.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
