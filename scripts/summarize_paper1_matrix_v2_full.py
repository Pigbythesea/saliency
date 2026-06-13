"""Audit and summarize the completed Paper 1 Matrix V2 full run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = [
    "deit_small_static",
    "dynamicvit_deit_small_keep_0_7",
    "tome_deit_small_r13",
]
ROIS = ["V1", "ventral", "lateral", "parietal"]
BEHAVIOR_METRICS = [
    "nss",
    "shuffled_auc",
    "auc_borji",
    "auc_judd",
    "cc",
    "similarity",
    "kl",
]
LOWER_IS_BETTER = {"kl"}
EXPECTED_COUNTS = {
    "neural_encoding": 12,
    "geometry": 48,
    "behavior": 9,
    "efficiency": 3,
}


def summarize_full_results(
    input_root: str | Path = "outputs/paper1_matrix_v2",
    output_dir: str | Path = "outputs/paper1_matrix_v2/summary/full",
) -> dict[str, Path]:
    root = Path(input_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    neural_rows = _load_neural_rows(root)
    geometry_rows = _load_geometry_rows(root)
    behavior_rows = _load_behavior_rows(root)
    efficiency_rows = _load_efficiency_rows(root)
    model_rows, quadrant_rows = _build_model_summaries(
        neural_rows,
        geometry_rows,
        behavior_rows,
        efficiency_rows,
    )
    audit = _build_audit(
        neural_rows,
        geometry_rows,
        behavior_rows,
        efficiency_rows,
    )

    paths = {
        "neural_encoding": output / "full_neural_encoding.csv",
        "geometry": output / "full_geometry.csv",
        "behavior": output / "full_behavior.csv",
        "efficiency": output / "matrix_v2_efficiency.csv",
        "model_summary": output / "full_model_summary.csv",
        "cross_axis_quadrants": output / "matrix_v2_cross_axis_quadrants.csv",
        "audit": output / "full_result_audit.json",
        "summary": output / "README.md",
    }
    _write_csv(paths["neural_encoding"], neural_rows)
    _write_csv(paths["geometry"], geometry_rows)
    _write_csv(paths["behavior"], behavior_rows)
    _write_csv(paths["efficiency"], efficiency_rows)
    _write_csv(paths["model_summary"], model_rows)
    _write_csv(paths["cross_axis_quadrants"], quadrant_rows)
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(
        _summary_markdown(model_rows, behavior_rows, audit),
        encoding="utf-8",
    )
    if not audit["passed"]:
        failed = [check["name"] for check in audit["checks"] if not check["passed"]]
        raise ValueError(f"Matrix V2 full-result audit failed: {', '.join(failed)}")
    return paths


def _load_neural_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "neural" / "full").glob("*/*/encoding_scores.csv")):
        records = _read_csv(path)
        if len(records) != 1:
            raise ValueError(f"Expected one selected encoding row in {path}")
        record = records[0]
        metadata = _read_json(path.parent / "metadata.json")
        selection = _read_json(path.parent / "selection_artifact.json")
        row = {
            "model": str(record["model"]),
            "roi": str(record["roi"]),
            "subject_id": str(record["subject_id"]),
            "layer": str(record["layer"]),
            "metric": str(record["metric"]),
            "metric_scope": str(record["metric_scope"]),
            "n_train": _as_int(record["n_train"]),
            "n_test": _as_int(record["n_test"]),
            "num_items": _as_int(metadata["num_items"]),
            "num_targets": _as_int(record["num_targets"]),
            "mean_score": _as_float(record["mean_score"]),
            "median_score": _as_float(record["median_score"]),
            "std_score": _as_float(record["std_score"]),
            "mean_noise_normalized_score": _as_float(
                record["mean_noise_normalized_score"]
            ),
            "median_noise_normalized_score": _as_float(
                record["median_noise_normalized_score"]
            ),
            "selected_ridge_alpha": _as_float(record["selected_ridge_alpha"]),
            "selection_score": _as_float(metadata["selection_score"]),
            "selection_primary_score": str(selection["primary_score"]),
            "selection_layer_matches": str(metadata["selected_layer"]) == record["layer"],
            "selection_alpha_matches": math.isclose(
                _as_float(metadata["selected_ridge_alpha"]),
                _as_float(record["selected_ridge_alpha"]),
            ),
            "source": path.as_posix(),
        }
        rows.append(row)
    return rows


def _load_geometry_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "neural" / "full").glob("*/*/geometry_scores.csv")):
        for record in _read_csv(path):
            rows.append(
                {
                    "model": str(record["model"]),
                    "roi": str(record["roi"]),
                    "subject_id": str(record["subject_id"]),
                    "layer": str(record["layer"]),
                    "geometry_method": str(record["geometry_method"]),
                    "score": _as_float(record["score"]),
                    "valid": _as_bool(record["valid"]),
                    "status": str(record["status"]),
                    "num_images_total": _as_int(record["num_images_total"]),
                    "num_images_used": _as_int(record["num_images_used"]),
                    "subset_seed": record["subset_seed"],
                    "subset_size": record["subset_size"],
                    "source": path.as_posix(),
                }
            )
    return rows


def _load_behavior_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = root / "behavior" / "full"
    for path in sorted(pattern.glob("*/*/aggregate_metrics.json")):
        aggregate = _read_json(path)
        metrics = aggregate["metrics"]
        raw = {metric: _as_float(metrics[metric]) for metric in BEHAVIOR_METRICS}
        per_image_path = path.parent / "per_image_metrics.csv"
        correction = _uniform_map_auc_correction(per_image_path)
        corrected = dict(raw)
        if correction:
            corrected.update(
                {"shuffled_auc": 0.5, "auc_borji": 0.5, "auc_judd": 0.5}
            )
        row: dict[str, Any] = {
            "dataset": str(aggregate["dataset"]),
            "behavior_scope": (
                "task_search"
                if str(aggregate["dataset"]).startswith("coco_search18")
                else "free_viewing"
            ),
            "model": str(aggregate["model"]),
            "num_items": _as_int(aggregate["num_items"]),
            "fixation_protocol": str(aggregate["fixation_protocol"]),
            "saliency_method": str(aggregate["saliency_method"]),
            "auc_tie_correction_applied": correction,
            "auc_tie_correction_reason": (
                "verified uniform-map signature; tied ROC scores equal chance"
                if correction
                else ""
            ),
        }
        for metric in BEHAVIOR_METRICS:
            row[metric] = corrected[metric]
            row[f"raw_{metric}"] = raw[metric]
        row["source"] = path.as_posix()
        rows.append(row)
    return rows


def _uniform_map_auc_correction(path: Path) -> bool:
    rows = _read_csv(path)
    if not rows:
        return False
    return all(
        math.isclose(_as_float(row["nss"]), 0.0, abs_tol=1e-12)
        and math.isclose(_as_float(row["cc"]), 0.0, abs_tol=1e-12)
        and math.isclose(_as_float(row["shuffled_auc"]), 1.0, abs_tol=1e-12)
        and math.isclose(_as_float(row["auc_borji"]), 1.0, abs_tol=1e-12)
        for row in rows
    )


def _load_efficiency_rows(root: Path) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for path in sorted((root / "external_artifacts" / "full").glob("*/efficiency.json")):
        payload = _read_json(path)
        resource_summary = payload.get("resource_summary", {})
        token_counts = [
            _as_float(value["mean"])
            for key, value in resource_summary.items()
            if key.startswith("realized_token_counts.") and "mean" in value
        ]
        raw_rows.append(
            {
                "model": path.parent.name,
                "parameters": _as_int(payload["parameters"]),
                "theoretical_flops": _as_int(payload["theoretical_flops"]),
                "realized_flops": _as_int(payload["realized_flops"]),
                "realized_gflops": _as_float(payload["realized_flops"]) / 1e9,
                "latency_ms_per_image": _as_float(payload["latency_ms_per_image"]),
                "peak_memory_bytes": _as_int(payload["peak_memory_bytes"]),
                "peak_memory_mib": _as_float(payload["peak_memory_bytes"]) / 2**20,
                "minimum_mean_tokens": min(token_counts) if token_counts else "",
                "maximum_mean_tokens": max(token_counts) if token_counts else "",
                "source": path.as_posix(),
            }
        )
    baseline = next(
        (row for row in raw_rows if row["model"] == "deit_small_static"),
        None,
    )
    if baseline is None:
        return raw_rows
    for row in raw_rows:
        row["realized_flops_reduction_vs_static"] = 1.0 - (
            row["realized_flops"] / baseline["realized_flops"]
        )
        row["latency_change_vs_static"] = (
            row["latency_ms_per_image"] / baseline["latency_ms_per_image"] - 1.0
        )
        row["peak_memory_reduction_vs_static"] = 1.0 - (
            row["peak_memory_bytes"] / baseline["peak_memory_bytes"]
        )
    return raw_rows


def _build_model_summaries(
    neural_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    behavior_rank = _rank_composite(
        behavior_rows,
        group_fields=("dataset",),
        value_fields=tuple(BEHAVIOR_METRICS),
    )
    neural_rank = _rank_composite(
        neural_rows,
        group_fields=("roi",),
        value_fields=("mean_noise_normalized_score",),
    )
    geometry_primary = [
        row
        for row in geometry_rows
        if row["geometry_method"] == "linear_cka"
        or (
            row["geometry_method"] == "subset_rsa"
            and _as_int(row["subset_size"]) == 2048
        )
    ]
    geometry_rank = _rank_composite(
        geometry_primary,
        group_fields=("roi", "geometry_method"),
        value_fields=("score",),
    )
    efficiency_by_model = {row["model"]: row for row in efficiency_rows}
    model_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_neural = [row for row in neural_rows if row["model"] == model]
        model_geometry = [row for row in geometry_primary if row["model"] == model]
        cka = [
            row["score"]
            for row in model_geometry
            if row["geometry_method"] == "linear_cka"
        ]
        rsa = [
            row["score"]
            for row in model_geometry
            if row["geometry_method"] == "subset_rsa"
        ]
        neural_geometry_rank = statistics.mean(
            [neural_rank[model], geometry_rank[model]]
        )
        efficiency = efficiency_by_model[model]
        model_rows.append(
            {
                "model": model,
                "behavior_rank_score": behavior_rank[model],
                "encoding_rank_score": neural_rank[model],
                "geometry_rank_score": geometry_rank[model],
                "neural_geometry_rank_score": neural_geometry_rank,
                "mean_encoding_score": statistics.mean(
                    row["mean_score"] for row in model_neural
                ),
                "mean_noise_normalized_encoding_score": statistics.mean(
                    row["mean_noise_normalized_score"] for row in model_neural
                ),
                "mean_linear_cka": statistics.mean(cka),
                "mean_subset_rsa_2048": statistics.mean(rsa),
                "realized_gflops": efficiency["realized_gflops"],
                "latency_ms_per_image": efficiency["latency_ms_per_image"],
                "behavior_rank_per_gflop": behavior_rank[model]
                / efficiency["realized_gflops"],
                "neural_geometry_rank_per_gflop": neural_geometry_rank
                / efficiency["realized_gflops"],
                "model_n": len(MODELS),
                "subject_n": 1,
            }
        )

    behavior_threshold = statistics.median(
        row["behavior_rank_score"] for row in model_rows
    )
    neural_threshold = statistics.median(
        row["neural_geometry_rank_score"] for row in model_rows
    )
    quadrant_rows: list[dict[str, Any]] = []
    for row in model_rows:
        row["aggregate_behavior_alignment"] = (
            "high"
            if row["behavior_rank_score"] >= behavior_threshold
            else "low"
        )
        row["aggregate_neural_geometry_alignment"] = (
            "high"
            if row["neural_geometry_rank_score"] >= neural_threshold
            else "low"
        )
        row["aggregate_quadrant"] = (
            f"{row['aggregate_behavior_alignment']}/"
            f"{row['aggregate_neural_geometry_alignment']}"
        )

    behavior_scope_ranks = {
        scope: _rank_composite(
            [row for row in behavior_rows if row["behavior_scope"] == scope],
            group_fields=("dataset",),
            value_fields=tuple(BEHAVIOR_METRICS),
        )
        for scope in ("free_viewing", "task_search")
    }
    neural_axis_rows = {
        "encoding": [
            {
                "model": row["model"],
                "roi": row["roi"],
                "axis_score": row["mean_noise_normalized_score"],
            }
            for row in neural_rows
        ],
        "linear_cka": [
            {
                "model": row["model"],
                "roi": row["roi"],
                "axis_score": row["score"],
            }
            for row in geometry_primary
            if row["geometry_method"] == "linear_cka"
        ],
        "subset_rsa_2048": [
            {
                "model": row["model"],
                "roi": row["roi"],
                "axis_score": row["score"],
            }
            for row in geometry_primary
            if row["geometry_method"] == "subset_rsa"
        ],
    }
    for behavior_scope, scope_ranks in behavior_scope_ranks.items():
        for neural_axis, axis_rows in neural_axis_rows.items():
            axis_ranks = _observation_ranks(
                axis_rows,
                group_fields=("roi",),
                value_field="axis_score",
            )
            values = {
                (row["model"], row["roi"]): row["axis_score"] for row in axis_rows
            }
            for roi in ROIS:
                for model in MODELS:
                    behavior_score = scope_ranks[model]
                    neural_score = axis_ranks[(roi, model)]
                    behavior_level = "high" if behavior_score >= 0.5 else "low"
                    neural_level = "high" if neural_score >= 0.5 else "low"
                    quadrant_rows.append(
                        {
                            "model": model,
                            "behavior_scope": behavior_scope,
                            "neural_axis": neural_axis,
                            "roi": roi,
                            "behavior_rank_score": behavior_score,
                            "neural_axis_value": values[(model, roi)],
                            "neural_axis_rank_score": neural_score,
                            "behavior_alignment": behavior_level,
                            "neural_alignment": neural_level,
                            "quadrant": f"{behavior_level}/{neural_level}",
                            "classification_method": (
                                "descriptive_panel_median_rank_n3"
                            ),
                            "model_n": len(MODELS),
                            "subject_n": 1,
                            "uncertainty": (
                                "not estimated; one subject and three-model "
                                "deterministic panel"
                            ),
                        }
                    )
    return model_rows, quadrant_rows


def _rank_composite(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> dict[str, float]:
    observations: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        for value_field in value_fields:
            key = tuple(row[field] for field in group_fields) + (value_field,)
            observations[key][str(row["model"])] = _as_float(row[value_field])

    scores: dict[str, list[float]] = defaultdict(list)
    for key, values in observations.items():
        if set(values) != set(MODELS):
            raise ValueError(f"Incomplete model panel for rank observation {key}")
        metric = str(key[-1])
        lower_is_better = metric in LOWER_IS_BETTER
        for model, value in values.items():
            others = [other for name, other in values.items() if name != model]
            better_than = sum(
                value < other if lower_is_better else value > other
                for other in others
            )
            tied_with = sum(math.isclose(value, other) for other in others)
            scores[model].append(
                (better_than + 0.5 * tied_with) / max(len(others), 1)
            )
    return {model: statistics.mean(scores[model]) for model in MODELS}


def _observation_ranks(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    value_field: str,
) -> dict[tuple[Any, ...], float]:
    observations: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        observations[key][str(row["model"])] = _as_float(row[value_field])

    output: dict[tuple[Any, ...], float] = {}
    for key, values in observations.items():
        if set(values) != set(MODELS):
            raise ValueError(f"Incomplete model panel for rank observation {key}")
        for model, value in values.items():
            others = [other for name, other in values.items() if name != model]
            better_than = sum(value > other for other in others)
            tied_with = sum(math.isclose(value, other) for other in others)
            output[(*key, model)] = (
                better_than + 0.5 * tied_with
            ) / max(len(others), 1)
    return output


def _build_audit(
    neural_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = [
        _check("neural_encoding_count", len(neural_rows) == 12, len(neural_rows)),
        _check("geometry_count", len(geometry_rows) == 48, len(geometry_rows)),
        _check("behavior_count", len(behavior_rows) == 9, len(behavior_rows)),
        _check("efficiency_count", len(efficiency_rows) == 3, len(efficiency_rows)),
        _check(
            "expected_model_roi_cells",
            {(row["model"], row["roi"]) for row in neural_rows}
            == {(model, roi) for model in MODELS for roi in ROIS},
            "3 models x 4 ROIs",
        ),
        _check(
            "neural_item_counts",
            all(row["num_items"] == 9841 for row in neural_rows),
            sorted({row["num_items"] for row in neural_rows}),
        ),
        _check(
            "behavior_item_counts",
            all(row["num_items"] == 2000 for row in behavior_rows),
            sorted({row["num_items"] for row in behavior_rows}),
        ),
        _check(
            "selection_consistency",
            all(
                row["selection_layer_matches"] and row["selection_alpha_matches"]
                for row in neural_rows
            ),
            "selected layer and alpha match metadata",
        ),
        _check(
            "geometry_valid",
            all(
                row["valid"]
                and row["status"] == "ok"
                and row["num_images_total"] == 9841
                for row in geometry_rows
            ),
            "all geometry rows valid over 9841 images",
        ),
        _check(
            "all_numeric_results_finite",
            _all_finite(neural_rows, geometry_rows, behavior_rows, efficiency_rows),
            "finite",
        ),
        _check(
            "uniform_baseline_auc_corrected",
            sum(row["auc_tie_correction_applied"] for row in behavior_rows) == 3,
            sum(row["auc_tie_correction_applied"] for row in behavior_rows),
        ),
    ]
    return {
        "schema_version": "hma.paper1_matrix_v2.full_result_audit.v1",
        "passed": all(check["passed"] for check in checks),
        "expected_counts": EXPECTED_COUNTS,
        "checks": checks,
        "notes": [
            (
                "Raw aggregate AUC values are preserved in raw_* columns. "
                "The three verified uniform static-routing cells use tie-aware "
                "chance AUC=0.5 in analysis columns."
            ),
            (
                "Cross-axis quadrants are descriptive within this three-model, "
                "single-subject panel and are not inferential population claims."
            ),
        ],
    }


def _all_finite(*tables: list[dict[str, Any]]) -> bool:
    for table in tables:
        for row in table:
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    return False
    return True


def _summary_markdown(
    model_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    by_model = {row["model"]: row for row in model_rows}
    static = by_model["deit_small_static"]
    dynamic = by_model["dynamicvit_deit_small_keep_0_7"]
    tome = by_model["tome_deit_small_r13"]
    corrections = sum(row["auc_tie_correction_applied"] for row in behavior_rows)
    return f"""# Paper 1 Matrix V2 Full-Run Summary

Audit status: **{"passed" if audit["passed"] else "failed"}**

- Complete panel: 3 models, 4 neural ROIs, 3 behavioral datasets.
- Aggregate encoding rank: static {static["encoding_rank_score"]:.3f},
  DynamicViT {dynamic["encoding_rank_score"]:.3f}, ToMe
  {tome["encoding_rank_score"]:.3f}.
- Aggregate geometry rank: static {static["geometry_rank_score"]:.3f},
  DynamicViT {dynamic["geometry_rank_score"]:.3f}, ToMe
  {tome["geometry_rank_score"]:.3f}.
- Aggregate behavioral rank: static {static["behavior_rank_score"]:.3f},
  DynamicViT {dynamic["behavior_rank_score"]:.3f}, ToMe
  {tome["behavior_rank_score"]:.3f}.
- DynamicViT realized FLOPs reduction versus static: {_percent(dynamic, static, "realized_gflops")}.
- ToMe realized FLOPs reduction versus static: {_percent(tome, static, "realized_gflops")}.
- DynamicViT measured latency change versus static: {_percent_change(dynamic, static, "latency_ms_per_image")}.
- ToMe measured latency change versus static: {_percent_change(tome, static, "latency_ms_per_image")}.
- Static uniform-routing AUC corrections: {corrections} dataset cells. Raw values remain in `full_behavior.csv`.

The cross-axis classifications are descriptive panel-median splits for a
three-model, single-subject result. They should be used to organize follow-up
analysis, not as inferential evidence by themselves.
"""


def _percent(row: dict[str, Any], baseline: dict[str, Any], field: str) -> str:
    reduction = 1.0 - _as_float(row[field]) / _as_float(baseline[field])
    return f"{100.0 * reduction:.2f}%"


def _percent_change(
    row: dict[str, Any],
    baseline: dict[str, Any],
    field: str,
) -> str:
    change = _as_float(row[field]) / _as_float(baseline[field]) - 1.0
    return f"{100.0 * change:+.2f}%"


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _print_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/paper1_matrix_v2")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper1_matrix_v2/summary/full",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _print_paths(summarize_full_results(args.input_root, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
