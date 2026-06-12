"""Create paper-facing attribution-family cross-axis interpretation tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

from hma.utils.paths import ensure_dir


DEFAULT_BEHAVIORAL_CSV = (
    "outputs/real_matrix_v2/aggregated/"
    "results_with_ssl_behavior_plus_transformer_relevance.csv"
)
DEFAULT_KEY_COMPARISONS = (
    "outputs/real_matrix_v2/aggregated/"
    "results_with_ssl_behavior_plus_transformer_relevance_summary/key_comparisons.csv"
)
DEFAULT_FAMILY_RANKINGS = (
    "outputs/real_matrix_v2/aggregated/"
    "results_with_ssl_behavior_plus_transformer_relevance_summary/family_rankings.csv"
)
DEFAULT_CROSS_LEVEL_CORRELATIONS = (
    "outputs/neural_roi_summary/matched_cross_level_correlations.csv"
)
DEFAULT_LEADER_OVERLAP = "outputs/neural_roi_summary/behavior_neural_leader_overlap.csv"
DEFAULT_OUTPUT_CSV = (
    "outputs/paper1_experiment_v1/summary/"
    "attribution_family_cross_axis_interpretation.csv"
)
DEFAULT_OUTPUT_MD = (
    "outputs/paper1_experiment_v1/summary/"
    "attribution_family_cross_axis_interpretation.md"
)

FIELDNAMES = [
    "dataset",
    "viewing_regime",
    "saliency_family",
    "included_methods",
    "behavior_metric_rows",
    "behavior_nss_rows",
    "behavior_total_n",
    "best_nss_method",
    "best_nss_model",
    "best_nss_mean",
    "best_nss_ci95_low",
    "best_nss_ci95_high",
    "leader_overlap_rows",
    "encoding_leader_match_count",
    "encoding_leader_match_rate",
    "geometry_or_rsa_leader_match_count",
    "geometry_or_rsa_leader_match_rate",
    "cross_level_rows",
    "cross_level_complete_rows",
    "cross_level_insufficient_rows",
    "cross_level_n_models_min",
    "cross_level_n_models_max",
    "mean_spearman_behavior_vs_encoding",
    "mean_spearman_behavior_vs_geometry",
    "transformer_relevance_vs_rollout_rows",
    "transformer_relevance_vs_rollout_all_metrics_better",
    "transformer_relevance_vs_rollout_nss_delta",
    "paper_interpretation",
]

REFERENCE_FAMILIES = {"reference", "baseline"}
TASK_SEARCH_FAMILIES = {"task_search_baseline"}
ATTRIBUTION_FAMILIES = {
    "class_localization",
    "evidence_sensitivity",
    "internal_routing",
    "perturbation",
    "transformer_relevance",
}
TRANSFORMER_RELEVANCE_FAMILY = "transformer_relevance"
ROLLOUT_FAMILY = "internal_routing"
TRANSFORMER_RELEVANCE_COMPARISON = "best_transformer_relevance_vs_best_internal_routing"
BANNED_TRANSFORMER_INTERPRETATION_TERMS = {
    "operational attention",
    "causal attention",
    "human-like attention",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-facing attribution-family interpretation tables."
    )
    parser.add_argument("--behavioral-csv", default=DEFAULT_BEHAVIORAL_CSV)
    parser.add_argument("--key-comparisons", default=DEFAULT_KEY_COMPARISONS)
    parser.add_argument("--family-rankings", default=DEFAULT_FAMILY_RANKINGS)
    parser.add_argument(
        "--cross-level-correlations", default=DEFAULT_CROSS_LEVEL_CORRELATIONS
    )
    parser.add_argument("--leader-overlap", default=DEFAULT_LEADER_OVERLAP)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = create_attribution_family_interpretation(
        behavioral_csv=args.behavioral_csv,
        key_comparisons=args.key_comparisons,
        family_rankings=args.family_rankings,
        cross_level_correlations=args.cross_level_correlations,
        leader_overlap=args.leader_overlap,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(
        "Attribution-family interpretation: "
        f"{Path(args.output_csv).resolve()} ({len(rows)} rows)"
    )


def create_attribution_family_interpretation(
    *,
    behavioral_csv: str | Path = DEFAULT_BEHAVIORAL_CSV,
    key_comparisons: str | Path = DEFAULT_KEY_COMPARISONS,
    family_rankings: str | Path = DEFAULT_FAMILY_RANKINGS,
    cross_level_correlations: str | Path = DEFAULT_CROSS_LEVEL_CORRELATIONS,
    leader_overlap: str | Path = DEFAULT_LEADER_OVERLAP,
    output_csv: str | Path | None = DEFAULT_OUTPUT_CSV,
    output_md: str | Path | None = DEFAULT_OUTPUT_MD,
) -> list[dict[str, Any]]:
    """Build and optionally write the attribution-family interpretation table."""
    rows = build_attribution_family_interpretation(
        behavioral_rows=_load_csv_rows(Path(behavioral_csv)),
        key_comparison_rows=_load_optional_csv_rows(Path(key_comparisons)),
        family_ranking_rows=_load_optional_csv_rows(Path(family_rankings)),
        cross_level_rows=_load_optional_csv_rows(Path(cross_level_correlations)),
        leader_overlap_rows=_load_optional_csv_rows(Path(leader_overlap)),
    )
    if output_csv is not None:
        _write_rows(Path(output_csv), rows, FIELDNAMES)
    if output_md is not None:
        _write_markdown_table(Path(output_md), rows)
    return rows


def build_attribution_family_interpretation(
    *,
    behavioral_rows: Iterable[dict[str, Any]],
    key_comparison_rows: Iterable[dict[str, Any]] = (),
    family_ranking_rows: Iterable[dict[str, Any]] = (),
    cross_level_rows: Iterable[dict[str, Any]] = (),
    leader_overlap_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return one compact row per dataset and saliency family."""
    behavioral = list(behavioral_rows)
    key_comparisons = list(key_comparison_rows)
    family_rankings = list(family_ranking_rows)
    cross_level = list(cross_level_rows)
    leader_overlap = list(leader_overlap_rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in behavioral:
        dataset = str(row.get("dataset", ""))
        family = str(row.get("saliency_family", ""))
        if dataset and family:
            grouped.setdefault((dataset, family), []).append(row)

    output: list[dict[str, Any]] = []
    for dataset, family in sorted(grouped):
        behavior_group = grouped[(dataset, family)]
        family_ranking_group = [
            row
            for row in family_rankings
            if row.get("dataset") == dataset and row.get("saliency_family") == family
        ]
        cross_group = [
            row
            for row in cross_level
            if row.get("behavior_dataset") == dataset
            and row.get("behavior_saliency_family") == family
        ]
        overlap_group = [
            row
            for row in leader_overlap
            if row.get("behavior_dataset") == dataset
            and row.get("behavior_saliency_family") == family
        ]
        comparison_group = [
            row
            for row in key_comparisons
            if row.get("dataset") == dataset
            and row.get("comparison") == TRANSFORMER_RELEVANCE_COMPARISON
        ]
        output.append(
            _interpretation_row(
                dataset=dataset,
                family=family,
                behavior_rows=behavior_group,
                family_ranking_rows=family_ranking_group,
                cross_level_rows=cross_group,
                leader_overlap_rows=overlap_group,
                comparison_rows=comparison_group,
            )
        )
    return output


def audit_attribution_family_interpretation(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return deterministic audit checks for the interpretation table."""
    table = list(rows)
    transformer_rows = [
        row for row in table if row.get("saliency_family") == TRANSFORMER_RELEVANCE_FAMILY
    ]
    rollout_rows = [row for row in table if row.get("saliency_family") == ROLLOUT_FAMILY]
    transformer_coco_rows = [
        row
        for row in transformer_rows
        if row.get("dataset") == "coco_search18_static2000"
    ]
    rows_missing_counts = [
        row
        for row in table
        if row.get("behavior_metric_rows", "") == ""
        or row.get("cross_level_rows", "") == ""
        or row.get("leader_overlap_rows", "") == ""
    ]
    bad_text_rows = [
        row
        for row in transformer_rows
        if any(
            term in str(row.get("paper_interpretation", "")).lower()
            for term in BANNED_TRANSFORMER_INTERPRETATION_TERMS
        )
    ]
    return [
        _audit_row(
            "transformer_relevance_present",
            passed=bool(transformer_rows),
            observed=str(len(transformer_rows)),
            expected="at least one transformer_relevance row",
        ),
        _audit_row(
            "rollout_family_separate",
            passed=bool(rollout_rows)
            and all(row.get("saliency_family") != TRANSFORMER_RELEVANCE_FAMILY for row in rollout_rows),
            observed=str(len(rollout_rows)),
            expected="internal_routing rows remain separate",
        ),
        _audit_row(
            "family_counts_present",
            passed=not rows_missing_counts,
            observed=str(len(rows_missing_counts)),
            expected="all rows include behavior/cross-level/leader-overlap counts",
        ),
        _audit_row(
            "no_coco_transformer_relevance",
            passed=not transformer_coco_rows,
            observed=str(len(transformer_coco_rows)),
            expected="no COCO-Search18 transformer_relevance rows",
        ),
        _audit_row(
            "transformer_interpretation_claim_hygiene",
            passed=not bad_text_rows,
            observed=str(len(bad_text_rows)),
            expected="no operational/causal/human-like attention language",
        ),
    ]


def _interpretation_row(
    *,
    dataset: str,
    family: str,
    behavior_rows: list[dict[str, Any]],
    family_ranking_rows: list[dict[str, Any]],
    cross_level_rows: list[dict[str, Any]],
    leader_overlap_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    nss_rows = [row for row in behavior_rows if row.get("metric") == "nss"]
    best_nss = _best_behavior_row(nss_rows, metric="nss")
    included_methods = sorted(
        {str(row.get("saliency_method", "")) for row in behavior_rows if row.get("saliency_method")}
    )
    complete_cross = [row for row in cross_level_rows if row.get("status") == "complete"]
    insufficient_cross = [
        row for row in cross_level_rows if row.get("status") == "insufficient_models"
    ]
    n_models = [
        value
        for value in (_optional_int(row.get("n_models")) for row in cross_level_rows)
        if value is not None
    ]
    encoding_matches = sum(
        1 for row in leader_overlap_rows if row.get("matches_encoding_leader") == "true"
    )
    rsa_matches = sum(
        1 for row in leader_overlap_rows if row.get("matches_rsa_leader") == "true"
    )
    transformer_comparison_rows = (
        comparison_rows if family == TRANSFORMER_RELEVANCE_FAMILY else []
    )
    nss_delta = _transformer_nss_delta(transformer_comparison_rows)
    all_better = _all_transformer_metrics_better(transformer_comparison_rows)
    interpretation = _paper_interpretation(
        dataset=dataset,
        family=family,
        complete_cross_count=len(complete_cross),
        n_models_max=max(n_models) if n_models else None,
        all_transformer_metrics_better=all_better,
        transformer_nss_delta=nss_delta,
    )
    return {
        "dataset": dataset,
        "viewing_regime": _viewing_regime(dataset),
        "saliency_family": family,
        "included_methods": ";".join(included_methods),
        "behavior_metric_rows": str(len(behavior_rows)),
        "behavior_nss_rows": str(len(nss_rows)),
        "behavior_total_n": str(_behavior_total_n(behavior_rows, family_ranking_rows)),
        "best_nss_method": best_nss.get("saliency_method", ""),
        "best_nss_model": best_nss.get("model", ""),
        "best_nss_mean": _fmt(best_nss.get("mean", "")),
        "best_nss_ci95_low": _fmt(best_nss.get("ci95_low", "")),
        "best_nss_ci95_high": _fmt(best_nss.get("ci95_high", "")),
        "leader_overlap_rows": str(len(leader_overlap_rows)),
        "encoding_leader_match_count": str(encoding_matches),
        "encoding_leader_match_rate": _rate(encoding_matches, len(leader_overlap_rows)),
        "geometry_or_rsa_leader_match_count": str(rsa_matches),
        "geometry_or_rsa_leader_match_rate": _rate(rsa_matches, len(leader_overlap_rows)),
        "cross_level_rows": str(len(cross_level_rows)),
        "cross_level_complete_rows": str(len(complete_cross)),
        "cross_level_insufficient_rows": str(len(insufficient_cross)),
        "cross_level_n_models_min": str(min(n_models)) if n_models else "",
        "cross_level_n_models_max": str(max(n_models)) if n_models else "",
        "mean_spearman_behavior_vs_encoding": _mean_metric(
            complete_cross,
            "spearman_behavior_vs_noise_normalized",
        ),
        "mean_spearman_behavior_vs_geometry": _mean_metric(
            complete_cross,
            "spearman_behavior_vs_geometry",
        ),
        "transformer_relevance_vs_rollout_rows": str(len(transformer_comparison_rows)),
        "transformer_relevance_vs_rollout_all_metrics_better": _bool_text(all_better),
        "transformer_relevance_vs_rollout_nss_delta": _fmt(nss_delta),
        "paper_interpretation": interpretation,
    }


def _paper_interpretation(
    *,
    dataset: str,
    family: str,
    complete_cross_count: int,
    n_models_max: int | None,
    all_transformer_metrics_better: bool | None,
    transformer_nss_delta: float | None,
) -> str:
    if family in TASK_SEARCH_FAMILIES:
        return "task_search_control_family"
    if family in REFERENCE_FAMILIES:
        return "reference_control_family"
    if family == TRANSFORMER_RELEVANCE_FAMILY:
        if (
            dataset == "coco_search18_static2000"
            or complete_cross_count == 0
            or (n_models_max or 0) < 3
        ):
            return "insufficient_cross_axis_evidence"
        if all_transformer_metrics_better and (transformer_nss_delta or 0.0) > 0.0:
            return "transformer_relevance_improves_rollout_behavior_only"
        return "attribution_family_descriptive_only"
    if complete_cross_count == 0 or (n_models_max or 0) < 3:
        return "insufficient_cross_axis_evidence"
    if family in ATTRIBUTION_FAMILIES:
        return "attribution_family_descriptive_only"
    return "insufficient_cross_axis_evidence"


def _viewing_regime(dataset: str) -> str:
    if dataset == "coco_search18_static2000":
        return "task_search"
    if dataset in {"salicon_static2000", "cat2000_static2000"}:
        return "free_viewing"
    return "unknown"


def _best_behavior_row(
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    if not rows:
        return {}
    reverse = metric not in {"kl", "emd", "emd_2d", "mae", "mse", "rmse", "loss"}
    return sorted(rows, key=lambda row: _float(row.get("mean")), reverse=reverse)[0]


def _behavior_total_n(
    behavior_rows: list[dict[str, Any]],
    family_ranking_rows: list[dict[str, Any]],
) -> int:
    if family_ranking_rows:
        return sum(_optional_int(row.get("total_n")) or 0 for row in family_ranking_rows)
    return sum(_optional_int(row.get("n")) or 0 for row in behavior_rows)


def _transformer_nss_delta(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if row.get("metric") == "nss":
            return _optional_float(row.get("delta_positive_favors_left"))
    return None


def _all_transformer_metrics_better(rows: list[dict[str, Any]]) -> bool | None:
    if not rows:
        return None
    deltas = [_optional_float(row.get("delta_positive_favors_left")) for row in rows]
    valid_deltas = [delta for delta in deltas if delta is not None]
    return bool(valid_deltas) and len(valid_deltas) == len(rows) and all(
        delta > 0.0 for delta in valid_deltas
    )


def _mean_metric(rows: list[dict[str, Any]], field: str) -> str:
    values = [
        value
        for value in (_optional_float(row.get(field)) for row in rows)
        if value is not None
    ]
    if not values:
        return ""
    return _fmt(sum(values) / len(values))


def _rate(count: int, total: int) -> str:
    if total <= 0:
        return ""
    return _fmt(count / total)


def _bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _fmt(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.4g}"


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else 0.0


def _optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _audit_row(
    check: str,
    *,
    passed: bool,
    observed: str,
    expected: str,
) -> dict[str, str]:
    return {
        "check": check,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "expected": expected,
    }


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return _load_csv_rows(path)


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return path
    fieldnames = list(rows[0])
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
