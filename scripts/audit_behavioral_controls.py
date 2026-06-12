"""Audit behavioral-control gaps for Paper 1 claim hardening."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from hma.utils.paths import ensure_dir


DEFAULT_BEHAVIORAL_AGGREGATE = (
    "outputs/real_matrix_v2/aggregated/"
    "results_with_ssl_behavior_plus_transformer_relevance.csv"
)
DEFAULT_OBSERVER_CONTROL_SUMMARY = (
    "outputs/paper1_experiment_v1/summary/behavioral_observer_control_summary.csv"
)
DEFAULT_OUTPUT = (
    "outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv"
)
DEFAULT_FREE_VIEWING_REFERENCE_FEASIBILITY = (
    "outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv"
)

FIELDNAMES = [
    "claim_axis",
    "viewing_regime",
    "dataset_scope",
    "required_control",
    "current_artifact",
    "status",
    "evidence_role",
    "next_action",
    "detail",
]

FREE_VIEWING_DATASET_PREFIXES = ("salicon", "cat2000")
TASK_SEARCH_DATASET_PREFIXES = ("coco_search18",)
POINT_METRICS = {"nss", "auc_judd", "auc_borji", "shuffled_auc"}
MAP_METRICS = {"cc", "similarity", "kl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reviewer-facing behavioral-control gap audit."
    )
    parser.add_argument(
        "--behavioral-aggregate",
        default=DEFAULT_BEHAVIORAL_AGGREGATE,
        help="Merged behavioral aggregate CSV.",
    )
    parser.add_argument(
        "--observer-control-summary",
        default=DEFAULT_OBSERVER_CONTROL_SUMMARY,
        help="Paper-facing observer-control summary CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output behavioral-control gap audit CSV.",
    )
    parser.add_argument(
        "--free-viewing-reference-feasibility",
        default=DEFAULT_FREE_VIEWING_REFERENCE_FEASIBILITY,
        help="Optional modern free-viewing reference feasibility decision CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = audit_behavioral_controls(
        behavioral_aggregate=args.behavioral_aggregate,
        observer_control_summary=args.observer_control_summary,
        free_viewing_reference_feasibility=args.free_viewing_reference_feasibility,
        output=args.output,
    )
    print(f"Behavioral-control gap audit: {Path(args.output).resolve()} ({len(rows)} rows)")


def audit_behavioral_controls(
    *,
    behavioral_aggregate: str | Path = DEFAULT_BEHAVIORAL_AGGREGATE,
    observer_control_summary: str | Path = DEFAULT_OBSERVER_CONTROL_SUMMARY,
    free_viewing_reference_feasibility: str | Path = DEFAULT_FREE_VIEWING_REFERENCE_FEASIBILITY,
    output: str | Path | None = DEFAULT_OUTPUT,
) -> list[dict[str, str]]:
    """Return and optionally write the behavioral-control gap audit rows."""
    behavioral_path = Path(behavioral_aggregate)
    observer_path = Path(observer_control_summary)
    feasibility_path = Path(free_viewing_reference_feasibility)
    behavioral_rows = _load_optional_csv_rows(behavioral_path)
    observer_rows = _load_optional_csv_rows(observer_path)
    feasibility_rows = _load_optional_csv_rows(feasibility_path)

    rows = build_behavioral_control_gap_audit(
        behavioral_rows,
        observer_rows,
        feasibility_rows,
        behavioral_artifact=str(behavioral_path),
        observer_summary_artifact=str(observer_path),
        feasibility_artifact=str(feasibility_path),
    )
    if output is not None:
        _write_rows(Path(output), rows)
    return rows


def build_behavioral_control_gap_audit(
    behavioral_rows: Iterable[dict[str, str]],
    observer_rows: Iterable[dict[str, str]],
    feasibility_rows: Iterable[dict[str, str]] | None = None,
    *,
    behavioral_artifact: str,
    observer_summary_artifact: str,
    feasibility_artifact: str = "",
) -> list[dict[str, str]]:
    """Build deterministic audit rows from already-generated summaries."""
    behavior = list(behavioral_rows)
    observers = list(observer_rows)
    feasibility = list(feasibility_rows or [])
    return [
        _behavior_control_row(
            behavior,
            viewing_regime="free_viewing",
            dataset_scope="SALICON/CAT2000",
            required_control="DeepGaze IIE free-viewing fixation reference",
            model="deepgaze_reference",
            saliency_method="deepgaze_precomputed",
            accepted_status="accepted",
            missing_status="missing",
            evidence_role="dedicated free-viewing fixation reference",
            present_next_action="keep as current accepted DeepGaze-class free-viewing reference",
            missing_next_action="regenerate DeepGaze IIE reference rows before free-viewing claims",
            artifact=behavioral_artifact,
            detail_prefix="DeepGaze IIE rows for SALICON/CAT2000",
        ),
        _behavior_control_row(
            behavior,
            viewing_regime="free_viewing",
            dataset_scope="SALICON/CAT2000",
            required_control="center-bias free-viewing baseline",
            model="center_bias_baseline",
            saliency_method="center_bias",
            accepted_status="accepted",
            missing_status="missing",
            evidence_role="minimum saliency prior control",
            present_next_action="keep as accepted free-viewing baseline context",
            missing_next_action="regenerate center-bias rows before free-viewing claims",
            artifact=behavioral_artifact,
            detail_prefix="center-bias rows for SALICON/CAT2000",
        ),
        _observer_control_row(
            observers,
            dataset="SALICON",
            viewing_regime="free_viewing",
            dataset_scope="SALICON",
            required_control="SALICON leave-one-observer-out control",
            observer_summary_artifact=observer_summary_artifact,
        ),
        _modern_free_viewing_reference_row(
            behavior,
            feasibility,
            behavioral_artifact=behavioral_artifact,
            feasibility_artifact=feasibility_artifact,
        ),
        _behavior_control_row(
            behavior,
            viewing_regime="task_search",
            dataset_scope="COCO-Search18",
            required_control="DeepGaze IIE task-search diagnostic reference",
            model="deepgaze_reference",
            saliency_method="deepgaze_precomputed",
            accepted_status="diagnostic",
            missing_status="missing",
            evidence_role="free-viewing reference used only as task-search diagnostic",
            present_next_action=(
                "keep diagnostic only; do not interpret as a task-specific search baseline"
            ),
            missing_next_action="document absence of DeepGaze diagnostic row for COCO-Search18",
            artifact=behavioral_artifact,
            detail_prefix="DeepGaze IIE rows for COCO-Search18",
        ),
        _behavior_control_row(
            behavior,
            viewing_regime="task_search",
            dataset_scope="COCO-Search18",
            required_control="center-bias task-search baseline",
            model="center_bias_baseline",
            saliency_method="center_bias",
            accepted_status="accepted",
            missing_status="missing",
            evidence_role="minimum spatial-prior control for search data",
            present_next_action="keep as accepted task-search baseline context",
            missing_next_action="regenerate center-bias rows before task-search claims",
            artifact=behavioral_artifact,
            detail_prefix="center-bias rows for COCO-Search18",
        ),
        _observer_control_row(
            observers,
            dataset="COCO-Search18",
            viewing_regime="task_search",
            dataset_scope="COCO-Search18",
            required_control="COCO-Search18 leave-one-observer-out control",
            observer_summary_artifact=observer_summary_artifact,
        ),
        _behavior_control_row(
            behavior,
            viewing_regime="task_search",
            dataset_scope="COCO-Search18",
            required_control="task-specific COCO-Search18 baseline",
            model="coco_search18_task_prior_baseline",
            saliency_method="coco_search18_task_prior",
            accepted_status="accepted",
            missing_status="missing",
            evidence_role="task-conditioned behavioral baseline",
            present_next_action=(
                "use as task-search control before interpreting COCO-Search18 "
                "model-human agreement"
            ),
            missing_next_action=(
                "select and implement a task-specific search baseline, or document "
                "infeasibility as a limitation"
            ),
            artifact=behavioral_artifact,
            detail_prefix="task-conditioned COCO-Search18 prior rows",
        ),
        _metric_boundary_row(behavior, behavioral_artifact=behavioral_artifact),
    ]


def _modern_free_viewing_reference_row(
    behavior_rows: list[dict[str, str]],
    feasibility_rows: list[dict[str, str]],
    *,
    behavioral_artifact: str,
    feasibility_artifact: str,
) -> dict[str, str]:
    matches = [
        row
        for row in _rows_for_regime(behavior_rows, "free_viewing")
        if row.get("model") == "deepgaze_msdb_reference"
        and row.get("saliency_method") == "deepgaze_precomputed"
    ]
    msdb_datasets = sorted({_dataset_label(row.get("dataset", "")) for row in matches})
    msdb_metrics = sorted({row.get("metric", "") for row in matches if row.get("metric")})
    if {"SALICON", "CAT2000"} <= set(msdb_datasets) and "nss" in msdb_metrics:
        return {
            "claim_axis": "behavioral_fixation_alignment",
            "viewing_regime": "free_viewing",
            "dataset_scope": "SALICON/CAT2000",
            "required_control": "modern free-viewing fixation reference",
            "current_artifact": behavioral_artifact,
            "status": "accepted",
            "evidence_role": "stronger DeepGaze-class reviewer control",
            "next_action": (
                "use DeepGaze MSDB as the accepted modern free-viewing reference "
                "while preserving DeepGaze IIE as historical/reference context"
            ),
            "detail": (
                f"DeepGaze MSDB rows: {len(matches)} summary rows across "
                f"datasets={','.join(msdb_datasets)} metrics={','.join(msdb_metrics)}"
            ),
        }

    base = {
        "claim_axis": "behavioral_fixation_alignment",
        "viewing_regime": "free_viewing",
        "dataset_scope": "SALICON/CAT2000",
        "required_control": "modern free-viewing fixation reference",
        "current_artifact": feasibility_artifact if feasibility_rows else "",
        "evidence_role": "stronger DeepGaze-class reviewer control",
    }
    msdb = next(
        (
            row
            for row in feasibility_rows
            if row.get("candidate_reference") == "DeepGaze MSDB"
        ),
        None,
    )
    if msdb is None:
        return {
            **base,
            "status": "needs_feasibility_decision",
            "next_action": (
                "decide whether DeepGaze MSDB or a comparable modern reference can be "
                "added without expanding into a generic saliency leaderboard"
            ),
            "detail": (
                "DeepGaze IIE is present as the current reference; no concrete DeepGaze "
                "MSDB feasibility decision is present in the current accepted outputs"
            ),
        }
    decision = msdb.get("decision", "")
    if decision == "feasible_now":
        return {
            **base,
            "status": "needs_export_and_evaluation",
            "next_action": (
                "run SALICON/CAT2000-only DeepGaze MSDB smoke export, full export, "
                "benchmark scoring, and separate aggregation before treating it as evidence"
            ),
            "detail": (
                "DeepGaze MSDB feasibility decision is feasible_now; this is not an "
                "accepted behavioral result until maps are exported and scored. "
                f"Feasibility detail: {msdb.get('detail', '')}"
            ),
        }
    if decision == "defer_or_document_limitation":
        return {
            **base,
            "status": "defer_or_document_limitation",
            "next_action": (
                "keep DeepGaze IIE as the accepted DeepGaze-class free-viewing reference "
                "and document the modern-reference limitation"
            ),
            "detail": msdb.get("detail", "DeepGaze MSDB feasibility was deferred."),
        }
    return {
        **base,
        "status": "requires_download_or_dependency",
        "next_action": (
            "resolve the required download/dependency before adding a modern free-viewing reference"
        ),
        "detail": msdb.get("detail", "DeepGaze MSDB requires additional setup."),
    }


def _behavior_control_row(
    rows: list[dict[str, str]],
    *,
    viewing_regime: str,
    dataset_scope: str,
    required_control: str,
    model: str,
    saliency_method: str,
    accepted_status: str,
    missing_status: str,
    evidence_role: str,
    present_next_action: str,
    missing_next_action: str,
    artifact: str,
    detail_prefix: str,
) -> dict[str, str]:
    scoped = _rows_for_regime(rows, viewing_regime)
    matches = [
        row
        for row in scoped
        if row.get("model") == model and row.get("saliency_method") == saliency_method
    ]
    if matches:
        metrics = sorted({row.get("metric", "") for row in matches if row.get("metric")})
        datasets = sorted({_dataset_label(row.get("dataset", "")) for row in matches})
        return {
            "claim_axis": "behavioral_fixation_alignment",
            "viewing_regime": viewing_regime,
            "dataset_scope": dataset_scope,
            "required_control": required_control,
            "current_artifact": artifact,
            "status": accepted_status,
            "evidence_role": evidence_role,
            "next_action": present_next_action,
            "detail": (
                f"{detail_prefix}: {len(matches)} summary rows across "
                f"datasets={','.join(datasets)} metrics={','.join(metrics)}"
            ),
        }
    return {
        "claim_axis": "behavioral_fixation_alignment",
        "viewing_regime": viewing_regime,
        "dataset_scope": dataset_scope,
        "required_control": required_control,
        "current_artifact": artifact,
        "status": missing_status,
        "evidence_role": evidence_role,
        "next_action": missing_next_action,
        "detail": f"{detail_prefix}: no matching rows found",
    }


def _observer_control_row(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    viewing_regime: str,
    dataset_scope: str,
    required_control: str,
    observer_summary_artifact: str,
) -> dict[str, str]:
    match = next(
        (
            row
            for row in rows
            if row.get("dataset") == dataset and row.get("viewing_regime") == viewing_regime
        ),
        None,
    )
    complete = match is not None and match.get("status") == "complete"
    source_path = (match or {}).get("source_path") or observer_summary_artifact
    if complete:
        detail = (
            f"observer summary complete with row_count={(match or {}).get('row_count', '')}, "
            f"image_count={(match or {}).get('image_count', '')}, "
            f"median_auc={(match or {}).get('median_inter_observer_auc', '')}"
        )
    else:
        detail = "observer-control summary missing or incomplete"
    return {
        "claim_axis": "behavioral_fixation_alignment",
        "viewing_regime": viewing_regime,
        "dataset_scope": dataset_scope,
        "required_control": required_control,
        "current_artifact": source_path,
        "status": "accepted" if complete else "missing",
        "evidence_role": "human/interobserver context, not model performance",
        "next_action": (
            "keep as reviewer-facing observer context"
            if complete
            else "regenerate observer-control summary before treating observer context as complete"
        ),
        "detail": detail,
    }


def _metric_boundary_row(
    rows: list[dict[str, str]],
    *,
    behavioral_artifact: str,
) -> dict[str, str]:
    free_rows = _rows_for_regime(rows, "free_viewing")
    task_rows = _rows_for_regime(rows, "task_search")
    free_point = _metrics_present(free_rows, POINT_METRICS)
    free_map = _metrics_present(free_rows, MAP_METRICS)
    task_point = _metrics_present(task_rows, POINT_METRICS)
    task_map = _metrics_present(task_rows, MAP_METRICS)
    free_protocols = _protocols_present(free_rows)
    task_protocols = _protocols_present(task_rows)
    accepted = (
        bool(free_point)
        and bool(free_map)
        and bool(task_point)
        and bool(task_map)
        and free_protocols <= {"points"}
        and task_protocols <= {"task_points"}
        and free_protocols
        and task_protocols
    )
    return {
        "claim_axis": "metric_boundary_control",
        "viewing_regime": "all_regimes_separated",
        "dataset_scope": "free_viewing=SALICON/CAT2000;task_search=COCO-Search18",
        "required_control": "point-fixation metrics separated from map-distribution metrics",
        "current_artifact": behavioral_artifact,
        "status": "accepted" if accepted else "missing",
        "evidence_role": "claim-boundary control",
        "next_action": (
            "preserve separate NSS/AUC point claims and CC/SIM/KL map-distribution claims"
            if accepted
            else "repair protocol or metric coverage before using behavioral summaries"
        ),
        "detail": (
            "point-fixation metrics="
            f"{','.join(sorted(POINT_METRICS))}; map-distribution metrics="
            f"{','.join(sorted(MAP_METRICS))}; free_protocols="
            f"{','.join(sorted(free_protocols))}; task_protocols="
            f"{','.join(sorted(task_protocols))}"
        ),
    }


def _rows_for_regime(rows: list[dict[str, str]], regime: str) -> list[dict[str, str]]:
    if regime == "free_viewing":
        return [
            row
            for row in rows
            if _matches_any_prefix(row.get("dataset", ""), FREE_VIEWING_DATASET_PREFIXES)
        ]
    if regime == "task_search":
        return [
            row
            for row in rows
            if _matches_any_prefix(row.get("dataset", ""), TASK_SEARCH_DATASET_PREFIXES)
        ]
    raise ValueError(f"Unknown viewing regime: {regime}")


def _metrics_present(rows: list[dict[str, str]], metrics: set[str]) -> set[str]:
    return {row.get("metric", "") for row in rows if row.get("metric", "") in metrics}


def _protocols_present(rows: list[dict[str, str]]) -> set[str]:
    return {row.get("fixation_protocol", "") for row in rows if row.get("fixation_protocol")}


def _matches_any_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _dataset_label(value: str) -> str:
    if value.startswith("salicon"):
        return "SALICON"
    if value.startswith("cat2000"):
        return "CAT2000"
    if value.startswith("coco_search18"):
        return "COCO-Search18"
    return value


def _load_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
