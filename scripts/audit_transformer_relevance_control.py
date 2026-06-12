"""Audit scoped transformer relevance control outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from hma.utils.paths import ensure_dir


DEFAULT_TRANSFORMER_RELEVANCE_AGGREGATE = (
    "outputs/real_matrix_v2_transformer_relevance/aggregated/results.csv"
)
DEFAULT_OUTPUT = (
    "outputs/paper1_experiment_v1/summary/transformer_relevance_control_audit.csv"
)

EXPECTED_DATASETS = {"salicon_static2000", "cat2000_static2000"}
EXPECTED_MODELS = {
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
}
EXPECTED_METHOD = "transformer_relevance"
EXPECTED_FAMILY = "transformer_relevance"
EXPECTED_METRICS = {
    "nss",
    "shuffled_auc",
    "auc_borji",
    "auc_judd",
    "cc",
    "similarity",
    "kl",
}

FIELDNAMES = [
    "check",
    "status",
    "expected",
    "observed",
    "detail",
    "next_action",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit scoped transformer relevance benchmark outputs."
    )
    parser.add_argument(
        "--aggregate",
        default=DEFAULT_TRANSFORMER_RELEVANCE_AGGREGATE,
        help="Transformer relevance aggregate CSV to audit.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output audit CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = audit_transformer_relevance_control(
        aggregate=args.aggregate,
        output=args.output,
    )
    statuses = _status_counts(rows)
    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(statuses.items())
    )
    print(
        "Transformer relevance control audit: "
        f"{Path(args.output).resolve()} ({len(rows)} rows; {status_text})"
    )


def audit_transformer_relevance_control(
    *,
    aggregate: str | Path = DEFAULT_TRANSFORMER_RELEVANCE_AGGREGATE,
    output: str | Path | None = DEFAULT_OUTPUT,
) -> list[dict[str, str]]:
    """Return and optionally write transformer relevance control audit rows."""
    aggregate_path = Path(aggregate)
    rows = _load_optional_csv_rows(aggregate_path)
    audit_rows = build_transformer_relevance_control_audit(
        rows,
        aggregate_artifact=str(aggregate_path),
        aggregate_exists=aggregate_path.is_file(),
    )
    if output is not None:
        _write_rows(Path(output), audit_rows)
    return audit_rows


def build_transformer_relevance_control_audit(
    aggregate_rows: Iterable[dict[str, str]],
    *,
    aggregate_artifact: str,
    aggregate_exists: bool = True,
) -> list[dict[str, str]]:
    """Build deterministic audit rows from transformer relevance aggregate rows."""
    rows = list(aggregate_rows)
    scoped_rows = [
        row
        for row in rows
        if row.get("saliency_method") == EXPECTED_METHOD
        and row.get("saliency_family") == EXPECTED_FAMILY
    ]
    return [
        _artifact_row(rows, aggregate_artifact=aggregate_artifact, exists=aggregate_exists),
        _dataset_scope_row(rows),
        _model_scope_row(scoped_rows),
        _method_label_row(rows),
        _family_label_row(rows),
        _metric_coverage_row(scoped_rows),
        _expected_cell_coverage_row(scoped_rows),
        _evidence_decision_row(rows, scoped_rows, aggregate_exists=aggregate_exists),
    ]


def _artifact_row(
    rows: list[dict[str, str]],
    *,
    aggregate_artifact: str,
    exists: bool,
) -> dict[str, str]:
    return _row(
        "aggregate_artifact_exists",
        passed=exists and bool(rows),
        expected=aggregate_artifact,
        observed=f"exists={exists};rows={len(rows)}",
        pass_detail="Transformer relevance aggregate exists and contains rows.",
        fail_detail=(
            "Transformer relevance aggregate is missing or empty. This is expected "
            "while the long static benchmark is still running."
        ),
        next_action=(
            "run or finish the scoped SALICON/CAT2000 transformer relevance static "
            "benchmarks, then aggregate outputs"
        ),
    )


def _dataset_scope_row(rows: list[dict[str, str]]) -> dict[str, str]:
    datasets = {row.get("dataset", "") for row in rows if row.get("dataset")}
    unexpected = datasets - EXPECTED_DATASETS
    missing = EXPECTED_DATASETS - datasets
    return _row(
        "dataset_scope",
        passed=bool(rows) and not unexpected and not missing,
        expected=_join(EXPECTED_DATASETS),
        observed=_join(datasets),
        pass_detail="Only SALICON/CAT2000 static rows are present.",
        fail_detail=(
            f"Unexpected datasets={_join(unexpected)}; missing datasets={_join(missing)}."
        ),
        next_action=(
            "remove leaked datasets or regenerate the scoped config set before merging "
            "this control into accepted behavioral evidence"
        ),
    )


def _model_scope_row(rows: list[dict[str, str]]) -> dict[str, str]:
    models = {row.get("model", "") for row in rows if row.get("model")}
    unexpected = models - EXPECTED_MODELS
    missing = EXPECTED_MODELS - models
    return _row(
        "model_scope",
        passed=bool(rows) and not unexpected and not missing,
        expected=_join(EXPECTED_MODELS),
        observed=_join(models),
        pass_detail="Only the four planned transformer models are present.",
        fail_detail=f"Unexpected models={_join(unexpected)}; missing models={_join(missing)}.",
        next_action=(
            "complete or regenerate the planned four-model transformer relevance scope"
        ),
    )


def _method_label_row(rows: list[dict[str, str]]) -> dict[str, str]:
    methods = {row.get("saliency_method", "") for row in rows if row.get("saliency_method")}
    return _row(
        "saliency_method_label",
        passed=bool(rows) and methods == {EXPECTED_METHOD},
        expected=EXPECTED_METHOD,
        observed=_join(methods),
        pass_detail="Every row uses saliency_method=transformer_relevance.",
        fail_detail="Rows must not mix transformer relevance with rollout or gradients.",
        next_action="repair labels or aggregate only the transformer relevance output root",
    )


def _family_label_row(rows: list[dict[str, str]]) -> dict[str, str]:
    families = {row.get("saliency_family", "") for row in rows if row.get("saliency_family")}
    return _row(
        "saliency_family_label",
        passed=bool(rows) and families == {EXPECTED_FAMILY},
        expected=EXPECTED_FAMILY,
        observed=_join(families),
        pass_detail="Every row is labeled as a distinct transformer_relevance family.",
        fail_detail=(
            "Transformer relevance must not be collapsed into internal_routing, "
            "evidence_sensitivity, or unknown."
        ),
        next_action="fix family mapping before interpreting or merging results",
    )


def _metric_coverage_row(rows: list[dict[str, str]]) -> dict[str, str]:
    metrics_by_cell: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("model", ""))
        metrics_by_cell.setdefault(key, set()).add(row.get("metric", ""))
    missing_cells = [
        f"{dataset}:{model}:{_join(EXPECTED_METRICS - metrics)}"
        for (dataset, model), metrics in sorted(metrics_by_cell.items())
        if EXPECTED_METRICS - metrics
    ]
    passed = bool(rows) and not missing_cells
    return _row(
        "metric_coverage",
        passed=passed,
        expected=_join(EXPECTED_METRICS),
        observed=(
            f"complete_cells={len(metrics_by_cell) - len(missing_cells)};"
            f"incomplete_cells={len(missing_cells)}"
        ),
        pass_detail="Each present dataset/model cell has the expected metric set.",
        fail_detail=f"Missing metric coverage: {';'.join(missing_cells) or 'no cells present'}.",
        next_action="rerun incomplete configs or reaggregate after all per-image outputs finish",
    )


def _expected_cell_coverage_row(rows: list[dict[str, str]]) -> dict[str, str]:
    expected_cells = {
        (dataset, model)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
    }
    observed_cells = {
        (row.get("dataset", ""), row.get("model", ""))
        for row in rows
        if row.get("dataset") and row.get("model")
    }
    missing = expected_cells - observed_cells
    unexpected = observed_cells - expected_cells
    return _row(
        "expected_cell_coverage",
        passed=bool(rows) and not missing and not unexpected,
        expected=f"{len(expected_cells)} cells",
        observed=f"{len(observed_cells)} cells",
        pass_detail="All 8 planned dataset/model cells are present.",
        fail_detail=(
            f"Missing cells={_format_cells(missing)}; unexpected cells={_format_cells(unexpected)}."
        ),
        next_action="finish the missing static configs before evidence acceptance",
    )


def _evidence_decision_row(
    rows: list[dict[str, str]],
    scoped_rows: list[dict[str, str]],
    *,
    aggregate_exists: bool,
) -> dict[str, str]:
    checks = [
        aggregate_exists and bool(rows),
        _datasets_pass(rows),
        _models_pass(scoped_rows),
        _methods_pass(rows),
        _families_pass(rows),
        _metrics_pass(scoped_rows),
        _cells_pass(scoped_rows),
    ]
    accepted = all(checks)
    return {
        "check": "evidence_decision",
        "status": "accepted_evidence_ready" if accepted else "diagnostic_or_incomplete",
        "expected": "all scope/method/family/metric/cell checks pass",
        "observed": f"passed_checks={sum(1 for check in checks if check)}/{len(checks)}",
        "detail": (
            "Transformer relevance rows are scoped and ready for behavioral-control "
            "interpretation."
            if accepted
            else "Transformer relevance rows should remain diagnostic or incomplete."
        ),
        "next_action": (
            "compare against attention_rollout and vanilla_gradient without merging families"
            if accepted
            else "wait for full static results or repair failed audit checks before merging"
        ),
    }


def _datasets_pass(rows: list[dict[str, str]]) -> bool:
    datasets = {row.get("dataset", "") for row in rows if row.get("dataset")}
    return bool(rows) and datasets == EXPECTED_DATASETS


def _models_pass(rows: list[dict[str, str]]) -> bool:
    models = {row.get("model", "") for row in rows if row.get("model")}
    return bool(rows) and models == EXPECTED_MODELS


def _methods_pass(rows: list[dict[str, str]]) -> bool:
    methods = {row.get("saliency_method", "") for row in rows if row.get("saliency_method")}
    return bool(rows) and methods == {EXPECTED_METHOD}


def _families_pass(rows: list[dict[str, str]]) -> bool:
    families = {row.get("saliency_family", "") for row in rows if row.get("saliency_family")}
    return bool(rows) and families == {EXPECTED_FAMILY}


def _metrics_pass(rows: list[dict[str, str]]) -> bool:
    cells: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        cells.setdefault((row.get("dataset", ""), row.get("model", "")), set()).add(
            row.get("metric", "")
        )
    return bool(cells) and all(metrics == EXPECTED_METRICS for metrics in cells.values())


def _cells_pass(rows: list[dict[str, str]]) -> bool:
    expected = {
        (dataset, model)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
    }
    observed = {
        (row.get("dataset", ""), row.get("model", ""))
        for row in rows
        if row.get("dataset") and row.get("model")
    }
    return observed == expected


def _row(
    check: str,
    *,
    passed: bool,
    expected: str,
    observed: str,
    pass_detail: str,
    fail_detail: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "check": check,
        "status": "pass" if passed else "fail",
        "expected": expected,
        "observed": observed or "none",
        "detail": pass_detail if passed else fail_detail,
        "next_action": "none" if passed else next_action,
    }


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


def _join(values: Iterable[str]) -> str:
    return ",".join(sorted(str(value) for value in values if str(value))) or "none"


def _format_cells(cells: set[tuple[str, str]]) -> str:
    return ";".join(f"{dataset}:{model}" for dataset, model in sorted(cells)) or "none"


def _status_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
