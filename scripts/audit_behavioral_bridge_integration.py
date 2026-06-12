"""Audit accepted transformer relevance integration into the behavioral bridge."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from hma.utils.paths import ensure_dir


DEFAULT_BASE_BRIDGE = "outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv"
DEFAULT_TRANSFORMER_RELEVANCE = (
    "outputs/real_matrix_v2_transformer_relevance/aggregated/results.csv"
)
DEFAULT_MERGED_BRIDGE = (
    "outputs/real_matrix_v2/aggregated/"
    "results_with_ssl_behavior_plus_transformer_relevance.csv"
)
DEFAULT_OUTPUT = (
    "outputs/paper1_experiment_v1/summary/"
    "behavioral_bridge_integration_audit.csv"
)

EXPECTED_BASE_ROWS = 399
EXPECTED_TRANSFORMER_RELEVANCE_ROWS = 56
EXPECTED_MERGED_ROWS = 455
EXPECTED_TRANSFORMER_RELEVANCE_DATASETS = {
    "salicon_static2000",
    "cat2000_static2000",
}
EXPECTED_METHOD = "transformer_relevance"
EXPECTED_FAMILY = "transformer_relevance"
ROLLOUT_METHOD = "attention_rollout"
ROLLOUT_FAMILY = "internal_routing"

MERGE_KEY_FIELDS = [
    "dataset",
    "model",
    "saliency_method",
    "saliency_family",
    "metric",
]
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
        description="Audit transformer relevance rows in the accepted behavioral bridge."
    )
    parser.add_argument("--base-bridge", default=DEFAULT_BASE_BRIDGE)
    parser.add_argument("--transformer-relevance", default=DEFAULT_TRANSFORMER_RELEVANCE)
    parser.add_argument("--merged-bridge", default=DEFAULT_MERGED_BRIDGE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = audit_behavioral_bridge_integration(
        base_bridge=args.base_bridge,
        transformer_relevance=args.transformer_relevance,
        merged_bridge=args.merged_bridge,
        output=args.output,
    )
    statuses = _status_counts(rows)
    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(statuses.items())
    )
    print(
        "Behavioral bridge integration audit: "
        f"{Path(args.output).resolve()} ({len(rows)} rows; {status_text})"
    )


def audit_behavioral_bridge_integration(
    *,
    base_bridge: str | Path = DEFAULT_BASE_BRIDGE,
    transformer_relevance: str | Path = DEFAULT_TRANSFORMER_RELEVANCE,
    merged_bridge: str | Path = DEFAULT_MERGED_BRIDGE,
    output: str | Path | None = DEFAULT_OUTPUT,
) -> list[dict[str, str]]:
    """Return and optionally write rows auditing the merged behavioral bridge."""
    base_path = Path(base_bridge)
    relevance_path = Path(transformer_relevance)
    merged_path = Path(merged_bridge)

    base_rows = _load_optional_csv_rows(base_path)
    relevance_rows = _load_optional_csv_rows(relevance_path)
    merged_rows = _load_optional_csv_rows(merged_path)

    audit_rows = build_behavioral_bridge_integration_audit(
        base_rows,
        relevance_rows,
        merged_rows,
        base_artifact=str(base_path),
        transformer_relevance_artifact=str(relevance_path),
        merged_artifact=str(merged_path),
        base_exists=base_path.is_file(),
        transformer_relevance_exists=relevance_path.is_file(),
        merged_exists=merged_path.is_file(),
    )
    if output is not None:
        _write_rows(Path(output), audit_rows)
    return audit_rows


def build_behavioral_bridge_integration_audit(
    base_rows: Iterable[dict[str, str]],
    transformer_relevance_rows: Iterable[dict[str, str]],
    merged_rows: Iterable[dict[str, str]],
    *,
    base_artifact: str,
    transformer_relevance_artifact: str,
    merged_artifact: str,
    base_exists: bool = True,
    transformer_relevance_exists: bool = True,
    merged_exists: bool = True,
    expected_base_rows: int = EXPECTED_BASE_ROWS,
    expected_transformer_relevance_rows: int = EXPECTED_TRANSFORMER_RELEVANCE_ROWS,
    expected_merged_rows: int = EXPECTED_MERGED_ROWS,
) -> list[dict[str, str]]:
    """Build deterministic audit rows from base, relevance, and merged CSV rows."""
    base = list(base_rows)
    relevance = list(transformer_relevance_rows)
    merged = list(merged_rows)
    relevance_in_merged = [
        row
        for row in merged
        if row.get("saliency_method") == EXPECTED_METHOD
        or row.get("saliency_family") == EXPECTED_FAMILY
    ]
    return [
        _artifact_row(
            "base_bridge_artifact_exists",
            rows=base,
            artifact=base_artifact,
            exists=base_exists,
            label="Base behavioral bridge",
        ),
        _artifact_row(
            "transformer_relevance_artifact_exists",
            rows=relevance,
            artifact=transformer_relevance_artifact,
            exists=transformer_relevance_exists,
            label="Transformer relevance aggregate",
        ),
        _artifact_row(
            "merged_bridge_artifact_exists",
            rows=merged,
            artifact=merged_artifact,
            exists=merged_exists,
            label="Merged behavioral bridge candidate",
        ),
        _row_count_row("base_bridge_row_count", base, expected_base_rows),
        _row_count_row(
            "transformer_relevance_row_count",
            relevance,
            expected_transformer_relevance_rows,
        ),
        _row_count_row("merged_bridge_row_count", merged, expected_merged_rows),
        _retention_row("base_rows_retained", base, merged),
        _retention_row("transformer_relevance_rows_retained", relevance, merged),
        _transformer_relevance_label_row(
            "merged_transformer_relevance_method_count",
            merged,
            field="saliency_method",
            expected_value=EXPECTED_METHOD,
            expected_count=expected_transformer_relevance_rows,
        ),
        _transformer_relevance_label_row(
            "merged_transformer_relevance_family_count",
            merged,
            field="saliency_family",
            expected_value=EXPECTED_FAMILY,
            expected_count=expected_transformer_relevance_rows,
        ),
        _transformer_relevance_dataset_scope_row(relevance_in_merged),
        _attention_rollout_family_row(merged),
    ]


def _artifact_row(
    check: str,
    *,
    rows: list[dict[str, str]],
    artifact: str,
    exists: bool,
    label: str,
) -> dict[str, str]:
    return _row(
        check,
        passed=exists and bool(rows),
        expected=artifact,
        observed=f"exists={exists};rows={len(rows)}",
        pass_detail=f"{label} exists and contains rows.",
        fail_detail=f"{label} is missing or empty.",
        next_action=f"generate or restore {artifact}",
    )


def _row_count_row(
    check: str,
    rows: list[dict[str, str]],
    expected_count: int,
) -> dict[str, str]:
    return _row(
        check,
        passed=len(rows) == expected_count,
        expected=str(expected_count),
        observed=str(len(rows)),
        pass_detail="Row count matches the accepted integration contract.",
        fail_detail="Row count differs from the accepted integration contract.",
        next_action="inspect merge inputs and rerun the behavioral bridge merge",
    )


def _retention_row(
    check: str,
    source_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
) -> dict[str, str]:
    missing = _missing_key_counts(_key_counts(source_rows), _key_counts(merged_rows))
    return _row(
        check,
        passed=not missing and bool(source_rows) and bool(merged_rows),
        expected=f"{len(source_rows)} source rows retained by merge key",
        observed=f"missing_keys={len(missing)}",
        pass_detail="All source rows are present in the merged candidate by merge key.",
        fail_detail=_format_missing_keys(missing),
        next_action="rerun merge and inspect duplicate or dropped merge keys",
    )


def _transformer_relevance_label_row(
    check: str,
    rows: list[dict[str, str]],
    *,
    field: str,
    expected_value: str,
    expected_count: int,
) -> dict[str, str]:
    matches = [row for row in rows if row.get(field) == expected_value]
    return _row(
        check,
        passed=len(matches) == expected_count,
        expected=f"{field}={expected_value};count={expected_count}",
        observed=str(len(matches)),
        pass_detail="Transformer relevance label count matches the accepted scope.",
        fail_detail="Transformer relevance label count differs from the accepted scope.",
        next_action="repair labels or rerun the merge from the accepted relevance aggregate",
    )


def _transformer_relevance_dataset_scope_row(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    datasets = {row.get("dataset", "") for row in rows if row.get("dataset", "")}
    return _row(
        "merged_transformer_relevance_dataset_scope",
        passed=datasets == EXPECTED_TRANSFORMER_RELEVANCE_DATASETS,
        expected=", ".join(sorted(EXPECTED_TRANSFORMER_RELEVANCE_DATASETS)),
        observed=", ".join(sorted(datasets)),
        pass_detail="Transformer relevance remains scoped to SALICON/CAT2000.",
        fail_detail="Transformer relevance dataset scope changed.",
        next_action="remove out-of-scope rows; do not add COCO-Search18 relevance here",
    )


def _attention_rollout_family_row(rows: list[dict[str, str]]) -> dict[str, str]:
    rollout_rows = [row for row in rows if row.get("saliency_method") == ROLLOUT_METHOD]
    bad_rows = [
        row for row in rollout_rows if row.get("saliency_family") != ROLLOUT_FAMILY
    ]
    return _row(
        "attention_rollout_family_preserved",
        passed=bool(rollout_rows) and not bad_rows,
        expected=f"{ROLLOUT_METHOD} rows use saliency_family={ROLLOUT_FAMILY}",
        observed=(
            f"rollout_rows={len(rollout_rows)};"
            f"wrong_family_rows={len(bad_rows)}"
        ),
        pass_detail="Attention rollout remains separated from transformer relevance.",
        fail_detail="Attention rollout rows were relabeled or collapsed.",
        next_action="restore attention_rollout rows to internal_routing family",
    )


def _key_counts(rows: list[dict[str, str]]) -> Counter[tuple[str, ...]]:
    return Counter(_merge_key(row) for row in rows)


def _merge_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in MERGE_KEY_FIELDS)


def _missing_key_counts(
    expected: Counter[tuple[str, ...]],
    observed: Counter[tuple[str, ...]],
) -> Counter[tuple[str, ...]]:
    missing: Counter[tuple[str, ...]] = Counter()
    for key, count in expected.items():
        if observed[key] < count:
            missing[key] = count - observed[key]
    return missing


def _format_missing_keys(missing: Counter[tuple[str, ...]]) -> str:
    if not missing:
        return ""
    examples = [
        "|".join(key) + f" x{count}"
        for key, count in list(missing.items())[:5]
    ]
    suffix = "" if len(missing) <= 5 else f"; +{len(missing) - 5} more"
    return "Missing merge keys: " + "; ".join(examples) + suffix


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
        "observed": observed,
        "detail": pass_detail if passed else fail_detail,
        "next_action": "" if passed else next_action,
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


def _status_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
