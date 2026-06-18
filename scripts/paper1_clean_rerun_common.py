"""Shared validation helpers for Paper 1 clean rerun scripts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


PUBLICATION_ROOT = resolve_path("outputs/paper1_publication_v0")
PREFLIGHT_ROOT = PUBLICATION_ROOT / "preflight"
DRY_RUN_ROOT = PREFLIGHT_ROOT / "dry_runs"
AUDIT_ROOT = PUBLICATION_ROOT / "audits"
LOG_ROOT = PUBLICATION_ROOT / "logs"

FORBIDDEN_INPUT_MARKERS = (
    "admission_panel",
    "_admission",
    "paper1_admission",
    "matrix_v1",
    "matrix_v2",
    "real_matrix_v2",
    "paper1_v1",
    "outputs/neural_subj",
    "outputs/neural_large_smoke",
    "outputs/real_matrix",
)

AXIS_ELIGIBILITY_COLUMNS = {
    "behavioral": "eligible_behavioral_evidence",
    "latent_neural_encoding": "eligible_latent_neural_encoding",
    "geometry": "eligible_latent_geometry",
    "efficiency_resource": "eligible_efficiency_resource_evidence",
}

AUDIT_FIELDNAMES = [
    "lane",
    "check_id",
    "status",
    "artifact",
    "path",
    "detail",
    "next_action",
]


class CleanRerunValidationError(RuntimeError):
    """Raised when clean rerun validation fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def yesno(value: bool) -> str:
    return "yes" if value else "no"


def pipe_join(values: Iterable[Any]) -> str:
    kept = [str(value) for value in values if str(value)]
    return "|".join(kept) if kept else "none"


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else " " for ch in text)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = resolve_path(path)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field, "")) for field in fieldnames})
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def load_clean_config(config_path: str | Path, *, lane: str) -> dict[str, Any]:
    path = resolve_path(config_path)
    reject_forbidden_path(path, f"{lane}.config")
    if not path.is_file():
        raise CleanRerunValidationError(f"Missing clean rerun config: {path}")
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise CleanRerunValidationError(f"Config must be a mapping: {path}")
    if str(payload.get("lane", lane)) != lane:
        raise CleanRerunValidationError(
            f"Config lane mismatch for {path}: expected {lane}, got {payload.get('lane')}"
        )
    return payload


def validate_contract(config: dict[str, Any], *, require_authorized: bool) -> dict[str, Any]:
    contract_path = resolve_path(config.get("contract") or config.get("publication_contract") or "configs/paper1_publication_contract.yaml")
    reject_forbidden_path(contract_path, "publication_contract")
    if not contract_path.is_file():
        raise CleanRerunValidationError(f"Missing publication contract: {contract_path}")
    contract = load_yaml(contract_path)
    root = resolve_path(contract.get("evidence", {}).get("publication_root", "outputs/paper1_publication_v0"))
    if root != PUBLICATION_ROOT:
        raise CleanRerunValidationError(
            f"Publication root mismatch: contract={root}, expected={PUBLICATION_ROOT}"
        )
    if require_authorized and contract.get("execution_authorized") is not True:
        raise CleanRerunValidationError(
            "Publication contract execution_authorized is not true"
        )
    return contract


def validate_execution_enabled(config: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    if config.get("execution_enabled") is not True:
        raise CleanRerunValidationError("Config execution_enabled is not true")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_forbidden_path(path: str | Path, label: str) -> None:
    text = str(path).replace("\\", "/").lower()
    matched = [marker for marker in FORBIDDEN_INPUT_MARKERS if marker in text]
    if matched:
        raise CleanRerunValidationError(
            f"{label} path uses forbidden legacy/admission marker {matched}: {path}"
        )


def require_publication_path(path: str | Path, label: str) -> Path:
    resolved = resolve_path(path)
    reject_forbidden_path(resolved, label)
    if not is_under(resolved, PUBLICATION_ROOT):
        raise CleanRerunValidationError(
            f"{label} must be under {PUBLICATION_ROOT}: {resolved}"
        )
    return resolved


def validate_input_path(path: str | Path, label: str, *, must_exist: bool = True) -> Path:
    resolved = resolve_path(path)
    reject_forbidden_path(resolved, label)
    if must_exist and not resolved.exists():
        raise CleanRerunValidationError(f"{label} does not exist: {resolved}")
    return resolved


def output_paths_from_config(config: dict[str, Any], keys: Iterable[str]) -> dict[str, Path]:
    outputs = dict(config.get("outputs", {}))
    resolved: dict[str, Path] = {}
    for key in keys:
        if key not in outputs:
            raise CleanRerunValidationError(f"Missing output path in config.outputs: {key}")
        resolved[key] = require_publication_path(outputs[key], f"outputs.{key}")
    return resolved


def certification_rows_by_model() -> dict[str, dict[str, str]]:
    path = validate_input_path(
        "outputs/paper1_publication_v0/preflight/model_certification_summary.csv",
        "model_certification_summary",
    )
    rows = read_csv(path)
    return {row["model_id"]: row for row in rows if row.get("model_id")}


def eligible_model_rows(axis: str) -> list[dict[str, str]]:
    if axis not in AXIS_ELIGIBILITY_COLUMNS:
        raise CleanRerunValidationError(f"Unknown eligibility axis: {axis}")
    eligibility_path = validate_input_path(
        "outputs/paper1_publication_v0/preflight/model_rerun_eligibility_table.csv",
        "model_rerun_eligibility_table",
    )
    certification = certification_rows_by_model()
    column = AXIS_ELIGIBILITY_COLUMNS[axis]
    rows = []
    for row in read_csv(eligibility_path):
        if row.get(column) != "yes":
            continue
        if row.get("paper_evidence_status") != "adapter_certified" and row.get(column) != "yes":
            continue
        model_id = row.get("model_id", "")
        cert = certification.get(model_id, {})
        merged = {**cert, **row}
        if merged.get("certification_status") != "adapter_certified":
            raise CleanRerunValidationError(
                f"Eligible model is not adapter_certified for {axis}: {model_id}"
            )
        rows.append(merged)
    if not rows:
        raise CleanRerunValidationError(f"No eligible certified models for {axis}")
    return rows


def model_identity(row: dict[str, str]) -> dict[str, str]:
    return {
        "model_id": row.get("model_id", ""),
        "family": row.get("family", ""),
        "model_role": row.get("model_role", ""),
        "runtime_model_id": row.get("runtime_model_id", ""),
        "source_or_package": row.get("source_or_package", ""),
        "source_revision_or_version": row.get("source_revision_or_version", ""),
        "checkpoint_or_weights": row.get("checkpoint_or_weights", ""),
        "checkpoint_sha256_or_policy": row.get("checkpoint_sha256_or_policy", ""),
        "paper_evidence_status": row.get("paper_evidence_status", ""),
    }


def audit_row(
    *,
    lane: str,
    check_id: str,
    status: str,
    artifact: str,
    path: str | Path,
    detail: str,
    next_action: str = "none",
) -> dict[str, str]:
    return {
        "lane": lane,
        "check_id": check_id,
        "status": status,
        "artifact": artifact,
        "path": str(path),
        "detail": detail,
        "next_action": next_action,
    }


def write_lane_audit(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_csv(path, rows, AUDIT_FIELDNAMES)


def write_failure_log(lane: str, exc: BaseException) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"{lane}_clean_rerun_failure.log"
    path.write_text(
        "\n".join(
            [
                f"generated_at={utc_now()}",
                f"lane={lane}",
                f"error_type={type(exc).__name__}",
                f"error_message={exc}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def dry_run_report_path(lane: str) -> Path:
    return DRY_RUN_ROOT / f"{lane}_dry_run.json"


def dry_run_plan_path(lane: str) -> Path:
    return DRY_RUN_ROOT / f"{lane}_planned_rows.csv"


def write_dry_run_report(
    *,
    lane: str,
    config_path: str | Path,
    expected_outputs: dict[str, Path],
    planned_rows: list[dict[str, Any]],
    audit_path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    plan_path = dry_run_plan_path(lane)
    write_csv(plan_path, planned_rows)
    payload = {
        "schema_version": "hma.paper1.clean_rerun_dry_run.v1",
        "lane": lane,
        "status": "dry_run_passed",
        "generated_at": utc_now(),
        "config_path": str(resolve_path(config_path).relative_to(PROJECT_ROOT)),
        "expected_outputs": {
            key: str(path.relative_to(PROJECT_ROOT)) for key, path in expected_outputs.items()
        },
        "planned_row_count": len(planned_rows),
        "planned_rows_path": str(plan_path.relative_to(PROJECT_ROOT)),
        "audit_path": str(audit_path.relative_to(PROJECT_ROOT)),
        "forbidden_input_markers": list(FORBIDDEN_INPUT_MARKERS),
    }
    if extra:
        payload.update(extra)
    return write_json(dry_run_report_path(lane), payload)


def run_with_failure_log(lane: str, fn: Any) -> int:
    try:
        return int(fn())
    except Exception as exc:
        path = write_failure_log(lane, exc)
        print(f"{lane} failed: {exc}", file=sys.stderr)
        print(f"failure_log={path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        return 2
