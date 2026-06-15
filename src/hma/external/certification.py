"""Publication adapter certification and setup-blocker records."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hma.external.adapters import adapter_class_available
from hma.external.registry import ExternalModelRegistry, load_external_registry
from hma.utils.config import load_yaml
from hma.utils.paths import resolve_path


PUBLICATION_REGISTRY_SCHEMA = "hma.external.publication_registry.v1"
CERTIFICATION_SCHEMA = "hma.external.adapter_certification.v1"
SUPPORTED_INPUT_MODES = {
    "image_only",
    "task_conditioned",
    "gaze_history_conditioned",
    "scanpath",
    "foveated",
    "stochastic",
    "active_vision",
}


@dataclass(frozen=True)
class PublicationAdapterRegistry:
    path: Path
    models: dict[str, dict[str, Any]]


def load_publication_adapter_registry(
    path: str | Path = "configs/external_models/publication_registry.yaml",
) -> PublicationAdapterRegistry:
    registry_path = resolve_path(path)
    payload = load_yaml(registry_path)
    if payload.get("schema_version") != PUBLICATION_REGISTRY_SCHEMA:
        raise ValueError("Unsupported publication adapter registry schema")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Publication adapter registry must define models")
    for model_id, model in models.items():
        _validate_publication_entry(str(model_id), model)
    return PublicationAdapterRegistry(path=registry_path, models=dict(models))


def build_certification_records(
    *,
    publication_registry_path: str | Path = (
        "configs/external_models/publication_registry.yaml"
    ),
    runtime_registry_path: str | Path = "configs/external_models/registry.yaml",
) -> list[dict[str, Any]]:
    """Create one certification or structured blocker record per candidate."""
    publication = load_publication_adapter_registry(publication_registry_path)
    runtime = load_external_registry(runtime_registry_path)
    records = [
        _build_record(model_id, entry, runtime)
        for model_id, entry in sorted(publication.models.items())
    ]
    return records


def write_certification_records(
    records: list[dict[str, Any]],
    *,
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    jsonl = resolve_path(jsonl_path)
    summary = resolve_path(csv_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    fieldnames = [
        "model_id",
        "family",
        "runtime_model_id",
        "adapter_class",
        "certification_status",
        "paper_evidence_status",
        "input_modes",
        "behavioral_outputs",
        "latent_outputs",
        "resource_outputs",
        "setup_command",
        "report_command",
        "blocker_codes",
        "blocker_count",
    ]
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model_id": record["model_id"],
                    "family": record["family"],
                    "runtime_model_id": record.get("runtime_model_id", ""),
                    "adapter_class": record.get("adapter_class", ""),
                    "certification_status": record["certification_status"],
                    "paper_evidence_status": record["paper_evidence_status"],
                    "input_modes": "|".join(record["input_contract"]["modes"]),
                    "behavioral_outputs": "|".join(record["outputs"]["behavioral"]),
                    "latent_outputs": "|".join(record["outputs"]["latent"]),
                    "resource_outputs": "|".join(record["outputs"]["resource"]),
                    "setup_command": record["setup"]["command"],
                    "report_command": record["setup"]["report_command"],
                    "blocker_codes": "|".join(
                        blocker["code"] for blocker in record["blockers"]
                    ),
                    "blocker_count": len(record["blockers"]),
                }
            )
    return jsonl, summary


def update_scope_reset_adapter_tables(
    records: list[dict[str, Any]],
    *,
    comparability_path: str | Path,
    role_matrix_path: str | Path,
) -> None:
    """Synchronize certification status into the two publication scope tables."""
    by_model = {str(record["model_id"]): record for record in records}
    _update_csv_rows(
        resolve_path(comparability_path),
        lambda row: _updated_comparability_row(row, by_model),
    )
    _update_csv_rows(
        resolve_path(role_matrix_path),
        lambda row: _updated_role_row(row, by_model),
    )


def _build_record(
    model_id: str,
    entry: dict[str, Any],
    runtime: ExternalModelRegistry,
) -> dict[str, Any]:
    setup_kind = str(entry["setup_kind"])
    runtime_id = str(entry.get("runtime_model_id", ""))
    blockers: list[dict[str, str]] = []
    adapter_class = ""
    installation_stages: dict[str, bool] = {}
    setup_command = str(entry.get("setup_command", ""))
    report_command = str(entry.get("report_command", ""))
    if setup_kind == "builtin":
        if str(entry.get("implementation_status", "")) != "ready":
            blockers.append(
                _blocker(
                    "builtin_implementation_missing",
                    "adapter",
                    "Built-in control implementation is not publication-ready.",
                    str(entry.get("resolution", "implement and test the control")),
                )
            )
    elif setup_kind == "unavailable_source":
        blockers.append(
            _blocker(
                "source_release_unavailable",
                "source",
                "No official executable source/checkpoint API is currently registered.",
                str(entry.get("resolution", "identify and pin an official release")),
            )
        )
        if runtime_id:
            setup_command = setup_command or _setup_command(runtime_id)
            report_command = report_command or _report_command(runtime_id)
    else:
        if not runtime_id:
            blockers.append(
                _blocker(
                    "runtime_registry_entry_missing",
                    "registry",
                    "No runtime registry model is mapped to this publication candidate.",
                    "add a configs/external_models/registry.yaml entry",
                )
            )
        else:
            setup_command = setup_command or _setup_command(runtime_id)
            report_command = report_command or _report_command(runtime_id)
            try:
                model = runtime.model(runtime_id)
            except KeyError:
                blockers.append(
                    _blocker(
                        "runtime_registry_entry_missing",
                        "registry",
                        f"Runtime model '{runtime_id}' is not registered.",
                        "add and validate the runtime registry entry",
                    )
                )
            else:
                adapter_class = str(model["adapter"])
                blockers.extend(_runtime_blockers(runtime, model))
                installation = _installation_report(runtime, runtime_id)
                if installation:
                    installation_stages = {
                        str(key): bool(value)
                        for key, value in installation.get("stages", {}).items()
                    }
    if setup_kind == "builtin" and not blockers:
        status = "adapter_certified"
        paper_status = str(entry.get("paper_evidence_status", "diagnostic_only"))
    elif installation_stages.get("evidence_ready"):
        status = "adapter_certified"
        paper_status = "adapter_certified"
        blockers = []
    elif any(
        blocker["code"]
        in {"source_release_unavailable", "runtime_registry_entry_missing"}
        for blocker in blockers
    ):
        status = "setup_blocked"
        paper_status = str(entry.get("paper_evidence_status", "adapter_in_progress"))
    else:
        status = "setup_scaffold_ready"
        paper_status = str(entry.get("paper_evidence_status", "adapter_in_progress"))
    return {
        "schema_version": CERTIFICATION_SCHEMA,
        "model_id": model_id,
        "family": str(entry["family"]),
        "model_role": str(entry["model_role"]),
        "runtime_model_id": runtime_id,
        "adapter_class": adapter_class,
        "certification_status": status,
        "paper_evidence_status": paper_status,
        "input_contract": {
            "modes": [str(value) for value in entry["input_modes"]],
            "deterministic_condition": str(entry["deterministic_condition"]),
            "task_required": "task_conditioned" in entry["input_modes"],
            "gaze_history_required": (
                "gaze_history_conditioned" in entry["input_modes"]
            ),
            "stochastic_seed_required": "stochastic" in entry["input_modes"],
        },
        "outputs": {
            "behavioral": [str(value) for value in entry["behavioral_outputs"]],
            "latent": [str(value) for value in entry["latent_outputs"]],
            "resource": [str(value) for value in entry["resource_outputs"]],
        },
        "setup": {
            "kind": setup_kind,
            "command": setup_command,
            "report_command": report_command,
        },
        "installation_stages": installation_stages,
        "blockers": blockers,
    }


def _runtime_blockers(
    registry: ExternalModelRegistry,
    model: dict[str, Any],
) -> list[dict[str, str]]:
    model_id = str(model["id"])
    blockers: list[dict[str, str]] = []
    source = dict(model["source"])
    if source.get("commit") == "PIN_REQUIRED":
        blockers.append(
            _blocker(
                "source_commit_unpinned",
                "source",
                "Official source revision is not pinned.",
                "record a tested immutable source commit",
            )
        )
    if str(source.get("repository", "")).startswith("UNRESOLVED_"):
        blockers.append(
            _blocker(
                "source_release_unavailable",
                "source",
                "Official executable source repository is unresolved.",
                "identify the official repository and immutable revision",
            )
        )
    license_name = str(source.get("license", "")).lower()
    if any(marker in license_name for marker in ("audit_required", "unknown")):
        blockers.append(
            _blocker(
                "license_audit_required",
                "license",
                "Source license is not publication-cleared.",
                "record and review the official source/checkpoint license",
            )
        )
    environment = registry.environment_path(model_id)
    if not environment.is_file():
        blockers.append(
            _blocker(
                "environment_manifest_missing",
                "environment",
                f"Environment manifest is missing: {environment}",
                "add a pinned environment manifest",
            )
        )
    environment_lock = resolve_path(
        f"configs/external_models/environment_locks/{model_id}.txt"
    )
    if not environment_lock.is_file():
        blockers.append(
            _blocker(
                "environment_lock_missing",
                "environment",
                "Installed environment lock is missing.",
                "run the setup scaffold and commit the explicit environment lock",
            )
        )
    checkpoint = dict(model["checkpoint"])
    if not checkpoint.get("filename"):
        blockers.append(
            _blocker(
                "checkpoint_metadata_missing",
                "checkpoint",
                "Checkpoint filename or snapshot target is unresolved.",
                "record the official checkpoint location and local target",
            )
        )
    checkpoint_lock = resolve_path(
        f"configs/external_models/checkpoint_locks/{model_id}.json"
    )
    if not checkpoint_lock.is_file():
        blockers.append(
            _blocker(
                "checkpoint_lock_missing",
                "checkpoint",
                "Checkpoint hash lock is missing.",
                "download the official checkpoint and create its hash lock",
            )
        )
    adapter = str(model["adapter"])
    if not adapter_class_available(adapter):
        blockers.append(
            _blocker(
                "adapter_class_missing",
                "adapter",
                f"Adapter class is not importable: {adapter}",
                "implement the registered adapter class",
            )
        )
    if str(model.get("adapter_status", "")) != "implemented":
        blockers.append(
            _blocker(
                "adapter_execution_pending",
                "adapter",
                f"Adapter status is {model.get('adapter_status', 'unspecified')}.",
                "complete feature/output hooks and pass adapter smoke validation",
            )
        )
    report = _installation_report(registry, model_id)
    if not report or not report.get("stages", {}).get("smoke_passed"):
        blockers.append(
            _blocker(
                "adapter_smoke_missing",
                "smoke",
                "No current source/environment/checkpoint-bound smoke pass exists.",
                "run the registered adapter smoke after setup",
            )
        )
    return blockers


def _installation_report(
    registry: ExternalModelRegistry,
    model_id: str,
) -> dict[str, Any] | None:
    path = registry.workspace_path("reports") / f"{model_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_publication_entry(model_id: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise TypeError(f"Publication adapter '{model_id}' must be a mapping")
    required = {
        "family",
        "model_role",
        "setup_kind",
        "input_modes",
        "deterministic_condition",
        "behavioral_outputs",
        "latent_outputs",
        "resource_outputs",
        "paper_evidence_status",
    }
    missing = required - set(entry)
    if missing:
        raise ValueError(
            f"Publication adapter '{model_id}' is missing {sorted(missing)}"
        )
    modes = {str(value) for value in entry["input_modes"]}
    if not modes or not modes <= SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Publication adapter '{model_id}' has unsupported input modes: "
            f"{sorted(modes - SUPPORTED_INPUT_MODES)}"
        )
    for key in ("behavioral_outputs", "latent_outputs", "resource_outputs"):
        if not isinstance(entry[key], list):
            raise TypeError(f"Publication adapter '{model_id}' {key} must be a list")


def _blocker(code: str, stage: str, detail: str, resolution: str) -> dict[str, str]:
    return {
        "code": code,
        "stage": stage,
        "detail": detail,
        "resolution": resolution,
    }


def _setup_command(model_id: str) -> str:
    return (
        "python scripts/setup_external_model.py "
        f"--model {model_id} --install --download-checkpoint --smoke"
    )


def _report_command(model_id: str) -> str:
    return (
        "python scripts/setup_external_model.py "
        f"--model {model_id} --report-only"
    )


def _update_csv_rows(path: Path, update: Any) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [update(dict(row)) for row in reader]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _updated_comparability_row(
    row: dict[str, str],
    records: dict[str, dict[str, Any]],
) -> dict[str, str]:
    record = records.get(str(row.get("model_id", "")))
    if not record:
        return row
    blocker_codes = [blocker["code"] for blocker in record["blockers"]]
    return {
        **row,
        "adapter_implementation": record.get("adapter_class")
        or str(record["setup"]["kind"]),
        "certification_status": str(record["certification_status"]),
        "blocking_setup": "|".join(blocker_codes) or "publication_preflight",
    }


def _updated_role_row(
    row: dict[str, str],
    records: dict[str, dict[str, Any]],
) -> dict[str, str]:
    record = records.get(str(row.get("model_id", "")))
    if not record:
        return row
    status = str(record["certification_status"])
    current = {
        "adapter_certified": "ready_now",
        "setup_scaffold_ready": "setup_scaffold_ready",
        "setup_blocked": "availability_or_setup_blocked",
    }[status]
    blocker_codes = [blocker["code"] for blocker in record["blockers"]]
    return {
        **row,
        "current_implementation_status": current,
        "paper_evidence_status": str(record["paper_evidence_status"]),
        "certification_basis": (
            "publication_adapter_certification_record"
            if status == "adapter_certified"
            else "executable_setup_scaffold_and_machine_readable_blockers"
        ),
        "blocking_requirement": "|".join(blocker_codes) or "publication_preflight",
    }
