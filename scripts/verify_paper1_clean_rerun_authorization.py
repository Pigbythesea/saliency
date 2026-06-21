"""Verify Paper 1 clean-rerun authorization against concrete artifacts.

This script is intentionally stricter than the model/setup preflight. It checks
the regenerated authorization claim against the frozen contract, config flags,
DeepGaze tensor validations, checkpoint locks, expected outputs, and clean-run
script availability before any publication-root clean rerun is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.artifacts import validate_external_artifact  # noqa: E402
from hma.utils.config import load_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


PUBLICATION_ROOT = resolve_path("outputs/paper1_publication_v0")
PREFLIGHT_ROOT = PUBLICATION_ROOT / "preflight"
AUDIT_ROOT = PUBLICATION_ROOT / "audits"

AUTHORIZATION_TABLE = PREFLIGHT_ROOT / "authorization_verification_table.csv"
EXECUTION_TABLE = PREFLIGHT_ROOT / "execution_path_verification_table.csv"
CLUSTER_COMMANDS = PREFLIGHT_ROOT / "user_run_cluster_commands.md"
IMPORT_PLAN = PREFLIGHT_ROOT / "postrun_import_validation_plan.md"
CLEAN_AUDIT = AUDIT_ROOT / "clean_rerun_audit.csv"

AUTH_FIELDNAMES = [
    "claim_id",
    "source_artifact",
    "claim",
    "verification_method",
    "expected",
    "observed",
    "status",
    "repair_or_next_action",
]
EXEC_FIELDNAMES = [
    "lane",
    "config_path",
    "runner_script",
    "expected_output",
    "config_status",
    "runner_status",
    "dry_run_status",
    "dry_run_report",
    "publication_root_status",
    "admission_substitution_status",
    "execution_path_status",
    "local_verification_command_cmd",
    "cluster_full_command_cmd",
    "next_action",
]
AUDIT_FIELDNAMES = [
    "artifact",
    "path",
    "exists_now",
    "evidence_status",
    "contamination_check",
    "next_action",
]

USER_ACTION_FIELDNAMES = [
    "item_type",
    "item_id",
    "status",
    "next_step_owner",
    "error_log_path",
    "missing_external_requirement",
    "user_action_needed",
    "next_command_cmd",
]

CLEAN_RUNNER_REQUIREMENTS = {
    "primary_behavioral_latent_to_fixation_encoding": {
        "config": "configs/paper1_primary_behavioral_latent_fixation.yaml",
        "runner": "scripts/run_paper1_primary_behavioral_latent_fixation.py",
        "expected": "outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv",
    },
    "latent_neural_encoding": {
        "config": "configs/paper1_latent_neural_matrix.yaml",
        "runner": "scripts/run_paper1_clean_latent_neural_encoding.py",
        "expected": "outputs/paper1_publication_v0/neural_encoding/encoding_scores.csv",
    },
    "geometry": {
        "config": "configs/paper1_latent_neural_matrix.yaml",
        "runner": "scripts/run_paper1_clean_geometry.py",
        "expected": "outputs/paper1_publication_v0/geometry/geometry_scores.csv",
    },
    "efficiency_resource": {
        "config": "configs/paper1_efficiency_resource_rerun.yaml",
        "runner": "scripts/run_paper1_clean_efficiency_resource.py",
        "expected": "outputs/paper1_publication_v0/efficiency/efficiency_profiles.csv",
    },
    "cross_axis": {
        "config": "configs/paper1_cross_axis_assembly.yaml",
        "runner": "scripts/assemble_paper1_clean_cross_axis.py",
        "expected": "outputs/paper1_publication_v0/cross_axis/model_axis_scores.csv",
    },
}

FULL_RUN_BLOCKING_MARKERS = [
    "Full behavioral scoring is not launched",
    "Full latent neural encoding is cluster-backed and was not launched",
    "Full geometry computation is cluster-backed and was not launched",
    "Full efficiency/resource profiling was not launched",
]

DEEPGAZE_ARTIFACTS = {
    "deepgaze_iie": "outputs/paper1_publication_v0/external/deepgaze_iie_latent",
    "deepgaze_iii": "outputs/paper1_publication_v0/external/deepgaze_iii_conditional",
    "deepgaze_msdb": "outputs/paper1_publication_v0/external/deepgaze_msdb_latent",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when verification blocks clean execution.",
    )
    args = parser.parse_args(argv)

    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    authorization_rows = build_authorization_rows()
    execution_rows = build_execution_rows()
    audit_rows = build_clean_audit_rows()

    write_csv(AUTHORIZATION_TABLE, authorization_rows, AUTH_FIELDNAMES)
    write_csv(EXECUTION_TABLE, execution_rows, EXEC_FIELDNAMES)
    write_csv(CLEAN_AUDIT, audit_rows, AUDIT_FIELDNAMES)
    write_cluster_commands(execution_rows)
    write_import_plan()

    blocked = any(row["status"] == "fail" for row in authorization_rows) or any(
        row["execution_path_status"] != "ready" for row in execution_rows
    )
    status = "blocked" if blocked else "ready"
    print(f"authorization_verification_status={status}")
    print(AUTHORIZATION_TABLE.relative_to(PROJECT_ROOT))
    print(EXECUTION_TABLE.relative_to(PROJECT_ROOT))
    print(CLUSTER_COMMANDS.relative_to(PROJECT_ROOT))
    print(IMPORT_PLAN.relative_to(PROJECT_ROOT))
    if args.strict and blocked:
        return 2
    return 0


def build_authorization_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.append(check_preflight_authorization())
    rows.append(check_user_action_checklist())
    rows.append(check_method_gate_table())
    rows.append(check_model_certification())
    rows.append(check_model_eligibility())
    rows.append(check_data_preflight())
    rows.append(check_expected_output_manifest())
    rows.append(check_checkpoint_locks())
    rows.append(check_deepgaze_artifacts())
    rows.append(check_publication_contract())
    rows.append(check_clean_configs())
    rows.append(check_clean_runners())
    rows.append(check_admission_runner_exclusion())
    return rows


def check_preflight_authorization() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "rerun_readiness_table.csv"
    rows = read_csv(path)
    row = next((item for item in rows if item.get("gate") == "first_clean_rerun_authorization"), {})
    observed = row.get("status", "missing")
    return auth_row(
        "rerun_readiness_authorization",
        path,
        "Regenerated preflight must explicitly authorize the first bounded clean rerun.",
        "Read first_clean_rerun_authorization from rerun_readiness_table.csv.",
        "authorized",
        observed,
        observed == "authorized",
        "Do not run clean lanes until this gate is authorized by regenerated preflight.",
    )


def check_user_action_checklist() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "user_action_checklist.csv"
    rows = read_csv(path)
    header_ok = csv_header(path) == USER_ACTION_FIELDNAMES
    blocked_rows = [row for row in rows if row.get("status") not in {"", "ready", "authorized"}]
    observed = f"header_ok={yesno(header_ok)}; action_rows={len(rows)}; blocked_rows={len(blocked_rows)}"
    return auth_row(
        "user_action_checklist_closed",
        path,
        "User action checklist must be a valid CSV and contain no open action rows.",
        "Validate header and count blocked rows.",
        "header_ok=yes; blocked_rows=0",
        observed,
        header_ok and not blocked_rows,
        "Repair checklist generation, complete listed actions, and regenerate preflight.",
    )


def check_method_gate_table() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "method_rerun_readiness_table.csv"
    rows = read_csv(path)
    failing = [
        f"{row.get('gate')}:{row.get('check_id')}={row.get('rerun_gate_status')}"
        for row in rows
        if row.get("rerun_gate_status") != "pass"
    ]
    observed = "failing=" + pipe_join(failing) + f"; rows={len(rows)}"
    return auth_row(
        "method_gates_pass",
        path,
        "Behavioral, latent neural, geometry, efficiency, and cross-axis method gates must pass or be accepted with explicit limitations.",
        "Check rerun_gate_status values.",
        "all pass",
        observed,
        bool(rows) and not failing,
        "Repair failed method-gate row or keep rerun blocked.",
    )


def check_model_certification() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "model_certification_summary.csv"
    rows = read_csv(path)
    unresolved = [
        row.get("model_id", "")
        for row in rows
        if row.get("certification_status") != "adapter_certified"
        and row.get("paper_evidence_status") != "diagnostic_only"
    ]
    certified = sum(1 for row in rows if row.get("certification_status") == "adapter_certified")
    observed = f"certified={certified}; unresolved={pipe_join(unresolved)}"
    return auth_row(
        "model_certification_closed",
        path,
        "No non-diagnostic publication candidate may remain uncertified.",
        "Scan certification_status and paper_evidence_status.",
        "unresolved=none",
        observed,
        bool(rows) and not unresolved,
        "Repair only the failed model claim, then regenerate preflight.",
    )


def check_model_eligibility() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "model_rerun_eligibility_table.csv"
    rows = read_csv(path)
    blocked = [
        f"{row.get('model_id')}:{row.get('blocking_requirement')}"
        for row in rows
        if row.get("blocking_requirement") not in {"", "none"}
    ]
    observed = f"eligible_rows={len(rows)}; blocked={pipe_join(blocked)}"
    return auth_row(
        "model_eligibility_has_no_blockers",
        path,
        "Eligibility table must contain no active blocking requirements.",
        "Scan blocking_requirement.",
        "blocked=none",
        observed,
        bool(rows) and not blocked,
        "Repair blocking eligibility rows and regenerate preflight.",
    )


def check_data_preflight() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "data_preflight.csv"
    rows = read_csv(path)
    failures = []
    for row in rows:
        data_path = resolve_path(row.get("path", ""))
        if row.get("status") != "ready" or row.get("exists") != "yes" or not data_path.is_file():
            failures.append(row.get("data_object", "unknown"))
    observed = f"rows={len(rows)}; failures={pipe_join(failures)}"
    return auth_row(
        "data_preflight_ready",
        path,
        "All data manifests required by the bounded clean rerun must exist and be ready.",
        "Check data_preflight rows and local paths.",
        "failures=none",
        observed,
        bool(rows) and not failures,
        "Repair data manifest paths/counts and regenerate preflight.",
    )


def check_expected_output_manifest() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "expected_outputs_manifest.csv"
    rows = read_csv(path)
    required = {
        "method_preflight",
        "model_setup",
        "model_certification",
        "model_eligibility",
        "data_preflight",
        "checkpoint_cache",
        "environment_preflight",
        "roi_stream_grouping",
        "roi_availability",
        "rerun_readiness",
        "authorization_verification",
        "execution_path_verification",
        "user_run_cluster_commands",
        "postrun_import_validation_plan",
        "primary_behavioral_latent_fixation",
        "primary_behavioral_latent_fixation_image_scores",
        "primary_behavioral_latent_fixation_feature_reduction",
        "primary_behavioral_latent_fixation_readout_selection",
        "legacy_behavioral_pipeline_exclusion_audit",
        "clean_neural_encoding",
        "clean_geometry",
        "clean_efficiency",
        "clean_cross_axis",
    }
    present = {row.get("artifact", "") for row in rows}
    missing = sorted(required - present)
    prerun_missing = [
        row.get("artifact", "")
        for row in rows
        if row.get("artifact_status") == "pre_rerun_required"
        and row.get("exists_now") != "yes"
        and row.get("artifact") not in {
            "authorization_verification",
            "execution_path_verification",
            "user_run_cluster_commands",
            "postrun_import_validation_plan",
        }
    ]
    observed = f"manifest_rows={len(rows)}; missing_entries={pipe_join(missing)}; missing_prerun={pipe_join(prerun_missing)}"
    return auth_row(
        "expected_output_manifest_complete",
        path,
        "Expected-output manifest must enumerate pre-rerun verification artifacts and post-rerun clean outputs.",
        "Check required artifact entries and pre-rerun existence flags.",
        "missing_entries=none; missing_prerun=none",
        observed,
        bool(rows) and not missing and not prerun_missing,
        "Regenerate preflight with verification artifact entries.",
    )


def check_checkpoint_locks() -> dict[str, str]:
    path = PREFLIGHT_ROOT / "checkpoint_cache_preflight.csv"
    rows = read_csv(path)
    runtime_by_model = {
        row.get("model_id", ""): row.get("runtime_model_id", "") or row.get("model_id", "")
        for row in read_csv(PREFLIGHT_ROOT / "model_certification_summary.csv")
    }
    failures = []
    checked = 0
    for row in rows:
        model_id = row.get("model_id", "")
        expected_hash = row.get("sha256_or_policy", "")
        if row.get("checkpoint_or_weights") == "not_required":
            continue
        lock_id = runtime_by_model.get(model_id, model_id)
        lock_path = resolve_path(f"configs/external_models/checkpoint_locks/{lock_id}.json")
        if not lock_path.is_file():
            failures.append(f"{model_id}:missing_lock:{lock_id}")
            continue
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "hma.external.checkpoint_lock.v1":
            failures.append(f"{model_id}:bad_lock_schema")
            continue
        if expected_hash and payload.get("sha256") != expected_hash:
            failures.append(f"{model_id}:sha256_mismatch")
            continue
        if lock_id.startswith("deepgaze_") and not deepgaze_lock_components_exist(payload):
            failures.append(f"{model_id}:deepgaze_cache_component_missing")
            continue
        checked += 1
    observed = f"checked_locks={checked}; failures={pipe_join(failures)}"
    return auth_row(
        "checkpoint_cache_locks_valid",
        path,
        "Checkpoint/cache rows must be backed by checkpoint lock files and actual DeepGaze cache components.",
        "Compare preflight hashes with checkpoint_locks and resolve DeepGaze cache component paths.",
        "failures=none",
        observed,
        bool(rows) and not failures,
        "Repair checkpoint/cache preflight rows or rerun model certification.",
    )


def deepgaze_lock_components_exist(payload: dict[str, Any]) -> bool:
    for file_record in payload.get("files", []):
        logical = str(file_record.get("path", ""))
        if logical.startswith("torch_hub/checkpoints/"):
            candidate = torch_hub_checkpoint_dir() / Path(logical).name
        elif logical.startswith("clip_cache/"):
            candidate = Path.home() / ".cache" / "clip" / Path(logical).name
        elif logical.startswith("data/"):
            candidate = PROJECT_ROOT / logical
        else:
            candidate = PROJECT_ROOT / logical
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return False
    return True


def torch_hub_checkpoint_dir() -> Path:
    try:
        import torch  # type: ignore

        return Path(torch.hub.get_dir()) / "checkpoints"
    except Exception:
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            return Path(torch_home) / "hub" / "checkpoints"
        return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def check_deepgaze_artifacts() -> dict[str, str]:
    failures = []
    checked = []
    for model_id, relative in DEEPGAZE_ARTIFACTS.items():
        root = resolve_path(relative)
        validation_path = root / "determinism_validation.json"
        if not validation_path.is_file():
            failures.append(f"{model_id}:missing_validation")
            continue
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "deterministic":
            failures.append(f"{model_id}:validation_status={validation.get('status')}")
            continue
        try:
            artifact = validate_external_artifact(root, verify_hashes=True)
        except Exception as exc:
            failures.append(f"{model_id}:artifact_validation={type(exc).__name__}")
            continue
        if artifact.get("schema_version") != "hma.external.artifact.v1":
            failures.append(f"{model_id}:bad_artifact_schema")
            continue
        if artifact.get("not_valid_external_artifact"):
            failures.append(f"{model_id}:placeholder_manifest")
            continue
        checked.append(f"{model_id}:images={artifact.get('num_images')}")
    observed = f"checked={pipe_join(checked)}; failures={pipe_join(failures)}"
    return auth_row(
        "deepgaze_tensor_exports_validated",
        "outputs/paper1_publication_v0/external/deepgaze_*",
        "DeepGaze validation files must prove deterministic non-placeholder tensor exports.",
        "Read determinism_validation.json and validate external artifact hashes.",
        "failures=none",
        observed,
        len(checked) == len(DEEPGAZE_ARTIFACTS) and not failures,
        "Rerun scripts/export_deepgaze_repair_tensors.py for failed DeepGaze rows.",
    )


def check_publication_contract() -> dict[str, str]:
    path = resolve_path("configs/paper1_publication_contract.yaml")
    payload = load_yaml(path) if path.is_file() else {}
    observed = f"execution_authorized={payload.get('execution_authorized', 'missing')}"
    return auth_row(
        "publication_contract_execution_authorized",
        path,
        "Frozen publication contract must allow execution before clean publication-root evidence is generated.",
        "Read execution_authorized from configs/paper1_publication_contract.yaml.",
        "execution_authorized=True",
        observed,
        payload.get("execution_authorized") is True,
        "Keep clean lanes blocked or update the frozen contract with an explicit authorization change.",
    )


def check_clean_configs() -> dict[str, str]:
    disabled = []
    missing = []
    for lane, spec in CLEAN_RUNNER_REQUIREMENTS.items():
        config = resolve_path(spec["config"])
        if not config.is_file():
            missing.append(f"{lane}:{spec['config']}")
            continue
        payload = load_yaml(config)
        if payload.get("execution_enabled") is not True:
            disabled.append(f"{lane}:execution_enabled={payload.get('execution_enabled')}")
    observed = f"missing={pipe_join(missing)}; disabled={pipe_join(disabled)}"
    return auth_row(
        "clean_rerun_configs_enabled",
        "configs/paper1_*clean*",
        "Every clean lane must have a config that is explicitly execution-enabled.",
        "Read execution_enabled in lane configs.",
        "missing=none; disabled=none",
        observed,
        not missing and not disabled,
        "Create missing lane configs or set execution_enabled only after clean runners are ready.",
    )


def check_clean_runners() -> dict[str, str]:
    missing = [
        f"{lane}:{spec['runner']}"
        for lane, spec in CLEAN_RUNNER_REQUIREMENTS.items()
        if not resolve_path(spec["runner"]).is_file()
    ]
    observed = "missing=" + pipe_join(missing)
    return auth_row(
        "clean_rerun_runner_scripts_exist",
        "scripts/",
        "Clean behavioral, latent neural, geometry, efficiency/resource, and cross-axis lanes must have dedicated runner scripts.",
        "Check required runner script paths and reject admission-panel substitution.",
        "missing=none",
        observed,
        not missing,
        "Implement dedicated clean runners before launching cluster jobs.",
    )


def check_admission_runner_exclusion() -> dict[str, str]:
    path = resolve_path("scripts/run_paper1_admission_panel.py")
    used_as_clean = any(
        spec["runner"] == "scripts/run_paper1_admission_panel.py"
        for spec in CLEAN_RUNNER_REQUIREMENTS.values()
    )
    observed = f"admission_runner_exists={yesno(path.is_file())}; used_as_clean={yesno(used_as_clean)}"
    return auth_row(
        "admission_panel_not_used_as_clean_substitute",
        path,
        "Admission-panel runner must not be used as a substitute for clean publication-root evidence.",
        "Inspect clean runner requirements.",
        "used_as_clean=no",
        observed,
        not used_as_clean,
        "Keep admission-panel artifacts as provenance only.",
    )


def build_execution_rows() -> list[dict[str, str]]:
    rows = []
    for lane, spec in CLEAN_RUNNER_REQUIREMENTS.items():
        config = resolve_path(spec["config"])
        runner = resolve_path(spec["runner"])
        expected = resolve_path(spec["expected"])
        config_status = config_status_for(config)
        runner_status = "ready" if runner.is_file() else "missing"
        implementation_status = full_run_implementation_status(runner)
        dry_run_status, dry_run_report = runner_dry_run_status(
            lane=lane,
            config=spec["config"],
            runner=spec["runner"],
            can_run=config_status == "ready" and runner_status == "ready",
        )
        root_status = (
            "publication_root"
            if str(expected).startswith(str(PUBLICATION_ROOT))
            else "outside_publication_root"
        )
        admission_status = (
            "rejected"
            if spec["runner"] == "scripts/run_paper1_admission_panel.py"
            else "not_used"
        )
        ready = (
            config_status == "ready"
            and runner_status == "ready"
            and implementation_status == "implemented"
            and dry_run_status == "pass"
            and root_status == "publication_root"
            and admission_status == "not_used"
        )
        local_command = (
            f".\\.venv\\Scripts\\python.exe {spec['runner']} --config {spec['config']} --dry-run"
            if runner.is_file()
            else ""
        )
        cluster_command = (
            f"ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python {spec['runner']} --config {spec['config']}\""
            if runner.is_file()
            else ""
        )
        next_action = "none"
        if implementation_status != "implemented":
            next_action = implementation_status
        elif not ready:
            next_action = "implement/enable dedicated clean lane before execution"
        rows.append(
            {
                "lane": lane,
                "config_path": spec["config"],
                "runner_script": spec["runner"],
                "expected_output": spec["expected"],
                "config_status": config_status,
                "runner_status": runner_status,
                "dry_run_status": dry_run_status,
                "dry_run_report": dry_run_report,
                "publication_root_status": root_status,
                "admission_substitution_status": admission_status,
                "execution_path_status": "ready" if ready else "blocked",
                "local_verification_command_cmd": local_command,
                "cluster_full_command_cmd": cluster_command,
                "next_action": next_action,
            }
        )
    return rows


def full_run_implementation_status(runner: Path) -> str:
    if not runner.is_file():
        return "missing_runner"
    text = runner.read_text(encoding="utf-8", errors="replace")
    markers = [marker for marker in FULL_RUN_BLOCKING_MARKERS if marker in text]
    if markers:
        return "full_run_not_implemented:" + pipe_join(markers)
    return "implemented"


def runner_dry_run_status(
    *,
    lane: str,
    config: str,
    runner: str,
    can_run: bool,
) -> tuple[str, str]:
    report = PREFLIGHT_ROOT / "dry_runs" / f"{lane}_dry_run.json"
    if not can_run:
        return "not_run", str(report.relative_to(PROJECT_ROOT))
    command = [sys.executable, runner, "--config", config, "--dry-run"]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if result.returncode != 0:
        failure = PREFLIGHT_ROOT / "dry_runs" / f"{lane}_dry_run_failure.log"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(
            "\n".join(
                [
                    "command_cmd=" + " ".join(command),
                    f"exit_code={result.returncode}",
                    "stdout:",
                    result.stdout,
                    "stderr:",
                    result.stderr,
                ]
            ),
            encoding="utf-8",
        )
        return f"fail:{failure.relative_to(PROJECT_ROOT)}", str(report.relative_to(PROJECT_ROOT))
    if not report.is_file():
        return "fail:missing_dry_run_report", str(report.relative_to(PROJECT_ROOT))
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "fail:invalid_dry_run_report_json", str(report.relative_to(PROJECT_ROOT))
    if payload.get("status") != "dry_run_passed":
        return f"fail:status={payload.get('status')}", str(report.relative_to(PROJECT_ROOT))
    return "pass", str(report.relative_to(PROJECT_ROOT))


def config_status_for(path: Path) -> str:
    if not path.is_file():
        return "missing"
    payload = load_yaml(path)
    if payload.get("execution_enabled") is True:
        return "ready"
    return f"disabled:execution_enabled={payload.get('execution_enabled')}"


def build_clean_audit_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(PREFLIGHT_ROOT / "expected_outputs_manifest.csv"):
        if row.get("artifact_status") != "expected_after_authorized_rerun":
            continue
        path = resolve_path(row.get("path", ""))
        exists = path.is_file() or path.is_dir()
        rows.append(
            {
                "artifact": row.get("artifact", ""),
                "path": row.get("path", ""),
                "exists_now": yesno(exists),
                "evidence_status": "clean_publication_evidence_pending" if not exists else "exists_requires_postrun_validation",
                "contamination_check": "not_merged_from_admission_or_legacy",
                "next_action": "run verified clean lane" if not exists else "validate rows and provenance before interpretation",
            }
        )
    return rows


def write_cluster_commands(execution_rows: list[dict[str, str]]) -> None:
    ready_commands = (
        [
            "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && bash outputs/paper1_publication_v0/preflight/cluster_jobs/full/submit_clean_full.sh\""
        ]
        if all(row["execution_path_status"] == "ready" for row in execution_rows)
        else []
    )
    blocked_lanes = [
        row["lane"] for row in execution_rows if row["execution_path_status"] != "ready"
    ]
    lines = [
        "# User-Run Cluster Commands",
        "",
        "These are cmd.exe-compatible commands. They intentionally do not use PowerShell.",
        "",
        "The cluster environment is `/scratch/tshu2/zzhan330/miniconda3` with `conda activate hma`.",
        "",
        "Run full clean jobs only after `authorization_verification_table.csv` and `execution_path_verification_table.csv` contain only passing/ready rows. Do not substitute `scripts/run_paper1_admission_panel.py` for clean publication-root evidence.",
        "",
        "## 1. Sync Workspace",
        "",
        "```cmd",
        "cd /d D:\\Git\\saliency",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"mkdir -p /scratch/tshu2/zzhan330/saliency\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && git pull\"",
        "scp configs\\paper1_primary_behavioral_latent_fixation.yaml zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/configs/",
        "scp configs\\paper1_latent_neural_matrix.yaml zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/configs/",
        "scp configs\\paper1_efficiency_resource_rerun.yaml zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/configs/",
        "scp configs\\paper1_cross_axis_assembly.yaml zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/configs/",
        "scp configs\\paper1_publication_contract.yaml zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/configs/",
        "scp scripts\\generate_paper1_publication_preflight.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\verify_paper1_clean_rerun_authorization.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\paper1_clean_rerun_common.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\run_paper1_primary_behavioral_latent_fixation.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\run_paper1_clean_latent_neural_encoding.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\run_paper1_clean_geometry.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\run_paper1_clean_efficiency_resource.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\profile_external_model_efficiency.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\assemble_paper1_clean_cross_axis.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp src\\hma\\behavioral\\latent_fixation.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/src/hma/behavioral/",
        "scp src\\hma\\behavioral\\__init__.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/src/hma/behavioral/",
        "scp src\\hma\\experiments\\neural_alignment.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/src/hma/experiments/",
        "scp src\\hma\\external\\adapters.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/src/hma/external/",
        "scp src\\hma\\saliency\\baselines.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/src/hma/saliency/",
        "scp scripts\\export_external_prediction_maps.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\prepare_paper1_clean_behavior_cell.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\run_paper1_clean_cell_job.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "scp scripts\\create_paper1_clean_cluster_jobs.py zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/scripts/",
        "```",
        "",
        "## 2. Environment Activation",
        "",
        "```cmd",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python --version\"",
        "```",
        "",
        "## 3. Cache And Checkpoint Verification",
        "",
        "```cmd",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/generate_paper1_publication_preflight.py\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/verify_paper1_clean_rerun_authorization.py\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && cat outputs/paper1_publication_v0/preflight/authorization_verification_table.csv\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && cat outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv\"",
        "```",
        "",
        "## 4. Small Verification Job",
        "",
        "This job validates the clean runner dry-run paths only. It does not create clean publication-root evidence.",
        "",
        "```cmd",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/run_paper1_primary_behavioral_latent_fixation.py --dry-run\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/run_paper1_primary_behavioral_latent_fixation.py --profile local_smoke\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/run_paper1_clean_latent_neural_encoding.py --dry-run\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/run_paper1_clean_geometry.py --dry-run\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/run_paper1_clean_efficiency_resource.py --dry-run\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/assemble_paper1_clean_cross_axis.py --dry-run\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/verify_paper1_clean_rerun_authorization.py --strict\"",
        "```",
        "",
        "## 5. Generate Clean Cell Cluster Jobs",
        "",
        "These commands generate Slurm scripts and job tables only. They do not run the clean rerun.",
        "",
        "```cmd",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/create_paper1_clean_cluster_jobs.py --mode smoke --max-items 8\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/create_paper1_clean_cluster_jobs.py --mode full\"",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && cat outputs/paper1_publication_v0/preflight/cluster_jobs/smoke/clean_cluster_job_manifest.json && cat outputs/paper1_publication_v0/preflight/cluster_jobs/full/clean_cluster_job_manifest.json\"",
        "```",
        "",
        "## 6. Submit Smoke Verification Jobs",
        "",
        "This submits a tiny bounded cluster verification subset under `outputs/paper1_publication_v0/preflight/smoke_cells/`; it is not clean publication evidence.",
        "",
        "```cmd",
        "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && bash outputs/paper1_publication_v0/preflight/cluster_jobs/smoke/submit_clean_smoke.sh\"",
        "```",
        "",
        "## 7. Full Clean Job Commands",
        "",
    ]
    if ready_commands:
        lines.extend(["```cmd", *ready_commands, "```"])
    else:
        lines.extend(
            [
                f"Full clean job commands are blocked because these lanes have no ready execution path: {pipe_join(blocked_lanes)}.",
                "",
                "Do not run admission-panel commands as substitutes. Implement dedicated clean runners, regenerate preflight, rerun authorization verification, and only then add full job commands here.",
            ]
        )
    lines.extend(
        [
            "",
            "## 8. Log Monitoring",
            "",
            "Use these only after a verified full job command has been launched.",
            "",
            "```cmd",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && tail -n 100 outputs/paper1_publication_v0/logs/primary_behavioral_latent_to_fixation_encoding_clean_rerun_failure.log\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && tail -n 100 outputs/paper1_publication_v0/logs/neural_encoding_clean_rerun.log\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && tail -n 100 outputs/paper1_publication_v0/logs/geometry_clean_rerun.log\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && tail -n 100 outputs/paper1_publication_v0/logs/efficiency_clean_rerun.log\"",
            "```",
            "",
            "## 9. Output Verification",
            "",
        "```cmd",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"source /scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh && conda activate hma && cd /scratch/tshu2/zzhan330/saliency && python scripts/verify_paper1_clean_rerun_authorization.py\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && test -f outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv && test -f outputs/paper1_publication_v0/neural_encoding/encoding_scores.csv && test -f outputs/paper1_publication_v0/geometry/geometry_scores.csv && test -f outputs/paper1_publication_v0/efficiency/efficiency_profiles.csv && test -f outputs/paper1_publication_v0/cross_axis/model_axis_scores.csv\"",
            "```",
            "",
            "## 10. Copy Back",
            "",
            "```cmd",
            "cd /d D:\\Git\\saliency",
            "scp -r zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/outputs/paper1_publication_v0 outputs\\",
            "```",
            "",
            "## 11. Import Validation",
            "",
            "```cmd",
            "cd /d D:\\Git\\saliency",
            ".\\.venv\\Scripts\\python.exe scripts\\generate_paper1_publication_preflight.py",
            ".\\.venv\\Scripts\\python.exe scripts\\verify_paper1_clean_rerun_authorization.py",
            "type outputs\\paper1_publication_v0\\preflight\\authorization_verification_table.csv",
            "type outputs\\paper1_publication_v0\\preflight\\execution_path_verification_table.csv",
            "```",
            "",
        ]
    )
    CLUSTER_COMMANDS.write_text("\n".join(lines), encoding="utf-8")


def write_import_plan() -> None:
    lines = [
        "# Postrun Import Validation Plan",
        "",
        "Do not interpret scientific results during import validation.",
        "",
        "Required checks after user-run cluster outputs are copied back:",
        "",
        "1. Regenerate preflight with `cmd.exe` command `.\\.venv\\Scripts\\python.exe scripts\\generate_paper1_publication_preflight.py`.",
        "2. Rerun authorization verification with `.\\.venv\\Scripts\\python.exe scripts\\verify_paper1_clean_rerun_authorization.py`.",
        "3. Confirm `outputs\\paper1_publication_v0\\behavioral_latent_fixation\\fixation_encoding_scores.csv` exists and contains primary latent-fixation behavioral rows.",
        "4. Confirm `outputs\\paper1_publication_v0\\neural_encoding\\encoding_scores.csv` exists and retains subject/ROI/stream scope.",
        "5. Confirm `outputs\\paper1_publication_v0\\geometry\\geometry_scores.csv` exists and retains subject/ROI/stream scope.",
        "6. Confirm `outputs\\paper1_publication_v0\\efficiency\\efficiency_profiles.csv` and `resource_allocation_profiles.csv` exist or missingness is audited.",
        "7. Confirm `outputs\\paper1_publication_v0\\cross_axis\\model_axis_scores.csv` uses only publication-root inputs.",
        "8. Confirm no `_admission_panel`, Matrix V1, Matrix V2, legacy map/saliency/scanpath, or scaffold paths are merged into clean evidence.",
        "9. Update `outputs\\paper1_publication_v0\\audits\\clean_rerun_audit.csv` with exact missing/failing rows before any interpretation.",
        "",
    ]
    IMPORT_PLAN.write_text("\n".join(lines), encoding="utf-8")


def auth_row(
    claim_id: str,
    source_artifact: str | Path,
    claim: str,
    verification_method: str,
    expected: str,
    observed: str,
    passed: bool,
    repair_or_next_action: str,
) -> dict[str, str]:
    source = str(source_artifact)
    try:
        path = Path(source_artifact)
        if path.is_absolute():
            source = str(path.relative_to(PROJECT_ROOT))
    except (TypeError, ValueError):
        source = str(source_artifact)
    return {
        "claim_id": claim_id,
        "source_artifact": source,
        "claim": claim,
        "verification_method": verification_method,
        "expected": expected,
        "observed": observed,
        "status": "pass" if passed else "fail",
        "repair_or_next_action": "none" if passed else repair_or_next_action,
    }


def read_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = resolve_path(path)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def csv_header(path: str | Path) -> list[str]:
    resolved = resolve_path(path)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field, "")) for field in fieldnames})


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else " " for ch in text)


def yesno(value: bool) -> str:
    return "yes" if value else "no"


def pipe_join(values: Iterable[str]) -> str:
    kept = [str(value) for value in values if str(value)]
    return "|".join(kept) if kept else "none"


if __name__ == "__main__":
    raise SystemExit(main())
