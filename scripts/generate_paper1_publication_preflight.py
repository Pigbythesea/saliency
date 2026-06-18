"""Generate Paper 1 Publication Matrix V0 pre-rerun readiness artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.certification import (  # noqa: E402
    build_certification_records,
    update_scope_reset_adapter_tables,
    write_certification_records,
)
from hma.external.hashing import sha256_tree  # noqa: E402
from hma.external.registry import load_external_registry  # noqa: E402
from hma.utils.config import load_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


PUBLICATION_ROOT = resolve_path("outputs/paper1_publication_v0")
PREFLIGHT_ROOT = PUBLICATION_ROOT / "preflight"
ROI_ROOT = PUBLICATION_ROOT / "roi_stream"
EXTERNAL_ROOT = PUBLICATION_ROOT / "external"
SCOPE_ROOT = resolve_path("outputs/paper1_scope_reset")
SETUP_ATTEMPT_RECORDS = PREFLIGHT_ROOT / "setup_attempt_records.jsonl"

SUBJECTS = ["subj01", "subj02", "subj03", "subj04"]
PRF_ROIS = ["V1", "V2", "V3", "hV4"]
STREAM_ROIS = ["midventral", "ventral", "midlateral", "lateral", "midparietal", "parietal"]
STREAM_GROUPS = {
    "early_retinotopic": PRF_ROIS,
    "ventral_category": ["midventral", "ventral"],
    "lateral_object": ["midlateral", "lateral"],
    "dorsal_parietal_spatial_selection": ["midparietal", "parietal"],
    "floc_category": ["EBA", "FBA", "OFA", "FFA", "OWFA", "VWFA", "OPA", "PPA", "RSC"],
}
SCANPATH_OR_FOVEATED = {
    "deepgaze_iii",
    "scandiff_freeview",
    "scandiff_visual_search",
    "hat",
    "adaptivenn_deit_small",
}
CLEAN_RERUN_CONFIGS = {
    "behavioral": "configs/paper1_clean_behavioral_rerun.yaml",
    "latent_neural_encoding": "configs/paper1_latent_neural_matrix.yaml",
    "geometry": "configs/paper1_latent_neural_matrix.yaml",
    "efficiency_resource": "configs/paper1_efficiency_resource_rerun.yaml",
    "cross_axis": "configs/paper1_cross_axis_assembly.yaml",
}
CLEAN_RERUN_RUNNERS = {
    "behavioral": "scripts/run_paper1_clean_behavioral_rerun.py",
    "latent_neural_encoding": "scripts/run_paper1_clean_latent_neural_encoding.py",
    "geometry": "scripts/run_paper1_clean_geometry.py",
    "efficiency_resource": "scripts/run_paper1_clean_efficiency_resource.py",
    "cross_axis": "scripts/assemble_paper1_clean_cross_axis.py",
}
METHOD_READY_VALUES = {
    "ready",
    "accepted_with_explicit_limitation",
    "not_applicable_to_this_axis",
}
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
KNOWN_ROW_COUNTS = {
    "data/manifests/salicon_observer_annotations_manifest.csv": 893854,
    "data/manifests/coco_search18_manifest.csv": 74646,
    "data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv": 150172,
    "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv": 65592,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-registry",
        default="configs/external_models/publication_registry.yaml",
    )
    parser.add_argument(
        "--runtime-registry",
        default="configs/external_models/registry.yaml",
    )
    args = parser.parse_args()

    publication = load_yaml(resolve_path(args.publication_registry))
    runtime = load_external_registry(args.runtime_registry)
    records = build_certification_records(
        publication_registry_path=args.publication_registry,
        runtime_registry_path=args.runtime_registry,
    )
    write_certification_records(
        records,
        jsonl_path=SCOPE_ROOT / "adapter_certification_records.jsonl",
        csv_path=SCOPE_ROOT / "adapter_certification_summary.csv",
    )
    update_scope_reset_adapter_tables(
        records,
        comparability_path=SCOPE_ROOT / "model_adapter_comparability_table.csv",
        role_matrix_path=SCOPE_ROOT / "model_role_matrix.csv",
    )

    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    ROI_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)

    method_rows = method_readiness_rows()
    write_csv(PREFLIGHT_ROOT / "method_rerun_readiness_table.csv", method_rows)

    setup_attempts = load_setup_attempt_records()
    model_rows = model_certification_rows(
        publication["models"],
        runtime,
        records,
        setup_attempts=setup_attempts,
    )
    write_csv(PREFLIGHT_ROOT / "model_certification_summary.csv", model_rows)
    setup_rows = model_setup_attempt_rows(
        model_rows,
        records,
        setup_attempts=setup_attempts,
    )
    write_csv(PREFLIGHT_ROOT / "model_setup_attempts.csv", setup_rows)
    eligibility_rows = model_eligibility_rows(model_rows)
    write_csv(PREFLIGHT_ROOT / "model_rerun_eligibility_table.csv", eligibility_rows)

    scanpath_rows = [row for row in setup_rows if row["model_id"] in SCANPATH_OR_FOVEATED]
    write_csv(PREFLIGHT_ROOT / "scanpath_foveated_setup_attempts.csv", scanpath_rows)
    write_csv(
        PREFLIGHT_ROOT / "scanpath_foveated_certification_summary.csv",
        [row for row in model_rows if row["model_id"] in SCANPATH_OR_FOVEATED],
    )
    write_scanpath_or_foveated_manifest(
        EXTERNAL_ROOT / "scanpath_or_foveated_certified" / "manifest.json",
        model_rows=model_rows,
        setup_rows=scanpath_rows,
    )

    for model_id, path_name in [
        ("dynamicvit_deit_small_keep_0_7", "dynamicvit_setup_preflight.csv"),
        ("tome_deit_small_r13", "tome_setup_preflight.csv"),
    ]:
        write_csv(
            PREFLIGHT_ROOT / path_name,
            [row for row in model_rows if row["runtime_model_id"] == model_id],
        )
    write_csv(
        PREFLIGHT_ROOT / "efficient_model_resource_schema_audit.csv",
        efficient_resource_schema_rows(model_rows),
    )

    data_rows = data_preflight_rows()
    write_csv(PREFLIGHT_ROOT / "data_preflight.csv", data_rows)
    write_csv(PREFLIGHT_ROOT / "environment_preflight.csv", environment_rows(model_rows))
    write_csv(PREFLIGHT_ROOT / "checkpoint_cache_preflight.csv", checkpoint_rows(model_rows))
    write_csv(PREFLIGHT_ROOT / "expected_outputs_manifest.csv", expected_output_rows())

    stream_rows, manifest_rows, availability_rows = roi_stream_rows()
    write_csv(ROI_ROOT / "stream_roi_grouping_spec.csv", stream_rows)
    write_csv(ROI_ROOT / "roi_manifest_audit.csv", manifest_rows)
    write_csv(ROI_ROOT / "subject_roi_availability.csv", availability_rows)
    write_stream_config(availability_rows)

    write_deepgaze_hook_audit(model_rows)
    write_deepgaze_certification_placeholders(model_rows)

    readiness_rows = rerun_readiness_rows(
        method_rows=method_rows,
        model_rows=model_rows,
        data_rows=data_rows,
        availability_rows=availability_rows,
    )
    write_csv(PREFLIGHT_ROOT / "rerun_readiness_table.csv", readiness_rows)
    write_first_clean_rerun_plan(
        readiness_rows=readiness_rows,
        model_rows=model_rows,
        availability_rows=availability_rows,
    )
    write_user_action_checklist(setup_rows, readiness_rows)
    print(PREFLIGHT_ROOT)
    return 0


def method_readiness_rows() -> list[dict[str, str]]:
    rows = [
        method_row(
            "behavioral_evaluation",
            "map_metrics",
            "probabilistic log-likelihood, information gain, NSS, shuffled AUC, CC, SIM, KL, AUC Judd, AUC Borji",
            "src/hma/metrics/saliency_metrics.py",
            "ready",
            "Metric implementations and constant-map policies are present.",
            "none",
        ),
        method_row(
            "behavioral_evaluation",
            "behavioral_uncertainty",
            "image-cluster bootstrap, SALICON worker-within-image, COCO-Search18 subject-within-image/task uncertainty",
            "src/hma/behavioral/uncertainty.py",
            "ready",
            "Hierarchical interval interfaces are implemented.",
            "none",
        ),
        method_row(
            "behavioral_evaluation",
            "conditional_scanpath_interfaces",
            "cIG/cNSS/cAUC-compatible conditional maps, sequence score hooks, sampled scanpath schema, seed and stopping fields",
            "src/hma/behavioral/sequence.py; src/hma/external/adapters.py",
            "accepted_with_explicit_limitation",
            "Interfaces and DeepGaze III bridge are implemented; full evidence remains blocked until a certified scanpath/foveated model artifact exists.",
            "complete scanpath/foveated setup attempts or certify one candidate",
        ),
        method_row(
            "behavioral_evaluation",
            "empirical_prior_sensitivity",
            "declared provenance for center/spatial/task priors and planned empirical-prior sensitivity",
            "src/hma/saliency/baselines.py",
            "ready",
            "Center/random/task priors and leakage-safe empirical spatial prior are implemented.",
            "none",
        ),
        method_row(
            "latent_feature_neural_encoding",
            "leakage_safe_encoding",
            "PCA train-only fitting, ridge-alpha validation, layer selection on validation, raw/noise-normalized separation, nonfinite ceiling handling",
            "src/hma/neural/encoding.py; src/hma/experiments/neural_alignment.py",
            "ready",
            "Controlled frozen-feature PCA/ridge method is available for certified tensors.",
            "none",
        ),
        method_row(
            "latent_feature_neural_encoding",
            "spatial_readout_sensitivity",
            "voxel-specific or learned spatial readout sensitivity",
            "src/hma/neural/learned_readout.py; configs/experiments/neural_large_smoke",
            "accepted_with_explicit_limitation",
            "Sensitivity path exists as smoke/config target; broad publication sensitivity remains a planned follow-up gate.",
            "include learned spatial-readout sensitivity in stream admission if compute permits",
        ),
        method_row(
            "representational_geometry",
            "geometry_controls",
            "debiased CKA, biased CKA diagnostic, subset RSA, response permutations, image-resampling intervals",
            "src/hma/neural/geometry.py; src/hma/neural/rsa.py; scripts/compute_matched_geometry.py",
            "ready",
            "Geometry methods are accepted for publication rerun on certified latent tensors.",
            "none",
        ),
        method_row(
            "efficiency_resource_allocation",
            "matched_efficiency_schema",
            "latency, FLOP, memory, parameters, total-cost schema for pruning, merging, foveated, recurrent, diffusion, and scanpath models",
            "src/hma/external/adapters.py; src/hma/external/artifacts.py",
            "accepted_with_explicit_limitation",
            "Static/token metrics and total-cost schema are implemented; foveated/diffusion models need certified artifacts before comparison.",
            "certify one scanpath/foveated model or keep those rows blocked",
        ),
        method_row(
            "cross_axis_inference",
            "family_aware_missingness",
            "family/role coverage checks, leave-one-model, leave-one-family, family-block sensitivity, missingness reporting",
            "src/hma/analysis/cross_axis.py; scripts/summarize_paper1_matrix_v2_full.py",
            "accepted_with_explicit_limitation",
            "Readiness/missingness reporting is allowed; interpretation is blocked until same-model overlap and stream coverage exist.",
            "use rerun_readiness_table.csv before any clean rerun",
        ),
    ]
    return rows


def method_row(
    gate: str,
    check_id: str,
    requirement: str,
    code_path: str,
    status: str,
    evidence: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "gate": gate,
        "check_id": check_id,
        "requirement": requirement,
        "code_path": code_path,
        "status": status,
        "rerun_gate_status": "pass" if status in METHOD_READY_VALUES else "blocked",
        "evidence_or_limitation": evidence,
        "next_action": next_action,
    }


def model_certification_rows(
    publication_models: dict[str, dict[str, Any]],
    runtime: Any,
    records: list[dict[str, Any]],
    *,
    setup_attempts: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    record_by_model = {record["model_id"]: record for record in records}
    rows: list[dict[str, str]] = []
    for model_id, entry in sorted(publication_models.items()):
        record = record_by_model[model_id]
        runtime_id = str(entry.get("runtime_model_id", ""))
        runtime_model: dict[str, Any] = {}
        if runtime_id and str(entry.get("setup_kind")) != "builtin":
            try:
                runtime_model = runtime.model(runtime_id)
            except KeyError:
                runtime_model = {}
        report = read_installation_report(runtime, runtime_id)
        blockers = record.get("blockers", [])
        attempt = setup_attempts.get(runtime_id) or setup_attempts.get(model_id)
        if not blockers:
            attempt = None
        paper_status = paper_status_for_record(record, entry)
        command = setup_or_cert_command(model_id, runtime_id, entry)
        next_command = str((attempt or {}).get("next_command_cmd", "")) or (
            next_command_after_action(model_id, runtime_id, entry)
        )
        rows.append(
            {
                "model_id": model_id,
                "family": str(entry["family"]),
                "model_role": str(entry["model_role"]),
                "runtime_model_id": runtime_id,
                "source_or_package": source_identity(runtime_model, entry),
                "source_revision_or_version": source_revision(runtime_model),
                "source_license_or_access": source_license(runtime_model, entry),
                "environment_manifest_or_package": environment_identity(runtime_model),
                "install_path": install_path(runtime, runtime_id, entry),
                "install_command_cmd": command,
                "checkpoint_or_weights": checkpoint_identity(runtime_model),
                "checkpoint_download_url": checkpoint_url(runtime_model),
                "checkpoint_expected_path": checkpoint_path(runtime, runtime_model),
                "checkpoint_sha256_or_policy": checkpoint_hash_or_policy(
                    report,
                    runtime,
                    runtime_model,
                ),
                "preprocessing_contract": json.dumps(
                    runtime_model.get("preprocessing", {"kind": entry.get("setup_kind")}),
                    sort_keys=True,
                ),
                "setup_report_diagnostics": report_attempt_text(
                    attempt=attempt,
                    fallback=report_diagnostics(report),
                ),
                "deterministic_input_condition": str(entry["deterministic_condition"]),
                "input_modes": "|".join(str(value) for value in entry["input_modes"]),
                "latent_tensor_status": output_status(record, "latent"),
                "latent_tensor_layers": "|".join(record["outputs"]["latent"]),
                "behavioral_output_status": output_status(record, "behavioral"),
                "behavioral_output_type": "|".join(record["outputs"]["behavioral"]),
                "conditioning_fields": conditioning_fields(record),
                "scanpath_glimpse_export_status": scanpath_status(record),
                "resource_allocation_output_status": output_status(record, "resource"),
                "resource_allocation_outputs": "|".join(record["outputs"]["resource"]),
                "efficiency_metadata_status": efficiency_status(record, report),
                "local_tiny_certification_command_cmd": local_cert_command(model_id, runtime_id, entry),
                "cluster_readiness_status": cluster_status(record, runtime_model),
                "certification_status": str(record["certification_status"]),
                "paper_evidence_status": paper_status,
                "blocker_codes": "|".join(blocker["code"] for blocker in blockers),
                "blocker_details": " | ".join(blocker["detail"] for blocker in blockers),
                "user_action_needed": user_action(runtime, runtime_model, record, entry),
                "next_command_after_user_action_cmd": next_command,
            }
        )
    return rows


def model_setup_attempt_rows(
    model_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    *,
    setup_attempts: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    record_by_model = {record["model_id"]: record for record in records}
    rows: list[dict[str, str]] = []
    for row in model_rows:
        record = record_by_model[row["model_id"]]
        blockers = record.get("blockers", [])
        attempt = setup_attempts.get(row["runtime_model_id"]) or setup_attempts.get(
            row["model_id"]
        )
        if not blockers:
            attempt = None
        command = str(
            (attempt or {}).get("command_cmd")
            or row["local_tiny_certification_command_cmd"]
        )
        log_path = str((attempt or {}).get("log_path", ""))
        exit_code = str((attempt or {}).get("exit_code", ""))
        attempt_error = str((attempt or {}).get("error_summary", ""))
        missing = refined_missing_requirement(
            attempt=attempt,
            fallback=" | ".join(blocker["detail"] for blocker in blockers) or "none",
        )
        next_owner = (
            "none"
            if not blockers
            else str((attempt or {}).get("next_step_owner", ""))
            or next_step_owner_for(row, blockers)
        )
        next_command = str((attempt or {}).get("next_command_cmd", "")) or row[
            "next_command_after_user_action_cmd"
        ]
        log_text = report_attempt_text(
            attempt=attempt,
            fallback=row.get("setup_report_diagnostics")
            or blocker_log_text(record, blockers),
        )
        rows.append(
            {
                "model_id": row["model_id"],
                "family": row["family"],
                "certification_status": row["certification_status"],
                "paper_evidence_status": row["paper_evidence_status"],
                "source_or_package": row["source_or_package"],
                "source_revision_or_version": row["source_revision_or_version"],
                "source_license_or_access": row["source_license_or_access"],
                "checkpoint_download_url": row["checkpoint_download_url"],
                "checkpoint_expected_path": row["checkpoint_expected_path"],
                "command_attempted_cmd": command,
                "exit_code": exit_code,
                "error_or_log": log_text,
                "error_log_path": log_path,
                "missing_external_requirement": missing,
                "next_step_owner": next_owner,
                "user_action_needed": row["user_action_needed"],
                "next_command_after_user_action_cmd": next_command,
                "setup_record_source": (
                    "setup_attempt_records_plus_publication_registry"
                    if attempt
                    else "publication_registry_plus_installation_report"
                ),
            }
        )
    return rows


def load_setup_attempt_records() -> dict[str, dict[str, Any]]:
    if not SETUP_ATTEMPT_RECORDS.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in SETUP_ATTEMPT_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("runtime_model_id") or record.get("model_id") or "")
        if key:
            records[key] = record
    return records


def report_attempt_text(
    *,
    attempt: dict[str, Any] | None,
    fallback: str,
) -> str:
    if not attempt:
        return fallback
    pieces = []
    if attempt.get("log_path"):
        pieces.append(f"log: {attempt['log_path']}")
    if attempt.get("exit_code") not in (None, ""):
        pieces.append(f"exit_code: {attempt['exit_code']}")
    if attempt.get("error_summary"):
        pieces.append(str(attempt["error_summary"]))
    return " | ".join(pieces) or fallback


def refined_missing_requirement(
    *,
    attempt: dict[str, Any] | None,
    fallback: str,
) -> str:
    if not attempt:
        return fallback
    missing = str(attempt.get("missing_external_requirement", "")) or fallback
    log_path = attempt.get("log_path")
    if not log_path:
        return missing
    path = resolve_path(str(log_path))
    if not path.is_file():
        return missing
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return missing
    if "timeout after" in text:
        if (
            "huggingface.co" in text
            or "fetching 5 files:" in text
            or "fetching 8 files:" in text
            or "fetching 10 files:" in text
        ):
            return "public Hugging Face snapshot download did not complete before the recorded timeout"
        if "transaction starting" in text or "libmamba" in text or "resolving environment" in text:
            return "pinned external conda/micromamba environment build did not complete before the recorded timeout"
    if "gatedrepoerror" in text:
        return "official Hugging Face model access is gated and requires user authentication/access approval"
    if "unresolved_official_source" in text or "pin_required" in text:
        return "official executable source/checkpoint API is unresolved"
    if "download returned an html page" in text or "drive.google.com" in text:
        return "manual Google Drive checkpoint download is required"
    if "gnutls recv error" in text or "failed to connect to github.com" in text:
        return "official GitHub source clone/fetch failed with network/TLS errors"
    return missing


def next_step_owner_for(
    row: dict[str, str],
    blockers: list[dict[str, str]],
) -> str:
    text = " ".join(
        [
            row.get("checkpoint_download_url", ""),
            row.get("source_license_or_access", ""),
            " ".join(blocker["code"] for blocker in blockers),
            " ".join(blocker["detail"] for blocker in blockers),
        ]
    ).lower()
    user_markers = (
        "manual_gated",
        "gated",
        "unverified",
        "source_release_unavailable",
        "unresolved",
        "google drive",
        "drive.google",
    )
    if any(marker in text for marker in user_markers):
        return "user_required"
    if blockers:
        return "codex_implementable"
    return "none"


def model_eligibility_rows(model_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in model_rows:
        certified = row["certification_status"] == "adapter_certified"
        diagnostic = row["paper_evidence_status"] == "diagnostic_only"
        behavior_ready = certified and row["behavioral_output_status"] != "not_applicable"
        latent_ready = certified and row["latent_tensor_status"] != "not_applicable"
        resource_ready = certified and (
            row["resource_allocation_output_status"] != "not_applicable"
            or row["efficiency_metadata_status"] == "ready"
        )
        rows.append(
            {
                "model_id": row["model_id"],
                "family": row["family"],
                "model_role": row["model_role"],
                "eligible_behavioral_evidence": yesno(behavior_ready or diagnostic),
                "eligible_latent_neural_encoding": yesno(latent_ready and not diagnostic),
                "eligible_latent_geometry": yesno(latent_ready and not diagnostic),
                "eligible_task_search_or_scanpath": yesno(
                    behavior_ready and row["scanpath_glimpse_export_status"] == "ready"
                ),
                "eligible_efficiency_resource_evidence": yesno(resource_ready),
                "paper_evidence_status": row["paper_evidence_status"],
                "blocking_requirement": row["blocker_codes"] or "none",
            }
        )
    return rows


def efficient_resource_schema_rows(model_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for model_id in ("dynamicvit_deit_small_keep_0_7", "tome_deit_small_r13", "deepgaze_iii", "hat", "scandiff_freeview", "adaptivenn_deit_small"):
        matches = [row for row in model_rows if row["runtime_model_id"] == model_id or row["model_id"] == model_id]
        for row in matches:
            rows.append(
                {
                    "model_id": row["model_id"],
                    "resource_outputs": row["resource_allocation_outputs"],
                    "resource_allocation_output_status": row["resource_allocation_output_status"],
                    "efficiency_metadata_status": row["efficiency_metadata_status"],
                    "total_cost_schema": "total_cost_per_image_task" if "scanpath" in row["resource_allocation_outputs"] or "selected_glimpses" in row["resource_allocation_outputs"] else "per_image_static_or_token_count",
                    "diagnostic_allocation_label": "diagnostic_allocation_object",
                    "rerun_status": "ready" if row["certification_status"] == "adapter_certified" else "blocked_until_setup_certified",
                }
            )
    return rows


def data_preflight_rows() -> list[dict[str, str]]:
    paths = [
        ("SALICON", "free_viewing", "data/manifests/salicon_manifest.csv"),
        ("SALICON_observers", "free_viewing_worker_uncertainty", "data/manifests/salicon_observer_annotations_manifest.csv"),
        ("CAT2000", "free_viewing", "data/manifests/cat2000_manifest.csv"),
        ("COCO-Search18", "task_search", "data/manifests/coco_search18_manifest.csv"),
        ("NSD_Algonauts_subj01-04_PRF", "latent_neural_geometry", "data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv"),
        ("NSD_Algonauts_subj01_streams", "stream_scope", "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv"),
    ]
    rows = []
    for name, scope, relative in paths:
        path = resolve_path(relative)
        rows.append(
            {
                "data_object": name,
                "scope": scope,
                "path": relative,
                "exists": yesno(path.is_file()),
                "row_count": str(count_csv_rows(path, relative=relative) if path.is_file() else 0),
                "status": "ready" if path.is_file() else "blocked_missing_file",
                "user_action_needed": "none" if path.is_file() else f"place required data manifest at {relative}",
            }
        )
    return rows


def environment_rows(model_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in model_rows:
        environment_missing = (
            "environment_lock_missing" in row["blocker_codes"]
            or "environment_ready: integration work remains"
            in row.get("setup_report_diagnostics", "")
        )
        rows.append(
            {
                "model_id": row["model_id"],
                "runtime_model_id": row["runtime_model_id"],
                "environment_manifest_or_package": row["environment_manifest_or_package"],
                "environment_lock_status": "missing" if environment_missing else "ready",
                "environment_status": "blocked" if environment_missing else "ready",
                "next_command_cmd": row["next_command_after_user_action_cmd"],
            }
        )
    return rows


def checkpoint_rows(model_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in model_rows:
        lock_missing = (
            "checkpoint_lock_missing" in row["blocker_codes"]
            or "checkpoint_ready: integration work remains"
            in row.get("setup_report_diagnostics", "")
        )
        rows.append(
            {
                "model_id": row["model_id"],
                "checkpoint_or_weights": row["checkpoint_or_weights"],
                "download_url_or_source": row["checkpoint_download_url"],
                "expected_local_path": row["checkpoint_expected_path"],
                "sha256_or_policy": row["checkpoint_sha256_or_policy"],
                "cache_lock_status": "missing" if lock_missing else "ready_or_not_required",
                "user_action_needed": row["user_action_needed"] if lock_missing else "none",
                "next_command_cmd": row["next_command_after_user_action_cmd"],
            }
        )
    return rows


def expected_output_rows() -> list[dict[str, str]]:
    outputs = [
        ("method_preflight", PREFLIGHT_ROOT / "method_rerun_readiness_table.csv", "pre_rerun_required"),
        ("model_setup", PREFLIGHT_ROOT / "model_setup_attempts.csv", "pre_rerun_required"),
        ("model_certification", PREFLIGHT_ROOT / "model_certification_summary.csv", "pre_rerun_required"),
        ("model_eligibility", PREFLIGHT_ROOT / "model_rerun_eligibility_table.csv", "pre_rerun_required"),
        ("data_preflight", PREFLIGHT_ROOT / "data_preflight.csv", "pre_rerun_required"),
        ("checkpoint_cache", PREFLIGHT_ROOT / "checkpoint_cache_preflight.csv", "pre_rerun_required"),
        ("environment_preflight", PREFLIGHT_ROOT / "environment_preflight.csv", "pre_rerun_required"),
        ("roi_stream_grouping", ROI_ROOT / "stream_roi_grouping_spec.csv", "pre_rerun_required"),
        ("roi_availability", ROI_ROOT / "subject_roi_availability.csv", "pre_rerun_required"),
        ("rerun_readiness", PREFLIGHT_ROOT / "rerun_readiness_table.csv", "pre_rerun_required"),
        ("authorization_verification", PREFLIGHT_ROOT / "authorization_verification_table.csv", "pre_rerun_required"),
        ("execution_path_verification", PREFLIGHT_ROOT / "execution_path_verification_table.csv", "pre_rerun_required"),
        ("user_run_cluster_commands", PREFLIGHT_ROOT / "user_run_cluster_commands.md", "pre_rerun_required"),
        ("postrun_import_validation_plan", PREFLIGHT_ROOT / "postrun_import_validation_plan.md", "pre_rerun_required"),
        ("clean_behavioral", PUBLICATION_ROOT / "behavioral" / "aggregate.csv", "expected_after_authorized_rerun"),
        ("clean_neural_encoding", PUBLICATION_ROOT / "neural_encoding" / "encoding_scores.csv", "expected_after_authorized_rerun"),
        ("clean_geometry", PUBLICATION_ROOT / "geometry" / "geometry_scores.csv", "expected_after_authorized_rerun"),
        ("clean_efficiency", PUBLICATION_ROOT / "efficiency" / "efficiency_profiles.csv", "expected_after_authorized_rerun"),
        ("clean_cross_axis", PUBLICATION_ROOT / "cross_axis" / "model_axis_scores.csv", "expected_after_authorized_rerun"),
    ]
    return [
        {
            "artifact": name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "artifact_status": status,
            "exists_now": yesno(path.exists()),
        }
        for name, path, status in outputs
    ]


def roi_stream_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest_paths = [
        "data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv",
        "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv",
    ]
    manifest_rows = [manifest_audit_row(path) for path in manifest_paths]
    availability = availability_by_subject_roi(manifest_paths)
    stream_rows = []
    for stream, rois in STREAM_GROUPS.items():
        for roi in rois:
            available_subjects = [
                row["subject_id"]
                for row in availability
                if row["roi"] == roi and row["available"] == "yes"
            ]
            stream_rows.append(
                {
                    "stream": stream,
                    "roi": roi,
                    "roi_class": roi_class(roi),
                    "available_subjects": "|".join(sorted(available_subjects)),
                    "primary_manifest": manifest_for_roi(roi),
                    "availability_status": "available" if available_subjects else "unavailable",
                    "exclusion_reason_if_unavailable": ""
                    if available_subjects
                    else "no local subject manifest or mask/response directory found",
                }
            )
    return stream_rows, manifest_rows, availability


def manifest_audit_row(relative: str) -> dict[str, str]:
    path = resolve_path(relative)
    subjects: set[str] = set()
    rois: set[str] = set()
    rows = 0
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                rows += 1
                subjects.add(str(record.get("subject_id", "")))
                rois.add(str(record.get("roi", "")))
    return {
        "manifest_path": relative,
        "exists": yesno(path.is_file()),
        "row_count": str(rows),
        "subjects": "|".join(sorted(value for value in subjects if value)),
        "rois": "|".join(sorted(value for value in rois if value)),
        "status": "ready" if path.is_file() and rows else "blocked_missing_or_empty",
    }


def availability_by_subject_roi(manifest_paths: list[str]) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    manifests: dict[tuple[str, str], str] = {}
    response_examples: dict[tuple[str, str], str] = {}
    for relative in manifest_paths:
        path = resolve_path(relative)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                key = (str(record.get("subject_id", "")), str(record.get("roi", "")))
                if not all(key):
                    continue
                counts[key] += 1
                manifests.setdefault(key, relative)
                response_examples.setdefault(key, str(record.get("roi_response_path", "")))
    rows = []
    all_rois = PRF_ROIS + STREAM_ROIS + STREAM_GROUPS["floc_category"]
    for subject in SUBJECTS:
        for roi in all_rois:
            key = (subject, roi)
            available = counts.get(key, 0) > 0
            rows.append(
                {
                    "subject_id": subject,
                    "roi": roi,
                    "stream": stream_for_roi(roi),
                    "roi_class": roi_class(roi),
                    "available": yesno(available),
                    "row_count": str(counts.get(key, 0)),
                    "manifest_path": manifests.get(key, ""),
                    "example_response_path": response_examples.get(key, ""),
                    "exclusion_reason_if_unavailable": ""
                    if available
                    else unavailable_roi_reason(subject, roi),
                }
            )
    return rows


def write_stream_config(availability_rows: list[dict[str, str]]) -> None:
    available = defaultdict(list)
    for row in availability_rows:
        if row["available"] == "yes":
            available[row["subject_id"]].append(row["roi"])
    payload = {
        "schema_version": "hma.paper1.latent_neural_stream_admission.v0",
        "publication_contract": "configs/paper1_publication_contract.yaml",
        "publication_root": "outputs/paper1_publication_v0",
        "model_eligibility_table": "outputs/paper1_publication_v0/preflight/model_rerun_eligibility_table.csv",
        "roi_stream_grouping": "outputs/paper1_publication_v0/roi_stream/stream_roi_grouping_spec.csv",
        "subject_roi_availability": "outputs/paper1_publication_v0/roi_stream/subject_roi_availability.csv",
        "forbid_v1_only_summary": True,
        "bounded_stream_admission": {
            "split": "train",
            "max_images_per_subject_roi": 64,
            "required_model_status": "adapter_certified",
            "feature_source": "external_artifact_manifest",
            "analysis": ["latent_feature_neural_encoding", "latent_feature_geometry"],
        },
        "roi_streams": {
            "early_retinotopic": ["V1", "V2", "V3", "hV4"],
            "ventral_category": ["midventral", "ventral"],
            "lateral_object": ["midlateral", "lateral"],
            "dorsal_parietal_spatial_selection": ["midparietal", "parietal"],
            "floc_category": STREAM_GROUPS["floc_category"],
        },
        "subjects": {
            subject: sorted(rois)
            for subject, rois in sorted(available.items())
        },
        "missing_regions_record": "outputs/paper1_publication_v0/roi_stream/subject_roi_availability.csv",
    }
    write_yaml(resolve_path("configs/paper1_latent_neural_stream_admission.yaml"), payload)


def write_deepgaze_hook_audit(model_rows: list[dict[str, str]]) -> None:
    rows_by_model = {row["model_id"]: row for row in model_rows}
    path = PREFLIGHT_ROOT / "deepgaze_hook_audit.md"
    iie = rows_by_model.get("deepgaze_iie", {})
    iii = rows_by_model.get("deepgaze_iii", {})
    msdb = rows_by_model.get("deepgaze_msdb", {})
    content = "\n".join(
        [
            "# DeepGaze Hook Audit",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## DeepGaze IIE",
            "",
            f"- Adapter: `{iie.get('certification_status', '')}`",
            f"- Candidate latent hooks: `{iie.get('latent_tensor_layers', '')}`",
            f"- Behavioral output: `{iie.get('behavioral_output_type', '')}`",
            f"- User action: `{iie.get('user_action_needed', '')}`",
            "",
            "## DeepGaze III",
            "",
            f"- Adapter: `{iii.get('certification_status', '')}`",
            f"- Conditional hooks: `{iii.get('latent_tensor_layers', '')}`",
            f"- Behavioral outputs: `{iii.get('behavioral_output_type', '')}`",
            f"- Resource outputs: `{iii.get('resource_allocation_outputs', '')}`",
            f"- Command attempted: `{iii.get('local_tiny_certification_command_cmd', '')}`",
            f"- Blockers: `{iii.get('blocker_details', '') or 'none'}`",
            f"- Next command: `{iii.get('next_command_after_user_action_cmd', '')}`",
            "",
            "## DeepGaze MSDB",
            "",
            f"- Adapter: `{msdb.get('certification_status', '')}`",
            f"- Candidate latent hooks: `{msdb.get('latent_tensor_layers', '')}`",
            f"- Behavioral output: `{msdb.get('behavioral_output_type', '')}`",
            f"- User action: `{msdb.get('user_action_needed', '')}`",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_deepgaze_certification_placeholders(model_rows: list[dict[str, str]]) -> None:
    by_model = {row["model_id"]: row for row in model_rows}
    for model_id, directory in [
        ("deepgaze_iie", EXTERNAL_ROOT / "deepgaze_iie_latent"),
        ("deepgaze_iii", EXTERNAL_ROOT / "deepgaze_iii_conditional"),
        ("deepgaze_msdb", EXTERNAL_ROOT / "deepgaze_msdb_latent"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        row = by_model.get(model_id, {})
        write_csv(
            directory / "certification.csv",
            [
                {
                    "model_id": model_id,
                    "certification_status": row.get("certification_status", ""),
                    "paper_evidence_status": row.get("paper_evidence_status", ""),
                    "expected_manifest": str((directory / "manifest.json").relative_to(PROJECT_ROOT)),
                    "blocker_codes": row.get("blocker_codes", ""),
                    "user_action_needed": row.get("user_action_needed", ""),
                    "next_command_after_user_action_cmd": row.get("next_command_after_user_action_cmd", ""),
                }
            ],
        )
        manifest_path = directory / "manifest.json"
        if not existing_deepgaze_artifact_ready(manifest_path):
            write_deepgaze_export_manifest(manifest_path, model_id, row)


def existing_deepgaze_artifact_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("not_valid_external_artifact"):
        return False
    return payload.get("schema_version") == "hma.external.artifact.v1"


def write_deepgaze_export_manifest(
    path: Path,
    model_id: str,
    row: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    certified = row.get("certification_status") == "adapter_certified"
    payload = {
        "schema_version": "hma.external.deepgaze_preflight_manifest.v1",
        "artifact_name": model_id,
        "artifact_status": (
            "adapter_certified_export_pending"
            if certified
            else "blocked_until_adapter_certified"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "certification_status": row.get("certification_status", ""),
        "paper_evidence_status": row.get("paper_evidence_status", ""),
        "latent_tensor_layers": row.get("latent_tensor_layers", ""),
        "behavioral_output_type": row.get("behavioral_output_type", ""),
        "resource_allocation_outputs": row.get("resource_allocation_outputs", ""),
        "blocker_codes": row.get("blocker_codes", ""),
        "user_action_needed": row.get("user_action_needed", ""),
        "next_command_after_user_action_cmd": row.get("next_command_after_user_action_cmd", ""),
        "not_valid_external_artifact": True,
        "export_note": (
            "adapter certification is complete; tensor export has not been run"
            if certified
            else "adapter certification is required before tensor export"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def rerun_readiness_rows(
    *,
    method_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
    data_rows: list[dict[str, str]],
    availability_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    method_ready = all(row["status"] in METHOD_READY_VALUES for row in method_rows)
    data_ready = all(row["status"] == "ready" for row in data_rows)
    certified_latent = [
        row for row in model_rows
        if row["certification_status"] == "adapter_certified"
        and row["latent_tensor_status"] == "ready"
        and row["paper_evidence_status"] != "diagnostic_only"
    ]
    scanpath_certified = [
        row for row in model_rows
        if row["model_id"] in SCANPATH_OR_FOVEATED
        and row["certification_status"] == "adapter_certified"
        and row["scanpath_glimpse_export_status"] == "ready"
    ]
    dynamicvit_ready = any(
        row["runtime_model_id"] == "dynamicvit_deit_small_keep_0_7"
        and row["certification_status"] == "adapter_certified"
        for row in model_rows
    )
    tome_ready = any(
        row["runtime_model_id"] == "tome_deit_small_r13"
        and row["certification_status"] == "adapter_certified"
        for row in model_rows
    )
    early_ready = all(
        any(row["subject_id"] == subject and row["roi"] == roi and row["available"] == "yes" for row in availability_rows)
        for subject in SUBJECTS
        for roi in PRF_ROIS
    )
    stream_ready = any(
        row["subject_id"] == "subj01"
        and row["roi"] in STREAM_ROIS
        and row["available"] == "yes"
        for row in availability_rows
    )
    unresolved_models = [
        row["model_id"]
        for row in model_rows
        if row["certification_status"] != "adapter_certified"
        and row["paper_evidence_status"] != "diagnostic_only"
    ]
    checks = [
        readiness_row("method_leftovers", method_ready, "method_rerun_readiness_table.csv"),
        readiness_row("data_manifests", data_ready, "data_preflight.csv"),
        readiness_row(
            "blocked_model_repair",
            not unresolved_models,
            "unresolved_models=" + pipe_join(unresolved_models),
        ),
        readiness_row("broad_latent_model_minimum", len(certified_latent) >= 5, f"certified_latent_models={len(certified_latent)}"),
        readiness_row("scanpath_foveated_model", bool(scanpath_certified), f"certified_scanpath_or_foveated={len(scanpath_certified)}"),
        readiness_row("dynamicvit_setup", dynamicvit_ready, "dynamicvit_setup_preflight.csv"),
        readiness_row("tome_setup", tome_ready, "tome_setup_preflight.csv"),
        readiness_row("roi_early_subject_scope", early_ready, "subject_roi_availability.csv"),
        readiness_row("roi_stream_scope", stream_ready, "stream_roi_grouping_spec.csv"),
    ]
    contract_authorized, contract_basis = publication_contract_execution_status()
    configs_enabled, configs_basis = clean_rerun_config_status()
    runners_ready, runners_basis = clean_rerun_runner_status()
    checks.extend(
        [
            readiness_row(
                "publication_contract_execution_flag",
                contract_authorized,
                contract_basis,
            ),
            readiness_row(
                "clean_rerun_configs_execution_enabled",
                configs_enabled,
                configs_basis,
            ),
            readiness_row(
                "clean_rerun_runner_scripts",
                runners_ready,
                runners_basis,
            ),
        ]
    )
    authorized = all(row["status"] == "ready" for row in checks)
    checks.append(
        {
            "gate": "first_clean_rerun_authorization",
            "status": "authorized" if authorized else "blocked",
            "basis": "all preflight gates ready" if authorized else "one or more preflight gates remain blocked",
            "next_action": "launch bounded clean rerun" if authorized else "complete user_action_checklist.csv and regenerate preflight",
        }
    )
    return checks


def readiness_row(gate: str, ready: bool, basis: str) -> dict[str, str]:
    return {
        "gate": gate,
        "status": "ready" if ready else "blocked",
        "basis": basis,
        "next_action": "none" if ready else "resolve blocking rows and regenerate preflight",
    }


def publication_contract_execution_status() -> tuple[bool, str]:
    path = resolve_path("configs/paper1_publication_contract.yaml")
    if not path.is_file():
        return False, "missing configs/paper1_publication_contract.yaml"
    payload = load_yaml(path)
    authorized = bool(payload.get("execution_authorized"))
    value = str(payload.get("execution_authorized", ""))
    return authorized, f"execution_authorized={value}"


def clean_rerun_config_status() -> tuple[bool, str]:
    disabled = []
    missing = []
    for lane, relative in CLEAN_RERUN_CONFIGS.items():
        path = resolve_path(relative)
        if not path.is_file():
            missing.append(f"{lane}:{relative}")
            continue
        payload = load_yaml(path)
        if payload.get("execution_enabled") is not True:
            disabled.append(f"{lane}:execution_enabled={payload.get('execution_enabled')}")
    if missing or disabled:
        return False, "missing=" + pipe_join(missing) + "; disabled=" + pipe_join(disabled)
    return True, "all configured clean rerun lanes have execution_enabled=true"


def clean_rerun_runner_status() -> tuple[bool, str]:
    missing = [
        f"{lane}:{relative}"
        for lane, relative in CLEAN_RERUN_RUNNERS.items()
        if not resolve_path(relative).is_file()
    ]
    blocked = []
    full_run_blocking_markers = [
        "Full behavioral scoring is not launched",
        "Full latent neural encoding is cluster-backed and was not launched",
        "Full geometry computation is cluster-backed and was not launched",
        "Full efficiency/resource profiling was not launched",
    ]
    for lane, relative in CLEAN_RERUN_RUNNERS.items():
        path = resolve_path(relative)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matched = [marker for marker in full_run_blocking_markers if marker in text]
        if matched:
            blocked.append(f"{lane}:full_run_not_implemented")
    if missing or blocked:
        return (
            False,
            "missing_clean_runners="
            + pipe_join(missing)
            + "; blocked_clean_runners="
            + pipe_join(blocked),
        )
    return True, "all clean rerun runner scripts exist and expose full-run paths"


def write_first_clean_rerun_plan(
    *,
    readiness_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
    availability_rows: list[dict[str, str]],
) -> None:
    authorized = any(
        row["gate"] == "first_clean_rerun_authorization" and row["status"] == "authorized"
        for row in readiness_rows
    )
    behavior = [row["model_id"] for row in model_rows if row["behavioral_output_status"] == "ready" and row["certification_status"] == "adapter_certified"]
    latent = [row["model_id"] for row in model_rows if row["latent_tensor_status"] == "ready" and row["certification_status"] == "adapter_certified"]
    scanpath = [row["model_id"] for row in model_rows if row["scanpath_glimpse_export_status"] == "ready" and row["certification_status"] == "adapter_certified"]
    efficiency = [row["model_id"] for row in model_rows if row["efficiency_metadata_status"] == "ready" and row["certification_status"] == "adapter_certified"]
    roi_scope = sorted(
        {
            f"{row['subject_id']}:{row['roi']}"
            for row in availability_rows
            if row["available"] == "yes"
        }
    )
    if authorized:
        cluster_lines = [
            "Preflight authorizes only the bounded clean rerun scope in this file. Codex did not launch a cluster job.",
            "Run these cmd.exe commands to sync the cluster workspace and re-check preflight before any axis-specific clean rerun job:",
            "",
            "```cmd",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && git pull\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && ./.venv/bin/python scripts/generate_paper1_publication_preflight.py\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && cat outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv\"",
            "```",
            "",
            "No single full publication-matrix runner is registered in `scripts/`; do not use the admission-panel runner as a substitute.",
        ]
    else:
        cluster_lines = [
            "No clean rerun cluster command is authorized while `rerun_readiness_table.csv` contains blocked gates.",
            "After all gates are ready, use cmd.exe commands only:",
            "",
            "```cmd",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && git pull\"",
            "ssh zzhan330@dsailogin.arch.jhu.edu \"cd /scratch/tshu2/zzhan330/saliency && ./.venv/bin/python scripts/generate_paper1_publication_preflight.py\"",
            "```",
        ]
    lines = [
        "# First Clean Rerun Plan",
        "",
        f"authorization_status: {'authorized' if authorized else 'blocked'}",
        "",
        "## Eligible Models",
        "",
        f"- behavioral: {pipe_join(behavior)}",
        f"- latent_neural_geometry: {pipe_join(latent)}",
        f"- task_search_or_scanpath: {pipe_join(scanpath)}",
        f"- efficiency_resource: {pipe_join(efficiency)}",
        "",
        "## ROI Stream Scope",
        "",
        f"- available_subject_roi_pairs: {pipe_join(roi_scope)}",
        "",
        "## Cluster Commands",
        "",
        *cluster_lines,
        "",
        "## Expected Output Validation",
        "",
        "`outputs/paper1_publication_v0/preflight/expected_outputs_manifest.csv` lists required pre-rerun and post-rerun artifacts.",
        "",
    ]
    (PREFLIGHT_ROOT / "first_clean_rerun_plan.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_user_action_checklist(
    setup_rows: list[dict[str, str]],
    readiness_rows: list[dict[str, str]],
) -> None:
    rows = []
    for row in setup_rows:
        if row["certification_status"] == "adapter_certified":
            continue
        action = row["user_action_needed"]
        if action == "none":
            continue
        rows.append(
            {
                "item_type": "model_setup",
                "item_id": row["model_id"],
                "status": "blocked",
                "next_step_owner": row.get("next_step_owner", ""),
                "error_log_path": row.get("error_log_path", ""),
                "missing_external_requirement": row.get("missing_external_requirement", ""),
                "user_action_needed": action,
                "next_command_cmd": row["next_command_after_user_action_cmd"],
            }
        )
    for row in readiness_rows:
        if row["status"] in {"ready", "authorized"}:
            continue
        rows.append(
            {
                "item_type": "rerun_gate",
                "item_id": row["gate"],
                "status": row["status"],
                "next_step_owner": "codex_implementable",
                "error_log_path": "",
                "missing_external_requirement": row["basis"],
                "user_action_needed": row["next_action"],
                "next_command_cmd": ".\\.venv\\Scripts\\python.exe scripts\\generate_paper1_publication_preflight.py",
            }
        )
    path = PREFLIGHT_ROOT / "user_action_checklist.csv"
    if rows:
        write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=USER_ACTION_FIELDNAMES).writeheader()


def paper_status_for_record(record: dict[str, Any], entry: dict[str, Any]) -> str:
    if record["certification_status"] == "adapter_certified":
        return str(entry.get("paper_evidence_status", "adapter_certified")).replace("adapter_in_progress", "adapter_certified")
    blocker_codes = {blocker["code"] for blocker in record.get("blockers", [])}
    if "license_audit_required" in blocker_codes:
        return "license_blocked"
    if str(entry.get("setup_kind")) == "builtin":
        return str(entry.get("paper_evidence_status", "diagnostic_only"))
    if "source_release_unavailable" in blocker_codes:
        return "setup_attempt_failed"
    return "adapter_in_progress"


def setup_or_cert_command(model_id: str, runtime_id: str, entry: dict[str, Any]) -> str:
    if str(entry.get("setup_kind")) == "builtin":
        return "builtin control; no install command"
    if str(entry.get("setup_command", "")).startswith("python scripts/certify_local"):
        return f".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models {runtime_id or model_id}"
    return (
        'wsl -e bash -lc "cd /mnt/d/Git/saliency && '
        f'python3 scripts/setup_external_model.py --model {runtime_id} --install --download-checkpoint --smoke"'
    )


def local_cert_command(model_id: str, runtime_id: str, entry: dict[str, Any]) -> str:
    if str(entry.get("setup_kind")) == "builtin":
        return "builtin control; run benchmark config after rerun authorization"
    if str(entry.get("setup_command", "")).startswith("python scripts/certify_local"):
        return f".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models {runtime_id or model_id}"
    if runtime_id:
        return f".\\.venv\\Scripts\\python.exe scripts\\setup_external_model.py --model {runtime_id} --report-only"
    return "source unavailable; no local certification command until source is supplied"


def next_command_after_action(model_id: str, runtime_id: str, entry: dict[str, Any]) -> str:
    if str(entry.get("setup_kind")) == "builtin":
        return ".\\.venv\\Scripts\\python.exe scripts\\generate_paper1_publication_preflight.py"
    if str(entry.get("setup_command", "")).startswith("python scripts/certify_local"):
        return f".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models {runtime_id or model_id}"
    if runtime_id:
        return (
            'wsl -e bash -lc "cd /mnt/d/Git/saliency && '
            f'python3 scripts/setup_external_model.py --model {runtime_id} --install --download-checkpoint --smoke"'
        )
    return ".\\.venv\\Scripts\\python.exe scripts\\generate_paper1_publication_preflight.py"


def user_action(
    runtime: Any,
    runtime_model: dict[str, Any],
    record: dict[str, Any],
    entry: dict[str, Any],
) -> str:
    blockers = record.get("blockers", [])
    if not blockers:
        return "none"
    actions = []
    for blocker in blockers:
        code = blocker["code"]
        if code == "checkpoint_lock_missing":
            actions.append(
                f"download official weights from {checkpoint_url(runtime_model) or 'registered source'} to {checkpoint_path(runtime, runtime_model) or 'registered checkpoint path'} and rerun the next command"
            )
        elif code == "environment_lock_missing":
            actions.append("build the pinned environment and write its lock file")
        elif code == "source_commit_unpinned":
            actions.append(
                f"pin the official source revision for {source_identity(runtime_model, entry)} in configs/external_models/registry.yaml"
            )
        elif code == "source_release_unavailable":
            actions.append(
                "supply an official executable source/checkpoint/license bundle; no public source is registered for this candidate"
            )
        elif code == "license_audit_required":
            license_text = source_license(runtime_model, entry)
            if "no_root_license_file_found" in license_text:
                actions.append(
                    "obtain ScanDiff license terms from the authors or an official release file; "
                    "the official GitHub root LICENSE request returned 404 on 2026-06-16"
                )
            else:
                actions.append(
                    f"record official license/access text for {source_identity(runtime_model, entry)}; current registry value is {license_text}"
                )
        elif code == "adapter_execution_pending":
            actions.append("complete adapter hooks and pass smoke validation")
        elif code == "adapter_smoke_missing":
            actions.append("run the local tiny-certification command and keep the generated report")
        else:
            actions.append(blocker["resolution"])
    return " | ".join(dict.fromkeys(actions))


def blocker_log_text(record: dict[str, Any], blockers: list[dict[str, str]]) -> str:
    if not blockers:
        return "no blocker; installation/certification report is ready"
    last_error = record.get("last_error")
    if last_error:
        return f"{last_error.get('type', '')}: {last_error.get('message', '')}"
    return "static preflight blockers: " + "|".join(blocker["code"] for blocker in blockers)


def output_status(record: dict[str, Any], key: str) -> str:
    outputs = record["outputs"][key]
    if not outputs:
        return "not_applicable"
    return "ready" if record["certification_status"] == "adapter_certified" else "defined_pending_certification"


def scanpath_status(record: dict[str, Any]) -> str:
    values = set(record["outputs"]["behavioral"]) | set(record["outputs"]["resource"])
    has_scanpath = any(
        "scanpath" in value or "glimpse" in value or value in {"fixation_count", "stopping_behavior"}
        for value in values
    )
    if not has_scanpath:
        return "not_applicable"
    return "ready" if record["certification_status"] == "adapter_certified" else "defined_pending_certification"


def efficiency_status(record: dict[str, Any], report: dict[str, Any] | None) -> str:
    if record["certification_status"] == "adapter_certified":
        return "ready"
    if report and report.get("stages", {}).get("environment_ready"):
        return "environment_ready_pending_smoke_or_checkpoint"
    return "defined_pending_certification"


def conditioning_fields(record: dict[str, Any]) -> str:
    contract = record["input_contract"]
    fields = []
    if contract["task_required"]:
        fields.append("task_id_or_target")
    if contract["gaze_history_required"]:
        fields.append("x_hist|y_hist|durations")
    if contract["stochastic_seed_required"]:
        fields.append("seed")
    if "foveated" in contract["modes"]:
        fields.append("glimpse_window|foveation_level")
    return "|".join(fields) or "image_only"


def cluster_status(record: dict[str, Any], runtime_model: dict[str, Any]) -> str:
    if str(runtime_model.get("execution", "")) in {"cluster_required", "linux_gpu"}:
        return "cluster_required_after_setup_preflight"
    if record["certification_status"] == "adapter_certified":
        return "local_ready_cluster_optional"
    return "not_ready"


def write_scanpath_or_foveated_manifest(
    path: Path,
    *,
    model_rows: list[dict[str, str]],
    setup_rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    certified = [
        row
        for row in model_rows
        if row["model_id"] in SCANPATH_OR_FOVEATED
        and row["certification_status"] == "adapter_certified"
        and row["scanpath_glimpse_export_status"] == "ready"
    ]
    payload = {
        "schema_version": "hma.external.scanpath_foveated_preflight_manifest.v1",
        "artifact_name": "scanpath_or_foveated_certified",
        "artifact_status": "certified" if certified else "blocked_no_scanpath_foveated_model_certified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "certified_models": [
            {
                "model_id": row["model_id"],
                "behavioral_output_type": row["behavioral_output_type"],
                "scanpath_glimpse_export_status": row["scanpath_glimpse_export_status"],
                "resource_allocation_outputs": row["resource_allocation_outputs"],
                "next_command_cmd": row["next_command_after_user_action_cmd"],
            }
            for row in certified
        ],
        "blocked_models": [
            row
            for row in setup_rows
            if row["certification_status"] != "adapter_certified"
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def source_identity(runtime_model: dict[str, Any], entry: dict[str, Any]) -> str:
    if runtime_model:
        return str(runtime_model.get("source", {}).get("repository", ""))
    return str(entry.get("setup_kind", "builtin"))


def source_revision(runtime_model: dict[str, Any]) -> str:
    return str(runtime_model.get("source", {}).get("commit", ""))


def source_license(runtime_model: dict[str, Any], entry: dict[str, Any]) -> str:
    if runtime_model:
        return str(runtime_model.get("source", {}).get("license", ""))
    return str(entry.get("setup_kind", "builtin"))


def install_path(runtime: Any, runtime_id: str, entry: dict[str, Any]) -> str:
    if str(entry.get("setup_kind")) == "builtin":
        return "src/hma/saliency/baselines.py"
    if not runtime_id:
        return ""
    try:
        return str((runtime.workspace_path("sources") / runtime_id).relative_to(PROJECT_ROOT))
    except Exception:
        return ""


def checkpoint_identity(runtime_model: dict[str, Any]) -> str:
    checkpoint = runtime_model.get("checkpoint", {})
    return str(checkpoint.get("filename") or checkpoint.get("access") or "not_required")


def checkpoint_url(runtime_model: dict[str, Any]) -> str:
    checkpoint = runtime_model.get("checkpoint", {})
    urls = [str(checkpoint["url"])] if checkpoint.get("url") else []
    urls.extend(str(value) for value in checkpoint.get("additional_urls", []) or [])
    if urls:
        return " | ".join(urls)
    return str(checkpoint.get("access") or "")


def checkpoint_path(runtime: Any, runtime_model: dict[str, Any]) -> str:
    if not runtime_model:
        return ""
    filename = runtime_model.get("checkpoint", {}).get("filename")
    if not filename:
        return ""
    return str((runtime.workspace_path("checkpoints") / str(runtime_model["id"]) / str(filename)).relative_to(PROJECT_ROOT))


def checkpoint_hash_or_policy(
    report: dict[str, Any] | None,
    runtime: Any,
    runtime_model: dict[str, Any],
) -> str:
    live_hash = live_checkpoint_lock_hash(runtime, runtime_model)
    if live_hash:
        return live_hash
    if report:
        checkpoint = report.get("checkpoint", {})
        return str(checkpoint.get("sha256") or checkpoint.get("hash_policy") or "")
    return str(runtime_model.get("checkpoint", {}).get("hash_policy", ""))


def live_checkpoint_lock_hash(runtime: Any, runtime_model: dict[str, Any]) -> str:
    if not runtime_model:
        return ""
    model_id = str(runtime_model.get("id") or "")
    filename = str(runtime_model.get("checkpoint", {}).get("filename") or "")
    if not model_id or not filename:
        return ""
    checkpoint_path = runtime.workspace_path("checkpoints") / model_id / filename
    lock_path = (
        PROJECT_ROOT
        / "configs"
        / "external_models"
        / "checkpoint_locks"
        / f"{model_id}.json"
    )
    if not checkpoint_path.exists() or not lock_path.is_file():
        return ""
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = str(lock.get("sha256") or "")
        if expected and sha256_tree(checkpoint_path) == expected:
            return expected
    except (OSError, ValueError):
        return ""
    return ""


def report_diagnostics(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    if report.get("stages", {}).get("evidence_ready"):
        return ""
    return " | ".join(str(value) for value in report.get("diagnostics", []))


def environment_identity(runtime_model: dict[str, Any]) -> str:
    if not runtime_model:
        return "builtin_or_unavailable"
    return str(runtime_model.get("environment", "environment_not_registered"))


def read_installation_report(runtime: Any, runtime_id: str) -> dict[str, Any] | None:
    if not runtime_id:
        return None
    path = runtime.workspace_path("reports") / f"{runtime_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_for_roi(roi: str) -> str:
    if roi in PRF_ROIS:
        return "data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv"
    if roi in STREAM_ROIS:
        return "data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv"
    return ""


def roi_class(roi: str) -> str:
    if roi in PRF_ROIS:
        return "prf_visualrois"
    if roi in STREAM_ROIS:
        return "streams"
    return "floc_category_unavailable"


def stream_for_roi(roi: str) -> str:
    for stream, rois in STREAM_GROUPS.items():
        if roi in rois:
            return stream
    return "unassigned"


def unavailable_roi_reason(subject: str, roi: str) -> str:
    if roi in STREAM_ROIS and subject != "subj01":
        return "stream ROI manifests/responses are local only for subj01"
    if roi in STREAM_GROUPS["floc_category"]:
        return "fLOC category ROI masks/manifests are deferred and unavailable locally"
    return "manifest row absent"


def write_blocked_manifest(
    path: Path,
    *,
    artifact_name: str,
    status: str,
    blockers: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "hma.external.blocked_artifact_manifest.v1",
        "artifact_name": artifact_name,
        "artifact_status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "not_valid_external_artifact": True,
        "blockers": blockers,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else " " for ch in text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def write_yaml(path: Path, value: Any) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def count_csv_rows(path: Path, *, relative: str = "") -> int:
    if relative in KNOWN_ROW_COUNTS:
        return KNOWN_ROW_COUNTS[relative]
    if path.stat().st_size > 100_000_000:
        raise RuntimeError(
            f"Refusing to count large CSV without a known count: {path}"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def yesno(value: bool) -> str:
    return "yes" if value else "no"


def pipe_join(values: Iterable[str]) -> str:
    values = [str(value) for value in values if str(value)]
    return "|".join(values) if values else "none"


if __name__ == "__main__":
    raise SystemExit(main())
