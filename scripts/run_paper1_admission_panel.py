"""Run and assemble the bounded Paper 1 publication admission panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.behavioral import image_cluster_bootstrap
from hma.datasets.salicon import SALICONDataset
from hma.experiments.neural_alignment import run_neural_alignment
from hma.external.adapters import build_adapter
from hma.external.registry import load_external_registry
from hma.metrics.saliency_metrics import (
    auc_judd,
    cc,
    information_gain,
    kl_divergence,
    nss,
    probabilistic_log_likelihood,
    similarity,
    simple_center_bias_map,
)
from hma.utils.config import load_yaml, save_yaml


BEHAVIOR_METRICS = (
    "log_likelihood_bits",
    "information_gain_bits",
    "nss",
    "auc_judd",
    "cc",
    "similarity",
    "kl_target_to_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/paper1_admission_panel_v1.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "behavioral", "neural", "efficiency", "assemble", "blockers"],
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = PROJECT_ROOT / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    phases = (
        ["blockers", "behavioral", "neural", "efficiency", "assemble"]
        if args.phase == "all"
        else [args.phase]
    )
    for phase in phases:
        if phase == "blockers":
            run_blocker_probes(config)
        elif phase == "behavioral":
            run_behavioral_panel(config)
        elif phase == "neural":
            run_neural_panel(config)
        elif phase == "efficiency":
            run_efficiency_panel(config)
        elif phase == "assemble":
            assemble_panel(config)
    return 0


def run_behavioral_panel(config: dict[str, Any]) -> None:
    section = dict(config["behavioral"])
    output_root = PROJECT_ROOT / str(config["output_root"])
    behavior_root = output_root / "behavioral"
    per_image_root = behavior_root / "per_image_metrics" / "admission_panel"
    map_root = behavior_root / "maps" / "admission_panel" / "deepgaze_iie"
    per_image_root.mkdir(parents=True, exist_ok=True)
    map_root.mkdir(parents=True, exist_ok=True)
    dataset = SALICONDataset(
        root=PROJECT_ROOT / str(section["root"]),
        manifest_path=PROJECT_ROOT / str(section["manifest"]),
        split=str(section["split"]),
        max_items=int(section["max_items"]),
        image_size=section["image_size"],
        validate_files=True,
    )
    registry = load_external_registry()
    deepgaze_config = registry.model("deepgaze_iie")
    deepgaze = build_adapter(
        str(deepgaze_config["adapter"]),
        model_id="deepgaze_iie",
        model_config=deepgaze_config,
        source_dir=registry.workspace_path("sources") / "deepgaze_iie",
        checkpoint_path=None,
        device=_resolve_device("auto"),
        seed=int(config["seed"]),
    )
    model_specs = _model_specs(config)
    rows_by_model: dict[str, list[dict[str, Any]]] = {
        model_id: [] for model_id in section["models"]
    }
    for index, item in enumerate(dataset):
        image = item["image"]
        target = np.asarray(item["fixation_map"], dtype=np.float32)
        fixations = _xy_to_yx(np.asarray(item["fixation_points"], dtype=np.float32))
        center = simple_center_bias_map(*target.shape)
        rng = np.random.default_rng(_stable_seed(int(config["seed"]), index))
        random_map = rng.random(target.shape, dtype=np.float32)
        deepgaze_output = deepgaze.run_batch([image], [str(item["image_id"])])
        deepgaze_map = np.asarray(
            deepgaze_output.task_outputs["human_gaze_density"].detach().cpu()[0],
            dtype=np.float32,
        )
        np.save(map_root / f"{item['map_key']}.npy", deepgaze_map)
        predictions = {
            "deepgaze_iie": deepgaze_map,
            "center_prior": center,
            "random_baseline": random_map,
        }
        for model_id, prediction in predictions.items():
            spec = model_specs[model_id]
            row = {
                "evidence_label": str(config["evidence_label"]),
                "dataset": "salicon",
                "behavioral_regime": "free_viewing",
                "behavioral_object": _behavioral_object(model_id),
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "map_key": item["map_key"],
                "matched_prior_id": "analytic_center_bias_224",
                "log_likelihood_bits": probabilistic_log_likelihood(
                    prediction,
                    target,
                    positive_fixations=fixations,
                ),
                "information_gain_bits": information_gain(
                    prediction,
                    center,
                    target,
                    positive_fixations=fixations,
                ),
                "nss": nss(
                    prediction,
                    target,
                    positive_fixations=fixations,
                ),
                "auc_judd": auc_judd(
                    prediction,
                    target,
                    positive_fixations=fixations,
                ),
                "cc": cc(prediction, target),
                "similarity": similarity(prediction, target),
                "kl_target_to_prediction": kl_divergence(target, prediction),
            }
            rows_by_model[model_id].append(row)

    all_rows = [row for rows in rows_by_model.values() for row in rows]
    _write_csv(per_image_root / "salicon_free_viewing.csv", all_rows)
    aggregate_rows = []
    uncertainty_rows = []
    for model_id, rows in rows_by_model.items():
        spec = model_specs[model_id]
        aggregate_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "dataset": "salicon",
                "behavioral_regime": "free_viewing",
                "behavioral_object": _behavioral_object(model_id),
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "num_images": len(rows),
                "matched_prior_id": "analytic_center_bias_224",
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in BEHAVIOR_METRICS
                },
            }
        )
        for metric in BEHAVIOR_METRICS:
            interval = image_cluster_bootstrap(
                rows,
                value_key=metric,
                resamples=int(section["bootstrap_resamples"]),
                seed=int(config["seed"]),
                confidence=float(section["bootstrap_confidence"]),
            )
            uncertainty_rows.append(
                {
                    "evidence_label": str(config["evidence_label"]),
                    "dataset": "salicon",
                    "behavioral_regime": "free_viewing",
                    "behavioral_object": _behavioral_object(model_id),
                    "model_id": model_id,
                    "family": spec["family"],
                    "role": spec["role"],
                    "metric": metric,
                    **interval.as_dict(),
                }
            )
    _write_csv(behavior_root / "aggregate_admission_panel.csv", aggregate_rows)
    _write_csv(behavior_root / "uncertainty_admission_panel.csv", uncertainty_rows)


def run_neural_panel(config: dict[str, Any]) -> None:
    section = dict(config["neural"])
    output_root = PROJECT_ROOT / str(config["output_root"])
    neural_root = output_root / "neural_encoding"
    geometry_root = output_root / "geometry"
    run_root = neural_root / "admission_runs"
    snapshot_root = output_root / "audits" / "admission_config_snapshots"
    run_root.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    encoding_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for spec in _latent_model_specs(config):
        model_name = str(spec.get("timm_model_name", spec["runtime_model_id"]))
        experiment_config = {
            "seed": int(config["seed"]),
            "device": "cuda",
            "experiment": {
                "name": f"admission_{spec['model_id']}_{section['subject_id']}_{section['roi']}"
            },
            "dataset": {
                "name": "nsd_algonauts",
                "label": "nsd_algonauts_admission",
                "root": section["root"],
                "manifest_path": section["manifest"],
                "split": section["split"],
                "subject_id": section["subject_id"],
                "roi": section["roi"],
                "max_items": int(section["max_items"]),
                "validate_files": True,
            },
            "model": {
                "name": model_name,
                "backend": "timm",
                "pretrained": True,
                "eval_mode": True,
            },
            "preprocessing": {
                "input_size": spec["input_size"],
                "mean": spec["mean"],
                "std": spec["std"],
            },
            "neural": {
                "layers": [spec["layer"]],
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "feature_reduction": "flatten_pca",
                "pca_components": int(section["pca_components"]),
                "pca_solver": "randomized",
                "pca_whiten": False,
                "feature_reduction_seed": int(config["seed"]),
                "train_fraction": float(section["train_fraction"]),
                "validation_fraction": float(section["validation_fraction"]),
                "ridge_alphas": list(section["ridge_alphas"]),
                "metric": "correlation",
                "selection": {"enabled": False},
                "rsa": {"enabled": False},
                "geometry": {
                    "enabled": True,
                    "methods": [
                        "debiased_linear_cka",
                        "linear_cka",
                        "subset_rsa",
                    ],
                    "subset_sizes": [int(section["geometry_subset_size"])],
                    "subset_seeds": [int(config["seed"])],
                    "null_control_seeds": [int(config["seed"])],
                    "bootstrap_resamples": int(
                        section["geometry_bootstrap_resamples"]
                    ),
                    "bootstrap_seed": int(config["seed"]),
                    "bootstrap_confidence": float(
                        section["geometry_bootstrap_confidence"]
                    ),
                },
            },
            "output": {
                "dir": str(
                    (
                        run_root
                        / str(spec["model_id"])
                    ).relative_to(PROJECT_ROOT)
                )
            },
        }
        snapshot = snapshot_root / f"{spec['model_id']}.yaml"
        save_yaml(experiment_config, snapshot)
        result = run_neural_alignment(snapshot)
        for row in result["score_rows"]:
            encoding_rows.append(
                {
                    "evidence_label": str(config["evidence_label"]),
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "role": spec["role"],
                    "fixed_admission_layer": spec["layer"],
                    "layer_selection_status": "fixed_admission_layer_not_final_selection",
                    **row,
                }
            )
        for row in result["geometry_rows"]:
            geometry_rows.append(
                {
                    "evidence_label": str(config["evidence_label"]),
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "role": spec["role"],
                    "fixed_admission_layer": spec["layer"],
                    **row,
                }
            )
    _write_csv(neural_root / "encoding_scores_admission_panel.csv", encoding_rows)
    _write_csv(geometry_root / "geometry_scores_admission_panel.csv", geometry_rows)


def run_efficiency_panel(config: dict[str, Any]) -> None:
    section = dict(config["efficiency"])
    output_root = PROJECT_ROOT / str(config["output_root"])
    efficiency_root = output_root / "efficiency"
    efficiency_root.mkdir(parents=True, exist_ok=True)
    registry = load_external_registry()
    profile_rows = []
    resource_rows = []
    for spec in _executable_model_specs(config):
        runtime_id = str(spec["runtime_model_id"])
        model_config = registry.model(runtime_id)
        adapter = build_adapter(
            str(model_config["adapter"]),
            model_id=runtime_id,
            model_config=model_config,
            source_dir=registry.workspace_path("sources") / runtime_id,
            checkpoint_path=None,
            device=_resolve_device(str(section["device"])),
            seed=int(config["seed"]),
        )
        image = Image.new("RGB", (224, 224), color=(127, 127, 127))
        profile = adapter.profile_efficiency(
            [image],
            warmup_runs=int(section["warmup_runs"]),
            measured_runs=int(section["measured_runs"]),
            repeats=int(section["repeats"]),
        )
        profile_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": spec["model_id"],
                "family": spec["family"],
                "role": spec["role"],
                "comparability_group": "static_full_image_224_batch1_local_gpu",
                **_flatten_profile(profile),
            }
        )
        realized_flops = profile.get("realized_flops")
        latency_ms = profile.get("latency_ms_per_image")
        if realized_flops is not None:
            total_cost_value = realized_flops
            total_cost_unit = "fvcore_flops"
        elif latency_ms is not None:
            total_cost_value = latency_ms
            total_cost_unit = "measured_latency_ms_per_image"
        else:
            total_cost_value = ""
            total_cost_unit = ""
        resource_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": spec["model_id"],
                "family": spec["family"],
                "role": spec["role"],
                "resource_mode": "static_full_image",
                "input_height": 224,
                "input_width": 224,
                "batch_size": 1,
                "sequential_or_adaptive": "false",
                "total_cost_scope": "per_image",
                "total_cost_value": total_cost_value,
                "total_cost_unit": total_cost_unit,
                "resource_trace_status": (
                    "complete_static_profile"
                    if total_cost_value != ""
                    else "partial_static_profile"
                ),
            }
        )
    for spec in _control_model_specs(config):
        profile_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": spec["model_id"],
                "family": spec["family"],
                "role": spec["role"],
                "comparability_group": "analytic_control_not_profiled",
                "profile_status": "not_applicable_control",
            }
        )
        resource_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": spec["model_id"],
                "family": spec["family"],
                "role": spec["role"],
                "resource_mode": "analytic_control",
                "sequential_or_adaptive": "false",
                "total_cost_scope": "not_applicable",
                "resource_trace_status": "not_applicable_control",
            }
        )
    _write_csv(
        efficiency_root / "efficiency_profiles_admission_panel.csv",
        profile_rows,
    )
    _write_csv(
        efficiency_root / "resource_allocation_profiles_admission_panel.csv",
        resource_rows,
    )


def run_blocker_probes(config: dict[str, Any]) -> None:
    registry = load_external_registry()
    records = []
    for requested_id in config.get("cluster_pending_models", []):
        try:
            model = registry.model(str(requested_id))
            canonical_id = str(model["id"])
            adapter = build_adapter(
                str(model["adapter"]),
                model_id=canonical_id,
                model_config=model,
                source_dir=registry.workspace_path("sources") / canonical_id,
                checkpoint_path=None,
                device="cpu",
                seed=int(config["seed"]),
            )
            smoke = adapter.smoke()
        except Exception as exc:
            records.append(
                {
                    "schema_version": "hma.paper1.admission_role_blocker.v1",
                    "model_id": str(requested_id),
                    "attempt": "adapter_construction_and_smoke",
                    "status": "blocked",
                    "blocker_type": type(exc).__name__,
                    "blocker_detail": str(exc),
                    "remediation": _blocker_remediation(str(requested_id)),
                }
            )
        else:
            records.append(
                {
                    "schema_version": "hma.paper1.admission_role_blocker.v1",
                    "model_id": str(requested_id),
                    "attempt": "adapter_construction_and_smoke",
                    "status": "smoke_passed",
                    "smoke": smoke,
                }
            )
    path = PROJECT_ROOT / "outputs/paper1_scope_reset/admission_role_blockers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def assemble_panel(config: dict[str, Any]) -> None:
    output_root = PROJECT_ROOT / str(config["output_root"])
    behavior = _read_csv(output_root / "behavioral/aggregate_admission_panel.csv")
    neural = _read_csv(
        output_root / "neural_encoding/encoding_scores_admission_panel.csv"
    )
    geometry = _read_csv(output_root / "geometry/geometry_scores_admission_panel.csv")
    efficiency = _read_csv(
        output_root / "efficiency/efficiency_profiles_admission_panel.csv"
    )
    behavior_by_model = {row["model_id"]: row for row in behavior}
    neural_by_model = {row["model_id"]: row for row in neural}
    geometry_by_model = {
        row["model_id"]: row
        for row in geometry
        if row.get("geometry_method") == "debiased_linear_cka"
        and row.get("control_type") == "observed"
    }
    efficiency_by_model = {row["model_id"]: row for row in efficiency}
    rows = []
    for spec in config["models"]:
        model_id = str(spec["model_id"])
        behavior_row = behavior_by_model.get(model_id, {})
        neural_row = neural_by_model.get(model_id, {})
        geometry_row = geometry_by_model.get(model_id, {})
        efficiency_row = efficiency_by_model.get(model_id, {})
        efficiency_available = bool(efficiency_row) and (
            efficiency_row.get("profile_status") != "not_applicable_control"
        )
        rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "behavioral_regime": behavior_row.get("behavioral_regime", ""),
                "behavioral_object": behavior_row.get("behavioral_object", ""),
                "behavioral_available": _available(behavior_row),
                "behavior_information_gain_bits": behavior_row.get(
                    "information_gain_bits", ""
                ),
                "latent_neural_available": _available(neural_row),
                "neural_subject_id": neural_row.get("subject_id", ""),
                "neural_roi": neural_row.get("roi", ""),
                "neural_mean_score": neural_row.get("mean_score", ""),
                "geometry_available": _available(geometry_row),
                "geometry_method": geometry_row.get("geometry_method", ""),
                "geometry_score": geometry_row.get("score", ""),
                "efficiency_available": str(efficiency_available).lower(),
                "latency_ms_per_image": efficiency_row.get(
                    "latency_ms_per_image", ""
                ),
                "axis_count_available": sum(
                    (
                        bool(behavior_row),
                        bool(neural_row),
                        bool(geometry_row),
                        efficiency_available,
                    )
                ),
                "paper_evidence_status": "admission_panel_not_final_paper_result",
            }
        )
    cross_root = output_root / "cross_axis"
    cross_root.mkdir(parents=True, exist_ok=True)
    _write_csv(cross_root / "model_axis_scores_admission_panel.csv", rows)
    _write_preflight(cross_root / "admission_panel_preflight.md", rows, config)
    _write_admission_audit(output_root / "audits/admission_panel_audit.csv", config, rows)


def _write_preflight(
    path: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    axis_counts = {
        axis: sum(row[f"{axis}_available"] == "true" for row in rows)
        for axis in ("behavioral", "geometry", "efficiency")
    }
    axis_counts["latent_neural"] = sum(
        row["latent_neural_available"] == "true" for row in rows
    )
    families = sorted({str(row["family"]) for row in rows})
    roles = sorted({str(row["role"]) for row in rows})
    complete = [row["model_id"] for row in rows if row["axis_count_available"] == 4]
    covered_model_ids = {
        str(row["model_id"])
        for row in rows
        if int(row["axis_count_available"]) > 0
    }
    pending = [
        str(value)
        for value in config.get("cluster_pending_models", [])
        if str(value) not in covered_model_ids
    ]
    lines = [
        "# Admission Panel Preflight",
        "",
        "Status: `admission_panel_not_final_paper_result`",
        "",
        f"- Models represented: {len(rows)}",
        f"- Families represented: {len(families)}",
        f"- Roles represented: {len(roles)}",
        f"- Behavioral rows available: {axis_counts['behavioral']}",
        f"- Latent neural rows available: {axis_counts['latent_neural']}",
        f"- Corrected geometry rows available: {axis_counts['geometry']}",
        f"- Efficiency rows available: {axis_counts['efficiency']}",
        f"- Models complete on all four admission axes: {len(complete)}",
        "- Behavioral regimes present: free_viewing",
        "- Neural scope present: subj01 V1 early-retinotopic",
        "",
        "## Missingness",
        "",
        "Cluster-pending model executions: " + ", ".join(pending),
        "",
        "DeepGaze IIE is admitted for behavioral output and efficiency only; its "
        "latent-feature neural and geometry axes remain unavailable in this panel.",
        "",
        "No convergence, dissociation, causal, or paper-facing interpretation is "
        "authorized from this preflight.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_admission_audit(
    path: Path,
    config: dict[str, Any],
    panel_rows: list[dict[str, Any]],
) -> None:
    certification_path = (
        PROJECT_ROOT / "outputs/paper1_scope_reset/adapter_certification_records.jsonl"
    )
    records = [
        json.loads(line)
        for line in certification_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    panel = {row["model_id"]: row for row in panel_rows}
    rows = []
    for record in records:
        model_id = str(record["model_id"])
        panel_row = panel.get(model_id, {})
        rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": model_id,
                "family": record["family"],
                "certification_status": record["certification_status"],
                "paper_evidence_status": record["paper_evidence_status"],
                "included_in_admission_panel": str(bool(panel_row)).lower(),
                "behavioral_available": panel_row.get("behavioral_available", "false"),
                "latent_neural_available": panel_row.get(
                    "latent_neural_available", "false"
                ),
                "geometry_available": panel_row.get("geometry_available", "false"),
                "efficiency_available": panel_row.get("efficiency_available", "false"),
                "blocker_codes": "|".join(
                    blocker["code"] for blocker in record["blockers"]
                ),
                "evidence_classification": (
                    "admission_panel_not_final_paper_result"
                    if panel_row
                    else "not_admitted"
                ),
            }
        )
    _write_csv(path, rows)


def _model_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["model_id"]): dict(row) for row in config["models"]}


def _latent_model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in config["models"]
        if not row.get("control")
        and not row.get("behavioral_only")
        and not row.get("cluster_only")
    ]


def _executable_model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in config["models"]
        if not row.get("control") and not row.get("cluster_only")
    ]


def _control_model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in config["models"] if row.get("control")]


def _behavioral_object(model_id: str) -> str:
    return {
        "deepgaze_iie": "human_gaze_density",
        "center_prior": "center_prior",
        "random_baseline": "random_baseline",
    }[model_id]


def _xy_to_yx(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)[:, [1, 0]]


def _stable_seed(seed: int, index: int) -> int:
    return (int(seed) * 1_000_003 + int(index)) % (2**32)


def _resolve_device(value: str) -> str:
    import torch

    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _flatten_profile(profile: dict[str, Any]) -> dict[str, Any]:
    hardware = dict(profile.get("hardware", {}))
    return {
        "profile_status": "complete" if profile else "missing",
        "parameters": profile.get("parameters", ""),
        "trainable_parameters": profile.get("trainable_parameters", ""),
        "latency_ms_per_image": profile.get("latency_ms_per_image", ""),
        "latency_ms_per_image_std": profile.get("latency_ms_per_image_std", ""),
        "latency_ms_per_image_cv": profile.get("latency_ms_per_image_cv", ""),
        "peak_memory_bytes": profile.get("peak_memory_bytes", ""),
        "peak_reserved_memory_bytes": profile.get("peak_reserved_memory_bytes", ""),
        "theoretical_flops": profile.get("theoretical_flops", ""),
        "realized_flops": profile.get("realized_flops", ""),
        "batch_size": profile.get("batch_size", ""),
        "warmup_batches": profile.get("warmup_batches", ""),
        "measured_batches_per_repeat": profile.get(
            "measured_batches_per_repeat", ""
        ),
        "timing_repeats": profile.get("timing_repeats", ""),
        "device_name": hardware.get("device_name", ""),
        "torch_version": hardware.get("torch_version", ""),
        "cuda_version": hardware.get("cuda_version", ""),
    }


def _blocker_remediation(model_id: str) -> str:
    if model_id in {"dynamicvit_deit_small_keep_0_7", "tome_deit_small_r13"}:
        return "run the pinned Linux environment inference and matched profiler on the cluster"
    if model_id == "hat":
        return "install Detectron2/MSDeformableAttn, lock checkpoint/license, and implement HAT execution hooks"
    if model_id in {"scandiff", "scandiff_freeview"}:
        return "lock the official Hugging Face snapshot and implement seeded scanpath export hooks"
    if model_id == "adaptivenn_deit_small":
        return "pin source/license/checkpoint and implement policy, latent, and total-cost hooks"
    return "complete the registered setup and adapter smoke"


def _available(row: dict[str, Any]) -> str:
    return str(bool(row)).lower()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(key)
    if not names:
        raise ValueError("cannot write an empty admission artifact")
    return names


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    raise SystemExit(main())
