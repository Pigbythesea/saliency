"""Import validated cluster cells into the bounded Paper 1 admission panel."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.behavioral import image_cluster_bootstrap
from hma.external.artifacts import validate_external_artifact
from hma.utils.config import load_yaml
from scripts.run_paper1_admission_panel import (
    BEHAVIOR_METRICS,
    _read_csv,
    _write_csv,
    assemble_panel,
)


MODEL_SPECS = {
    "dynamicvit_deit_small_keep_0_7": {
        "family": "DynamicViT",
        "role": "generic_efficient_computation",
        "layer": "blocks.11",
        "behavioral_object": "operational_resource_allocation",
        "resource_mode": "learned_token_pruning",
    },
    "tome_deit_small_r13": {
        "family": "ToMe",
        "role": "generic_efficient_computation",
        "layer": "blocks.11",
        "behavioral_object": "operational_resource_allocation",
        "resource_mode": "token_merging",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/paper1_admission_panel_v1.yaml",
    )
    parser.add_argument(
        "--cluster-root",
        default="outputs/paper1_admission_v1/cluster",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    cluster_root = PROJECT_ROOT / args.cluster_root
    imported = []
    for model_id in args.models:
        _validate_model_outputs(cluster_root, model_id)
        imported.append(model_id)
    _import_behavior(config, cluster_root, imported)
    _import_neural(config, cluster_root, imported)
    _import_efficiency(config, cluster_root, imported)
    assemble_panel(config)
    audit_path = (
        PROJECT_ROOT
        / str(config["output_root"])
        / "audits"
        / "cluster_import_audit.json"
    )
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "hma.paper1.admission_cluster_import.v1",
                "evidence_label": config["evidence_label"],
                "models": imported,
                "cluster_root": str(cluster_root),
                "status": "imported_not_final_paper_result",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(audit_path)
    return 0


def _validate_model_outputs(cluster_root: Path, model_id: str) -> None:
    required = [
        cluster_root / "behavior" / model_id / "per_image_metrics.csv",
        cluster_root / "neural" / model_id / "encoding_scores.csv",
        cluster_root / "neural" / model_id / "geometry_scores.csv",
        cluster_root / "external" / "neural" / model_id / "efficiency.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing cluster admission outputs: " + ", ".join(missing))
    validate_external_artifact(
        cluster_root / "external" / "neural" / model_id,
        verify_hashes=True,
    )
    validate_external_artifact(
        cluster_root / "external" / "behavior" / model_id,
        verify_hashes=True,
    )


def _import_behavior(
    config: dict[str, Any],
    cluster_root: Path,
    model_ids: list[str],
) -> None:
    output_root = PROJECT_ROOT / str(config["output_root"])
    section = dict(config["behavioral"])
    per_image_path = (
        output_root
        / "behavioral"
        / "per_image_metrics"
        / "admission_panel"
        / "salicon_free_viewing.csv"
    )
    existing = [
        row for row in _read_csv(per_image_path) if row["model_id"] not in model_ids
    ]
    imported_rows: list[dict[str, Any]] = []
    aggregate_rows = [
        row
        for row in _read_csv(output_root / "behavioral/aggregate_admission_panel.csv")
        if row["model_id"] not in model_ids
    ]
    uncertainty_rows = [
        row
        for row in _read_csv(output_root / "behavioral/uncertainty_admission_panel.csv")
        if row["model_id"] not in model_ids
    ]
    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        source = _read_csv(
            cluster_root / "behavior" / model_id / "per_image_metrics.csv"
        )
        rows = []
        for row in source:
            converted = {
                "evidence_label": str(config["evidence_label"]),
                "dataset": "salicon",
                "behavioral_regime": "free_viewing",
                "behavioral_object": spec["behavioral_object"],
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "image_id": row["image_id"],
                "image_path": row.get("image_path", ""),
                "map_key": row.get("map_key", ""),
                "matched_prior_id": row.get(
                    "matched_prior_id", "analytic_center_bias"
                ),
                "log_likelihood_bits": row["log_likelihood_bits"],
                "information_gain_bits": row[
                    "information_gain_vs_matched_prior"
                ],
                "nss": row["nss"],
                "auc_judd": row["auc_judd"],
                "cc": row["cc"],
                "similarity": row["similarity"],
                "kl_target_to_prediction": row["kl"],
            }
            rows.append(converted)
        imported_rows.extend(rows)
        aggregate_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "dataset": "salicon",
                "behavioral_regime": "free_viewing",
                "behavioral_object": spec["behavioral_object"],
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "num_images": len(rows),
                "matched_prior_id": rows[0]["matched_prior_id"],
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
                    "behavioral_object": spec["behavioral_object"],
                    "model_id": model_id,
                    "family": spec["family"],
                    "role": spec["role"],
                    "metric": metric,
                    **interval.as_dict(),
                }
            )
    _write_csv(per_image_path, existing + imported_rows)
    _write_csv(
        output_root / "behavioral/aggregate_admission_panel.csv",
        aggregate_rows,
    )
    _write_csv(
        output_root / "behavioral/uncertainty_admission_panel.csv",
        uncertainty_rows,
    )


def _import_neural(
    config: dict[str, Any],
    cluster_root: Path,
    model_ids: list[str],
) -> None:
    output_root = PROJECT_ROOT / str(config["output_root"])
    encoding_path = output_root / "neural_encoding/encoding_scores_admission_panel.csv"
    geometry_path = output_root / "geometry/geometry_scores_admission_panel.csv"
    encoding_rows = [
        row for row in _read_csv(encoding_path) if row["model_id"] not in model_ids
    ]
    geometry_rows = [
        row for row in _read_csv(geometry_path) if row["model_id"] not in model_ids
    ]
    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        common = {
            "evidence_label": str(config["evidence_label"]),
            "model_id": model_id,
            "family": spec["family"],
            "role": spec["role"],
            "fixed_admission_layer": spec["layer"],
        }
        for row in _read_csv(
            cluster_root / "neural" / model_id / "encoding_scores.csv"
        ):
            encoding_rows.append(
                {
                    **common,
                    "layer_selection_status": "fixed_admission_layer_not_final_selection",
                    **row,
                }
            )
        for row in _read_csv(
            cluster_root / "neural" / model_id / "geometry_scores.csv"
        ):
            geometry_rows.append({**common, **row})
    _write_csv(encoding_path, encoding_rows)
    _write_csv(geometry_path, geometry_rows)


def _import_efficiency(
    config: dict[str, Any],
    cluster_root: Path,
    model_ids: list[str],
) -> None:
    output_root = PROJECT_ROOT / str(config["output_root"])
    profile_path = output_root / "efficiency/efficiency_profiles_admission_panel.csv"
    resource_path = (
        output_root / "efficiency/resource_allocation_profiles_admission_panel.csv"
    )
    profile_rows = [
        row for row in _read_csv(profile_path) if row["model_id"] not in model_ids
    ]
    resource_rows = [
        row for row in _read_csv(resource_path) if row["model_id"] not in model_ids
    ]
    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        artifact_root = cluster_root / "external" / "neural" / model_id
        profile = json.loads(
            (artifact_root / "efficiency.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (artifact_root / "manifest.json").read_text(encoding="utf-8")
        )
        hardware = dict(manifest.get("provenance", {}).get("hardware", {}))
        profile_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "comparability_group": "adaptive_token_224_cluster_l40s",
                "profile_status": "complete",
                "parameters": profile.get("parameters", ""),
                "latency_ms_per_image": profile.get("latency_ms_per_image", ""),
                "peak_memory_bytes": profile.get("peak_memory_bytes", ""),
                "theoretical_flops": profile.get("theoretical_flops", ""),
                "realized_flops": profile.get("realized_flops", ""),
                "device_name": hardware.get("device_name", ""),
                "torch_version": hardware.get("torch_version", ""),
                "cuda_version": hardware.get("cuda_version", ""),
            }
        )
        resource_rows.append(
            {
                "evidence_label": str(config["evidence_label"]),
                "model_id": model_id,
                "family": spec["family"],
                "role": spec["role"],
                "resource_mode": spec["resource_mode"],
                "input_height": 224,
                "input_width": 224,
                "batch_size": 1,
                "sequential_or_adaptive": "true",
                "total_cost_scope": "per_image",
                "total_cost_value": profile.get("realized_flops", ""),
                "total_cost_unit": "realized_flops",
                "resource_trace_status": "complete_operational_trace",
                "resource_summary_json": json.dumps(
                    profile.get("resource_summary", {}),
                    sort_keys=True,
                ),
            }
        )
    _write_csv(profile_path, profile_rows)
    _write_csv(resource_path, resource_rows)


if __name__ == "__main__":
    raise SystemExit(main())
