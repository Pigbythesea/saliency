"""Prepare one cluster routing-map behavioral admission cell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml, save_yaml
from hma.utils.paths import resolve_path
from scripts.export_external_routing_maps import export_routing_maps


def prepare_behavior_cell(model_id: str, artifact_dir: str | Path) -> Path:
    base = resolve_path(
        "configs/experiments/paper1_matrix_v2/behavior/"
        f"salicon_static2000_{model_id}_operational_routing.yaml"
    )
    config = load_yaml(base)
    map_dir = resolve_path(
        f"outputs/paper1_admission_v1/cluster/routing_maps/{model_id}"
    )
    export_routing_maps(artifact_dir, map_dir)
    config["experiment"]["name"] = f"admission_{model_id}_salicon_free_viewing"
    config["experiment"]["matrix_version"] = "paper1_admission_v1"
    config["dataset"]["label"] = "salicon_admission"
    config["dataset"]["max_items"] = 128
    config["saliency"]["root"] = str(map_dir)
    config["metric_controls"]["matched_prior"] = {"type": "center_bias"}
    config["metrics"] = [
        "log_likelihood_bits",
        "information_gain_vs_matched_prior",
        "nss",
        "auc_judd",
        "cc",
        "similarity",
        "kl",
    ]
    config["output"]["dir"] = str(
        resolve_path(f"outputs/paper1_admission_v1/cluster/behavior/{model_id}")
    )
    runtime_config = resolve_path(
        f"outputs/paper1_admission_v1/cluster/runtime_configs/{model_id}.yaml"
    )
    save_yaml(config, runtime_config)
    return runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(prepare_behavior_cell(args.model, args.artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
