"""Prepare operational routing maps and a runtime behavioral config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml, save_yaml
from hma.utils.paths import resolve_path
from scripts.create_paper1_matrix_v2_behavior_configs import DATASETS
from scripts.export_external_routing_maps import export_routing_maps


def prepare_behavior_cell(
    *,
    dataset_name: str,
    model_id: str,
    artifact_dir: str | Path,
    stage: str,
    max_items: int | None,
) -> Path:
    dataset = DATASETS[dataset_name]
    label = str(dataset["label"])
    base = resolve_path(
        "configs/experiments/paper1_matrix_v2/behavior/"
        f"{label}_{model_id}_operational_routing.yaml"
    )
    config = load_yaml(base)
    map_dir = resolve_path(
        f"outputs/paper1_matrix_v2/routing_maps/{stage}/{label}/{model_id}"
    )
    export_routing_maps(artifact_dir, map_dir)
    config["dataset"]["max_items"] = max_items
    config["saliency"]["root"] = str(map_dir)
    config["output"]["dir"] = str(
        resolve_path(
            f"outputs/paper1_matrix_v2/behavior/{stage}/{label}/"
            f"{model_id}_operational_routing"
        )
    )
    runtime_config = resolve_path(
        f"outputs/paper1_matrix_v2/runtime_configs/{stage}/"
        f"{label}_{model_id}.yaml"
    )
    save_yaml(config, runtime_config)
    return runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = prepare_behavior_cell(
        dataset_name=args.dataset,
        model_id=args.model,
        artifact_dir=args.artifact,
        stage=args.stage,
        max_items=args.max_items,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
