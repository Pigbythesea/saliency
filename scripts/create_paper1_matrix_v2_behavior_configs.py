"""Generate routing-map behavioral configs for the first Matrix V2 models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml, save_yaml
from hma.utils.paths import resolve_path


DATASETS = {
    "salicon": {
        "label": "salicon_static2000",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/v2/salicon_static2000_manifest.csv",
        "split": "val",
    },
    "cat2000": {
        "label": "cat2000_static2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/v2/cat2000_static2000_manifest.csv",
        "split": "train",
        "categories": None,
    },
    "coco_search18": {
        "label": "coco_search18_static2000",
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/v2/coco_search18_static2000_manifest.csv",
        "split": "val",
        "generate_fixation_map": True,
        "fixation_sigma": 10.0,
    },
}


def create_behavior_configs(
    *,
    matrix_path: str | Path = "configs/paper1_matrix_v2.yaml",
    output_root: str | Path = "configs/experiments/paper1_matrix_v2/behavior",
) -> list[Path]:
    matrix = load_yaml(resolve_path(matrix_path))
    models = matrix["first_evidence_run"]["models"]
    paths = []
    for dataset_name, dataset in DATASETS.items():
        for model_id in models:
            name = f"{dataset['label']}_{model_id}_operational_routing"
            config = _behavior_config(
                name=name,
                dataset_name=dataset_name,
                dataset=dict(dataset),
                model_id=str(model_id),
            )
            path = resolve_path(output_root) / f"{name}.yaml"
            save_yaml(config, path)
            paths.append(path)
    return paths


def _behavior_config(
    *,
    name: str,
    dataset_name: str,
    dataset: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    dataset_config = {
        "name": dataset_name,
        **dataset,
        "image_size": [224, 224],
        "max_items": None,
        "validate_files": True,
    }
    return {
        "seed": 123,
        "device": "cpu",
        "experiment": {
            "name": name,
            "matrix_version": "paper1_matrix_v2",
            "object_type": "operational_resource_allocation",
        },
        "dataset": dataset_config,
        "model": {"name": model_id},
        "saliency": {
            "method": "precomputed_map",
            "root": (
                f"outputs/paper1_matrix_v2/routing_maps/"
                f"{dataset['label']}/{model_id}"
            ),
            "path_template": "{image_id}.npy",
            "object_type": "operational_resource_allocation",
        },
        "metric_controls": {
            "seed": 123,
            "auc_borji_splits": 100,
            "shuffled_auc_splits": 100,
            "max_positive_fixations_per_image": 256,
            "shuffled_auc_pool_points_per_image": 256,
            "emd_downsample": 32,
        },
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": [
            "nss",
            "shuffled_auc",
            "auc_borji",
            "auc_judd",
            "cc",
            "similarity",
            "kl",
        ],
        "output": {
            "dir": (
                f"outputs/paper1_matrix_v2/behavior/{dataset['label']}/"
                f"{model_id}_operational_routing"
            ),
            "save_visualizations": False,
            "num_visualizations": 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/paper1_matrix_v2.yaml")
    parser.add_argument(
        "--output-root", default="configs/experiments/paper1_matrix_v2/behavior"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = create_behavior_configs(
        matrix_path=args.matrix,
        output_root=args.output_root,
    )
    print(f"Generated {len(paths)} Matrix V2 behavioral configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
