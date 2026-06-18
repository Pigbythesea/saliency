"""Prepare one clean Paper 1 behavioral runtime config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import save_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


DATASETS: dict[str, dict[str, Any]] = {
    "salicon": {
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/salicon_manifest.csv",
        "split": "val",
    },
    "cat2000": {
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/cat2000_manifest.csv",
        "split": "train",
        "categories": None,
    },
    "coco_search18": {
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/coco_search18_manifest.csv",
        "split": "val",
        "generate_fixation_map": True,
        "fixation_sigma": 10.0,
    },
}

BUILTIN_METHODS = {
    "center_prior": "center_bias",
    "random_baseline": "random_saliency",
    "empirical_spatial_prior": "empirical_spatial_prior",
    "coco_search18_task_prior": "coco_search18_task_prior",
}


def prepare_behavior_cell(
    *,
    dataset_name: str,
    model_id: str,
    output_dir: str | Path,
    runtime_config: str | Path,
    map_root: str | Path | None = None,
    builtin_method: str | None = None,
    max_items: int | None = None,
    seed: int = 123,
) -> Path:
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown clean behavioral dataset: {dataset_name}")
    dataset = dict(DATASETS[dataset_name])
    dataset["name"] = dataset_name
    dataset["label"] = f"paper1_clean_{dataset_name}"
    dataset["image_size"] = [224, 224]
    dataset["max_items"] = max_items
    dataset["validate_files"] = True
    saliency = _saliency_config(
        dataset_name=dataset_name,
        model_id=model_id,
        map_root=map_root,
        builtin_method=builtin_method,
        seed=seed,
    )
    config = {
        "seed": seed,
        "device": "cpu",
        "experiment": {
            "name": f"paper1_clean_{dataset_name}_{model_id}",
            "matrix_version": "paper1_publication_v0",
            "object_type": "clean_behavioral_map",
        },
        "dataset": dataset,
        "model": {"name": model_id},
        "preprocessing": {
            "input_size": [224, 224],
            "mean": "imagenet",
            "std": "imagenet",
        },
        "saliency": saliency,
        "metric_controls": {
            "seed": seed,
            "auc_borji_splits": 100,
            "shuffled_auc_splits": 100,
            "max_positive_fixations_per_image": 256,
            "shuffled_auc_pool_points_per_image": 256,
            "emd_downsample": 32,
        },
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": ["nss", "shuffled_auc", "auc_borji", "auc_judd", "cc", "similarity", "kl"],
        "output": {
            "dir": str(output_dir),
            "save_visualizations": False,
            "num_visualizations": 0,
        },
    }
    path = resolve_path(runtime_config)
    save_yaml(config, path)
    return path


def _saliency_config(
    *,
    dataset_name: str,
    model_id: str,
    map_root: str | Path | None,
    builtin_method: str | None,
    seed: int,
) -> dict[str, Any]:
    if builtin_method:
        method = BUILTIN_METHODS.get(builtin_method, builtin_method)
        config: dict[str, Any] = {"method": method}
        if method == "random_saliency":
            config["seed"] = seed
        if method == "empirical_spatial_prior":
            config.update(
                {
                    "prior_manifest_path": DATASETS[dataset_name]["manifest_path"],
                    "prior_split": "train",
                    "image_size": [224, 224],
                    "fixation_sigma": 10.0,
                }
            )
        if method == "coco_search18_task_prior":
            config.update(
                {
                    "prior_manifest_path": "data/manifests/coco_search18_manifest.csv",
                    "prior_split": "train",
                    "image_size": [224, 224],
                    "fixation_sigma": 10.0,
                }
            )
        return config
    if map_root is None:
        raise ValueError(f"External behavioral cell requires map_root for {model_id}")
    return {
        "method": "precomputed_map",
        "root": str(map_root),
        "path_template": "{map_key}.npy",
        "object_type": "clean_external_prediction_map",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--map-root")
    parser.add_argument("--builtin-method")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = prepare_behavior_cell(
        dataset_name=args.dataset,
        model_id=args.model,
        output_dir=args.output_dir,
        runtime_config=args.runtime_config,
        map_root=args.map_root,
        builtin_method=args.builtin_method,
        max_items=args.max_items,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
