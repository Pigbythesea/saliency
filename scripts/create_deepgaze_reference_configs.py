"""Generate V2 reference configs for precomputed DeepGaze-style maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from hma.utils.config import save_yaml


DATASETS = [
    {
        "label": "salicon_static2000",
        "name": "salicon",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/v2/salicon_static2000_manifest.csv",
        "split": "val",
        "extra": {},
    },
    {
        "label": "cat2000_static2000",
        "name": "cat2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/v2/cat2000_static2000_manifest.csv",
        "split": "train",
        "extra": {"categories": None},
    },
    {
        "label": "coco_search18_static2000",
        "name": "coco_search18",
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/v2/coco_search18_static2000_manifest.csv",
        "split": "val",
        "extra": {"generate_fixation_map": True, "fixation_sigma": 10.0},
    },
]
DATASET_BY_LABEL = {dataset["label"]: dataset for dataset in DATASETS}

METRICS = ["nss", "shuffled_auc", "auc_borji", "auc_judd", "cc", "similarity", "kl"]
METRIC_CONTROLS = {
    "seed": 123,
    "auc_borji_splits": 100,
    "shuffled_auc_splits": 100,
    "max_positive_fixations_per_image": 256,
    "shuffled_auc_pool_points_per_image": 256,
    "emd_downsample": 32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DeepGaze reference configs.")
    parser.add_argument(
        "--config-dir",
        default="configs/experiments/real_matrix_v2_references",
        help="Directory for generated reference YAML configs.",
    )
    parser.add_argument(
        "--precomputed-root",
        default="data/precomputed/deepgaze",
        help="Root containing one subdirectory per dataset label.",
    )
    parser.add_argument(
        "--path-template",
        default=None,
        help=(
            "Map path template relative to each dataset-label precomputed root. "
            "Defaults to {image_id}.npy for SALICON and {map_key}.npy for datasets "
            "with repeated image_id values."
        ),
    )
    parser.add_argument("--npz-key", default=None, help="Optional NPZ key for map arrays.")
    parser.add_argument(
        "--reference-name",
        default="deepgaze_reference",
        help="Model name written into generated configs.",
    )
    parser.add_argument(
        "--reference-label",
        default="deepgaze_precomputed",
        help=(
            "Output-dir and filename label for generated configs. The saliency method "
            "stays deepgaze_precomputed so the existing precomputed-map loader is used."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_BY_LABEL),
        default=[dataset["label"] for dataset in DATASETS],
        help="Dataset labels to generate. Use SALICON/CAT2000 only for MSDB free-viewing controls.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    for dataset_label in args.datasets:
        dataset = DATASET_BY_LABEL[dataset_label]
        config = _reference_config(
            dataset,
            precomputed_root=Path(args.precomputed_root),
            path_template=args.path_template or _default_path_template(dataset),
            npz_key=args.npz_key,
            reference_name=args.reference_name,
            reference_label=args.reference_label,
        )
        output = config_dir / f"{dataset['label']}__{args.reference_name}_{args.reference_label}.yaml"
        save_yaml(config, output)
        print(output)


def _reference_config(
    dataset: dict,
    *,
    precomputed_root: Path,
    path_template: str,
    npz_key: str | None,
    reference_name: str = "deepgaze_reference",
    reference_label: str = "deepgaze_precomputed",
) -> dict:
    saliency = {
        "method": "deepgaze_precomputed",
        "root": (precomputed_root / dataset["label"]).as_posix(),
        "path_template": path_template,
    }
    if npz_key:
        saliency["npz_key"] = npz_key

    return {
        "seed": 123,
        "device": "auto",
        "experiment": {"name": f"{dataset['label']}_{reference_name}"},
        "dataset": {
            "name": dataset["name"],
            "label": dataset["label"],
            "root": dataset["root"],
            "manifest_path": dataset["manifest_path"],
            "split": dataset["split"],
            "image_size": [224, 224],
            "max_items": None,
            "validate_files": True,
            **dataset["extra"],
        },
        "model": {"name": reference_name},
        "saliency": saliency,
        "metric_controls": dict(METRIC_CONTROLS),
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": list(METRICS),
        "output": {
            "dir": f"outputs/real_matrix_v2/{dataset['label']}/{reference_name}_{reference_label}",
            "save_visualizations": False,
            "num_visualizations": 0,
        },
    }


def _default_path_template(dataset: dict) -> str:
    if dataset["label"] in {"cat2000_static2000", "coco_search18_static2000"}:
        return "{map_key}.npy"
    return "{image_id}.npy"


if __name__ == "__main__":
    main()
