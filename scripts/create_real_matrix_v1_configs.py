"""Generate experiment configs for the first real-data model matrix."""

from __future__ import annotations

from pathlib import Path

from hma.utils.config import save_yaml


MATRIX_ROOT = Path("configs/experiments/real_matrix_v1")
OUTPUT_ROOT = "outputs/real_matrix_v1"
METRICS = ["nss", "auc_judd", "cc", "similarity", "kl"]

DATASETS = [
    {
        "label": "salicon_pilot500",
        "name": "salicon",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/pilot/salicon_pilot500_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "extra": {},
    },
    {
        "label": "cat2000_pilot500",
        "name": "cat2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/pilot/cat2000_pilot500_manifest.csv",
        "split": "train",
        "image_size": [224, 224],
        "extra": {"categories": None},
    },
    {
        "label": "coco_search18_pilot500",
        "name": "coco_search18",
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/pilot/coco_search18_pilot500_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "extra": {"generate_fixation_map": True, "fixation_sigma": 10.0},
    },
]

BASELINES = [
    {"model": "center_bias_baseline", "method": "center_bias"},
    {"model": "random_baseline", "method": "random_saliency", "seed": 123},
]

MODELS = [
    "resnet50",
    "convnext_tiny",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def main() -> None:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for baseline in BASELINES:
            _write_config(dataset, baseline["model"], baseline["method"], baseline=baseline)
        for model in MODELS:
            _write_config(dataset, model, "vanilla_gradient")

    for dataset in DATASETS[:2]:
        _write_config(
            dataset,
            "resnet50",
            "gradcam",
            saliency_extra={"target_layer": "layer4"},
            output_suffix="resnet50_gradcam",
        )

    save_yaml(
        {
            "models": [
                {
                    "name": model,
                    "backend": "timm",
                    "pretrained": True,
                    "eval_mode": True,
                }
                for model in MODELS
            ]
        },
        "configs/models/selected_pretrained_matrix.yaml",
    )
    print(f"Generated configs under {MATRIX_ROOT}")
    print("Generated configs/models/selected_pretrained_matrix.yaml")


def _write_config(
    dataset: dict,
    model_name: str,
    method: str,
    *,
    baseline: dict | None = None,
    saliency_extra: dict | None = None,
    output_suffix: str | None = None,
) -> None:
    run_name = output_suffix or f"{model_name}_{method}"
    config = {
        "seed": 123,
        "device": "auto",
        "experiment": {"name": f"{dataset['label']}_{run_name}"},
        "dataset": {
            "name": dataset["name"],
            "label": dataset["label"],
            "root": dataset["root"],
            "manifest_path": dataset["manifest_path"],
            "split": dataset["split"],
            "image_size": dataset["image_size"],
            "max_items": None,
            "validate_files": True,
            **dataset["extra"],
        },
        "model": {"name": model_name},
        "preprocessing": {
            "input_size": [224, 224],
            "mean": "imagenet",
            "std": "imagenet",
        },
        "saliency": {"method": method, **(saliency_extra or {})},
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": list(METRICS),
        "output": {
            "dir": f"{OUTPUT_ROOT}/{dataset['label']}/{run_name}",
            "save_visualizations": True,
            "num_visualizations": 5,
        },
    }

    if baseline is None:
        config["model"] = {
            "name": model_name,
            "backend": "timm",
            "pretrained": True,
            "eval_mode": True,
        }
    elif "seed" in baseline:
        config["saliency"]["seed"] = baseline["seed"]

    save_yaml(config, MATRIX_ROOT / f"{dataset['label']}__{run_name}.yaml")


if __name__ == "__main__":
    main()
