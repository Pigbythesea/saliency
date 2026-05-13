"""Generate experiment configs for the controlled V2 real-data matrix."""

from __future__ import annotations

from pathlib import Path

from hma.utils.config import save_yaml


MATRIX_ROOT = Path("configs/experiments/real_matrix_v2")
OUTPUT_ROOT = "outputs/real_matrix_v2"
METRICS = ["nss", "shuffled_auc", "auc_borji", "auc_judd", "cc", "similarity", "kl"]
PILOT_METRICS = [*METRICS, "emd"]
METRIC_CONTROLS = {
    "seed": 123,
    "auc_borji_splits": 100,
    "shuffled_auc_splits": 100,
    "max_positive_fixations_per_image": 256,
    "shuffled_auc_pool_points_per_image": 256,
    "emd_downsample": 32,
}

DATASETS = [
    {
        "label": "salicon_pilot500",
        "name": "salicon",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/pilot/salicon_pilot500_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "scale": "pilot",
        "extra": {},
    },
    {
        "label": "cat2000_pilot500",
        "name": "cat2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/pilot/cat2000_pilot500_manifest.csv",
        "split": "train",
        "image_size": [224, 224],
        "scale": "pilot",
        "extra": {"categories": None},
    },
    {
        "label": "coco_search18_pilot500",
        "name": "coco_search18",
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/pilot/coco_search18_pilot500_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "scale": "pilot",
        "extra": {"generate_fixation_map": True, "fixation_sigma": 10.0},
    },
    {
        "label": "salicon_static2000",
        "name": "salicon",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/v2/salicon_static2000_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "scale": "static2000",
        "extra": {},
    },
    {
        "label": "cat2000_static2000",
        "name": "cat2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/v2/cat2000_static2000_manifest.csv",
        "split": "train",
        "image_size": [224, 224],
        "scale": "static2000",
        "extra": {"categories": None},
    },
    {
        "label": "coco_search18_static2000",
        "name": "coco_search18",
        "root": "data/raw/COCO-Search18",
        "manifest_path": "data/manifests/v2/coco_search18_static2000_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "scale": "static2000",
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

VANILLA_GRADIENT_MODELS = list(MODELS)
ATTENTION_ROLLOUT_MODELS = ["vit_base_patch16_224", "deit_small_patch16_224"]
GRADCAM_CONFIGS = {
    "resnet50": {"target_layer": "layer4"},
    "convnext_tiny": {"target_layer": "stages.3"},
}
PILOT_INTEGRATED_GRADIENT_MODELS = ["resnet50", "vit_base_patch16_224"]


def main() -> None:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for baseline in BASELINES:
            _write_config(dataset, baseline["model"], baseline["method"], baseline=baseline)
        for model in VANILLA_GRADIENT_MODELS:
            _write_config(dataset, model, "vanilla_gradient")
        for model in ATTENTION_ROLLOUT_MODELS:
            _write_config(dataset, model, "attention_rollout")
        for model, saliency_extra in GRADCAM_CONFIGS.items():
            _write_config(dataset, model, "gradcam", saliency_extra=saliency_extra)
        if dataset["scale"] == "pilot":
            for model in PILOT_INTEGRATED_GRADIENT_MODELS:
                _write_config(
                    dataset,
                    model,
                    "integrated_gradients",
                    saliency_extra={"steps": 16},
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


def _write_config(
    dataset: dict,
    model_name: str,
    method: str,
    *,
    baseline: dict | None = None,
    saliency_extra: dict | None = None,
) -> None:
    run_name = f"{model_name}_{method}"
    metrics = PILOT_METRICS if dataset["scale"] == "pilot" else METRICS
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
        "metric_controls": dict(METRIC_CONTROLS),
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": list(metrics),
        "output": {
            "dir": f"{OUTPUT_ROOT}/{dataset['label']}/{run_name}",
            "save_visualizations": dataset["scale"] == "pilot",
            "num_visualizations": 5 if dataset["scale"] == "pilot" else 0,
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
