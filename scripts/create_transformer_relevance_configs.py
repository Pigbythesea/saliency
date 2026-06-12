"""Generate scoped transformer relevance behavioral benchmark configs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hma.utils.config import save_yaml

try:
    from scripts.create_real_matrix_v2_configs import METRIC_CONTROLS, METRICS
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from create_real_matrix_v2_configs import METRIC_CONTROLS, METRICS


DEBUG_CONFIG_ROOT = Path("configs/experiments/transformer_relevance_debug")
STATIC_CONFIG_ROOT = Path("configs/experiments/real_matrix_v2_transformer_relevance")
DEBUG_OUTPUT_ROOT = "outputs/transformer_relevance_debug"
STATIC_OUTPUT_ROOT = "outputs/real_matrix_v2_transformer_relevance"

STATIC_DATASETS = [
    {
        "label": "salicon_static2000",
        "name": "salicon",
        "root": "data/raw/SALICON",
        "manifest_path": "data/manifests/v2/salicon_static2000_manifest.csv",
        "split": "val",
        "image_size": [224, 224],
        "extra": {},
    },
    {
        "label": "cat2000_static2000",
        "name": "cat2000",
        "root": "data/raw/CAT2000",
        "manifest_path": "data/manifests/v2/cat2000_static2000_manifest.csv",
        "split": "train",
        "image_size": [224, 224],
        "extra": {"categories": None},
    },
]

DEBUG_DATASET = {
    "label": "salicon_transformer_relevance_smoke",
    "name": "salicon",
    "root": "data/raw/SALICON",
    "manifest_path": "data/manifests/pilot/salicon_pilot500_manifest.csv",
    "split": "val",
    "image_size": [224, 224],
    "max_items": 8,
    "extra": {},
}

TRANSFORMER_MODEL_SPECS = [
    {
        "model": "vit_small_patch14_dinov2",
        "input_size": [518, 518],
        "grid_size": [37, 37],
    },
    {
        "model": "vit_base_patch16_clip_224",
        "input_size": [224, 224],
        "grid_size": [14, 14],
    },
    {
        "model": "vit_base_patch16_224",
        "input_size": [224, 224],
        "grid_size": [14, 14],
    },
    {
        "model": "deit_small_patch16_224",
        "input_size": [224, 224],
        "grid_size": [14, 14],
    },
]


def create_transformer_relevance_configs(
    *,
    debug_config_root: str | Path = DEBUG_CONFIG_ROOT,
    static_config_root: str | Path = STATIC_CONFIG_ROOT,
    debug_output_root: str | Path = DEBUG_OUTPUT_ROOT,
    static_output_root: str | Path = STATIC_OUTPUT_ROOT,
    static_datasets: Iterable[dict] | None = None,
    model_specs: Iterable[dict] | None = None,
) -> list[Path]:
    """Write debug and static transformer relevance configs."""
    debug_root = Path(debug_config_root)
    static_root = Path(static_config_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    specs = list(model_specs) if model_specs is not None else list(TRANSFORMER_MODEL_SPECS)
    datasets = list(static_datasets) if static_datasets is not None else list(STATIC_DATASETS)

    written: list[Path] = []
    dino_spec = next(spec for spec in specs if spec["model"] == "vit_small_patch14_dinov2")
    written.append(
        _write_config(
            root=debug_root,
            output_root=str(debug_output_root),
            dataset=DEBUG_DATASET,
            spec=dino_spec,
            filename="salicon_vit_small_patch14_dinov2_transformer_relevance_smoke.yaml",
            metrics=["nss", "auc_judd", "cc", "similarity", "kl"],
            save_visualizations=True,
            num_visualizations=3,
        )
    )
    for dataset in datasets:
        for spec in specs:
            run_name = f"{spec['model']}_transformer_relevance"
            written.append(
                _write_config(
                    root=static_root,
                    output_root=str(static_output_root),
                    dataset=dataset,
                    spec=spec,
                    filename=f"{dataset['label']}__{run_name}.yaml",
                    metrics=list(METRICS),
                    save_visualizations=False,
                    num_visualizations=0,
                )
            )
    return written


def _write_config(
    *,
    root: Path,
    output_root: str,
    dataset: dict,
    spec: dict,
    filename: str,
    metrics: list[str],
    save_visualizations: bool,
    num_visualizations: int,
) -> Path:
    model_name = str(spec["model"])
    run_name = f"{model_name}_transformer_relevance"
    max_items = dataset.get("max_items")
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
            "max_items": max_items,
            "validate_files": True,
            **dataset["extra"],
        },
        "model": {
            "name": model_name,
            "backend": "timm",
            "pretrained": True,
            "eval_mode": True,
        },
        "preprocessing": {
            "input_size": list(spec["input_size"]),
            "mean": "imagenet",
            "std": "imagenet",
        },
        "saliency": {
            "method": "transformer_relevance",
            "grid_size": list(spec["grid_size"]),
            "head_fusion": "mean",
            "discard_ratio": 0.0,
        },
        "metric_controls": dict(METRIC_CONTROLS),
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": metrics,
        "output": {
            "dir": f"{output_root}/{dataset['label']}/{run_name}",
            "save_visualizations": save_visualizations,
            "num_visualizations": num_visualizations,
        },
    }

    path = root / filename
    save_yaml(config, path)
    return path


def main() -> None:
    written = create_transformer_relevance_configs()
    print(f"Generated {len(written)} transformer relevance configs")


if __name__ == "__main__":
    main()
