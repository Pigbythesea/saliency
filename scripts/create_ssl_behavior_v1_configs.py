"""Generate behavioral saliency configs for pretrained SSL/VLM bridge rows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hma.utils.config import save_yaml

try:
    from scripts.create_real_matrix_v2_configs import (
        DATASETS,
        METRIC_CONTROLS,
        METRICS,
        PILOT_METRICS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from create_real_matrix_v2_configs import (
        DATASETS,
        METRIC_CONTROLS,
        METRICS,
        PILOT_METRICS,
    )


CONFIG_ROOT = Path("configs/experiments/real_matrix_v2_ssl_behavior")
OUTPUT_ROOT = "outputs/real_matrix_v2_ssl_behavior"

MODEL_SALIENCY_SPECS = [
    {
        "model": "vit_small_patch14_dinov2",
        "method": "vanilla_gradient",
        "input_size": [518, 518],
    },
    {
        "model": "vit_small_patch14_dinov2",
        "method": "attention_rollout",
        "input_size": [518, 518],
    },
    {
        "model": "vit_base_patch16_clip_224",
        "method": "vanilla_gradient",
        "input_size": [224, 224],
    },
    {
        "model": "vit_base_patch16_clip_224",
        "method": "attention_rollout",
        "input_size": [224, 224],
    },
    {
        "model": "resnet50_clip",
        "method": "vanilla_gradient",
        "input_size": [224, 224],
    },
    {
        "model": "resnet50_clip",
        "method": "gradcam",
        "input_size": [224, 224],
        "saliency_extra": {"target_layer": "stages.3"},
    },
]


def create_ssl_behavior_v1_configs(
    *,
    config_root: str | Path = CONFIG_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
    datasets: Iterable[dict] | None = None,
    model_saliency_specs: Iterable[dict] | None = None,
) -> list[Path]:
    """Write SSL/VLM behavioral bridge configs and return their paths."""
    root = Path(config_root)
    root.mkdir(parents=True, exist_ok=True)
    selected_datasets = list(datasets) if datasets is not None else list(DATASETS)
    selected_specs = (
        list(model_saliency_specs)
        if model_saliency_specs is not None
        else list(MODEL_SALIENCY_SPECS)
    )

    written: list[Path] = []
    for dataset in selected_datasets:
        for spec in selected_specs:
            written.append(
                _write_config(
                    root=root,
                    output_root=str(output_root),
                    dataset=dataset,
                    model_name=str(spec["model"]),
                    method=str(spec["method"]),
                    input_size=list(spec["input_size"]),
                    saliency_extra=dict(spec.get("saliency_extra") or {}),
                )
            )
    return written


def _write_config(
    *,
    root: Path,
    output_root: str,
    dataset: dict,
    model_name: str,
    method: str,
    input_size: list[int],
    saliency_extra: dict,
) -> Path:
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
        "model": {
            "name": model_name,
            "backend": "timm",
            "pretrained": True,
            "eval_mode": True,
        },
        "preprocessing": {
            "input_size": input_size,
            "mean": "imagenet",
            "std": "imagenet",
        },
        "saliency": {"method": method, **saliency_extra},
        "metric_controls": dict(METRIC_CONTROLS),
        "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
        "metrics": list(metrics),
        "output": {
            "dir": f"{output_root}/{dataset['label']}/{run_name}",
            "save_visualizations": dataset["scale"] == "pilot",
            "num_visualizations": 5 if dataset["scale"] == "pilot" else 0,
        },
    }

    path = root / f"{dataset['label']}__{run_name}.yaml"
    save_yaml(config, path)
    return path


def main() -> None:
    written = create_ssl_behavior_v1_configs()
    print(f"Generated {len(written)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
