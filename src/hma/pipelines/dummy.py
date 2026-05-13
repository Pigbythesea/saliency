"""Dummy end-to-end saliency pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from hma.config import load_yaml_config
from hma.data import DummySaliencyDataset
from hma.metrics import mean_absolute_error, pearson_correlation
from hma.models import DummySaliencyModel


def run_dummy_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run a deterministic synthetic saliency benchmark."""
    config = load_yaml_config(config_path)
    dataset = DummySaliencyDataset.from_config(config.get("dataset", {}))
    model = DummySaliencyModel.from_config(config.get("model", {}))

    mae_values: list[float] = []
    pearson_values: list[float] = []

    for item_index, item in enumerate(dataset):
        prediction = model.predict(np.asarray(item["image"]), item_index=item_index)
        target = np.asarray(item["fixation_map"])
        mae_values.append(mean_absolute_error(prediction, target))
        pearson_values.append(pearson_correlation(prediction, target))

    return {
        "experiment": config.get("experiment", {}).get("name", "dummy_pipeline"),
        "num_items": len(dataset),
        "mae": float(np.mean(mae_values)) if mae_values else 0.0,
        "pearson": float(np.mean(pearson_values)) if pearson_values else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HMA dummy saliency pipeline.")
    parser.add_argument(
        "--config",
        default="configs/experiments/dummy_pipeline.yaml",
        help="Path to a dummy experiment YAML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_dummy_pipeline(args.config)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
