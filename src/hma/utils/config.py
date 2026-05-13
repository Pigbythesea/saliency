"""YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "device": "cpu",
    "dataset": {
        "name": None,
        "root": "data",
        "max_items": None,
    },
    "model": {
        "name": None,
    },
    "saliency": {
        "method": None,
    },
    "metrics": [],
    "output": {
        "dir": "outputs/default",
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level of {config_path}")
    return data


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to YAML, creating the parent directory if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    merged = deepcopy(base)

    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = merge_dicts(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)

    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config over the default experiment skeleton."""
    return merge_dicts(DEFAULT_EXPERIMENT_CONFIG, load_yaml(path))
