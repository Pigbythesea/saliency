"""Dataset registry helpers."""

from __future__ import annotations

from typing import Any, TypeVar

from hma.datasets.base import BaseVisionDataset


DatasetT = TypeVar("DatasetT", bound=BaseVisionDataset)

_DATASET_REGISTRY: dict[str, type[BaseVisionDataset]] = {}


def register_dataset(name: str, cls: type[DatasetT]) -> type[DatasetT]:
    """Register a dataset class under a stable config name."""
    existing = _DATASET_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"Dataset name already registered: {name}")
    _DATASET_REGISTRY[name] = cls
    return cls


def get_dataset_class(name: str) -> type[BaseVisionDataset]:
    """Return the dataset class registered for a name."""
    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc


def _extract_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    if "name" in config:
        return config
    dataset_config = config.get("dataset")
    if isinstance(dataset_config, dict):
        return dataset_config
    raise KeyError("Dataset config must contain a dataset name")


def build_dataset(config: dict[str, Any]) -> BaseVisionDataset:
    """Build a dataset from either a dataset config or full experiment config."""
    dataset_config = _extract_dataset_config(config)
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise KeyError("Dataset config must contain a non-empty 'name'")

    dataset_cls = get_dataset_class(str(dataset_name))
    from_config = getattr(dataset_cls, "from_config", None)
    if callable(from_config):
        return from_config(dataset_config)
    return dataset_cls(**dataset_config)
