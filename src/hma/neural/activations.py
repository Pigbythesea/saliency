"""Activation extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def extract_activations(
    model_wrapper: Any,
    dataloader: Iterable,
    layers: list[str],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Extract layer activations from item or batch dictionaries."""
    image_ids: list[str] = []
    collected: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}

    for batch in dataloader:
        images = batch["image"]
        image_ids.extend(_as_id_list(batch.get("image_id", "")))
        images = _move_to_device(images, device)
        features = model_wrapper.get_features(images, layers=layers)
        feature_dict = _normalize_feature_output(features, layers)
        for layer in layers:
            collected[layer].append(_to_numpy(feature_dict[layer]))

    activations: dict[str, np.ndarray] = {
        "image_ids": np.asarray(image_ids, dtype=object)
    }
    for layer, values in collected.items():
        activations[layer] = np.concatenate(values, axis=0) if values else np.empty((0,))
    return activations


def save_activations(activations: Any, path: str | Path) -> Path:
    """Save activations as .npy for arrays or .npz for dictionaries."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(activations, dict):
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        np.savez(output, **activations)
        return output

    if output.suffix != ".npy":
        output = output.with_suffix(".npy")
    np.save(output, np.asarray(activations))
    return output


def _normalize_feature_output(features: Any, layers: list[str]) -> dict[str, Any]:
    if isinstance(features, dict):
        return features
    if len(layers) == 1:
        return {layers[0]: features}
    if isinstance(features, (list, tuple)) and len(features) == len(layers):
        return dict(zip(layers, features))
    raise ValueError("Model features must be a dict or match requested layers")


def _as_id_list(image_id: Any) -> list[str]:
    if isinstance(image_id, (list, tuple)):
        return [str(value) for value in image_id]
    array = np.asarray(image_id)
    if array.ndim > 0:
        return [str(value) for value in array.tolist()]
    return [str(image_id)]


def _move_to_device(values: Any, device: str) -> Any:
    to = getattr(values, "to", None)
    if callable(to):
        return values.to(device)
    return values


def _to_numpy(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        return values.detach().cpu().numpy()
    return np.asarray(values)
