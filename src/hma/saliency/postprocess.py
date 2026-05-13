"""Shared saliency map postprocessing."""

from __future__ import annotations

import numpy as np

from hma.metrics.saliency_metrics import normalize_map


def normalize_saliency_map(values: np.ndarray) -> np.ndarray:
    """Normalize a saliency map into the [0, 1] range."""
    return normalize_map(values)


def resize_saliency_map(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D saliency map with nearest-neighbor sampling."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D saliency map, got shape {array.shape}")

    target_height, target_width = target_shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_shape dimensions must be positive")

    source_height, source_width = array.shape
    if (source_height, source_width) == (target_height, target_width):
        return array.copy()

    row_indices = np.floor(
        np.linspace(0, source_height, target_height, endpoint=False)
    ).astype(int)
    col_indices = np.floor(
        np.linspace(0, source_width, target_width, endpoint=False)
    ).astype(int)
    row_indices = np.clip(row_indices, 0, source_height - 1)
    col_indices = np.clip(col_indices, 0, source_width - 1)
    return array[np.ix_(row_indices, col_indices)].astype(np.float32)


def postprocess_saliency_map(
    values: np.ndarray,
    target_shape: tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Apply standard saliency resizing and normalization."""
    processed = np.asarray(values, dtype=np.float32)
    if target_shape is not None:
        processed = resize_saliency_map(processed, target_shape)
    if normalize:
        processed = normalize_saliency_map(processed)
    return processed.astype(np.float32)
