"""Small saliency metrics for synthetic arrays."""

from __future__ import annotations

import numpy as np


def _as_flat_arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape}, target {true.shape}")
    return pred.ravel(), true.ravel()


def mean_absolute_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return mean absolute error between two maps."""
    pred, true = _as_flat_arrays(prediction, target)
    return float(np.mean(np.abs(pred - true)))


def pearson_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return Pearson correlation, using 0.0 when either map is constant."""
    pred, true = _as_flat_arrays(prediction, target)
    pred_centered = pred - float(np.mean(pred))
    true_centered = true - float(np.mean(true))

    denominator = float(np.linalg.norm(pred_centered) * np.linalg.norm(true_centered))
    if np.isclose(denominator, 0.0):
        return 0.0

    return float(np.dot(pred_centered, true_centered) / denominator)
