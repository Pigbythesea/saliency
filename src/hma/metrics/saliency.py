"""Small saliency metrics for synthetic arrays."""

from __future__ import annotations

import numpy as np

from hma.metrics.saliency_metrics import cc


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
    return cc(prediction, target)
