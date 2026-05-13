"""Shared saliency map postprocessing."""

from __future__ import annotations

import numpy as np


def normalize_saliency_map(values: np.ndarray) -> np.ndarray:
    """Normalize a saliency map into the [0, 1] range."""
    array = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))

    if np.isclose(maximum, minimum):
        return np.zeros_like(array, dtype=np.float32)

    return ((array - minimum) / (maximum - minimum)).astype(np.float32)
