"""Core saliency and fixation metrics."""

from __future__ import annotations

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = _as_float_array(a)
    second = _as_float_array(b)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} != {second.shape}")
    return first, second


def _fixation_mask(fixation_map: np.ndarray) -> np.ndarray:
    return _as_float_array(fixation_map) > 0


def _sum_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.maximum(_as_float_array(values), 0.0)
    total = float(np.sum(array))
    if not np.isfinite(total) or total <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array / total).astype(np.float32)


def normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize a map into [0, 1]."""
    array = _as_float_array(x)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return np.zeros_like(array, dtype=np.float32)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum <= eps:
        return np.zeros_like(array, dtype=np.float32)

    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def auc_judd(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """Compute Judd AUC using positive fixation-map values as fixation locations."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    saliency = normalize_map(saliency).ravel()
    fixation_mask = _fixation_mask(fixations).ravel()

    num_fixations = int(np.sum(fixation_mask))
    num_pixels = fixation_mask.size
    num_nonfixations = num_pixels - num_fixations
    if num_fixations == 0 or num_nonfixations == 0:
        return 0.0

    order = np.argsort(-saliency, kind="mergesort")
    sorted_fixations = fixation_mask[order].astype(np.float32)
    true_positive = np.cumsum(sorted_fixations) / float(num_fixations)
    false_positive = np.cumsum(1.0 - sorted_fixations) / float(num_nonfixations)

    true_positive = np.concatenate([[0.0], true_positive, [1.0]])
    false_positive = np.concatenate([[0.0], false_positive, [1.0]])
    return float(np.trapezoid(true_positive, false_positive))


def nss(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """Compute normalized scanpath saliency."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    fixation_mask = _fixation_mask(fixations)
    if not np.any(fixation_mask):
        return 0.0

    std = float(np.std(saliency))
    if not np.isfinite(std) or np.isclose(std, 0.0):
        return 0.0

    normalized = (saliency - float(np.mean(saliency))) / std
    return float(np.mean(normalized[fixation_mask]))


def cc(saliency_map_a: np.ndarray, saliency_map_b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two saliency maps."""
    first, second = _check_same_shape(saliency_map_a, saliency_map_b)
    first = first.ravel()
    second = second.ravel()
    first_centered = first - float(np.mean(first))
    second_centered = second - float(np.mean(second))

    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return 0.0

    return float(np.dot(first_centered, second_centered) / denominator)


def similarity(saliency_map_a: np.ndarray, saliency_map_b: np.ndarray) -> float:
    """Compute histogram intersection similarity between two saliency maps."""
    first, second = _check_same_shape(saliency_map_a, saliency_map_b)
    first = _sum_normalize(first)
    second = _sum_normalize(second)
    return float(np.sum(np.minimum(first, second)))


def kl_divergence(
    target_map: np.ndarray,
    predicted_map: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute stable KL(target || predicted) between two saliency distributions."""
    target, predicted = _check_same_shape(target_map, predicted_map)
    target_prob = _sum_normalize(target, eps=eps)
    predicted_prob = _sum_normalize(predicted, eps=eps)

    if float(np.sum(target_prob)) <= eps:
        return 0.0

    predicted_prob = np.maximum(predicted_prob, eps)
    target_prob = np.maximum(target_prob, eps)
    return float(np.sum(target_prob * np.log(target_prob / predicted_prob)))


def simple_center_bias_map(
    height: int,
    width: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Create a normalized Gaussian center-bias map."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    sigma = float(sigma) if sigma is not None else min(height, width) / 6.0
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    squared_distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
    center_map = np.exp(-squared_distance / (2.0 * sigma**2))
    return normalize_map(center_map)
