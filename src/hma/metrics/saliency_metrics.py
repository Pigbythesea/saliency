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


def is_constant_map(
    values: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    """Return whether a finite map is constant within numerical tolerance."""
    array = _as_float_array(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    return bool(np.allclose(array, float(array.flat[0]), rtol=rtol, atol=atol))


def _defined_distribution(values: np.ndarray) -> np.ndarray:
    array = _as_float_array(values)
    if is_constant_map(array) and float(array.flat[0]) > 0.0:
        return np.ones_like(array, dtype=np.float32)
    return array


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


def auc_judd(
    saliency_map: np.ndarray,
    fixation_map: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    constant_policy: str = "defined",
) -> float:
    """Compute Judd AUC using discrete fixation locations when available."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    if constant_policy == "defined" and is_constant_map(saliency):
        return 0.5
    saliency = normalize_map(saliency)
    fixation_mask = _coords_to_mask(
        _positive_coords(fixations, positive_fixations),
        saliency.shape,
    ).ravel()
    saliency = saliency.ravel()

    num_fixations = int(np.sum(fixation_mask))
    num_pixels = fixation_mask.size
    num_nonfixations = num_pixels - num_fixations
    if num_fixations == 0 or num_nonfixations == 0:
        return 0.0

    return _auc_from_scores(saliency[fixation_mask], saliency[~fixation_mask])


def auc_borji(
    saliency_map: np.ndarray,
    fixation_map: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    splits: int = 100,
    seed: int = 0,
    constant_policy: str = "defined",
) -> float:
    """Compute Borji AUC with random non-fixation negatives."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    if splits <= 0:
        raise ValueError("splits must be positive")

    if constant_policy == "defined" and is_constant_map(saliency):
        return 0.5
    saliency = normalize_map(saliency)
    positive_scores = _positive_scores(saliency, fixations, positive_fixations)
    if positive_scores.size == 0:
        return 0.0

    positive_mask = _coords_to_mask(
        _positive_coords(fixations, positive_fixations),
        saliency.shape,
    )
    negative_scores = saliency[~positive_mask].ravel()
    if negative_scores.size == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(int(splits)):
        sampled = rng.choice(
            negative_scores,
            size=positive_scores.size,
            replace=negative_scores.size < positive_scores.size,
        )
        scores.append(_auc_from_scores(positive_scores, sampled))
    return float(np.mean(scores)) if scores else 0.0


def shuffled_auc(
    saliency_map: np.ndarray,
    fixation_map: np.ndarray,
    negative_fixations: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    splits: int = 100,
    seed: int = 0,
    constant_policy: str = "defined",
) -> float:
    """Compute shuffled AUC using fixations from other images as negatives."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    if splits <= 0:
        raise ValueError("splits must be positive")

    if constant_policy == "defined" and is_constant_map(saliency):
        return 0.5
    saliency = normalize_map(saliency)
    positive_scores = _positive_scores(saliency, fixations, positive_fixations)
    negative_coords = _as_yx_coords(negative_fixations, saliency.shape)
    negative_scores = _scores_at_coords(saliency, negative_coords)
    if positive_scores.size == 0 or negative_scores.size == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(int(splits)):
        sampled = rng.choice(
            negative_scores,
            size=positive_scores.size,
            replace=negative_scores.size < positive_scores.size,
        )
        scores.append(_auc_from_scores(positive_scores, sampled))
    return float(np.mean(scores)) if scores else 0.0


def emd_2d(
    target_map: np.ndarray,
    predicted_map: np.ndarray,
    *,
    downsample: int | None = 32,
) -> float:
    """Compute a lightweight 2D EMD approximation from x/y map marginals."""
    target, predicted = _check_same_shape(target_map, predicted_map)
    if downsample is not None:
        size = int(downsample)
        if size <= 0:
            raise ValueError("downsample must be positive")
        target = _downsample_sum(target, size)
        predicted = _downsample_sum(predicted, size)

    target_prob = _sum_normalize(target)
    predicted_prob = _sum_normalize(predicted)
    if float(np.sum(target_prob)) == 0.0 and float(np.sum(predicted_prob)) == 0.0:
        return 0.0

    y_distance = _wasserstein_1d_from_marginals(
        target_prob.sum(axis=1),
        predicted_prob.sum(axis=1),
    )
    x_distance = _wasserstein_1d_from_marginals(
        target_prob.sum(axis=0),
        predicted_prob.sum(axis=0),
    )
    normalizer = max(target_prob.shape[0] + target_prob.shape[1] - 2, 1)
    return float((y_distance + x_distance) / normalizer)


def nss(
    saliency_map: np.ndarray,
    fixation_map: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    constant_policy: str = "defined",
) -> float:
    """Compute normalized scanpath saliency at fixation locations."""
    saliency, fixations = _check_same_shape(saliency_map, fixation_map)
    if constant_policy == "defined" and is_constant_map(saliency):
        return 0.0
    fixation_mask = _coords_to_mask(
        _positive_coords(fixations, positive_fixations),
        saliency.shape,
    )
    if not np.any(fixation_mask):
        return 0.0

    std = float(np.std(saliency))
    if not np.isfinite(std) or np.isclose(std, 0.0):
        return 0.0

    normalized = (saliency - float(np.mean(saliency))) / std
    return float(np.mean(normalized[fixation_mask]))


def cc(
    saliency_map_a: np.ndarray,
    saliency_map_b: np.ndarray,
    *,
    constant_policy: str = "defined",
) -> float:
    """Compute Pearson correlation coefficient between two saliency maps."""
    first, second = _check_same_shape(saliency_map_a, saliency_map_b)
    if constant_policy == "defined" and (
        is_constant_map(first) or is_constant_map(second)
    ):
        return 0.0
    first = first.ravel()
    second = second.ravel()
    first_centered = first - float(np.mean(first))
    second_centered = second - float(np.mean(second))

    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return 0.0

    return float(np.dot(first_centered, second_centered) / denominator)


def similarity(
    saliency_map_a: np.ndarray,
    saliency_map_b: np.ndarray,
    *,
    constant_policy: str = "defined",
) -> float:
    """Compute histogram intersection similarity between two saliency maps."""
    first, second = _check_same_shape(saliency_map_a, saliency_map_b)
    if constant_policy == "defined":
        first = _defined_distribution(first)
        second = _defined_distribution(second)
    first = _sum_normalize(first)
    second = _sum_normalize(second)
    return float(np.sum(np.minimum(first, second)))


def kl_divergence(
    target_map: np.ndarray,
    predicted_map: np.ndarray,
    eps: float = 1e-8,
    constant_policy: str = "defined",
) -> float:
    """Compute stable KL(target || predicted) between two saliency distributions."""
    target, predicted = _check_same_shape(target_map, predicted_map)
    if constant_policy == "defined":
        target = _defined_distribution(target)
        predicted = _defined_distribution(predicted)
    target_prob = _sum_normalize(target, eps=eps)
    predicted_prob = _sum_normalize(predicted, eps=eps)

    if float(np.sum(target_prob)) <= eps:
        return 0.0

    predicted_prob = np.maximum(predicted_prob, eps)
    target_prob = np.maximum(target_prob, eps)
    return float(np.sum(target_prob * np.log(target_prob / predicted_prob)))


def probabilistic_log_likelihood(
    predicted_map: np.ndarray,
    target_map: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return mean base-2 log likelihood under a probabilistic saliency map."""
    predicted, target = _check_same_shape(predicted_map, target_map)
    probability = _probability_distribution(predicted, epsilon=epsilon)
    if positive_fixations is not None:
        coords = _as_yx_coords_preserve_repeats(
            positive_fixations,
            probability.shape,
        )
        if coords.size == 0:
            return 0.0
        values = probability[coords[:, 0], coords[:, 1]]
        return float(np.mean(np.log2(np.maximum(values, epsilon))))

    target_probability = _probability_distribution(target, epsilon=epsilon)
    return float(
        np.sum(
            target_probability
            * np.log2(np.maximum(probability, epsilon))
        )
    )


def information_gain(
    predicted_map: np.ndarray,
    baseline_map: np.ndarray,
    target_map: np.ndarray,
    *,
    positive_fixations: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return mean information gain in bits against a matched prior."""
    predicted, target = _check_same_shape(predicted_map, target_map)
    baseline, _ = _check_same_shape(baseline_map, target)
    return probabilistic_log_likelihood(
        predicted,
        target,
        positive_fixations=positive_fixations,
        epsilon=epsilon,
    ) - probabilistic_log_likelihood(
        baseline,
        target,
        positive_fixations=positive_fixations,
        epsilon=epsilon,
    )


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


def _positive_scores(
    saliency: np.ndarray,
    fixation_map: np.ndarray,
    positive_fixations: np.ndarray | None,
) -> np.ndarray:
    return _scores_at_coords(saliency, _positive_coords(fixation_map, positive_fixations))


def _positive_coords(
    fixation_map: np.ndarray,
    positive_fixations: np.ndarray | None,
) -> np.ndarray:
    if positive_fixations is not None:
        return _as_yx_coords(positive_fixations, fixation_map.shape)
    return np.argwhere(_fixation_mask(fixation_map)).astype(np.int64)


def _as_yx_coords(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if array.ndim == 2 and array.shape == shape:
        return np.argwhere(_fixation_mask(array)).astype(np.int64)
    coords = np.asarray(array, dtype=np.float32).reshape(-1, 2)
    rounded = np.rint(coords).astype(np.int64)
    height, width = shape
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 0] < height)
        & (rounded[:, 1] < width)
    )
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(rounded[valid], axis=0)


def _as_yx_coords_preserve_repeats(
    values: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if array.ndim == 2 and array.shape == shape:
        return np.argwhere(_fixation_mask(array)).astype(np.int64)
    coords = np.asarray(array, dtype=np.float32).reshape(-1, 2)
    rounded = np.rint(coords).astype(np.int64)
    height, width = shape
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 0] < height)
        & (rounded[:, 1] < width)
    )
    return rounded[valid] if np.any(valid) else np.zeros((0, 2), dtype=np.int64)


def _probability_distribution(
    values: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("probabilistic saliency maps must be finite 2D arrays")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= epsilon:
        return np.full(array.shape, 1.0 / array.size, dtype=np.float64)
    probability = array / total
    probability = np.maximum(probability, epsilon)
    return probability / float(np.sum(probability))


def _scores_at_coords(saliency: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return saliency[coords[:, 0], coords[:, 1]].astype(np.float32)


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _auc_from_scores(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    if positive_scores.size == 0 or negative_scores.size == 0:
        return 0.0
    num_positive = int(positive_scores.size)
    num_negative = int(negative_scores.size)
    scores = np.concatenate([positive_scores, negative_scores]).astype(np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    positive_rank_sum = float(np.sum(ranks[:num_positive]))
    auc = (
        positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    ) / float(num_positive * num_negative)
    return float(np.clip(auc, 0.0, 1.0))


def _downsample_sum(values: np.ndarray, target_size: int) -> np.ndarray:
    array = _as_float_array(values)
    height, width = array.shape
    if height <= target_size and width <= target_size:
        return array
    y_edges = np.linspace(0, height, min(target_size, height) + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, min(target_size, width) + 1, dtype=np.int64)
    output = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.float32)
    for y_index in range(output.shape[0]):
        for x_index in range(output.shape[1]):
            patch = array[
                y_edges[y_index] : y_edges[y_index + 1],
                x_edges[x_index] : x_edges[x_index + 1],
            ]
            output[y_index, x_index] = float(np.sum(patch))
    return output


def _wasserstein_1d_from_marginals(first: np.ndarray, second: np.ndarray) -> float:
    first_prob = _sum_normalize(first)
    second_prob = _sum_normalize(second)
    return float(np.sum(np.abs(np.cumsum(first_prob - second_prob))))
