"""Conditional-map and generated-scanpath metric interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


VALID_REGIMES = {"free_viewing", "task_search", "scanpath"}


@dataclass(frozen=True)
class BehavioralSequenceResult:
    """Metrics for one behavioral object without cross-regime pooling."""

    behavioral_regime: str
    behavioral_object: str
    metrics: dict[str, float]
    num_steps: int
    task_id: str = ""
    seed: int | str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "behavioral_regime": self.behavioral_regime,
            "behavioral_object": self.behavioral_object,
            "num_steps": self.num_steps,
            "task_id": self.task_id,
            "seed": self.seed,
            **self.metrics,
        }


def evaluate_conditional_maps(
    conditional_maps: Sequence[np.ndarray],
    observed_fixations: Sequence[Sequence[float]],
    *,
    regime: str,
    baseline_maps: Sequence[np.ndarray] | None = None,
    task_id: str = "",
    epsilon: float = 1e-12,
) -> BehavioralSequenceResult:
    """Score each next-fixation map against the matching observed fixation."""
    _validate_regime(regime)
    if regime == "scanpath":
        raise ValueError("conditional maps must use free_viewing or task_search")
    if len(conditional_maps) != len(observed_fixations):
        raise ValueError("conditional_maps and observed_fixations must have equal length")
    if baseline_maps is not None and len(baseline_maps) != len(conditional_maps):
        raise ValueError("baseline_maps must match conditional_maps")
    nss_scores: list[float] = []
    log_likelihoods: list[float] = []
    information_gains: list[float] = []
    for index, (prediction, fixation) in enumerate(
        zip(conditional_maps, observed_fixations)
    ):
        probability = _probability_map(prediction, epsilon=epsilon)
        y, x = _fixation_index(fixation, probability.shape)
        standardized = _standardized_map(prediction)
        nss_scores.append(float(standardized[y, x]))
        predicted_log = float(np.log2(max(float(probability[y, x]), epsilon)))
        log_likelihoods.append(predicted_log)
        if baseline_maps is not None:
            baseline = _probability_map(baseline_maps[index], epsilon=epsilon)
            baseline_log = float(np.log2(max(float(baseline[y, x]), epsilon)))
            information_gains.append(predicted_log - baseline_log)
    metrics = {
        "conditional_nss": float(np.mean(nss_scores)) if nss_scores else 0.0,
        "conditional_log_likelihood_bits": (
            float(np.mean(log_likelihoods)) if log_likelihoods else 0.0
        ),
    }
    if baseline_maps is not None:
        metrics["conditional_information_gain_bits"] = (
            float(np.mean(information_gains)) if information_gains else 0.0
        )
    return BehavioralSequenceResult(
        behavioral_regime=regime,
        behavioral_object="conditional_next_fixation",
        metrics=metrics,
        num_steps=len(conditional_maps),
        task_id=task_id,
    )


def evaluate_scanpath(
    predicted_fixations: Sequence[Sequence[float]],
    observed_fixations: Sequence[Sequence[float]],
    *,
    image_shape: tuple[int, int],
    regime: str,
    task_id: str = "",
    target_bbox: tuple[float, float, float, float] | None = None,
    seed: int | str = "",
) -> BehavioralSequenceResult:
    """Evaluate a generated scanpath with a normalized sequence metric."""
    _validate_regime(regime)
    predicted = _points(predicted_fixations)
    observed = _points(observed_fixations)
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    diagonal = float(np.hypot(width, height))
    dtw = _dtw_distance(predicted, observed) / max(diagonal, 1e-12)
    predicted_length = _scanpath_length(predicted) / max(diagonal, 1e-12)
    observed_length = _scanpath_length(observed) / max(diagonal, 1e-12)
    metrics = {
        "sequence_score": float(np.exp(-dtw)),
        "normalized_dtw_distance": float(dtw),
        "predicted_scanpath_length": float(predicted_length),
        "observed_scanpath_length": float(observed_length),
        "scanpath_length_error": float(abs(predicted_length - observed_length)),
        "fixation_count_error": float(abs(len(predicted) - len(observed))),
    }
    if target_bbox is not None:
        predicted_step = _first_target_step(predicted, target_bbox)
        observed_step = _first_target_step(observed, target_bbox)
        metrics["target_fixated"] = float(predicted_step is not None)
        metrics["target_fixation_step"] = (
            float(predicted_step) if predicted_step is not None else -1.0
        )
        metrics["target_fixation_step_error"] = (
            float(abs(predicted_step - observed_step))
            if predicted_step is not None and observed_step is not None
            else -1.0
        )
    return BehavioralSequenceResult(
        behavioral_regime=regime,
        behavioral_object="generated_scanpath",
        metrics=metrics,
        num_steps=len(predicted),
        task_id=task_id,
        seed=seed,
    )


def _validate_regime(regime: str) -> None:
    if regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {sorted(VALID_REGIMES)}")


def _probability_map(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("conditional maps must be finite 2D arrays")
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= epsilon:
        return np.full(array.shape, 1.0 / array.size, dtype=np.float64)
    return array / total


def _standardized_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std())
    if standard_deviation <= 1e-12:
        return np.zeros_like(array)
    return (array - float(array.mean())) / standard_deviation


def _fixation_index(
    fixation: Sequence[float],
    shape: tuple[int, int],
) -> tuple[int, int]:
    if len(fixation) < 2:
        raise ValueError("fixations must contain x and y")
    x, y = float(fixation[0]), float(fixation[1])
    return (
        int(np.clip(round(y), 0, shape[0] - 1)),
        int(np.clip(round(x), 0, shape[1] - 1)),
    )


def _points(fixations: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(fixations, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values[:, :2]).all():
        raise ValueError("fixations must be a finite Nx2 array")
    return values[:, :2]


def _dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) == 0 and len(second) == 0:
        return 0.0
    if len(first) == 0 or len(second) == 0:
        return float("inf")
    costs = np.full((len(first) + 1, len(second) + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            distance = float(np.linalg.norm(first[i - 1] - second[j - 1]))
            costs[i, j] = distance + min(
                costs[i - 1, j],
                costs[i, j - 1],
                costs[i - 1, j - 1],
            )
    return float(costs[-1, -1] / max(len(first), len(second)))


def _scanpath_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _first_target_step(
    points: np.ndarray,
    target_bbox: tuple[float, float, float, float],
) -> int | None:
    x_min, y_min, x_max, y_max = (float(value) for value in target_bbox)
    for index, (x, y) in enumerate(points, start=1):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return index
    return None
