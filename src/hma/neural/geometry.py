"""Scalable representational-geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hma.neural.rsa import compare_rdms, compute_rdm


@dataclass(frozen=True)
class GeometryResult:
    """Metadata-friendly geometry score result."""

    method: str
    score: float | str
    valid: bool
    status: str
    num_images_total: int
    num_images_used: int
    subset_seed: int | str = ""
    subset_size: int | str = ""
    feature_rdm_metric: str = ""
    response_rdm_metric: str = ""
    rdm_compare_method: str = ""
    subset_index_policy: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "geometry_method": self.method,
            "score": self.score,
            "valid": str(self.valid).lower(),
            "status": self.status,
            "num_images_total": self.num_images_total,
            "num_images_used": self.num_images_used,
            "subset_seed": self.subset_seed,
            "subset_size": self.subset_size,
            "feature_rdm_metric": self.feature_rdm_metric,
            "response_rdm_metric": self.response_rdm_metric,
            "rdm_compare_method": self.rdm_compare_method,
            "subset_index_policy": self.subset_index_policy,
        }


def linear_cka(features: np.ndarray, responses: np.ndarray) -> GeometryResult:
    """Compute linear CKA without materializing an image x image kernel."""
    x, y = _validated_pair(features, responses)
    total = int(x.shape[0])
    status = _invalid_status(x, y)
    if status is not None:
        return GeometryResult("linear_cka", "", False, status, total, total)

    x_centered = _center_columns(x)
    y_centered = _center_columns(y)
    cross = x_centered.T @ y_centered
    x_self = x_centered.T @ x_centered
    y_self = y_centered.T @ y_centered
    numerator = float(np.sum(cross * cross))
    denominator = float(np.sqrt(np.sum(x_self * x_self) * np.sum(y_self * y_self)))
    if denominator <= 1e-12:
        return GeometryResult("linear_cka", "", False, "zero_norm", total, total)
    return GeometryResult("linear_cka", numerator / denominator, True, "ok", total, total)


def subset_rsa(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    subset_size: int,
    seed: int,
    feature_rdm_metric: str = "correlation",
    response_rdm_metric: str = "correlation",
    compare_method: str = "spearman",
) -> GeometryResult:
    """Compute RSA on a deterministic image subset."""
    x, y = _validated_pair(features, responses)
    total = int(x.shape[0])
    if subset_size <= 1:
        return GeometryResult(
            "subset_rsa",
            "",
            False,
            "subset_size_too_small",
            total,
            0,
            seed,
            subset_size,
            feature_rdm_metric,
            response_rdm_metric,
            compare_method,
            "deterministic_sorted_without_replacement",
        )
    if subset_size > total:
        return GeometryResult(
            "subset_rsa",
            "",
            False,
            "subset_size_exceeds_items",
            total,
            0,
            seed,
            subset_size,
            feature_rdm_metric,
            response_rdm_metric,
            compare_method,
            "deterministic_sorted_without_replacement",
        )
    status = _invalid_status(x, y)
    if status is not None:
        return GeometryResult(
            "subset_rsa",
            "",
            False,
            status,
            total,
            0,
            seed,
            subset_size,
            feature_rdm_metric,
            response_rdm_metric,
            compare_method,
            "deterministic_sorted_without_replacement",
        )

    indices = deterministic_subset_indices(total, subset_size=subset_size, seed=seed)
    x_subset = x[indices]
    y_subset = y[indices]
    try:
        feature_rdm = compute_rdm(x_subset, metric=feature_rdm_metric)
        response_rdm = compute_rdm(y_subset, metric=response_rdm_metric)
        score = compare_rdms(feature_rdm, response_rdm, method=compare_method)
    except ValueError as exc:
        return GeometryResult(
            "subset_rsa",
            "",
            False,
            f"invalid_rdm: {exc}",
            total,
            int(subset_size),
            seed,
            subset_size,
            feature_rdm_metric,
            response_rdm_metric,
            compare_method,
            "deterministic_sorted_without_replacement",
        )
    if not np.isfinite(score):
        return GeometryResult(
            "subset_rsa",
            "",
            False,
            "nonfinite_score",
            total,
            int(subset_size),
            seed,
            subset_size,
            feature_rdm_metric,
            response_rdm_metric,
            compare_method,
            "deterministic_sorted_without_replacement",
        )
    return GeometryResult(
        "subset_rsa",
        float(score),
        True,
        "ok",
        total,
        int(subset_size),
        seed,
        subset_size,
        feature_rdm_metric,
        response_rdm_metric,
        compare_method,
        "deterministic_sorted_without_replacement",
    )


def deterministic_subset_indices(total: int, *, subset_size: int, seed: int) -> np.ndarray:
    """Return deterministic sorted subset indices."""
    if subset_size > total:
        raise ValueError("subset_size cannot exceed total")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=subset_size, replace=False))


def _validated_pair(features: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and responses must be 2D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and responses must have the same number of rows")
    if x.shape[0] < 2:
        raise ValueError("at least two images are required")
    return x, y


def _invalid_status(x: np.ndarray, y: np.ndarray) -> str | None:
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return "nonfinite_input"
    if np.all(np.var(x, axis=0) <= 1e-12):
        return "constant_features"
    if np.all(np.var(y, axis=0) <= 1e-12):
        return "constant_responses"
    return None


def _center_columns(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)
