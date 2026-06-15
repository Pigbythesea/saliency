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


@dataclass(frozen=True)
class GeometryInterval:
    """Image-resampling interval for a geometry statistic."""

    method: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    valid_resamples: int
    seed: int
    uncertainty_unit: str = "image"

    def as_row(self) -> dict[str, Any]:
        return {
            "interval_method": self.method,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "bootstrap_resamples": self.resamples,
            "bootstrap_valid_resamples": self.valid_resamples,
            "bootstrap_seed": self.seed,
            "uncertainty_unit": self.uncertainty_unit,
        }


def linear_cka(features: np.ndarray, responses: np.ndarray) -> GeometryResult:
    """Compute biased linear CKA as a secondary diagnostic."""
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


def debiased_linear_cka(
    features: np.ndarray,
    responses: np.ndarray,
) -> GeometryResult:
    """Compute unbiased linear CKA using the diagonal-deleted HSIC estimator."""
    x, y = _validated_pair(features, responses)
    total = int(x.shape[0])
    if total < 4:
        return GeometryResult(
            "debiased_linear_cka",
            "",
            False,
            "requires_at_least_four_images",
            total,
            total,
        )
    status = _invalid_status(x, y)
    if status is not None:
        return GeometryResult(
            "debiased_linear_cka",
            "",
            False,
            status,
            total,
            total,
        )
    x_centered = _center_columns(x)
    y_centered = _center_columns(y)
    cross_hsic = _unbiased_linear_hsic(x_centered, y_centered)
    x_hsic = _unbiased_linear_hsic(x_centered, x_centered)
    y_hsic = _unbiased_linear_hsic(y_centered, y_centered)
    denominator = float(np.sqrt(max(x_hsic, 0.0) * max(y_hsic, 0.0)))
    if denominator <= 1e-12:
        return GeometryResult(
            "debiased_linear_cka",
            "",
            False,
            "zero_unbiased_norm",
            total,
            total,
        )
    score = float(cross_hsic / denominator)
    if not np.isfinite(score):
        return GeometryResult(
            "debiased_linear_cka",
            "",
            False,
            "nonfinite_score",
            total,
            total,
        )
    return GeometryResult(
        "debiased_linear_cka",
        score,
        True,
        "ok",
        total,
        total,
    )


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


def bootstrap_geometry_interval(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    method: str,
    resamples: int = 2000,
    seed: int = 123,
    confidence: float = 0.95,
    feature_rdm_metric: str = "correlation",
    response_rdm_metric: str = "correlation",
    compare_method: str = "spearman",
) -> GeometryInterval:
    """Resample paired images and return a percentile interval."""
    x, y = _validated_pair(features, responses)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    estimate = _geometry_score(
        x,
        y,
        method=method,
        feature_rdm_metric=feature_rdm_metric,
        response_rdm_metric=response_rdm_metric,
        compare_method=compare_method,
    )
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(resamples):
        indices = rng.choice(len(x), size=len(x), replace=True)
        try:
            score = _geometry_score(
                x[indices],
                y[indices],
                method=method,
                feature_rdm_metric=feature_rdm_metric,
                response_rdm_metric=response_rdm_metric,
                compare_method=compare_method,
            )
        except ValueError:
            continue
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        raise ValueError("geometry bootstrap produced no valid resamples")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(scores), [alpha, 1.0 - alpha])
    return GeometryInterval(
        method=method,
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        confidence=float(confidence),
        resamples=int(resamples),
        valid_resamples=len(scores),
        seed=int(seed),
    )


def geometry_method_agreement(
    cka_scores: np.ndarray,
    rsa_scores: np.ndarray,
) -> dict[str, Any]:
    """Report CKA/RSA agreement without selecting a favorable metric."""
    cka = np.asarray(cka_scores, dtype=np.float64).reshape(-1)
    rsa = np.asarray(rsa_scores, dtype=np.float64).reshape(-1)
    if cka.shape != rsa.shape:
        raise ValueError("CKA and RSA score arrays must have matching shapes")
    finite = np.isfinite(cka) & np.isfinite(rsa)
    cka = cka[finite]
    rsa = rsa[finite]
    if len(cka) < 3:
        return {
            "status": "insufficient_pairs",
            "n_pairs": int(len(cka)),
            "pearson": "",
            "spearman": "",
            "direction_agreement_fraction": "",
        }
    return {
        "status": "complete",
        "n_pairs": int(len(cka)),
        "pearson": _vector_correlation(cka, rsa),
        "spearman": _vector_correlation(_rankdata(cka), _rankdata(rsa)),
        "direction_agreement_fraction": float(
            np.mean(np.sign(cka - np.median(cka)) == np.sign(rsa - np.median(rsa)))
        ),
    }


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


def _unbiased_linear_hsic(first: np.ndarray, second: np.ndarray) -> float:
    """Unbiased linear HSIC without materializing image-by-image kernels."""
    n = int(first.shape[0])
    if first.shape[0] != second.shape[0] or n < 4:
        raise ValueError("unbiased HSIC requires paired feature rows with n >= 4")
    cross_norm = float(np.sum((first.T @ second) ** 2))
    first_diagonal = np.sum(first * first, axis=1)
    second_diagonal = np.sum(second * second, axis=1)
    diagonal_product = float(first_diagonal @ second_diagonal)
    diagonal_sum_product = float(first_diagonal.sum() * second_diagonal.sum())
    return (
        cross_norm
        + diagonal_sum_product / ((n - 1) * (n - 2))
        - (n * diagonal_product / (n - 2))
    )


def _geometry_score(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    method: str,
    feature_rdm_metric: str,
    response_rdm_metric: str,
    compare_method: str,
) -> float:
    if method == "debiased_linear_cka":
        result = debiased_linear_cka(features, responses)
    elif method == "linear_cka":
        result = linear_cka(features, responses)
    elif method in {"rsa", "subset_rsa"}:
        feature_rdm = compute_rdm(features, metric=feature_rdm_metric)
        response_rdm = compute_rdm(responses, metric=response_rdm_metric)
        score = compare_rdms(feature_rdm, response_rdm, method=compare_method)
        if not np.isfinite(score):
            raise ValueError("RSA score is non-finite")
        return float(score)
    else:
        raise ValueError(
            "method must be 'debiased_linear_cka', 'linear_cka', or 'rsa'"
        )
    if not result.valid or result.score == "":
        raise ValueError(f"{method} is invalid: {result.status}")
    return float(result.score)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _vector_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - float(first.mean())
    second_centered = second - float(second.mean())
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)
