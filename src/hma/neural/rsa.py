"""Representational similarity analysis utilities."""

from __future__ import annotations

import numpy as np


def compute_rdm(features: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Compute a square representational dissimilarity matrix."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a 2D array")

    if metric == "correlation":
        centered = values - values.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        normalized = np.divide(
            centered,
            np.maximum(norm, 1e-12),
            out=np.zeros_like(centered),
            where=norm > 1e-12,
        )
        similarity = normalized @ normalized.T
        rdm = 1.0 - similarity
    elif metric == "euclidean":
        diff = values[:, None, :] - values[None, :, :]
        rdm = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        raise ValueError("metric must be 'correlation' or 'euclidean'")

    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def compare_rdms(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """Compare upper triangles of two RDMs."""
    first = np.asarray(rdm_a, dtype=np.float64)
    second = np.asarray(rdm_b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("RDMs must be square arrays with matching shapes")

    indices = np.triu_indices(first.shape[0], k=1)
    a = first[indices]
    b = second[indices]
    if method == "spearman":
        a = _rankdata(a)
        b = _rankdata(b)
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'")
    return float(_correlation(a, b))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denominator)
