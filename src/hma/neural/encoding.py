"""Encoding model utilities."""

from __future__ import annotations

import numpy as np


def fit_ridge_encoding(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Fit a closed-form ridge encoding model with intercept."""
    X = np.asarray(X_train, dtype=np.float64)
    Y = _as_2d(np.asarray(Y_train, dtype=np.float64))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X_train and Y_train must have the same number of rows")

    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - x_mean
    Yc = Y - y_mean

    identity = np.eye(Xc.shape[1], dtype=np.float64)
    weights = np.linalg.solve(Xc.T @ Xc + float(alpha) * identity, Xc.T @ Yc)
    intercept = y_mean - x_mean @ weights
    return {
        "weights": weights.astype(np.float32),
        "intercept": intercept.ravel().astype(np.float32),
        "alpha": float(alpha),
    }


def predict_ridge_encoding(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    """Predict responses from a fitted ridge encoding model."""
    return np.asarray(X, dtype=np.float32) @ model["weights"] + model["intercept"]


def evaluate_encoding(
    pred_or_model: np.ndarray | dict[str, np.ndarray | float],
    X_or_Y: np.ndarray,
    Y_test: np.ndarray | None = None,
    metric: str = "correlation",
) -> np.ndarray:
    """Evaluate predictions with per-target correlation or R2."""
    if Y_test is None:
        predictions = _as_2d(np.asarray(pred_or_model, dtype=np.float64))
        target = _as_2d(np.asarray(X_or_Y, dtype=np.float64))
    else:
        predictions = _as_2d(predict_ridge_encoding(pred_or_model, X_or_Y))
        target = _as_2d(np.asarray(Y_test, dtype=np.float64))

    if predictions.shape != target.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    if metric == "correlation":
        return _columnwise_correlation(predictions, target).astype(np.float32)
    if metric == "r2":
        residual = np.sum((target - predictions) ** 2, axis=0)
        total = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
        return (1.0 - residual / np.maximum(total, 1e-12)).astype(np.float32)
    raise ValueError("metric must be 'correlation' or 'r2'")


def _columnwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(a_centered, axis=0) * np.linalg.norm(b_centered, axis=0)
    return np.divide(
        np.sum(a_centered * b_centered, axis=0),
        denominator,
        out=np.zeros(a.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )


def _as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a 1D or 2D response array")
    return values
