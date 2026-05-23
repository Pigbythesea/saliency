"""Encoding model utilities."""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


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


def benchmark_encoding_target_scores(
    predictions: np.ndarray,
    target: np.ndarray,
    noise_ceiling: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute per-target benchmark-style encoding scores."""
    pred = _as_2d(np.asarray(predictions, dtype=np.float64))
    true = _as_2d(np.asarray(target, dtype=np.float64))
    if pred.shape != true.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    ceilings = _noise_ceiling_array(noise_ceiling, pred.shape[1])
    correlations, pred_valid, target_valid = _columnwise_correlation_with_validity(pred, true)
    prediction_r2 = _columnwise_r2(pred, true, target_valid=target_valid)
    rows: list[dict[str, float | int | str]] = []
    for index, correlation in enumerate(correlations):
        squared = float(correlation * correlation)
        ceiling = ceilings[index] if ceilings is not None else None
        normalized: float | str = ""
        scope = "benchmark_style_non_noise_normalized"
        if ceiling is not None and np.isfinite(ceiling) and ceiling > _EPSILON:
            normalized = float(squared / ceiling)
            scope = "benchmark_style_noise_normalized"
        rows.append(
            {
                "target_index": index,
                "pearson_r": float(correlation),
                "r2_score_from_r": squared,
                "prediction_r2": float(prediction_r2[index]),
                "noise_ceiling": "" if ceiling is None or not np.isfinite(ceiling) else float(ceiling),
                "noise_normalized_score": normalized,
                "valid_prediction_variance": str(bool(pred_valid[index])).lower(),
                "valid_target_variance": str(bool(target_valid[index])).lower(),
                "metric_scope": scope,
            }
        )
    return rows


def _columnwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _columnwise_correlation_with_validity(a, b)[0]


def _columnwise_correlation_with_validity(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_norm = np.linalg.norm(a_centered, axis=0)
    b_norm = np.linalg.norm(b_centered, axis=0)
    prediction_valid = a_norm > _EPSILON
    target_valid = b_norm > _EPSILON
    denominator = a_norm * b_norm
    correlations = np.divide(
        np.sum(a_centered * b_centered, axis=0),
        denominator,
        out=np.zeros(a.shape[1], dtype=np.float64),
        where=denominator > _EPSILON,
    )
    return correlations, prediction_valid, target_valid


def _columnwise_r2(
    predictions: np.ndarray,
    target: np.ndarray,
    *,
    target_valid: np.ndarray,
) -> np.ndarray:
    residual = np.sum((target - predictions) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    scores = np.divide(
        residual,
        total,
        out=np.zeros(target.shape[1], dtype=np.float64),
        where=target_valid,
    )
    return np.where(target_valid, 1.0 - scores, 0.0)


def _noise_ceiling_array(values: np.ndarray | None, num_targets: int) -> np.ndarray | None:
    if values is None:
        return None
    ceilings = np.asarray(values, dtype=np.float64).ravel()
    if ceilings.size != num_targets:
        raise ValueError("noise_ceiling must have one value per target")
    return ceilings


def _as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a 1D or 2D response array")
    return values
