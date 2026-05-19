"""Small neural encoding experiment runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hma.datasets import build_dataset
from hma.models import build_model
from hma.neural import (
    evaluate_encoding,
    fit_ridge_encoding,
    predict_ridge_encoding,
    save_activations,
)
from hma.preprocessing import preprocess_image_for_model
from hma.utils.config import load_experiment_config
from hma.utils.device import resolve_device
from hma.utils.paths import ensure_dir, resolve_path


def run_neural_alignment(config_path: str | Path) -> dict[str, Any]:
    """Run a compact activation extraction plus ridge-encoding smoke experiment."""
    config = load_experiment_config(config_path)
    output_dir = ensure_dir(resolve_path(config["output"]["dir"]))
    neural_config = dict(config.get("neural", {}))
    layers = [str(layer) for layer in neural_config.get("layers", ["embedding"])]
    response_key = str(neural_config.get("response_key", "roi_responses"))
    alpha = float(neural_config.get("ridge_alpha", 1.0))
    train_fraction = float(neural_config.get("train_fraction", 0.8))
    metric = str(neural_config.get("metric", "correlation"))
    seed = int(neural_config.get("seed", config.get("seed", 0)))

    dataset = build_dataset(config)
    model = build_model(config)
    device = resolve_device(config.get("device", "auto"))

    image_ids, responses, features_by_layer = _collect_features_and_responses(
        dataset=dataset,
        model=model,
        config=config,
        layers=layers,
        response_key=response_key,
        device=device,
    )
    if len(image_ids) < 2:
        raise ValueError("Neural alignment requires at least two response-bearing items")

    activation_path = save_activations(
        {"image_ids": np.asarray(image_ids, dtype=object), **features_by_layer},
        output_dir / "activations.npz",
    )
    score_rows = _fit_and_score_layers(
        features_by_layer,
        responses,
        layers=layers,
        train_fraction=train_fraction,
        alpha=alpha,
        metric=metric,
        seed=seed,
    )
    scores_path = output_dir / "encoding_scores.csv"
    _write_score_rows(scores_path, score_rows)

    metadata = {
        "config_path": str(config_path),
        "dataset": config.get("dataset", {}).get("label") or config.get("dataset", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "num_items": len(image_ids),
        "layers": layers,
        "response_key": response_key,
        "ridge_alpha": alpha,
        "train_fraction": train_fraction,
        "metric": metric,
        "activations": str(activation_path),
        "encoding_scores": str(scores_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "metadata": str(metadata_path), "score_rows": score_rows}


def _collect_features_and_responses(
    *,
    dataset: Any,
    model: Any,
    config: dict[str, Any],
    layers: list[str],
    response_key: str,
    device: str,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    image_ids: list[str] = []
    responses: list[np.ndarray] = []
    collected: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}

    for index, item in enumerate(dataset):
        response = _response_for_item(item, response_key)
        if response is None:
            raise ValueError(
                f"Dataset item {item.get('image_id', index)} is missing neural response '{response_key}'"
            )
        tensor = preprocess_image_for_model(item["image"], config=config, device=device)
        features = model.get_features(tensor, layers=layers)
        feature_dict = _normalize_feature_output(features, layers)
        image_ids.append(str(item.get("image_id", f"item_{index:04d}")))
        responses.append(np.asarray(response, dtype=np.float32).ravel())
        for layer in layers:
            collected[layer].append(_flatten_feature(feature_dict[layer]))

    response_matrix = _stack_consistent(responses, "ROI responses")
    feature_matrix = {
        layer: _stack_consistent(values, f"features for layer {layer}")
        for layer, values in collected.items()
    }
    return image_ids, response_matrix, feature_matrix


def _fit_and_score_layers(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    train_fraction: float,
    alpha: float,
    metric: str,
    seed: int,
) -> list[dict[str, Any]]:
    n_items = responses.shape[0]
    n_train = min(max(1, int(round(n_items * train_fraction))), n_items - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    rows: list[dict[str, Any]] = []
    for layer in layers:
        features = features_by_layer[layer]
        model = fit_ridge_encoding(features[train_idx], responses[train_idx], alpha=alpha)
        predictions = predict_ridge_encoding(model, features[test_idx])
        scores = evaluate_encoding(predictions, responses[test_idx], metric=metric)
        rows.append(
            {
                "layer": layer,
                "metric": metric,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "num_targets": int(scores.size),
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "std_score": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
            }
        )
    return rows


def _response_for_item(item: dict[str, Any], response_key: str) -> Any:
    if response_key in item and item[response_key] is not None:
        return item[response_key]
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(response_key)
    return None


def _normalize_feature_output(features: Any, layers: list[str]) -> dict[str, Any]:
    if isinstance(features, dict):
        return features
    if len(layers) == 1:
        return {layers[0]: features}
    if isinstance(features, (list, tuple)) and len(features) == len(layers):
        return dict(zip(layers, features))
    raise ValueError("Model features must be a dict or match requested layers")


def _flatten_feature(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.shape[0] == 1 and array.ndim > 1:
        array = array[0]
    return array.reshape(-1)


def _stack_consistent(values: list[np.ndarray], label: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No {label} were collected")
    first_shape = values[0].shape
    if any(value.shape != first_shape for value in values):
        raise ValueError(f"All {label} must have matching shapes")
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _write_score_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer",
        "metric",
        "n_train",
        "n_test",
        "num_targets",
        "mean_score",
        "median_score",
        "std_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
