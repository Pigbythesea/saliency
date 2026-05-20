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
    compare_rdms,
    compute_rdm,
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
    feature_reduction = str(neural_config.get("feature_reduction", "flatten"))
    rsa_config = dict(neural_config.get("rsa", {}))

    dataset = build_dataset(config)
    model = build_model(config)
    device = resolve_device(config.get("device", "auto"))
    _move_model_to_device(model, device)

    collection = _collect_features_and_responses(
        dataset=dataset,
        model=model,
        config=config,
        layers=layers,
        response_key=response_key,
        device=device,
        feature_reduction=feature_reduction,
    )
    image_ids, responses, features_by_layer, item_metadata = collection
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
        row_context=_score_row_context(config, item_metadata),
    )
    scores_path = output_dir / "encoding_scores.csv"
    _write_score_rows(scores_path, score_rows)
    rsa_rows: list[dict[str, Any]] = []
    rsa_path: Path | None = None
    if bool(rsa_config.get("enabled", False)):
        rsa_rows = _compute_rsa_rows(
            features_by_layer,
            responses,
            layers=layers,
            feature_rdm_metric=str(rsa_config.get("rdm_metric", "correlation")),
            response_rdm_metric=str(rsa_config.get("response_rdm_metric", "correlation")),
            compare_method=str(rsa_config.get("compare_method", "spearman")),
            row_context=_score_row_context(config, item_metadata),
        )
        rsa_path = output_dir / "rsa_scores.csv"
        _write_rsa_rows(rsa_path, rsa_rows)

    metadata = {
        "config_path": str(config_path),
        "dataset": config.get("dataset", {}).get("label") or config.get("dataset", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "model_name": config.get("model", {}).get("name"),
        "model_backend": config.get("model", {}).get("backend", "timm"),
        "model_pretrained": bool(config.get("model", {}).get("pretrained", False)),
        "num_items": len(image_ids),
        "layers": layers,
        "response_key": response_key,
        "feature_reduction": feature_reduction,
        "ridge_alpha": alpha,
        "train_fraction": train_fraction,
        "metric": metric,
        "subjects": sorted(_unique_metadata_values(item_metadata, "subject_id")),
        "rois": sorted(_unique_metadata_values(item_metadata, "roi")),
        "activations": str(activation_path),
        "encoding_scores": str(scores_path),
        "rsa_scores": str(rsa_path) if rsa_path is not None else None,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        **metadata,
        "metadata": str(metadata_path),
        "score_rows": score_rows,
        "rsa_rows": rsa_rows,
    }


def _move_model_to_device(model: Any, device: str) -> None:
    torch_model = getattr(model, "model", model)
    to = getattr(torch_model, "to", None)
    if callable(to):
        to(device)


def _collect_features_and_responses(
    *,
    dataset: Any,
    model: Any,
    config: dict[str, Any],
    layers: list[str],
    response_key: str,
    device: str,
    feature_reduction: str,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    image_ids: list[str] = []
    responses: list[np.ndarray] = []
    item_metadata: list[dict[str, Any]] = []
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
        metadata = item.get("metadata", {})
        item_metadata.append(metadata if isinstance(metadata, dict) else {})
        for layer in layers:
            collected[layer].append(
                _reduce_feature(feature_dict[layer], method=feature_reduction)
            )

    response_matrix = _stack_consistent(responses, "ROI responses")
    feature_matrix = {
        layer: _stack_consistent(values, f"features for layer {layer}")
        for layer, values in collected.items()
    }
    return image_ids, response_matrix, feature_matrix, item_metadata


def _fit_and_score_layers(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    train_fraction: float,
    alpha: float,
    metric: str,
    seed: int,
    row_context: dict[str, Any],
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
                **row_context,
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


def _compute_rsa_rows(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    feature_rdm_metric: str,
    response_rdm_metric: str,
    compare_method: str,
    row_context: dict[str, Any],
) -> list[dict[str, Any]]:
    response_rdm = compute_rdm(responses, metric=response_rdm_metric)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        model_rdm = compute_rdm(features_by_layer[layer], metric=feature_rdm_metric)
        score = compare_rdms(model_rdm, response_rdm, method=compare_method)
        rows.append(
            {
                **row_context,
                "layer": layer,
                "model_rdm_metric": feature_rdm_metric,
                "response_rdm_metric": response_rdm_metric,
                "compare_method": compare_method,
                "n_items": int(responses.shape[0]),
                "score": float(score),
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


def _reduce_feature(values: Any, method: str = "flatten") -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.shape[0] == 1 and array.ndim > 1:
        array = array[0]
    if method == "flatten":
        return array.reshape(-1)
    if method == "spatial_mean":
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            return array.mean(axis=0)
        return array.mean(axis=tuple(range(1, array.ndim)))
    raise ValueError("feature_reduction must be 'flatten' or 'spatial_mean'")


def _score_row_context(
    config: dict[str, Any],
    item_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_config = config.get("dataset", {})
    return {
        "dataset": dataset_config.get("label") or dataset_config.get("name"),
        "model": config.get("model", {}).get("name"),
        "subject_id": _single_or_mixed(_unique_metadata_values(item_metadata, "subject_id")),
        "roi": _single_or_mixed(_unique_metadata_values(item_metadata, "roi")),
    }


def _unique_metadata_values(
    item_metadata: list[dict[str, Any]],
    key: str,
) -> set[str]:
    values = set()
    for metadata in item_metadata:
        value = metadata.get(key)
        if value is not None:
            values.add(str(value))
    return values


def _single_or_mixed(values: set[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _stack_consistent(values: list[np.ndarray], label: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No {label} were collected")
    first_shape = values[0].shape
    if any(value.shape != first_shape for value in values):
        raise ValueError(f"All {label} must have matching shapes")
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _write_score_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "subject_id",
        "roi",
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


def _write_rsa_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "subject_id",
        "roi",
        "layer",
        "model_rdm_metric",
        "response_rdm_metric",
        "compare_method",
        "n_items",
        "score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
