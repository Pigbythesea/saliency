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
    benchmark_encoding_target_scores,
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
    noise_ceiling_key = str(neural_config.get("noise_ceiling_key", "noise_ceiling"))
    alpha = float(neural_config.get("ridge_alpha", 1.0))
    ridge_alphas = _optional_float_list(neural_config.get("ridge_alphas"))
    validation_fraction = float(neural_config.get("validation_fraction", 0.2))
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
        noise_ceiling_key=noise_ceiling_key,
        device=device,
        feature_reduction=feature_reduction,
    )
    image_ids, responses, features_by_layer, item_metadata, noise_ceiling = collection
    if len(image_ids) < 2:
        raise ValueError("Neural alignment requires at least two response-bearing items")

    activation_path = save_activations(
        {"image_ids": np.asarray(image_ids, dtype=object), **features_by_layer},
        output_dir / "activations.npz",
    )
    score_rows, target_score_rows = _fit_and_score_layers(
        features_by_layer,
        responses,
        layers=layers,
        train_fraction=train_fraction,
        alpha=alpha,
        ridge_alphas=ridge_alphas,
        validation_fraction=validation_fraction,
        metric=metric,
        seed=seed,
        feature_reduction=feature_reduction,
        row_context=_score_row_context(config, item_metadata),
        noise_ceiling=noise_ceiling,
    )
    scores_path = output_dir / "encoding_scores.csv"
    _write_score_rows(scores_path, score_rows)
    target_scores_path = output_dir / "encoding_target_scores.csv"
    _write_target_score_rows(target_scores_path, target_score_rows)
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
        "noise_ceiling_key": noise_ceiling_key,
        "noise_ceiling_available": noise_ceiling is not None,
        "noise_ceiling_source": _single_or_mixed(
            _unique_metadata_values(item_metadata, "noise_ceiling_source")
        ),
        "feature_reduction": feature_reduction,
        "ridge_alpha": alpha,
        "ridge_alphas": ridge_alphas,
        "validation_fraction": validation_fraction,
        "train_fraction": train_fraction,
        "metric": metric,
        "metric_scope": _metric_scope_from_target_rows(target_score_rows),
        "alpha_selection_modes": sorted(
            {str(row.get("alpha_selection_mode", "")) for row in score_rows}
        ),
        "selected_ridge_alphas": {
            str(row["layer"]): float(row["selected_ridge_alpha"]) for row in score_rows
        },
        "subjects": sorted(_unique_metadata_values(item_metadata, "subject_id")),
        "rois": sorted(_unique_metadata_values(item_metadata, "roi")),
        "activations": str(activation_path),
        "encoding_scores": str(scores_path),
        "encoding_target_scores": str(target_scores_path),
        "rsa_scores": str(rsa_path) if rsa_path is not None else None,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        **metadata,
        "metadata": str(metadata_path),
        "score_rows": score_rows,
        "target_score_rows": target_score_rows,
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
    noise_ceiling_key: str,
    device: str,
    feature_reduction: str,
) -> tuple[
    list[str],
    np.ndarray,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    np.ndarray | None,
]:
    image_ids: list[str] = []
    responses: list[np.ndarray] = []
    noise_ceilings: list[np.ndarray | None] = []
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
        response_array = np.asarray(response, dtype=np.float32).ravel()
        responses.append(response_array)
        metadata = item.get("metadata", {})
        item_metadata.append(metadata if isinstance(metadata, dict) else {})
        noise_ceiling = _response_for_item(item, noise_ceiling_key)
        noise_ceilings.append(
            None
            if noise_ceiling is None
            else np.asarray(noise_ceiling, dtype=np.float32).ravel()
        )
        for layer in layers:
            collected[layer].append(
                _reduce_feature(feature_dict[layer], method=feature_reduction)
            )

    response_matrix = _stack_consistent(responses, "ROI responses")
    feature_matrix = {
        layer: _stack_consistent(values, f"features for layer {layer}")
        for layer, values in collected.items()
    }
    noise_ceiling_array = _validate_noise_ceiling(
        noise_ceilings,
        num_targets=response_matrix.shape[1],
        key=noise_ceiling_key,
    )
    return image_ids, response_matrix, feature_matrix, item_metadata, noise_ceiling_array


def _fit_and_score_layers(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    train_fraction: float,
    alpha: float,
    ridge_alphas: list[float] | None,
    validation_fraction: float,
    metric: str,
    seed: int,
    feature_reduction: str,
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_items = responses.shape[0]
    n_train = min(max(1, int(round(n_items * train_fraction))), n_items - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for layer in layers:
        features = features_by_layer[layer]
        alpha_result = _select_ridge_alpha(
            features,
            responses,
            train_idx=train_idx,
            alpha=alpha,
            ridge_alphas=ridge_alphas,
            validation_fraction=validation_fraction,
            rng=rng,
        )
        selected_alpha = alpha_result["selected_alpha"]
        model = fit_ridge_encoding(
            features[train_idx],
            responses[train_idx],
            alpha=selected_alpha,
        )
        predictions = predict_ridge_encoding(model, features[test_idx])
        scores = evaluate_encoding(predictions, responses[test_idx], metric=metric)
        layer_context = {
            **row_context,
            "layer": layer,
            "metric": metric,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "split_seed": int(seed),
            "feature_reduction": feature_reduction,
            "selected_ridge_alpha": float(selected_alpha),
            "alpha_selection_mode": alpha_result["mode"],
        }
        layer_target_rows = [
            {
                **layer_context,
                **target_row,
            }
            for target_row in benchmark_encoding_target_scores(
                predictions,
                responses[test_idx],
                noise_ceiling=noise_ceiling,
            )
        ]
        target_rows.extend(layer_target_rows)
        metric_scope = _metric_scope_from_target_rows(layer_target_rows)
        noise_summary = _noise_normalized_layer_summary(layer_target_rows)
        rows.append(
            {
                **layer_context,
                "num_targets": int(scores.size),
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "std_score": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
                "mean_r2_score_from_r": float(
                    np.mean([float(row["r2_score_from_r"]) for row in layer_target_rows])
                ),
                **noise_summary,
                "metric_scope": metric_scope,
            }
        )
    return rows, target_rows


def _noise_normalized_layer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_scores: list[float] = []
    zero_count = 0
    invalid_count = 0
    for row in rows:
        ceiling = _optional_float(row.get("noise_ceiling"))
        if ceiling is None or not np.isfinite(ceiling) or ceiling < 0.0:
            invalid_count += 1
            continue
        if ceiling == 0.0:
            zero_count += 1
            continue
        score = _optional_float(row.get("noise_normalized_score"))
        if score is None or not np.isfinite(score):
            invalid_count += 1
            continue
        valid_scores.append(score)
    return {
        "mean_noise_normalized_score": float(np.mean(valid_scores)) if valid_scores else "",
        "median_noise_normalized_score": float(np.median(valid_scores)) if valid_scores else "",
        "valid_noise_ceiling_targets": len(valid_scores),
        "zero_noise_ceiling_targets": zero_count,
        "invalid_noise_ceiling_targets": invalid_count,
    }


def _select_ridge_alpha(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    train_idx: np.ndarray,
    alpha: float,
    ridge_alphas: list[float] | None,
    validation_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not ridge_alphas:
        return {"selected_alpha": float(alpha), "mode": "fixed"}
    if train_idx.size < 3:
        return {"selected_alpha": float(alpha), "mode": "fixed_insufficient_train"}

    n_val = int(round(train_idx.size * validation_fraction))
    n_val = min(max(1, n_val), train_idx.size - 1)
    if n_val < 1 or train_idx.size - n_val < 1:
        return {"selected_alpha": float(alpha), "mode": "fixed_insufficient_train"}

    inner_order = train_idx[rng.permutation(train_idx.size)]
    val_idx = inner_order[:n_val]
    inner_train_idx = inner_order[n_val:]
    best_alpha = float(ridge_alphas[0])
    best_score = -np.inf
    for candidate_alpha in ridge_alphas:
        model = fit_ridge_encoding(
            features[inner_train_idx],
            responses[inner_train_idx],
            alpha=float(candidate_alpha),
        )
        predictions = predict_ridge_encoding(model, features[val_idx])
        scores = evaluate_encoding(predictions, responses[val_idx], metric="correlation")
        score = float(np.mean(scores))
        if score > best_score + 1e-6 or (
            abs(score - best_score) <= 1e-6 and float(candidate_alpha) < best_alpha
        ):
            best_score = score
            best_alpha = float(candidate_alpha)
    return {"selected_alpha": best_alpha, "mode": "cv_inner_validation"}


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


def _optional_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError("neural.ridge_alphas must be a list of numbers")
    parsed = [float(value) for value in values]
    return parsed or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_scope_from_target_rows(rows: list[dict[str, Any]]) -> str:
    if rows and all(row.get("metric_scope") == "benchmark_style_noise_normalized" for row in rows):
        return "benchmark_style_noise_normalized"
    return "benchmark_style_non_noise_normalized"


def _validate_noise_ceiling(
    values: list[np.ndarray | None],
    *,
    num_targets: int,
    key: str,
) -> np.ndarray | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(
            f"Noise ceiling metadata '{key}' is partially missing across dataset items"
        )

    first = present[0]
    if first.size != num_targets:
        raise ValueError(
            f"Noise ceiling metadata '{key}' must have one value per target: "
            f"{first.size} values for {num_targets} targets"
        )
    for value in present[1:]:
        if value.size != num_targets:
            raise ValueError(
                f"Noise ceiling metadata '{key}' must have one value per target: "
                f"{value.size} values for {num_targets} targets"
            )
        if not np.allclose(value, first, equal_nan=True):
            raise ValueError(
                f"Noise ceiling metadata '{key}' must be consistent across dataset items"
            )
    return first.astype(np.float32, copy=False)


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
        "metric_scope",
        "n_train",
        "n_test",
        "num_targets",
        "mean_score",
        "median_score",
        "std_score",
        "mean_r2_score_from_r",
        "mean_noise_normalized_score",
        "median_noise_normalized_score",
        "valid_noise_ceiling_targets",
        "zero_noise_ceiling_targets",
        "invalid_noise_ceiling_targets",
        "selected_ridge_alpha",
        "alpha_selection_mode",
        "split_seed",
        "feature_reduction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_target_score_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "subject_id",
        "roi",
        "layer",
        "metric",
        "metric_scope",
        "target_index",
        "pearson_r",
        "r2_score_from_r",
        "prediction_r2",
        "noise_ceiling",
        "noise_normalized_score",
        "valid_noise_ceiling",
        "valid_prediction_variance",
        "valid_target_variance",
        "n_train",
        "n_test",
        "selected_ridge_alpha",
        "alpha_selection_mode",
        "split_seed",
        "feature_reduction",
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
