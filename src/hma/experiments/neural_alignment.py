"""Small neural encoding experiment runner."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from hma.datasets import build_dataset
from hma.external.artifacts import (
    load_external_features,
    load_external_features_to_memmaps,
)
from hma.models import build_model
from hma.neural import (
    SpatialReadoutConfig,
    benchmark_encoding_target_scores,
    compare_rdms,
    compute_rdm,
    evaluate_encoding,
    fit_ridge_encoding,
    fit_spatial_readout,
    fuse_spatial_feature_layers,
    linear_cka,
    predict_ridge_encoding,
    predict_spatial_readout,
    save_activations,
    subset_rsa,
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
    encoding_method = str(neural_config.get("encoding_method", "ridge"))
    feature_reduction = str(neural_config.get("feature_reduction", "flatten"))
    layer_fusion = str(neural_config.get("layer_fusion", ""))
    selection_config = dict(neural_config.get("selection", {}))
    selection_enabled = bool(selection_config.get("enabled", False))
    if encoding_method not in {"ridge", "learned_spatial_readout"}:
        raise ValueError("neural.encoding_method must be 'ridge' or 'learned_spatial_readout'")
    selection_candidates = (
        _selection_candidate_configs(
            neural_config,
            layers=layers,
            feature_reduction=(
                "learned_spatial_readout"
                if encoding_method == "learned_spatial_readout"
                else feature_reduction
            ),
            default_seed=seed,
        )
        if selection_enabled
        else []
    )
    if selection_enabled:
        layers = _unique_preserving([str(candidate["layer"]) for candidate in selection_candidates])
    feature_reduction_config = (
        {}
        if selection_enabled
        else _feature_reduction_config(
            neural_config,
            feature_reduction=feature_reduction,
            default_seed=seed,
        )
    )
    rsa_config = dict(neural_config.get("rsa", {}))
    if (
        encoding_method == "learned_spatial_readout"
        and layer_fusion
        and bool(rsa_config.get("enabled", False))
    ):
        raise ValueError("Multi-layer learned_spatial_readout smoke runs must keep RSA disabled")

    dataset = build_dataset(config)
    external_artifact_config = dict(config.get("external_artifact", {}))
    external_artifact_path = external_artifact_config.get("path")
    external_artifact_manifest: dict[str, Any] | None = None
    collection_feature_reduction = (
        "selection_raw"
        if selection_enabled or encoding_method == "learned_spatial_readout"
        else feature_reduction
    )
    if external_artifact_path:
        device = "external_artifact"
        configured_feature_cache = external_artifact_config.get("feature_cache_dir")
        collection, external_artifact_manifest = _collect_external_features_and_responses(
            dataset=dataset,
            artifact_path=resolve_path(external_artifact_path),
            layers=layers,
            response_key=response_key,
            noise_ceiling_key=noise_ceiling_key,
            feature_reduction=collection_feature_reduction,
            verify_hashes=bool(external_artifact_config.get("verify_hashes", True)),
            feature_storage_dir=(
                resolve_path(configured_feature_cache)
                if configured_feature_cache
                else output_dir / "feature_cache"
                if collection_feature_reduction == "selection_raw"
                else None
            ),
        )
    else:
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
            feature_reduction=collection_feature_reduction,
            feature_storage_dir=(
                output_dir / "feature_cache"
                if selection_enabled or encoding_method == "learned_spatial_readout"
                else None
            ),
        )
    image_ids, responses, features_by_layer, feature_shapes, item_metadata, noise_ceiling = collection
    if len(image_ids) < 2:
        raise ValueError("Neural alignment requires at least two response-bearing items")
    if external_artifact_manifest is not None:
        _configure_pca_cache(
            selection_candidates,
            feature_reduction_config,
            neural_config=neural_config,
            artifact_manifest=external_artifact_manifest,
            image_ids=image_ids,
        )

    selection_result: dict[str, Any] | None = None
    learned_readout_result: dict[str, Any] | None = None
    if encoding_method == "learned_spatial_readout" and selection_enabled:
        (
            score_rows,
            target_score_rows,
            reduced_features_by_layer,
            feature_reduction_metadata,
            learned_readout_result,
            selection_result,
        ) = _select_learned_readout_candidate_and_score_final(
            features_by_layer,
            responses,
            image_ids=image_ids,
            candidates=selection_candidates,
            feature_shapes=feature_shapes,
            train_fraction=train_fraction,
            metric=metric,
            seed=seed,
            device=device,
            selection_config=selection_config,
            learned_config=dict(neural_config.get("learned_readout", {})),
            row_context=_score_row_context(config, item_metadata),
            noise_ceiling=noise_ceiling,
        )
    elif encoding_method == "learned_spatial_readout":
        (
            score_rows,
            target_score_rows,
            reduced_features_by_layer,
            feature_reduction_metadata,
            learned_readout_result,
        ) = _fit_and_score_learned_spatial_readout(
            features_by_layer,
            responses,
            layers=layers,
            feature_shapes=feature_shapes,
            train_fraction=train_fraction,
            metric=metric,
            seed=seed,
            device=device,
            learned_config=dict(neural_config.get("learned_readout", {})),
            layer_fusion=layer_fusion,
            row_context=_score_row_context(config, item_metadata),
            noise_ceiling=noise_ceiling,
            image_ids=image_ids,
        )
    elif selection_enabled:
        (
            score_rows,
            target_score_rows,
            reduced_features_by_layer,
            feature_reduction_metadata,
            selection_result,
        ) = _select_candidate_and_score_final(
            features_by_layer,
            responses,
            image_ids=image_ids,
            candidates=selection_candidates,
            feature_shapes=feature_shapes,
            train_fraction=train_fraction,
            alpha=alpha,
            ridge_alphas=ridge_alphas,
            metric=metric,
            seed=seed,
            selection_config=selection_config,
            row_context=_score_row_context(config, item_metadata),
            noise_ceiling=noise_ceiling,
        )
    else:
        (
            score_rows,
            target_score_rows,
            reduced_features_by_layer,
            feature_reduction_metadata,
        ) = _fit_and_score_layers(
            features_by_layer,
            responses,
            layers=layers,
            feature_shapes=feature_shapes,
            train_fraction=train_fraction,
            alpha=alpha,
            ridge_alphas=ridge_alphas,
            validation_fraction=validation_fraction,
            metric=metric,
            seed=seed,
            feature_reduction=feature_reduction,
            feature_reduction_config=feature_reduction_config,
            row_context=_score_row_context(config, item_metadata),
            noise_ceiling=noise_ceiling,
        )
    activation_payload = (
        {"image_ids": np.asarray(image_ids, dtype=object)}
        if encoding_method == "learned_spatial_readout"
        else {"image_ids": np.asarray(image_ids, dtype=object), **reduced_features_by_layer}
    )
    activation_path = save_activations(activation_payload, output_dir / "activations.npz")
    feature_reduction_metadata_path = output_dir / "feature_reduction_metadata.json"
    feature_reduction_metadata_path.write_text(
        json.dumps(feature_reduction_metadata, indent=2),
        encoding="utf-8",
    )
    scores_path = output_dir / "encoding_scores.csv"
    _write_score_rows(scores_path, score_rows)
    target_scores_path = output_dir / "encoding_target_scores.csv"
    _write_target_score_rows(target_scores_path, target_score_rows)
    rsa_rows: list[dict[str, Any]] = []
    rsa_path: Path | None = None
    if bool(rsa_config.get("enabled", False)):
        rsa_feature_source = reduced_features_by_layer if selection_enabled else features_by_layer
        rsa_layers = [str(selection_result["selected_candidate"]["layer"])] if selection_result else layers
        rsa_rows = _compute_rsa_rows(
            rsa_feature_source,
            responses,
            layers=rsa_layers,
            feature_rdm_metric=str(rsa_config.get("rdm_metric", "correlation")),
            response_rdm_metric=str(rsa_config.get("response_rdm_metric", "correlation")),
            compare_method=str(rsa_config.get("compare_method", "spearman")),
            row_context=_score_row_context(config, item_metadata),
        )
        rsa_path = output_dir / "rsa_scores.csv"
        _write_rsa_rows(rsa_path, rsa_rows)

    geometry_config = dict(neural_config.get("geometry", {}))
    geometry_rows: list[dict[str, Any]] = []
    geometry_path: Path | None = None
    if bool(geometry_config.get("enabled", False)):
        geometry_layers = (
            [str(selection_result["selected_candidate"]["layer"])]
            if selection_result
            else list(reduced_features_by_layer)
        )
        geometry_rows = _compute_geometry_rows(
            reduced_features_by_layer,
            responses,
            layers=geometry_layers,
            methods=[str(method) for method in geometry_config.get("methods", ["linear_cka"])],
            subset_sizes=[int(size) for size in geometry_config.get("subset_sizes", [])],
            subset_seeds=[
                int(value)
                for value in geometry_config.get(
                    "subset_seeds",
                    [geometry_config.get("subset_seed", seed)],
                )
            ],
            null_control_seeds=[
                int(value)
                for value in geometry_config.get(
                    "null_control_seeds",
                    geometry_config.get(
                        "subset_seeds",
                        [geometry_config.get("subset_seed", seed)],
                    ),
                )
            ],
            row_context=_score_row_context(config, item_metadata),
            model_feature_reduction=feature_reduction_metadata.get(
                "feature_reduction",
                feature_reduction,
            ),
        )
        geometry_path = output_dir / "geometry_scores.csv"
        _write_geometry_rows(geometry_path, geometry_rows)

    metadata = {
        "config_path": str(config_path),
        "dataset": config.get("dataset", {}).get("label") or config.get("dataset", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "model_name": config.get("model", {}).get("name"),
        "model_backend": config.get("model", {}).get("backend", "timm"),
        "model_pretrained": bool(config.get("model", {}).get("pretrained", False)),
        "external_artifact": (
            str(resolve_path(external_artifact_path)) if external_artifact_path else None
        ),
        "external_artifact_schema": (
            external_artifact_manifest.get("schema_version")
            if external_artifact_manifest
            else None
        ),
        "external_artifact_provenance": (
            external_artifact_manifest.get("provenance")
            if external_artifact_manifest
            else None
        ),
        "num_items": len(image_ids),
        "layers": layers,
        "response_key": response_key,
        "noise_ceiling_key": noise_ceiling_key,
        "noise_ceiling_available": noise_ceiling is not None,
        "noise_ceiling_source": _single_or_mixed(
            _unique_metadata_values(item_metadata, "noise_ceiling_source")
        ),
        "encoding_method": encoding_method,
        "feature_reduction": (
            feature_reduction_metadata.get("feature_reduction", "learned_spatial_readout")
            if encoding_method == "learned_spatial_readout"
            else feature_reduction
        ),
        "feature_reduction_metadata": str(feature_reduction_metadata_path),
        "ridge_alpha": alpha,
        "ridge_alphas": ridge_alphas,
        "validation_fraction": validation_fraction,
        "train_fraction": train_fraction,
        "metric": metric,
        "metric_scope": _metric_scope_from_target_rows(target_score_rows),
        "alpha_selection_modes": sorted(
            {str(row.get("alpha_selection_mode", "")) for row in score_rows}
        ),
        "selected_ridge_alphas": _selected_ridge_alpha_metadata(score_rows),
        "subjects": sorted(_unique_metadata_values(item_metadata, "subject_id")),
        "rois": sorted(_unique_metadata_values(item_metadata, "roi")),
        "activations": str(activation_path),
        "encoding_scores": str(scores_path),
        "encoding_target_scores": str(target_scores_path),
        "rsa_scores": str(rsa_path) if rsa_path is not None else None,
        "geometry_scores": str(geometry_path) if geometry_path is not None else None,
        "selection_enabled": bool(selection_enabled),
        "selection_artifact": "",
        "selection_candidates": "",
        "selected_layer": "",
        "selected_feature_reduction": "",
        "selected_ridge_alpha": "",
        "selection_score": "",
        "learned_readout_metadata": "",
    }
    if learned_readout_result is not None:
        learned_metadata_path = output_dir / "learned_readout_metadata.json"
        learned_metadata_path.write_text(
            json.dumps(learned_readout_result["metadata"], indent=2),
            encoding="utf-8",
        )
        metadata["learned_readout_metadata"] = str(learned_metadata_path)
    if selection_result is not None:
        selection_candidates_path = output_dir / "selection_candidates.csv"
        _write_selection_candidate_rows(
            selection_candidates_path,
            selection_result["candidate_rows"],
        )
        selection_artifact_path = output_dir / "selection_artifact.json"
        selection_artifact_path.write_text(
            json.dumps(selection_result["artifact"], indent=2),
            encoding="utf-8",
        )
        selected_candidate = selection_result["selected_candidate"]
        selected_alpha = selection_result.get("selected_alpha", "")
        metadata.update(
            {
                "selection_artifact": str(selection_artifact_path),
                "selection_candidates": str(selection_candidates_path),
                "selected_layer": str(selected_candidate["layer"]),
                "selected_feature_reduction": str(selected_candidate["feature_reduction"]),
                "selected_ridge_alpha": (
                    "" if selected_alpha == "" else float(selected_alpha)
                ),
                "selection_score": float(selection_result["selection_score"]),
            }
        )
        del collection
        del features_by_layer
        gc.collect()
        _cleanup_feature_cache(output_dir / "feature_cache")
    if learned_readout_result is not None and selection_result is None:
        del collection
        del features_by_layer
        gc.collect()
        _cleanup_feature_cache(output_dir / "feature_cache")
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        **metadata,
        "metadata": str(metadata_path),
        "score_rows": score_rows,
        "target_score_rows": target_score_rows,
        "rsa_rows": rsa_rows,
        "geometry_rows": geometry_rows,
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
    feature_storage_dir: Path | None = None,
) -> tuple[
    list[str],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, list[int]],
    list[dict[str, Any]],
    np.ndarray | None,
]:
    image_ids: list[str] = []
    responses: list[np.ndarray] = []
    noise_ceilings: list[np.ndarray | None] = []
    item_metadata: list[dict[str, Any]] = []
    collected: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}
    feature_memmaps: dict[str, np.ndarray] = {}
    feature_shapes: dict[str, list[int]] = {}
    expected_items = _dataset_len(dataset) if feature_reduction == "selection_raw" else None
    if feature_reduction == "selection_raw" and feature_storage_dir is not None:
        feature_storage_dir.mkdir(parents=True, exist_ok=True)

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
            feature_array = _feature_array(feature_dict[layer])
            feature_shapes.setdefault(layer, [int(dim) for dim in feature_array.shape])
            if feature_reduction == "selection_raw":
                if expected_items is not None and feature_storage_dir is not None:
                    if layer not in feature_memmaps:
                        feature_memmaps[layer] = _create_feature_memmap(
                            feature_storage_dir,
                            layer=layer,
                            shape=(expected_items, *feature_array.shape),
                        )
                    feature_memmaps[layer][index] = feature_array
                else:
                    collected[layer].append(feature_array)
            elif feature_reduction == "flatten_pca":
                collected[layer].append(feature_array.reshape(-1))
            else:
                collected[layer].append(
                    _reduce_feature_array(feature_array, method=feature_reduction)
                )

    response_matrix = _stack_consistent(responses, "ROI responses")
    if feature_reduction == "selection_raw" and feature_memmaps:
        feature_matrix = {
            layer: feature_memmaps[layer][: len(image_ids)]
            for layer in layers
        }
    else:
        feature_matrix = {
            layer: _stack_consistent(values, f"features for layer {layer}")
            for layer, values in collected.items()
        }
    noise_ceiling_array = _validate_noise_ceiling(
        noise_ceilings,
        num_targets=response_matrix.shape[1],
        key=noise_ceiling_key,
    )
    return (
        image_ids,
        response_matrix,
        feature_matrix,
        feature_shapes,
        item_metadata,
        noise_ceiling_array,
    )


def _collect_external_features_and_responses(
    *,
    dataset: Any,
    artifact_path: Path,
    layers: list[str],
    response_key: str,
    noise_ceiling_key: str,
    feature_reduction: str,
    verify_hashes: bool,
    feature_storage_dir: Path | None,
) -> tuple[
    tuple[
        list[str],
        np.ndarray,
        dict[str, np.ndarray],
        dict[str, list[int]],
        list[dict[str, Any]],
        np.ndarray | None,
    ],
    dict[str, Any],
]:
    if feature_reduction == "selection_raw":
        if feature_storage_dir is None:
            raise ValueError("External selection_raw import requires feature storage")
        artifact_image_ids, raw_features, manifest = (
            load_external_features_to_memmaps(
                artifact_path,
                layers=layers,
                storage_dir=feature_storage_dir,
                verify_hashes=verify_hashes,
            )
        )
    else:
        artifact_image_ids, raw_features, manifest = load_external_features(
            artifact_path,
            layers=layers,
            verify_hashes=verify_hashes,
        )
    dataset_image_ids: list[str] = []
    responses: list[np.ndarray] = []
    noise_ceilings: list[np.ndarray | None] = []
    item_metadata: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        response = _response_for_item(item, response_key)
        if response is None:
            raise ValueError(
                f"Dataset item {item.get('image_id', index)} is missing neural response "
                f"'{response_key}'"
            )
        dataset_image_ids.append(str(item.get("image_id", f"item_{index:04d}")))
        responses.append(np.asarray(response, dtype=np.float32).ravel())
        metadata = item.get("metadata", {})
        item_metadata.append(metadata if isinstance(metadata, dict) else {})
        noise_ceiling = _response_for_item(item, noise_ceiling_key)
        noise_ceilings.append(
            None
            if noise_ceiling is None
            else np.asarray(noise_ceiling, dtype=np.float32).ravel()
        )
    if dataset_image_ids != artifact_image_ids:
        mismatch = next(
            (
                index
                for index, (dataset_id, artifact_id) in enumerate(
                    zip(dataset_image_ids, artifact_image_ids)
                )
                if dataset_id != artifact_id
            ),
            min(len(dataset_image_ids), len(artifact_image_ids)),
        )
        raise ValueError(
            "External artifact image order does not match the neural dataset at "
            f"index {mismatch}; dataset={dataset_image_ids[mismatch:mismatch + 1]}, "
            f"artifact={artifact_image_ids[mismatch:mismatch + 1]}"
        )
    response_matrix = _stack_consistent(responses, "ROI responses")
    feature_shapes = {
        layer: [int(value) for value in np.asarray(values).shape[1:]]
        for layer, values in raw_features.items()
    }
    if feature_reduction == "selection_raw":
        features_by_layer = raw_features
    elif feature_reduction == "flatten_pca":
        features_by_layer = {
            layer: np.asarray(values).reshape(len(dataset_image_ids), -1)
            for layer, values in raw_features.items()
        }
    else:
        features_by_layer = {
            layer: _stack_consistent(
                [
                    _reduce_feature_array(item, method=feature_reduction)
                    for item in np.asarray(values)
                ],
                f"features for layer {layer}",
            )
            for layer, values in raw_features.items()
        }
    noise_ceiling_array = _validate_noise_ceiling(
        noise_ceilings,
        num_targets=response_matrix.shape[1],
        key=noise_ceiling_key,
    )
    return (
        (
            dataset_image_ids,
            response_matrix,
            features_by_layer,
            feature_shapes,
            item_metadata,
            noise_ceiling_array,
        ),
        manifest,
    )


def _fit_and_score_layers(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    feature_shapes: dict[str, list[int]],
    train_fraction: float,
    alpha: float,
    ridge_alphas: list[float] | None,
    validation_fraction: float,
    metric: str,
    seed: int,
    feature_reduction: str,
    feature_reduction_config: dict[str, Any],
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    n_items = responses.shape[0]
    n_train = min(max(1, int(round(n_items * train_fraction))), n_items - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    reduced_features_by_layer: dict[str, np.ndarray] = {}
    feature_metadata_rows: list[dict[str, Any]] = []
    for layer in layers:
        raw_features = features_by_layer[layer]
        features, feature_metadata = _prepare_layer_features(
            raw_features,
            layer=layer,
            input_shape=feature_shapes.get(layer, list(raw_features.shape[1:])),
            train_idx=train_idx,
            feature_reduction=feature_reduction,
            feature_reduction_config=feature_reduction_config,
        )
        reduced_features_by_layer[layer] = features
        feature_metadata_rows.append(feature_metadata)
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
    return rows, target_rows, reduced_features_by_layer, {
        "feature_reduction": feature_reduction,
        "layers": feature_metadata_rows,
    }


def _fit_and_score_learned_spatial_readout(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    feature_shapes: dict[str, list[int]],
    train_fraction: float,
    metric: str,
    seed: int,
    device: str,
    learned_config: dict[str, Any],
    layer_fusion: str = "",
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
    image_ids: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    layer, raw_features, fusion_metadata = _learned_readout_features(
        features_by_layer,
        layers=layers,
        layer_fusion=layer_fusion,
    )
    n_items = responses.shape[0]
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _outer_train_test_indices(n_items, train_fraction, rng)
    readout_config = _learned_readout_config(
        learned_config,
        seed=seed,
        device=device,
    )
    feature_reduction_label = _learned_readout_feature_reduction(readout_config)
    inner_train_idx, validation_idx = _inner_train_validation_indices(
        train_idx,
        readout_config.validation_fraction,
        rng,
    )
    model_bundle = fit_spatial_readout(
        raw_features,
        responses,
        train_idx=inner_train_idx,
        validation_idx=validation_idx,
        config=readout_config,
    )
    predictions = predict_spatial_readout(model_bundle, raw_features, indices=test_idx)
    scores = evaluate_encoding(predictions, responses[test_idx], metric=metric)
    layer_context = {
        **row_context,
        "layer": layer,
        "metric": metric,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "split_seed": int(seed),
        "feature_reduction": feature_reduction_label,
        "selected_ridge_alpha": "",
        "alpha_selection_mode": "early_stopping_validation",
    }
    target_rows = [
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
    rows = [
        {
            **layer_context,
            "num_targets": int(scores.size),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
            "mean_r2_score_from_r": float(
                np.mean([float(target_row["r2_score_from_r"]) for target_row in target_rows])
            ),
            **_noise_normalized_layer_summary(target_rows),
            "metric_scope": _metric_scope_from_target_rows(target_rows),
        }
    ]
    readout_metadata = dict(model_bundle["metadata"])
    readout_metadata.update(
        {
            "layer": layer,
            "layers": list(layers),
            "feature_reduction": feature_reduction_label,
            "n_outer_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "outer_train_image_ids": _ids_for_indices(image_ids, train_idx),
            "outer_test_image_ids": _ids_for_indices(image_ids, test_idx),
            "selection_train_image_ids": _ids_for_indices(image_ids, inner_train_idx),
            "selection_validation_image_ids": _ids_for_indices(image_ids, validation_idx),
            **fusion_metadata,
        }
    )
    feature_metadata = {
        "feature_reduction": feature_reduction_label,
        "layers": [
            {
                "layer": layer,
                "method": feature_reduction_label,
                "readout_variant": readout_config.variant,
                "spatial_rank": int(readout_config.spatial_rank),
                "input_feature_shape": [
                    int(dim)
                    for dim in _learned_readout_input_shape(
                        layer=layer,
                        layers=layers,
                        feature_shapes=feature_shapes,
                        raw_features=raw_features,
                    )
                ],
                "output_feature_shape": [int(responses.shape[1])],
                "requested_components": "",
                "effective_components": int(responses.shape[1]),
                "explained_variance_ratio_sum": "",
                "train_only_fit": True,
                "n_train_fit": int(inner_train_idx.size),
                "random_seed": int(readout_config.seed),
                "pca_solver": "",
                "pca_whiten": "",
                **fusion_metadata,
            }
        ],
    }
    return (
        rows,
        target_rows,
        {},
        feature_metadata,
        {
            "metadata": readout_metadata,
        },
    )


def _learned_readout_features(
    features_by_layer: dict[str, np.ndarray],
    *,
    layers: list[str],
    layer_fusion: str,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    if len(layers) == 1:
        if layer_fusion and layer_fusion != "channel_concat":
            raise ValueError("learned_spatial_readout layer_fusion must be 'channel_concat'")
        layer = layers[0]
        return layer, features_by_layer[layer], {}
    if layer_fusion != "channel_concat":
        raise ValueError(
            "learned_spatial_readout requires neural.layer_fusion='channel_concat' "
            "when multiple layers are configured"
        )
    fused, metadata = fuse_spatial_feature_layers(
        features_by_layer,
        layers=layers,
        fusion_method=layer_fusion,
    )
    return _fused_layer_label(layers), fused, metadata


def _fused_layer_label(layers: list[str]) -> str:
    return "+".join(str(layer) for layer in layers)


def _learned_readout_input_shape(
    *,
    layer: str,
    layers: list[str],
    feature_shapes: dict[str, list[int]],
    raw_features: np.ndarray,
) -> list[int]:
    if len(layers) == 1:
        return [int(dim) for dim in feature_shapes.get(layer, list(raw_features.shape[1:]))]
    return [int(dim) for dim in raw_features.shape[1:]]


def _learned_readout_config(
    learned_config: dict[str, Any],
    *,
    seed: int,
    device: str,
) -> SpatialReadoutConfig:
    validation_fraction = float(learned_config.get("validation_fraction", 0.2))
    variant = str(learned_config.get("variant", "separable"))
    spatial_rank = int(learned_config.get("spatial_rank", 1 if variant == "separable" else 4))
    return SpatialReadoutConfig(
        variant=variant,
        spatial_rank=spatial_rank,
        max_epochs=int(learned_config.get("max_epochs", 100)),
        batch_size=int(learned_config.get("batch_size", 32)),
        target_batch_size=int(learned_config.get("target_batch_size", 256)),
        lr=float(learned_config.get("lr", 1e-3)),
        weight_decay=float(learned_config.get("weight_decay", 1e-4)),
        patience=int(learned_config.get("patience", 10)),
        min_delta=float(learned_config.get("min_delta", 1e-6)),
        validation_fraction=validation_fraction,
        objective=str(learned_config.get("objective", "pearson")),
        seed=int(learned_config.get("seed", seed)),
        device=str(learned_config.get("device", device)),
        progress=bool(learned_config.get("progress", False)),
        progress_every=int(learned_config.get("progress_every", 1)),
    )


def _learned_readout_feature_reduction(config: SpatialReadoutConfig) -> str:
    if config.variant == "voxel_specific_lowrank":
        return "voxel_specific_spatial_readout"
    return "learned_spatial_readout"


def _dataset_len(dataset: Any) -> int | None:
    try:
        length = len(dataset)
    except TypeError:
        return None
    return int(length) if length > 0 else None


def _create_feature_memmap(
    directory: Path,
    *,
    layer: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    path = directory / f"{_safe_name(layer)}.npy"
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )


def _cleanup_feature_cache(directory: Path) -> None:
    if directory.is_dir():
        shutil.rmtree(directory)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _select_candidate_and_score_final(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    image_ids: list[str],
    candidates: list[dict[str, Any]],
    feature_shapes: dict[str, list[int]],
    train_fraction: float,
    alpha: float,
    ridge_alphas: list[float] | None,
    metric: str,
    seed: int,
    selection_config: dict[str, Any],
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    n_items = responses.shape[0]
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _outer_train_test_indices(n_items, train_fraction, rng)
    selection_fraction = float(selection_config.get("validation_fraction", 0.2))
    selection_train_idx, validation_idx = _inner_train_validation_indices(
        train_idx,
        selection_fraction,
        rng,
    )
    primary_score = str(selection_config.get("primary_score", "mean_noise_normalized_score"))

    candidate_rows: list[dict[str, Any]] = []
    candidate_artifacts: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        layer = str(candidate["layer"])
        candidate_feature_reduction = str(candidate["feature_reduction"])
        candidate_config = _candidate_feature_reduction_config(candidate)
        features, feature_metadata = _prepare_layer_features(
            features_by_layer[layer],
            layer=layer,
            input_shape=feature_shapes.get(layer, list(features_by_layer[layer].shape[1:])),
            train_idx=selection_train_idx,
            feature_reduction=candidate_feature_reduction,
            feature_reduction_config=candidate_config,
        )
        alpha_result = _select_ridge_alpha_on_validation(
            features,
            responses,
            train_idx=selection_train_idx,
            validation_idx=validation_idx,
            alpha=alpha,
            ridge_alphas=ridge_alphas,
            metric=metric,
            noise_ceiling=noise_ceiling,
            primary_score=primary_score,
        )
        row = {
            **row_context,
            "candidate_index": int(candidate_index),
            "layer": layer,
            "feature_reduction": candidate_feature_reduction,
            "pca_components": candidate.get("pca_components", ""),
            "pca_solver": candidate.get("pca_solver", ""),
            "pca_whiten": candidate.get("pca_whiten", ""),
            "selected_ridge_alpha": float(alpha_result["selected_alpha"]),
            "alpha_selection_mode": alpha_result["mode"],
            "validation_score": float(alpha_result["score"]),
            "validation_score_type": alpha_result["score_type"],
            "primary_score": primary_score,
            "selection_n_train": int(selection_train_idx.size),
            "selection_n_validation": int(validation_idx.size),
            "outer_n_train": int(train_idx.size),
            "outer_n_test": int(test_idx.size),
            "selected": "false",
            "pca_n_train_fit": feature_metadata.get("n_train_fit", ""),
            "pca_effective_components": feature_metadata.get("effective_components", ""),
            "pca_explained_variance_ratio_sum": feature_metadata.get(
                "explained_variance_ratio_sum",
                "",
            ),
        }
        candidate_rows.append(row)
        candidate_artifacts.append(
            {
                "candidate": dict(candidate),
                "validation": dict(row),
                "feature_reduction_metadata": feature_metadata,
            }
        )

    if not candidate_rows:
        raise ValueError("neural.selection requires at least one candidate")
    selected_index = _selected_candidate_index(candidate_rows)
    candidate_rows[selected_index]["selected"] = "true"
    candidate_artifacts[selected_index]["validation"]["selected"] = "true"
    selected_candidate = candidates[selected_index]
    selected_alpha = float(candidate_rows[selected_index]["selected_ridge_alpha"])
    selected_layer = str(selected_candidate["layer"])
    selected_feature_reduction = str(selected_candidate["feature_reduction"])
    selected_features, selected_feature_metadata = _prepare_layer_features(
        features_by_layer[selected_layer],
        layer=selected_layer,
        input_shape=feature_shapes.get(
            selected_layer,
            list(features_by_layer[selected_layer].shape[1:]),
        ),
        train_idx=train_idx,
        feature_reduction=selected_feature_reduction,
        feature_reduction_config=_candidate_feature_reduction_config(selected_candidate),
    )
    final_rows, final_target_rows = _score_prepared_layer(
        selected_features,
        responses,
        train_idx=train_idx,
        test_idx=test_idx,
        layer=selected_layer,
        metric=metric,
        selected_alpha=selected_alpha,
        alpha_selection_mode="selection_validation",
        split_seed=seed,
        feature_reduction=selected_feature_reduction,
        row_context=row_context,
        noise_ceiling=noise_ceiling,
    )
    reduced_features_by_layer = {selected_layer: selected_features}
    feature_reduction_metadata = {
        "feature_reduction": selected_feature_reduction,
        "layers": [selected_feature_metadata],
        "selection_enabled": True,
    }
    artifact = {
        "selection_enabled": True,
        "primary_score": primary_score,
        "split_seed": int(seed),
        "outer_train_image_ids": _ids_for_indices(image_ids, train_idx),
        "outer_test_image_ids": _ids_for_indices(image_ids, test_idx),
        "selection_train_image_ids": _ids_for_indices(image_ids, selection_train_idx),
        "selection_validation_image_ids": _ids_for_indices(image_ids, validation_idx),
        "candidates": candidate_artifacts,
        "selected_candidate_index": int(selected_index),
        "selected_candidate": dict(selected_candidate),
        "selected_alpha": selected_alpha,
        "selection_score": float(candidate_rows[selected_index]["validation_score"]),
        "selection_score_type": candidate_rows[selected_index]["validation_score_type"],
        "final_test_config": {
            "layer": selected_layer,
            "feature_reduction": selected_feature_reduction,
            "selected_ridge_alpha": selected_alpha,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "feature_reduction_metadata": selected_feature_metadata,
        },
    }
    selection_result = {
        "candidate_rows": candidate_rows,
        "artifact": artifact,
        "selected_candidate": dict(selected_candidate),
        "selected_alpha": selected_alpha,
        "selection_score": float(candidate_rows[selected_index]["validation_score"]),
    }
    return (
        final_rows,
        final_target_rows,
        reduced_features_by_layer,
        feature_reduction_metadata,
        selection_result,
    )


def _select_learned_readout_candidate_and_score_final(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    image_ids: list[str],
    candidates: list[dict[str, Any]],
    feature_shapes: dict[str, list[int]],
    train_fraction: float,
    metric: str,
    seed: int,
    device: str,
    selection_config: dict[str, Any],
    learned_config: dict[str, Any],
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    n_items = responses.shape[0]
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _outer_train_test_indices(n_items, train_fraction, rng)
    selection_fraction = float(selection_config.get("validation_fraction", 0.2))
    selection_train_idx, validation_idx = _inner_train_validation_indices(
        train_idx,
        selection_fraction,
        rng,
    )
    primary_score = str(selection_config.get("primary_score", "mean_noise_normalized_score"))
    readout_config = _learned_readout_config(
        learned_config,
        seed=seed,
        device=device,
    )

    candidate_rows: list[dict[str, Any]] = []
    candidate_artifacts: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        layer = str(candidate["layer"])
        if str(candidate.get("feature_reduction", "")) != "learned_spatial_readout":
            raise ValueError(
                "learned_spatial_readout selection candidates must use "
                "feature_reduction='learned_spatial_readout'"
            )
        raw_features = features_by_layer[layer]
        validation_result = _score_learned_readout_on_validation(
            raw_features,
            responses,
            train_idx=selection_train_idx,
            validation_idx=validation_idx,
            metric=metric,
            readout_config=readout_config,
            noise_ceiling=noise_ceiling,
            primary_score=primary_score,
        )
        row = {
            **row_context,
            "candidate_index": int(candidate_index),
            "layer": layer,
            "feature_reduction": "learned_spatial_readout",
            "pca_components": "",
            "pca_solver": "",
            "pca_whiten": "",
            "selected_ridge_alpha": "",
            "alpha_selection_mode": "learned_readout_selection_validation",
            "validation_score": float(validation_result["score"]),
            "validation_score_type": validation_result["score_type"],
            "primary_score": primary_score,
            "selection_n_train": int(selection_train_idx.size),
            "selection_n_validation": int(validation_idx.size),
            "outer_n_train": int(train_idx.size),
            "outer_n_test": int(test_idx.size),
            "selected": "false",
            "pca_n_train_fit": "",
            "pca_effective_components": "",
            "pca_explained_variance_ratio_sum": "",
            "selection_best_epoch": validation_result["metadata"].get("best_epoch", ""),
            "selection_epochs_ran": validation_result["metadata"].get("epochs_ran", ""),
        }
        candidate_rows.append(row)
        candidate_artifacts.append(
            {
                "candidate": dict(candidate),
                "validation": dict(row),
                "learned_readout_metadata": validation_result["metadata"],
            }
        )

    if not candidate_rows:
        raise ValueError("neural.selection requires at least one candidate")
    selected_index = _selected_candidate_index(candidate_rows)
    candidate_rows[selected_index]["selected"] = "true"
    candidate_artifacts[selected_index]["validation"]["selected"] = "true"
    selected_candidate = candidates[selected_index]
    selected_layer = str(selected_candidate["layer"])

    (
        final_rows,
        final_target_rows,
        reduced_features_by_layer,
        feature_reduction_metadata,
        learned_readout_result,
    ) = _fit_and_score_learned_spatial_readout(
        features_by_layer,
        responses,
        layers=[selected_layer],
        feature_shapes=feature_shapes,
        train_fraction=train_fraction,
        metric=metric,
        seed=seed,
        device=device,
        learned_config=learned_config,
        row_context=row_context,
        noise_ceiling=noise_ceiling,
        image_ids=image_ids,
    )
    for row in final_rows:
        row["alpha_selection_mode"] = "learned_readout_selection_validation"
    for row in final_target_rows:
        row["alpha_selection_mode"] = "learned_readout_selection_validation"
    feature_reduction_metadata["selection_enabled"] = True

    artifact = {
        "selection_enabled": True,
        "primary_score": primary_score,
        "split_seed": int(seed),
        "outer_train_image_ids": _ids_for_indices(image_ids, train_idx),
        "outer_test_image_ids": _ids_for_indices(image_ids, test_idx),
        "selection_train_image_ids": _ids_for_indices(image_ids, selection_train_idx),
        "selection_validation_image_ids": _ids_for_indices(image_ids, validation_idx),
        "candidates": candidate_artifacts,
        "selected_candidate_index": int(selected_index),
        "selected_candidate": dict(selected_candidate),
        "selected_alpha": "",
        "selection_score": float(candidate_rows[selected_index]["validation_score"]),
        "selection_score_type": candidate_rows[selected_index]["validation_score_type"],
        "final_test_config": {
            "layer": selected_layer,
            "feature_reduction": "learned_spatial_readout",
            "selected_ridge_alpha": "",
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "feature_reduction_metadata": feature_reduction_metadata["layers"][0],
        },
    }
    selection_result = {
        "candidate_rows": candidate_rows,
        "artifact": artifact,
        "selected_candidate": {
            "layer": selected_layer,
            "feature_reduction": "learned_spatial_readout",
            "candidate_index": int(selected_index),
        },
        "selected_alpha": "",
        "selection_score": float(candidate_rows[selected_index]["validation_score"]),
    }
    return (
        final_rows,
        final_target_rows,
        reduced_features_by_layer,
        feature_reduction_metadata,
        learned_readout_result,
        selection_result,
    )


def _score_learned_readout_on_validation(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    metric: str,
    readout_config: SpatialReadoutConfig,
    noise_ceiling: np.ndarray | None,
    primary_score: str,
) -> dict[str, Any]:
    model_bundle = fit_spatial_readout(
        features,
        responses,
        train_idx=train_idx,
        validation_idx=validation_idx,
        config=readout_config,
    )
    predictions = predict_spatial_readout(model_bundle, features, indices=validation_idx)
    scores = evaluate_encoding(predictions, responses[validation_idx], metric=metric)
    target_rows = benchmark_encoding_target_scores(
        predictions,
        responses[validation_idx],
        noise_ceiling=noise_ceiling,
    )
    row = {
        "mean_score": float(np.mean(scores)),
        **_noise_normalized_layer_summary(target_rows),
    }
    score, score_type = _selection_primary_score(row, primary_score)
    return {
        "score": score,
        "score_type": score_type,
        "mean_score": row["mean_score"],
        "metadata": dict(model_bundle["metadata"]),
    }


def _score_prepared_layer(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    layer: str,
    metric: str,
    selected_alpha: float,
    alpha_selection_mode: str,
    split_seed: int,
    feature_reduction: str,
    row_context: dict[str, Any],
    noise_ceiling: np.ndarray | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        "split_seed": int(split_seed),
        "feature_reduction": feature_reduction,
        "selected_ridge_alpha": float(selected_alpha),
        "alpha_selection_mode": alpha_selection_mode,
    }
    target_rows = [
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
    row = {
        **layer_context,
        "num_targets": int(scores.size),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
        "mean_r2_score_from_r": float(
            np.mean([float(target_row["r2_score_from_r"]) for target_row in target_rows])
        ),
        **_noise_normalized_layer_summary(target_rows),
        "metric_scope": _metric_scope_from_target_rows(target_rows),
    }
    return [row], target_rows


def _feature_reduction_config(
    neural_config: dict[str, Any],
    *,
    feature_reduction: str,
    default_seed: int,
) -> dict[str, Any]:
    if feature_reduction != "flatten_pca":
        return {}
    if "pca_components" not in neural_config:
        raise ValueError(
            "neural.pca_components is required when feature_reduction is 'flatten_pca'"
        )
    components = int(neural_config["pca_components"])
    if components <= 0:
        raise ValueError("neural.pca_components must be a positive integer")
    return {
        "pca_components": components,
        "pca_solver": str(neural_config.get("pca_solver", "randomized")),
        "pca_whiten": bool(neural_config.get("pca_whiten", False)),
        "feature_reduction_seed": int(
            neural_config.get("feature_reduction_seed", default_seed)
        ),
    }


def _configure_pca_cache(
    selection_candidates: list[dict[str, Any]],
    feature_reduction_config: dict[str, Any],
    *,
    neural_config: dict[str, Any],
    artifact_manifest: dict[str, Any],
    image_ids: list[str],
) -> None:
    cache_dir = neural_config.get("pca_cache_dir")
    if not cache_dir:
        return
    source_payload = {
        "schema_version": artifact_manifest.get("schema_version"),
        "model_id": artifact_manifest.get("model_id"),
        "num_images": artifact_manifest.get("num_images"),
        "features": artifact_manifest.get("features"),
        "provenance": artifact_manifest.get("provenance"),
        "image_ids": image_ids,
    }
    source = hashlib.sha256(
        json.dumps(
            source_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolved_cache = str(resolve_path(cache_dir))
    feature_reduction_config.update(
        {
            "pca_cache_dir": resolved_cache,
            "pca_cache_source": source,
        }
    )
    for candidate in selection_candidates:
        if str(candidate.get("feature_reduction")) == "flatten_pca":
            candidate["pca_cache_dir"] = resolved_cache
            candidate["pca_cache_source"] = source


def _selection_candidate_configs(
    neural_config: dict[str, Any],
    *,
    layers: list[str],
    feature_reduction: str,
    default_seed: int,
) -> list[dict[str, Any]]:
    selection_config = dict(neural_config.get("selection", {}))
    raw_candidates = selection_config.get("candidates")
    if raw_candidates is None:
        raw_candidates = [{"layer": layer} for layer in layers]
    if not isinstance(raw_candidates, list):
        raise ValueError("neural.selection.candidates must be a list")

    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError("Each neural.selection candidate must be a mapping")
        layer = raw_candidate.get("layer")
        if layer is None:
            raise ValueError("Each neural.selection candidate requires a layer")
        candidate_feature_reduction = str(
            raw_candidate.get("feature_reduction", feature_reduction)
        )
        candidate: dict[str, Any] = {
            "layer": str(layer),
            "feature_reduction": candidate_feature_reduction,
            "candidate_index": int(index),
        }
        if candidate_feature_reduction == "flatten_pca":
            if "pca_components" in raw_candidate:
                pca_components = raw_candidate["pca_components"]
            elif "pca_components" in neural_config:
                pca_components = neural_config["pca_components"]
            else:
                raise ValueError(
                    "neural.selection flatten_pca candidates require pca_components"
                )
            candidate.update(
                {
                    "pca_components": int(pca_components),
                    "pca_solver": str(
                        raw_candidate.get(
                            "pca_solver",
                            neural_config.get("pca_solver", "randomized"),
                        )
                    ),
                    "pca_whiten": bool(
                        raw_candidate.get(
                            "pca_whiten",
                            neural_config.get("pca_whiten", False),
                        )
                    ),
                    "feature_reduction_seed": int(
                        raw_candidate.get(
                            "feature_reduction_seed",
                            neural_config.get("feature_reduction_seed", default_seed),
                        )
                    ),
                }
            )
        elif candidate_feature_reduction not in {
            "flatten",
            "spatial_mean",
            "learned_spatial_readout",
        }:
            raise ValueError(
                "feature_reduction must be 'flatten', 'spatial_mean', "
                "'flatten_pca', or 'learned_spatial_readout'"
            )
        candidates.append(candidate)
    return candidates


def _candidate_feature_reduction_config(candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("feature_reduction")) != "flatten_pca":
        return {}
    return {
        "pca_components": int(candidate["pca_components"]),
        "pca_solver": str(candidate.get("pca_solver", "randomized")),
        "pca_whiten": bool(candidate.get("pca_whiten", False)),
        "feature_reduction_seed": int(candidate.get("feature_reduction_seed", 0)),
        "pca_cache_dir": candidate.get("pca_cache_dir"),
        "pca_cache_source": candidate.get("pca_cache_source"),
    }


def _prepare_layer_features(
    features: np.ndarray,
    *,
    layer: str,
    input_shape: list[int],
    train_idx: np.ndarray,
    feature_reduction: str,
    feature_reduction_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_reduction == "flatten_pca":
        return _fit_transform_flatten_pca(
            features,
            layer=layer,
            input_shape=input_shape,
            train_idx=train_idx,
            pca_components=int(feature_reduction_config["pca_components"]),
            pca_solver=str(feature_reduction_config["pca_solver"]),
            pca_whiten=bool(feature_reduction_config["pca_whiten"]),
            random_seed=int(feature_reduction_config["feature_reduction_seed"]),
            cache_dir=feature_reduction_config.get("pca_cache_dir"),
            cache_source=feature_reduction_config.get("pca_cache_source"),
        )

    reduced = _reduce_feature_matrix(features, method=feature_reduction)
    metadata = {
        "layer": layer,
        "method": feature_reduction,
        "input_feature_shape": [int(dim) for dim in input_shape],
        "output_feature_shape": [int(dim) for dim in reduced.shape[1:]],
        "requested_components": "",
        "effective_components": int(reduced.shape[1]) if reduced.ndim == 2 else "",
        "explained_variance_ratio_sum": "",
        "train_only_fit": False,
        "n_train_fit": 0,
        "random_seed": "",
        "pca_solver": "",
        "pca_whiten": "",
    }
    return reduced, metadata


def _fit_transform_flatten_pca(
    features: np.ndarray,
    *,
    layer: str,
    input_shape: list[int],
    train_idx: np.ndarray,
    pca_components: int,
    pca_solver: str,
    pca_whiten: bool,
    random_seed: int,
    cache_dir: str | Path | None = None,
    cache_source: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "feature_reduction='flatten_pca' requires scikit-learn; "
            "install the project with the neural extra"
        ) from exc

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim < 2:
        raise ValueError("flatten_pca expects a batched feature matrix")
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    max_components = min(int(train_idx.size), int(matrix.shape[1]))
    if pca_components > max_components:
        raise ValueError(
            "neural.pca_components must be <= min(n_train, n_features): "
            f"{pca_components} > {max_components}"
        )
    cache_key = _pca_cache_key(
        cache_source=cache_source,
        layer=layer,
        input_shape=input_shape,
        matrix_shape=matrix.shape,
        matrix_dtype=matrix.dtype,
        train_idx=train_idx,
        pca_components=pca_components,
        pca_solver=pca_solver,
        pca_whiten=pca_whiten,
        random_seed=random_seed,
    )
    cached = _load_pca_cache(cache_dir, cache_key, expected_rows=matrix.shape[0])
    if cached is not None:
        reduced, metadata = cached
        return reduced, {
            **metadata,
            "cache_hit": True,
            "cache_key": cache_key,
        }

    train_matrix = np.asarray(matrix[train_idx], dtype=np.float32, order="C")
    pca = PCA(
        n_components=pca_components,
        svd_solver=pca_solver,
        whiten=pca_whiten,
        random_state=random_seed,
        copy=False,
    )
    pca.fit(train_matrix)
    reduced = _transform_pca_in_batches(matrix, pca).astype(np.float32, copy=False)
    metadata = {
        "layer": layer,
        "method": "flatten_pca",
        "input_feature_shape": [int(dim) for dim in input_shape],
        "output_feature_shape": [int(reduced.shape[1])],
        "requested_components": int(pca_components),
        "effective_components": int(reduced.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "train_only_fit": True,
        "n_train_fit": int(train_idx.size),
        "random_seed": int(random_seed),
        "pca_solver": pca_solver,
        "pca_whiten": bool(pca_whiten),
        "cache_hit": False,
        "cache_key": cache_key,
    }
    _write_pca_cache(cache_dir, cache_key, reduced, metadata)
    return reduced, metadata


def _transform_pca_in_batches(
    matrix: np.ndarray,
    pca: Any,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    components = np.asarray(pca.components_)
    mean = np.asarray(pca.mean_)
    n_items = int(matrix.shape[0])
    n_components = int(components.shape[0])
    reduced = np.empty((n_items, n_components), dtype=np.float32)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = np.asarray(matrix[start:end], dtype=np.float32).reshape(end - start, -1)
        centered = batch - mean
        transformed = centered @ components.T
        if bool(getattr(pca, "whiten", False)):
            scale = np.sqrt(np.asarray(pca.explained_variance_))
            scale[scale == 0.0] = 1.0
            transformed = transformed / scale
        reduced[start:end] = transformed.astype(np.float32, copy=False)
    return reduced


def _pca_cache_key(
    *,
    cache_source: str | None,
    layer: str,
    input_shape: list[int],
    matrix_shape: tuple[int, ...],
    matrix_dtype: np.dtype,
    train_idx: np.ndarray,
    pca_components: int,
    pca_solver: str,
    pca_whiten: bool,
    random_seed: int,
) -> str:
    payload = {
        "schema_version": "hma.neural.flatten_pca_cache.v1",
        "source": cache_source or "uncached",
        "layer": layer,
        "input_shape": [int(value) for value in input_shape],
        "matrix_shape": [int(value) for value in matrix_shape],
        "matrix_dtype": str(matrix_dtype),
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_idx, dtype=np.int64).tobytes()
        ).hexdigest(),
        "pca_components": int(pca_components),
        "pca_solver": pca_solver,
        "pca_whiten": bool(pca_whiten),
        "random_seed": int(random_seed),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_pca_cache(
    cache_dir: str | Path | None,
    cache_key: str,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser().resolve()
    array_path = root / f"{cache_key}.npy"
    metadata_path = root / f"{cache_key}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        reduced = np.load(array_path, mmap_mode="r", allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        reduced.ndim != 2
        or reduced.shape[0] != expected_rows
        or int(metadata.get("effective_components", -1)) != reduced.shape[1]
    ):
        return None
    return reduced, metadata


def _write_pca_cache(
    cache_dir: str | Path | None,
    cache_key: str,
    reduced: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if not cache_dir:
        return
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / f"{cache_key}.npy"
    metadata_path = root / f"{cache_key}.json"
    array_tmp = root / f"{cache_key}.npy.tmp"
    metadata_tmp = root / f"{cache_key}.json.tmp"
    with array_tmp.open("wb") as handle:
        np.save(handle, np.asarray(reduced, dtype=np.float32), allow_pickle=False)
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    array_tmp.replace(array_path)
    metadata_tmp.replace(metadata_path)


def _reduce_feature_matrix(features: np.ndarray, method: str = "flatten") -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim < 2:
        raise ValueError("Expected a batched feature matrix")
    reduced = [_reduce_feature_array(item, method=method) for item in matrix]
    return _stack_consistent(reduced, f"reduced {method} features")


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


def _select_ridge_alpha_on_validation(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    alpha: float,
    ridge_alphas: list[float] | None,
    metric: str,
    noise_ceiling: np.ndarray | None,
    primary_score: str,
) -> dict[str, Any]:
    candidate_alphas = ridge_alphas or [float(alpha)]
    mode = "selection_validation" if ridge_alphas else "fixed"
    best_alpha = float(candidate_alphas[0])
    best_score = -np.inf
    best_score_type = "raw"
    for candidate_alpha in candidate_alphas:
        model = fit_ridge_encoding(
            features[train_idx],
            responses[train_idx],
            alpha=float(candidate_alpha),
        )
        predictions = predict_ridge_encoding(model, features[validation_idx])
        scores = evaluate_encoding(predictions, responses[validation_idx], metric=metric)
        target_rows = benchmark_encoding_target_scores(
            predictions,
            responses[validation_idx],
            noise_ceiling=noise_ceiling,
        )
        row = {
            "mean_score": float(np.mean(scores)),
            **_noise_normalized_layer_summary(target_rows),
        }
        score, score_type = _selection_primary_score(row, primary_score)
        if score > best_score + 1e-6 or (
            abs(score - best_score) <= 1e-6 and float(candidate_alpha) < best_alpha
        ):
            best_score = score
            best_alpha = float(candidate_alpha)
            best_score_type = score_type
    return {
        "selected_alpha": best_alpha,
        "mode": mode,
        "score": best_score,
        "score_type": best_score_type,
    }


def _selection_primary_score(row: dict[str, Any], primary_score: str) -> tuple[float, str]:
    if primary_score == "mean_noise_normalized_score":
        normalized = _optional_float(row.get("mean_noise_normalized_score"))
        if normalized is not None:
            return normalized, "noise_normalized"
        raw = _optional_float(row.get("mean_score"))
        return (-np.inf if raw is None else raw), "raw"
    if primary_score == "mean_score":
        raw = _optional_float(row.get("mean_score"))
        return (-np.inf if raw is None else raw), "raw"
    raise ValueError(
        "neural.selection.primary_score must be 'mean_noise_normalized_score' or 'mean_score'"
    )


def _selected_candidate_index(rows: list[dict[str, Any]]) -> int:
    best_index = 0
    best_score = -np.inf
    for index, row in enumerate(rows):
        score = float(row["validation_score"])
        if score > best_score + 1e-6:
            best_score = score
            best_index = index
    return best_index


def _outer_train_test_indices(
    n_items: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = min(max(1, int(round(n_items * train_fraction))), n_items - 1)
    order = rng.permutation(n_items)
    return order[:n_train], order[n_train:]


def _inner_train_validation_indices(
    train_idx: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if train_idx.size < 3:
        raise ValueError("neural.selection requires at least three outer-train items")
    n_val = int(round(train_idx.size * validation_fraction))
    n_val = min(max(1, n_val), train_idx.size - 1)
    inner_order = train_idx[rng.permutation(train_idx.size)]
    return inner_order[n_val:], inner_order[:n_val]


def _ids_for_indices(image_ids: list[str], indices: np.ndarray) -> list[str]:
    return [str(image_ids[int(index)]) for index in indices]


def _unique_preserving(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def _compute_geometry_rows(
    features_by_layer: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    layers: list[str],
    methods: list[str],
    subset_sizes: list[int],
    subset_seeds: list[int],
    null_control_seeds: list[int],
    row_context: dict[str, Any],
    model_feature_reduction: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        features = features_by_layer[layer]
        for method in methods:
            if method == "linear_cka":
                result = linear_cka(features, responses)
                observed = _geometry_row(
                    result.as_row(),
                    row_context=row_context,
                    layer=layer,
                    model_feature_reduction=model_feature_reduction,
                )
                rows.append(_observed_geometry_row(observed))
                for control_seed in null_control_seeds:
                    null_result = linear_cka(
                        features,
                        _permuted_responses(responses, control_seed),
                    )
                    rows.append(
                        _null_geometry_row(
                            _geometry_row(
                                null_result.as_row(),
                                row_context=row_context,
                                layer=layer,
                                model_feature_reduction=model_feature_reduction,
                            ),
                            observed,
                            control_seed,
                        )
                    )
            elif method == "subset_rsa":
                for subset_size in subset_sizes:
                    for subset_seed in subset_seeds:
                        result = subset_rsa(
                            features,
                            responses,
                            subset_size=subset_size,
                            seed=subset_seed,
                        )
                        observed = _geometry_row(
                            result.as_row(),
                            row_context=row_context,
                            layer=layer,
                            model_feature_reduction=model_feature_reduction,
                        )
                        rows.append(_observed_geometry_row(observed))
                        null_result = subset_rsa(
                            features,
                            _permuted_responses(responses, subset_seed),
                            subset_size=subset_size,
                            seed=subset_seed,
                        )
                        rows.append(
                            _null_geometry_row(
                                _geometry_row(
                                    null_result.as_row(),
                                    row_context=row_context,
                                    layer=layer,
                                    model_feature_reduction=model_feature_reduction,
                                ),
                                observed,
                                subset_seed,
                            )
                        )
            else:
                raise ValueError("neural.geometry.methods must contain 'linear_cka' or 'subset_rsa'")
    return rows


def _permuted_responses(responses: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1_000_000)
    return np.asarray(responses)[rng.permutation(len(responses))]


def _observed_geometry_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "control_type": "observed",
        "control_seed": "",
        "paired_observed_score": "",
        "observed_minus_null": "",
    }


def _null_geometry_row(
    row: dict[str, Any],
    observed: dict[str, Any],
    control_seed: int,
) -> dict[str, Any]:
    observed_score = observed.get("score", "")
    null_score = row.get("score", "")
    delta: float | str = ""
    if observed_score != "" and null_score != "":
        delta = float(observed_score) - float(null_score)
    return {
        **row,
        "control_type": "response_permutation",
        "control_seed": int(control_seed),
        "paired_observed_score": observed_score,
        "observed_minus_null": delta,
    }


def _geometry_row(
    result_row: dict[str, Any],
    *,
    row_context: dict[str, Any],
    layer: str,
    model_feature_reduction: str,
) -> dict[str, Any]:
    return {
        **row_context,
        "layer": layer,
        **result_row,
        "model_feature_source": "activations.npz",
        "neural_response_source": "dataset_roi_responses",
        "centering": "image_centered_columns",
        "model_feature_reduction": model_feature_reduction,
        "response_metric": "raw_roi_responses",
    }


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


def _selected_ridge_alpha_metadata(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for row in rows:
        raw = row.get("selected_ridge_alpha", "")
        parsed = _optional_float(raw)
        values[str(row["layer"])] = "" if parsed is None else parsed
    return values


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
    return _reduce_feature_array(_feature_array(values), method=method)


def _feature_array(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.shape[0] == 1 and array.ndim > 1:
        array = array[0]
    return array


def _reduce_feature_array(array: np.ndarray, method: str = "flatten") -> np.ndarray:
    if method == "flatten":
        return array.reshape(-1)
    if method == "spatial_mean":
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            return array.mean(axis=0)
        return array.mean(axis=tuple(range(1, array.ndim)))
    raise ValueError("feature_reduction must be 'flatten', 'spatial_mean', or 'flatten_pca'")


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


def _write_selection_candidate_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "subject_id",
        "roi",
        "candidate_index",
        "layer",
        "feature_reduction",
        "pca_components",
        "pca_solver",
        "pca_whiten",
        "selected_ridge_alpha",
        "alpha_selection_mode",
        "validation_score",
        "validation_score_type",
        "primary_score",
        "selection_n_train",
        "selection_n_validation",
        "outer_n_train",
        "outer_n_test",
        "selected",
        "pca_n_train_fit",
        "pca_effective_components",
        "pca_explained_variance_ratio_sum",
        "selection_best_epoch",
        "selection_epochs_ran",
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


def _write_geometry_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "subject_id",
        "roi",
        "layer",
        "geometry_method",
        "score",
        "valid",
        "status",
        "num_images_total",
        "num_images_used",
        "subset_seed",
        "subset_size",
        "model_feature_source",
        "neural_response_source",
        "centering",
        "model_feature_reduction",
        "response_metric",
        "feature_rdm_metric",
        "response_rdm_metric",
        "rdm_compare_method",
        "subset_index_policy",
        "control_type",
        "control_seed",
        "paired_observed_score",
        "observed_minus_null",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
