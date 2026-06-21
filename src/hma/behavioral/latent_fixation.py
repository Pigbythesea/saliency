"""Latent feature to fixation-density encoding utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hma.datasets import build_dataset
from hma.external.artifacts import load_external_features
from hma.metrics.saliency_metrics import (
    cc,
    information_gain,
    kl_divergence,
    nss,
    probabilistic_log_likelihood,
    similarity,
    simple_center_bias_map,
)
from hma.neural.encoding import fit_ridge_encoding, predict_ridge_encoding
from hma.saliency.postprocess import postprocess_saliency_map


@dataclass(frozen=True)
class FixationDatasetBundle:
    """Dataset rows aligned to a fixed fixation target shape."""

    dataset: str
    regime: str
    object_label: str
    artifact_key: str
    image_ids: list[str]
    map_keys: list[str]
    row_keys: list[str]
    artifact_ids: list[str]
    targets: np.ndarray
    positive_fixations: list[np.ndarray | None]
    metadata: list[dict[str, Any]]


def load_fixation_dataset_bundle(
    dataset_name: str,
    dataset_config: dict[str, Any],
    *,
    target_size: tuple[int, int],
) -> FixationDatasetBundle:
    """Load fixation targets for one behavioral dataset."""
    config = dict(dataset_config)
    config.setdefault("name", dataset_name)
    config["image_size"] = [int(target_size[0]), int(target_size[1])]
    dataset = build_dataset(config)
    artifact_key = str(config.get("artifact_key", "map_key"))
    if artifact_key not in {"image_id", "map_key", "row_key"}:
        raise ValueError("artifact_key must be image_id, map_key, or row_key")

    image_ids: list[str] = []
    map_keys: list[str] = []
    row_keys: list[str] = []
    artifact_ids: list[str] = []
    targets: list[np.ndarray] = []
    positive_fixations: list[np.ndarray | None] = []
    metadata: list[dict[str, Any]] = []

    for item in dataset:
        image_id = str(item.get("image_id", ""))
        map_key = str(item.get("map_key") or image_id)
        row_key = str(item.get("row_key") or map_key)
        key = {"image_id": image_id, "map_key": map_key, "row_key": row_key}[artifact_key]
        fixation_map = item.get("fixation_map")
        if fixation_map is None:
            continue
        target = postprocess_saliency_map(
            np.asarray(fixation_map, dtype=np.float32),
            target_shape=target_size,
            normalize=False,
        )
        if target.shape != target_size:
            raise ValueError(
                f"Fixation target shape mismatch for {dataset_name}: {target.shape}"
            )
        image_ids.append(image_id)
        map_keys.append(map_key)
        row_keys.append(row_key)
        artifact_ids.append(str(key))
        targets.append(target)
        positive_fixations.append(
            xy_points_to_yx_coords(item.get("fixation_points"), target_size)
        )
        item_metadata = dict(item.get("metadata", {}))
        item_metadata.setdefault("dataset", dataset_name)
        item_metadata["image_path"] = str(item.get("image_path", ""))
        metadata.append(item_metadata)

    if not targets:
        raise ValueError(f"No fixation targets loaded for dataset {dataset_name}")
    return FixationDatasetBundle(
        dataset=dataset_name,
        regime=str(config.get("regime", "")),
        object_label=str(config.get("behavioral_object", "human_fixation_density")),
        artifact_key=artifact_key,
        image_ids=image_ids,
        map_keys=map_keys,
        row_keys=row_keys,
        artifact_ids=artifact_ids,
        targets=np.stack(targets, axis=0).astype(np.float32, copy=False),
        positive_fixations=positive_fixations,
        metadata=metadata,
    )


def run_latent_fixation_encoding(
    *,
    bundle: FixationDatasetBundle,
    artifact_dir: str | Path,
    model_id: str,
    layers: Iterable[str],
    ridge_alphas: Iterable[float],
    pca_components: int,
    train_fraction: float,
    validation_fraction_of_train: float,
    seed: int,
    verify_hashes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Train/select latent readouts and return aggregate rows plus image rows."""
    requested_layers = [str(layer) for layer in layers]
    if not requested_layers:
        raise ValueError("At least one latent layer is required")
    artifact_ids, features_by_layer, manifest = load_external_features(
        artifact_dir,
        layers=requested_layers,
        verify_hashes=verify_hashes,
    )
    aligned_indices = align_artifact_indices(
        bundle.artifact_ids,
        artifact_ids,
        label=f"{bundle.dataset}:{model_id}",
    )
    targets = bundle.targets.reshape(bundle.targets.shape[0], -1)
    train_idx, validation_idx, test_idx = deterministic_splits(
        len(bundle.artifact_ids),
        train_fraction=train_fraction,
        validation_fraction_of_train=validation_fraction_of_train,
        seed=seed,
    )
    split_by_index = {
        int(index): "train" for index in train_idx
    } | {int(index): "validation" for index in validation_idx} | {
        int(index): "test" for index in test_idx
    }
    candidate_rows: list[dict[str, Any]] = []
    reducer_metadata: list[dict[str, Any]] = []
    layer_cache: dict[str, np.ndarray] = {}

    for layer in requested_layers:
        raw = np.asarray(features_by_layer[layer])
        aligned = np.asarray(raw[aligned_indices], dtype=np.float32)
        reduced, metadata = reduce_features_train_only(
            aligned,
            train_idx=train_idx,
            requested_components=int(pca_components),
            seed=seed,
        )
        layer_cache[layer] = reduced
        reducer_metadata.append(
            {
                "dataset": bundle.dataset,
                "model_id": model_id,
                "layer": layer,
                "artifact_dir": str(Path(artifact_dir)),
                "artifact_model_id": manifest.get("model_id", ""),
                "artifact_num_images": manifest.get("num_images", ""),
                "artifact_key": bundle.artifact_key,
                **metadata,
            }
        )
        for alpha in ridge_alphas:
            candidate_rows.append(
                score_candidate(
                    layer=layer,
                    alpha=float(alpha),
                    features=reduced,
                    targets=targets,
                    target_shape=bundle.targets.shape[1:],
                    train_idx=train_idx,
                    validation_idx=validation_idx,
                    baseline_map=simple_center_bias_map(*bundle.targets.shape[1:]),
                    positive_fixations=bundle.positive_fixations,
                )
            )

    selected = select_candidate(candidate_rows)
    selected_layer = str(selected["layer"])
    selected_alpha = float(selected["alpha"])
    final_train_idx = np.concatenate([train_idx, validation_idx])
    final_model = fit_ridge_encoding(
        layer_cache[selected_layer][final_train_idx],
        targets[final_train_idx],
        alpha=selected_alpha,
    )
    predictions = predict_ridge_encoding(final_model, layer_cache[selected_layer][test_idx])
    baseline_map = simple_center_bias_map(*bundle.targets.shape[1:])
    image_rows = score_prediction_images(
        predictions=predictions,
        test_idx=test_idx,
        bundle=bundle,
        model_id=model_id,
        layer=selected_layer,
        alpha=selected_alpha,
        baseline_map=baseline_map,
        split_by_index=split_by_index,
    )
    aggregate_rows = aggregate_image_scores(
        image_rows,
        dataset=bundle.dataset,
        regime=bundle.regime,
        object_label=bundle.object_label,
        model_id=model_id,
        layer=selected_layer,
        alpha=selected_alpha,
        n_train=int(train_idx.size),
        n_validation=int(validation_idx.size),
        n_test=int(test_idx.size),
    )
    selection_artifact = {
        "dataset": bundle.dataset,
        "model_id": model_id,
        "artifact_dir": str(Path(artifact_dir)),
        "selected_layer": selected_layer,
        "selected_alpha": selected_alpha,
        "selection_metric": "validation_mean_latent_fixation_information_gain",
        "selection_value": selected["validation_mean_latent_fixation_information_gain"],
        "candidate_rows": candidate_rows,
        "split": {
            "train_indices": train_idx.astype(int).tolist(),
            "validation_indices": validation_idx.astype(int).tolist(),
            "test_indices": test_idx.astype(int).tolist(),
        },
    }
    return aggregate_rows, image_rows, selection_artifact, reducer_metadata


def score_candidate(
    *,
    layer: str,
    alpha: float,
    features: np.ndarray,
    targets: np.ndarray,
    target_shape: tuple[int, int],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    baseline_map: np.ndarray,
    positive_fixations: list[np.ndarray | None],
) -> dict[str, Any]:
    """Fit one candidate readout and score validation information gain."""
    if validation_idx.size == 0:
        validation_idx = train_idx
    model = fit_ridge_encoding(features[train_idx], targets[train_idx], alpha=alpha)
    predictions = predict_ridge_encoding(model, features[validation_idx])
    values = []
    for pred, index in zip(predictions, validation_idx):
        prediction_map = prediction_to_map(pred, target_shape)
        target_map = targets[int(index)].reshape(target_shape)
        values.append(
            information_gain(
                prediction_map,
                baseline_map,
                target_map,
                positive_fixations=positive_fixations[int(index)],
            )
        )
    return {
        "layer": layer,
        "alpha": float(alpha),
        "validation_mean_latent_fixation_information_gain": float(np.mean(values)) if values else 0.0,
        "validation_n": int(validation_idx.size),
    }


def score_prediction_images(
    *,
    predictions: np.ndarray,
    test_idx: np.ndarray,
    bundle: FixationDatasetBundle,
    model_id: str,
    layer: str,
    alpha: float,
    baseline_map: np.ndarray,
    split_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    """Compute held-out image-level fixation metrics."""
    rows: list[dict[str, Any]] = []
    target_shape = bundle.targets.shape[1:]
    for pred, index in zip(predictions, test_idx):
        item_index = int(index)
        prediction_map = prediction_to_map(pred, target_shape)
        target_map = bundle.targets[item_index]
        positives = bundle.positive_fixations[item_index]
        metric_values = {
            "latent_fixation_information_gain": information_gain(
                prediction_map,
                baseline_map,
                target_map,
                positive_fixations=positives,
            ),
            "latent_fixation_log_likelihood_bits": probabilistic_log_likelihood(
                prediction_map,
                target_map,
                positive_fixations=positives,
            ),
            "latent_fixation_nss": nss(
                prediction_map,
                target_map,
                positive_fixations=positives,
            ),
            "latent_fixation_cc": cc(prediction_map, target_map),
            "latent_fixation_similarity": similarity(prediction_map, target_map),
            "latent_fixation_kl": kl_divergence(target_map, prediction_map),
        }
        for metric, value in metric_values.items():
            rows.append(
                {
                    "dataset": bundle.dataset,
                    "behavioral_regime": bundle.regime,
                    "behavioral_object": bundle.object_label,
                    "model_id": model_id,
                    "layer": layer,
                    "selected_alpha": alpha,
                    "metric": metric,
                    "value": float(value),
                    "image_id": bundle.image_ids[item_index],
                    "map_key": bundle.map_keys[item_index],
                    "row_key": bundle.row_keys[item_index],
                    "artifact_id": bundle.artifact_ids[item_index],
                    "split_role": split_by_index.get(item_index, "test"),
                    "evidence_status": "clean_publication_bounded_rerun",
                    "score_source": "primary_behavioral_latent_to_fixation_encoding",
                }
            )
    return rows


def aggregate_image_scores(
    image_rows: list[dict[str, Any]],
    *,
    dataset: str,
    regime: str,
    object_label: str,
    model_id: str,
    layer: str,
    alpha: float,
    n_train: int,
    n_validation: int,
    n_test: int,
) -> list[dict[str, Any]]:
    """Aggregate held-out image rows by metric."""
    metrics = sorted({str(row["metric"]) for row in image_rows})
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = np.asarray(
            [float(row["value"]) for row in image_rows if row["metric"] == metric],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "behavioral_regime": regime,
                "behavioral_object": object_label,
                "model_id": model_id,
                "layer": layer,
                "selected_alpha": alpha,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "n": int(values.size),
                "n_train": n_train,
                "n_validation": n_validation,
                "n_test": n_test,
                "primary_behavioral_score": "yes"
                if metric == "latent_fixation_information_gain"
                else "no",
                "evidence_status": "clean_publication_bounded_rerun",
                "score_source": "primary_behavioral_latent_to_fixation_encoding",
            }
        )
    return rows


def align_artifact_indices(
    required_ids: list[str],
    artifact_ids: list[str],
    *,
    label: str,
) -> np.ndarray:
    """Return artifact indices in dataset order."""
    index_by_id = {str(value): idx for idx, value in enumerate(artifact_ids)}
    missing = sorted({value for value in required_ids if value not in index_by_id})
    if missing:
        preview = "|".join(missing[:10])
        raise ValueError(
            f"Feature artifact for {label} is missing {len(missing)} dataset ids: {preview}"
        )
    return np.asarray([index_by_id[value] for value in required_ids], dtype=np.int64)


def deterministic_splits(
    n_items: int,
    *,
    train_fraction: float,
    validation_fraction_of_train: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/validation/test indices for small and full runs."""
    if n_items < 2:
        raise ValueError("Latent fixation encoding requires at least two images")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items).astype(np.int64)
    test_count = max(1, int(round(n_items * (1.0 - float(train_fraction)))))
    if test_count >= n_items:
        test_count = 1
    train_validation = indices[:-test_count]
    test_idx = indices[-test_count:]
    if train_validation.size == 0:
        train_validation = test_idx[:1]
        test_idx = indices[1:]
    validation_count = int(round(train_validation.size * float(validation_fraction_of_train)))
    if train_validation.size >= 3:
        validation_count = max(1, validation_count)
    else:
        validation_count = 0
    if validation_count >= train_validation.size:
        validation_count = max(0, train_validation.size - 1)
    validation_idx = train_validation[-validation_count:] if validation_count else np.zeros((0,), dtype=np.int64)
    train_idx = train_validation[:-validation_count] if validation_count else train_validation
    if train_idx.size == 0:
        raise ValueError("Latent fixation encoding split produced an empty train set")
    if test_idx.size == 0:
        raise ValueError("Latent fixation encoding split produced an empty test set")
    return train_idx.astype(np.int64), validation_idx.astype(np.int64), test_idx.astype(np.int64)


def reduce_features_train_only(
    features: np.ndarray,
    *,
    train_idx: np.ndarray,
    requested_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flatten features and apply train-fit PCA when dimensionality requires it."""
    matrix = np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)
    finite = np.isfinite(matrix)
    if not bool(np.all(finite)):
        matrix = np.where(finite, matrix, 0.0).astype(np.float32, copy=False)
    max_components = min(int(requested_components), int(train_idx.size), int(matrix.shape[1]))
    if max_components <= 0:
        raise ValueError("PCA component count must be positive")
    if matrix.shape[1] <= max_components:
        return matrix, {
            "feature_reduction": "flatten",
            "input_feature_shape": [int(value) for value in features.shape[1:]],
            "flattened_feature_dim": int(matrix.shape[1]),
            "requested_pca_components": int(requested_components),
            "realized_pca_components": int(matrix.shape[1]),
            "pca_fit_scope": "not_applied_feature_dim_within_limit",
        }
    reduced, metadata = _pca_reduce(matrix, train_idx=train_idx, components=max_components, seed=seed)
    metadata.update(
        {
            "feature_reduction": "flatten_pca",
            "input_feature_shape": [int(value) for value in features.shape[1:]],
            "flattened_feature_dim": int(matrix.shape[1]),
            "requested_pca_components": int(requested_components),
            "realized_pca_components": int(reduced.shape[1]),
            "pca_fit_scope": "training_only",
        }
    )
    return reduced.astype(np.float32, copy=False), metadata


def _pca_reduce(
    matrix: np.ndarray,
    *,
    train_idx: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA  # type: ignore

        solver = "randomized" if components < min(matrix[train_idx].shape) else "full"
        pca = PCA(n_components=components, svd_solver=solver, random_state=seed)
        pca.fit(matrix[train_idx])
        reduced = pca.transform(matrix)
        return reduced, {
            "pca_backend": "sklearn",
            "pca_solver": solver,
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
    except Exception:
        train = matrix[train_idx].astype(np.float64, copy=False)
        mean = train.mean(axis=0, keepdims=True)
        centered_train = train - mean
        _u, singular_values, vt = np.linalg.svd(centered_train, full_matrices=False)
        basis = vt[:components].T
        reduced = (matrix.astype(np.float64, copy=False) - mean) @ basis
        variance = singular_values**2
        total = float(np.sum(variance))
        explained = float(np.sum(variance[:components]) / total) if total > 0.0 else 0.0
        return reduced.astype(np.float32, copy=False), {
            "pca_backend": "numpy_svd",
            "pca_solver": "full_svd_fallback",
            "pca_explained_variance_ratio_sum": explained,
        }


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the layer/alpha with the highest validation information gain."""
    if not rows:
        raise ValueError("No readout candidates were scored")
    return max(
        rows,
        key=lambda row: (
            float(row["validation_mean_latent_fixation_information_gain"]),
            str(row["layer"]),
            -float(row["alpha"]),
        ),
    )


def prediction_to_map(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert a vector prediction into a non-negative fixation map."""
    values = np.asarray(prediction, dtype=np.float32).reshape(shape)
    values = np.where(np.isfinite(values), values, 0.0)
    values = values - float(np.min(values))
    if float(np.max(values)) <= 1e-8:
        return np.ones(shape, dtype=np.float32)
    return values.astype(np.float32, copy=False)


def xy_points_to_yx_coords(
    points: Any,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Convert dataset fixation points from x/y to y/x coordinates."""
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    coords = np.rint(array.reshape(-1, 2)[:, [1, 0]]).astype(np.int64)
    height, width = target_shape
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 1] >= 0)
        & (coords[:, 0] < height)
        & (coords[:, 1] < width)
    )
    if not bool(np.any(valid)):
        return np.zeros((0, 2), dtype=np.int64)
    return coords[valid]


def json_dumps_stable(payload: dict[str, Any]) -> str:
    """Serialize runner metadata with numpy values converted."""
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
