"""Compute matched full-image geometry scores from existing neural outputs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hma.neural import linear_cka, subset_rsa
from hma.utils.config import load_experiment_config, load_yaml
from hma.utils.paths import resolve_path


DEFAULT_CONFIG = Path("configs/paper1_config.yaml")
GEOMETRY_FIELDNAMES = [
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
    "feature_rdm_metric",
    "response_rdm_metric",
    "rdm_compare_method",
    "subset_index_policy",
    "wall_time_sec",
    "feature_shape",
    "response_shape",
    "estimated_rdm_bytes",
    "model_feature_source",
    "neural_response_source",
    "centering",
    "model_feature_reduction",
    "response_metric",
]


def compute_matched_geometry(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    methods_override: list[str] | None = None,
    models: list[str] | None = None,
    rois: list[str] | None = None,
    subset_sizes_override: list[int] | None = None,
    subset_seeds_override: list[int] | None = None,
    skip_existing: bool = False,
    progress: bool = False,
    progress_label: str | None = None,
) -> list[Path]:
    """Compute geometry scores for every output directory in the Paper 1 scope."""
    scope = _geometry_scope(config_path)
    geometry = dict(scope.get("geometry", {}))
    methods = methods_override or [
        str(method) for method in geometry.get("methods", ["linear_cka"])
    ]
    subset_sizes = subset_sizes_override or [
        int(size) for size in geometry.get("subset_sizes", [])
    ]
    subset_seeds = subset_seeds_override or [
        int(seed)
        for seed in geometry.get("subset_seeds", [geometry.get("subset_seed", 123)])
    ]
    feature_rdm_metric = str(geometry.get("feature_rdm_metric", "correlation"))
    response_rdm_metric = str(geometry.get("response_rdm_metric", "correlation"))
    rdm_compare_method = str(geometry.get("rdm_compare_method", "spearman"))
    expected_num_items = int(scope.get("expected_num_items") or 0)
    expected_feature_reduction = str(scope.get("feature_reduction", "flatten_pca"))
    expected_models = set(str(model) for model in scope.get("models", []))
    expected_rois = set(str(roi) for roi in scope.get("rois", []))
    model_filter = set(models or [])
    roi_filter = set(rois or [])

    output_dirs = [resolve_path(raw_dir) for raw_dir in scope.get("neural_output_dirs", [])]
    label = progress_label or str(config_path)
    if progress:
        _print_progress(f"{label}: starting matched geometry over {len(output_dirs)} cells")

    written: list[Path] = []
    response_cache: dict[tuple[str, str, str, str, str, int], tuple[list[str], np.ndarray]] = {}
    for index, output_dir in enumerate(output_dirs, start=1):
        _validate_output_dir(
            output_dir,
            expected_models=expected_models,
            expected_rois=expected_rois,
            expected_num_items=expected_num_items,
            expected_feature_reduction=expected_feature_reduction,
        )
        metadata = _load_json(output_dir / "metadata.json")
        model = str(metadata.get("model_name") or metadata.get("model", ""))
        roi = _single(metadata.get("rois"))
        if model_filter and model not in model_filter:
            continue
        if roi_filter and roi not in roi_filter:
            continue
        path = output_dir / "geometry_scores.csv"
        cell_label = f"{label}: cell {index}/{len(output_dirs)} {model} {roi}"
        if skip_existing and path.is_file():
            if progress:
                _print_progress(f"{cell_label}: skipped existing {path}")
            written.append(path)
            continue
        if progress:
            _print_progress(f"{cell_label}: computing")
        rows = _compute_output_geometry(
            output_dir,
            metadata=metadata,
            methods=methods,
            subset_sizes=subset_sizes,
            subset_seeds=subset_seeds,
            feature_rdm_metric=feature_rdm_metric,
            response_rdm_metric=response_rdm_metric,
            rdm_compare_method=rdm_compare_method,
            response_cache=response_cache,
            progress=progress,
            progress_label=cell_label,
        )
        _write_rows(path, rows)
        if progress:
            _print_progress(f"{cell_label}: wrote {len(rows)} rows to {path}")
        written.append(path)
    if progress:
        _print_progress(f"{label}: finished matched geometry; outputs={len(written)}")
    return written


def _geometry_scope(config_path: str | Path) -> dict[str, Any]:
    """Return a legacy paper1_scope or V1 discovery_matrix geometry scope."""
    config = load_yaml(config_path)
    legacy_scope = config.get("paper1_scope")
    if isinstance(legacy_scope, dict):
        return legacy_scope

    discovery = config.get("discovery_matrix")
    if not isinstance(discovery, dict):
        raise ValueError(
            f"{config_path} must contain either a paper1_scope or discovery_matrix mapping"
        )
    geometry = config.get("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError(f"{config_path} must contain a geometry mapping")
    encoding = config.get("encoding", {})
    if not isinstance(encoding, dict):
        raise ValueError(f"{config_path} must contain an encoding mapping")

    config_root = resolve_path(discovery["config_root"])
    output_root = resolve_path(discovery["output_root"])
    config_paths = sorted(config_root.glob("*.yaml"))
    output_dirs = [output_root / path.stem for path in config_paths]
    roi_groups = discovery.get("roi_groups", {})
    rois = [
        str(roi)
        for group in roi_groups.values()
        for roi in group.get("rois", [])
    ]
    expected_cells = int(discovery.get("expected_cells", {}).get("model_roi_cells", 0))
    if expected_cells and len(output_dirs) != expected_cells:
        raise ValueError(
            f"V1 geometry expected {expected_cells} configs, found {len(output_dirs)} in {config_root}"
        )

    return {
        "subject_id": discovery.get("subject_id", ""),
        "models": [str(model) for model in discovery.get("models", [])],
        "rois": rois,
        "feature_reduction": encoding.get("method", "flatten_pca"),
        "expected_num_items": int(discovery.get("max_items") or 0),
        "geometry": geometry,
        "neural_output_dirs": [str(path) for path in output_dirs],
    }


def _validate_output_dir(
    output_dir: Path,
    *,
    expected_models: set[str],
    expected_rois: set[str],
    expected_num_items: int,
    expected_feature_reduction: str,
) -> None:
    metadata_path = output_dir / "metadata.json"
    encoding_path = output_dir / "encoding_scores.csv"
    activations_path = output_dir / "activations.npz"
    for path in [metadata_path, encoding_path, activations_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required geometry input not found: {path}")

    metadata = _load_json(metadata_path)
    model = str(metadata.get("model_name") or metadata.get("model", ""))
    roi = _single(metadata.get("rois"))
    if expected_models and model not in expected_models:
        raise ValueError(f"Unexpected model for V1 geometry: {model} in {output_dir}")
    if expected_rois and roi not in expected_rois:
        raise ValueError(f"Unexpected ROI for V1 geometry: {roi} in {output_dir}")
    feature_reduction = str(metadata.get("feature_reduction", ""))
    if expected_feature_reduction and feature_reduction != expected_feature_reduction:
        raise ValueError(
            f"Expected feature_reduction={expected_feature_reduction}, got {feature_reduction} "
            f"in {output_dir}"
        )
    if expected_num_items and int(metadata.get("num_items") or 0) != expected_num_items:
        raise ValueError(
            f"Expected num_items={expected_num_items}, got {metadata.get('num_items')} "
            f"in {output_dir}"
        )


def _compute_output_geometry(
    output_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
    methods: list[str],
    subset_sizes: list[int],
    subset_seeds: list[int],
    feature_rdm_metric: str,
    response_rdm_metric: str,
    rdm_compare_method: str,
    response_cache: dict[tuple[str, str, str, str, str, int], tuple[list[str], np.ndarray]],
    progress: bool = False,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or _load_json(output_dir / "metadata.json")
    activation_path = _artifact_path(
        metadata.get("activations"),
        output_dir=output_dir,
        filename="activations.npz",
    )
    selected_layer = str(metadata.get("selected_layer") or _selected_layer(output_dir))
    if not selected_layer:
        raise ValueError(f"No selected layer found for {output_dir}")

    activations = np.load(activation_path, allow_pickle=True)
    if selected_layer not in activations.files:
        raise ValueError(f"Layer '{selected_layer}' not found in {activation_path}")
    image_ids = [str(value) for value in activations["image_ids"].tolist()]
    features = np.asarray(activations[selected_layer], dtype=np.float32)

    config = load_experiment_config(metadata["config_path"])
    dataset_image_ids, responses = _dataset_image_ids_and_responses(config, response_cache)
    if image_ids != dataset_image_ids:
        raise ValueError(f"Activation image_ids do not match dataset order for {output_dir}")
    if responses.shape[0] != features.shape[0]:
        raise ValueError(f"Feature/response row mismatch for {output_dir}")

    common = {
        "dataset": metadata.get("dataset", ""),
        "model": metadata.get("model_name") or metadata.get("model", ""),
        "subject_id": _single(metadata.get("subjects")),
        "roi": _single(metadata.get("rois")),
        "layer": selected_layer,
        "model_feature_source": str(activation_path),
        "neural_response_source": "dataset_roi_responses",
        "centering": "image_centered_columns",
        "model_feature_reduction": metadata.get("feature_reduction", ""),
        "response_metric": "raw_roi_responses",
        "feature_shape": _shape_text(features.shape),
        "response_shape": _shape_text(responses.shape),
    }

    operation_total = sum(
        1 if method in {"linear_cka", "linear_cka_full9841"} else len(subset_sizes) * len(subset_seeds)
        for method in methods
    )
    operation_index = 0
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method in {"linear_cka", "linear_cka_full9841"}:
            operation_index += 1
            if progress:
                _print_progress(
                    f"{progress_label}: method {operation_index}/{operation_total} linear_cka_full9841 start"
                )
            row = _profiled_row(
                lambda: linear_cka(features, responses),
                estimated_rdm_bytes="0",
            )
            row["geometry_method"] = "linear_cka_full9841"
            rows.append({**common, **row})
            if progress:
                _print_progress(
                    f"{progress_label}: method {operation_index}/{operation_total} linear_cka_full9841 done "
                    f"score={row.get('score')} wall={row.get('wall_time_sec')}s"
                )
        elif method == "subset_rsa":
            for subset_size in subset_sizes:
                for subset_seed in subset_seeds:
                    operation_index += 1
                    method_label = _subset_rsa_method_label(
                        subset_size,
                        subset_seed,
                        feature_rdm_metric,
                        response_rdm_metric,
                        rdm_compare_method,
                    )
                    if progress:
                        _print_progress(
                            f"{progress_label}: method {operation_index}/{operation_total} {method_label} start"
                        )
                    row = _profiled_row(
                        lambda subset_size=subset_size, subset_seed=subset_seed: subset_rsa(
                            features,
                            responses,
                            subset_size=subset_size,
                            seed=subset_seed,
                            feature_rdm_metric=feature_rdm_metric,
                            response_rdm_metric=response_rdm_metric,
                            compare_method=rdm_compare_method,
                        ),
                        estimated_rdm_bytes=str(_estimated_two_rdm_bytes(subset_size)),
                    )
                    row["geometry_method"] = method_label
                    rows.append({**common, **row})
                    if progress:
                        _print_progress(
                            f"{progress_label}: method {operation_index}/{operation_total} {method_label} done "
                            f"score={row.get('score')} wall={row.get('wall_time_sec')}s"
                        )
        else:
            raise ValueError(f"Unsupported geometry method: {method}")
    return rows


def _profiled_row(callable_result: Any, *, estimated_rdm_bytes: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = callable_result()
    elapsed = time.perf_counter() - start
    row = result.as_row()
    row["wall_time_sec"] = f"{elapsed:.6f}"
    row["estimated_rdm_bytes"] = estimated_rdm_bytes
    return row


def _subset_rsa_method_label(
    subset_size: int,
    subset_seed: int,
    feature_rdm_metric: str,
    response_rdm_metric: str,
    compare_method: str,
) -> str:
    metric_label = (
        "corr_rdm"
        if feature_rdm_metric == response_rdm_metric == "correlation"
        else f"{feature_rdm_metric}_{response_rdm_metric}_rdm"
    )
    return (
        f"subset_rsa_{metric_label}_{compare_method}_"
        f"size{subset_size}_seed{subset_seed}"
    )


def _estimated_two_rdm_bytes(subset_size: int) -> int:
    return int(2 * subset_size * subset_size * np.dtype(np.float32).itemsize)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in shape)


def _dataset_image_ids_and_responses(
    config: dict[str, Any],
    cache: dict[tuple[str, str, str, str, str, int], tuple[list[str], np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    dataset = dict(config.get("dataset", {}))
    root = resolve_path(dataset.get("root", "data/raw/nsd_algonauts"))
    manifest_path = resolve_path(dataset["manifest_path"])
    split = str(dataset.get("split", "train"))
    subject_id = str(dataset.get("subject_id", ""))
    roi = str(dataset.get("roi", ""))
    max_items = int(dataset.get("max_items") or 0)
    key = (str(root), str(manifest_path), split, subject_id, roi, max_items)
    if key in cache:
        return cache[key]

    import csv as _csv

    image_ids: list[str] = []
    responses: list[np.ndarray] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.DictReader(handle)
        for record in reader:
            if record.get("split") != split:
                continue
            if subject_id and record.get("subject_id") != subject_id:
                continue
            if roi and record.get("roi") != roi:
                continue
            response_path = str(record.get("roi_response_path", ""))
            if not response_path:
                raise ValueError("Manifest row is missing roi_response_path")
            image_ids.append(str(record["image_id"]))
            responses.append(np.asarray(np.load(root / response_path), dtype=np.float32))
            if max_items and len(image_ids) >= max_items:
                break
    if not responses:
        raise ValueError("No neural responses found")
    first_shape = responses[0].shape
    if any(response.shape != first_shape for response in responses):
        raise ValueError("All neural responses must have matching shapes")
    result = (image_ids, np.stack(responses, axis=0))
    cache[key] = result
    return result


def _selected_layer(output_dir: Path) -> str:
    path = output_dir / "encoding_scores.csv"
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return str(rows[0].get("layer", "")) if rows else ""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _single(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _artifact_path(raw_path: Any, *, output_dir: Path, filename: str) -> Path:
    """Resolve synced artifacts even when metadata contains stale absolute paths."""
    local_path = output_dir / filename
    if raw_path is None:
        return local_path
    path = Path(str(raw_path))
    if not path.is_absolute():
        candidate = output_dir / path
        return candidate if candidate.is_file() else local_path
    if path.is_file():
        return path
    return local_path


def _print_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GEOMETRY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional geometry methods override, e.g. --methods linear_cka",
    )
    parser.add_argument("--models", nargs="+", help="Optional model-name filter.")
    parser.add_argument("--rois", nargs="+", help="Optional ROI filter.")
    parser.add_argument(
        "--subset-sizes",
        nargs="+",
        type=int,
        help="Optional subset sizes override for subset RSA.",
    )
    parser.add_argument(
        "--subset-seeds",
        nargs="+",
        type=int,
        help="Optional subset seeds override for subset RSA.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    written = compute_matched_geometry(
        args.config,
        methods_override=args.methods,
        models=args.models,
        rois=args.rois,
        subset_sizes_override=args.subset_sizes,
        subset_seeds_override=args.subset_seeds,
        skip_existing=args.skip_existing,
        progress=not args.no_progress,
    )
    print(f"Wrote {len(written)} geometry score files")


if __name__ == "__main__":
    main()
