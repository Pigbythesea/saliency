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
) -> list[Path]:
    """Compute geometry scores for every output directory in the Paper 1 scope."""
    scope = load_yaml(config_path).get("paper1_scope", {})
    if not isinstance(scope, dict):
        raise ValueError("configs/paper1_config.yaml must contain a paper1_scope mapping")

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
    model_filter = set(models or [])
    roi_filter = set(rois or [])

    written: list[Path] = []
    response_cache: dict[tuple[str, str, str, str, str, int], tuple[list[str], np.ndarray]] = {}
    for raw_dir in scope.get("neural_output_dirs", []):
        output_dir = resolve_path(raw_dir)
        metadata = _load_json(output_dir / "metadata.json")
        model = str(metadata.get("model_name") or metadata.get("model", ""))
        roi = _single(metadata.get("rois"))
        if model_filter and model not in model_filter:
            continue
        if roi_filter and roi not in roi_filter:
            continue
        path = output_dir / "geometry_scores.csv"
        if skip_existing and path.is_file():
            written.append(path)
            continue
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
        )
        _write_rows(path, rows)
        written.append(path)
    return written


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
) -> list[dict[str, Any]]:
    metadata = metadata or _load_json(output_dir / "metadata.json")
    activation_path = Path(metadata.get("activations") or output_dir / "activations.npz")
    if not activation_path.is_absolute():
        activation_path = output_dir / activation_path
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

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method in {"linear_cka", "linear_cka_full9841"}:
            row = _profiled_row(
                lambda: linear_cka(features, responses),
                estimated_rdm_bytes="0",
            )
            row["geometry_method"] = "linear_cka_full9841"
            rows.append({**common, **row})
        elif method == "subset_rsa":
            for subset_size in subset_sizes:
                for subset_seed in subset_seeds:
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
                    row["geometry_method"] = _subset_rsa_method_label(
                        subset_size,
                        subset_seed,
                        feature_rdm_metric,
                        response_rdm_metric,
                        rdm_compare_method,
                    )
                    rows.append({**common, **row})
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
    args = parser.parse_args()
    written = compute_matched_geometry(
        args.config,
        methods_override=args.methods,
        models=args.models,
        rois=args.rois,
        subset_sizes_override=args.subset_sizes,
        subset_seeds_override=args.subset_seeds,
        skip_existing=args.skip_existing,
    )
    print(f"Wrote {len(written)} geometry score files")


if __name__ == "__main__":
    main()
