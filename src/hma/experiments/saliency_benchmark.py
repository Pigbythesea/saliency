"""Static saliency benchmark runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from hma.datasets import build_dataset
from hma.metrics.saliency import mean_absolute_error, pearson_correlation
from hma.metrics.saliency_metrics import cc, kl_divergence, nss, similarity
from hma.models import build_model
from hma.saliency import build_saliency_method, postprocess_saliency_map
from hma.utils.config import load_experiment_config
from hma.utils.paths import ensure_dir, resolve_path


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def run_saliency_benchmark(config_path: str | Path) -> dict[str, Any]:
    """Run a static saliency benchmark and save CSV/JSON outputs."""
    config = load_experiment_config(config_path)
    output_dir = ensure_dir(resolve_path(config["output"]["dir"]))
    visualization_dir = output_dir / "visualizations"

    dataset = build_dataset(config)
    model = build_model(config)
    saliency_method = build_saliency_method(config)
    metric_names = list(config.get("metrics", []))
    metric_fns = _build_metric_functions(metric_names)

    rows: list[dict[str, Any]] = []
    save_visualizations = bool(config.get("output", {}).get("save_visualizations", False))
    num_visualizations = int(config.get("output", {}).get("num_visualizations", 0))
    if save_visualizations and num_visualizations > 0:
        ensure_dir(visualization_dir)

    for item_index, item in enumerate(dataset):
        image = item["image"]
        target = _as_2d_array(item.get("fixation_map"))
        prediction = saliency_method(
            model,
            image,
            item=item,
            item_index=item_index,
            target_map=target,
        )
        prediction_map = _prepare_prediction_map(prediction, target.shape)

        row: dict[str, Any] = {
            "image_id": item.get("image_id", f"item_{item_index:04d}"),
            "image_path": item.get("image_path", ""),
        }
        for metric_name, metric_fn in metric_fns.items():
            row[metric_name] = metric_fn(prediction_map, target)
        rows.append(row)

        if save_visualizations and item_index < num_visualizations:
            _save_visualization(
                visualization_dir / f"{row['image_id']}.png",
                image=image,
                target_map=target,
                prediction_map=prediction_map,
            )

    per_image_csv = output_dir / "per_image_metrics.csv"
    aggregate_json = output_dir / "aggregate_metrics.json"
    _write_per_image_csv(per_image_csv, rows, metric_names)
    aggregate = _build_aggregate(
        rows=rows,
        metric_names=metric_names,
        config=config,
        config_path=config_path,
        per_image_csv=per_image_csv,
        aggregate_json=aggregate_json,
    )
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def _build_metric_functions(metric_names: list[str]) -> dict[str, MetricFn]:
    registry: dict[str, MetricFn] = {
        "nss": lambda prediction, target: nss(prediction, target),
        "cc": lambda prediction, target: cc(prediction, target),
        "similarity": lambda prediction, target: similarity(prediction, target),
        "kl": lambda prediction, target: kl_divergence(target, prediction),
        "kl_divergence": lambda prediction, target: kl_divergence(target, prediction),
        "mae": lambda prediction, target: mean_absolute_error(prediction, target),
        "pearson": lambda prediction, target: pearson_correlation(prediction, target),
    }
    missing = [name for name in metric_names if name not in registry]
    if missing:
        raise KeyError(f"Unsupported saliency metrics: {missing}")
    return {name: registry[name] for name in metric_names}


def _prepare_prediction_map(prediction: Any, target_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(_to_numpy(prediction), dtype=np.float32)
    if array.ndim == 4:
        array = array[0, 0]
    elif array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = array.mean(axis=0)
    elif array.ndim != 2:
        raise ValueError(f"Expected saliency map with 2-4 dimensions, got {array.shape}")
    return postprocess_saliency_map(array, target_shape=target_shape)


def _as_2d_array(values: Any) -> np.ndarray:
    if values is None:
        raise ValueError("Dataset item must contain fixation_map for static benchmark")
    array = np.asarray(_to_numpy(values), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D fixation_map, got {array.shape}")
    return array


def _to_numpy(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _write_per_image_csv(
    path: Path,
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> None:
    fieldnames = ["image_id", "image_path", *metric_names]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_aggregate(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    config: dict[str, Any],
    config_path: str | Path,
    per_image_csv: Path,
    aggregate_json: Path,
) -> dict[str, Any]:
    metric_means = {
        metric: float(np.mean([float(row[metric]) for row in rows])) if rows else 0.0
        for metric in metric_names
    }
    return {
        "experiment": config.get("experiment", {}).get(
            "name", Path(config_path).stem
        ),
        "num_items": len(rows),
        "metrics": metric_means,
        "config_path": str(config_path),
        "dataset": config.get("dataset", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "saliency_method": config.get("saliency", {}).get("method"),
        "per_image_csv": str(per_image_csv),
        "aggregate_json": str(aggregate_json),
    }


def _save_visualization(
    path: Path,
    image: Any,
    target_map: np.ndarray,
    prediction_map: np.ndarray,
) -> None:
    panels = [
        _image_to_pil(image),
        _map_to_pil(target_map).convert("RGB"),
        _map_to_pil(prediction_map).convert("RGB"),
    ]
    width, height = panels[0].size
    panels = [panel.resize((width, height), Image.BILINEAR) for panel in panels]
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    canvas.save(path)


def _image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(_to_numpy(image), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    array = postprocess_saliency_map(array.mean(axis=2) if array.ndim == 3 else array)
    return _map_to_pil(array).convert("RGB")


def _map_to_pil(values: np.ndarray) -> Image.Image:
    array = postprocess_saliency_map(values)
    uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return ImageOps.autocontrast(Image.fromarray(uint8, mode="L"))
