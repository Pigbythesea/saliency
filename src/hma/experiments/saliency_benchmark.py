"""Static saliency benchmark runner."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from hma.datasets import build_dataset
from hma.metrics.saliency import mean_absolute_error, pearson_correlation
from hma.metrics.saliency_metrics import (
    auc_borji,
    auc_judd,
    cc,
    emd_2d,
    kl_divergence,
    nss,
    shuffled_auc,
    similarity,
)
from hma.models import build_model
from hma.preprocessing import preprocess_image_for_model
from hma.saliency import build_saliency_method, postprocess_saliency_map
from hma.utils.config import load_experiment_config
from hma.utils.device import resolve_device
from hma.utils.paths import ensure_dir, resolve_path


MetricContext = dict[str, Any]
MetricFn = Callable[[np.ndarray, np.ndarray, MetricContext], float]


def run_saliency_benchmark(config_path: str | Path) -> dict[str, Any]:
    """Run a static saliency benchmark and save CSV/JSON outputs."""
    config = load_experiment_config(config_path)
    output_dir = ensure_dir(resolve_path(config["output"]["dir"]))
    visualization_dir = output_dir / "visualizations"

    saliency_config = config.get("saliency", {})
    saliency_method_name = str(saliency_config.get("method") or saliency_config.get("name"))
    metric_names = list(config.get("metrics", []))
    shuffled_pool = _build_shuffled_fixation_pool(config) if "shuffled_auc" in metric_names else None
    metric_context_settings = _build_metric_context_settings(config)
    dataset = build_dataset(config)
    model = None if _saliency_is_model_independent(saliency_method_name) else build_model(config)
    saliency_method = build_saliency_method(config)
    target_class = saliency_config.get("target_class")
    resolved_device = resolve_device(config.get("device", "auto"))
    if _saliency_requires_torch(saliency_method_name):
        _move_model_to_device(model, resolved_device)

    metric_fns = _build_metric_functions(metric_names, config)
    cache_settings = _build_cache_settings(output_dir, config)

    rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_writes = 0
    save_visualizations = bool(config.get("output", {}).get("save_visualizations", False))
    num_visualizations = int(config.get("output", {}).get("num_visualizations", 0))
    if save_visualizations and num_visualizations > 0:
        ensure_dir(visualization_dir)

    for item_index, item in enumerate(dataset):
        image = item["image"]
        target = _as_2d_array(item.get("fixation_map"))
        cache_key = _build_saliency_cache_key(config, item, target.shape)
        prediction = _load_cached_saliency(cache_settings, cache_key)
        if prediction is not None:
            cache_hits += 1
        else:
            saliency_input = _prepare_saliency_input(
                image=image,
                config=config,
                device=resolved_device,
                method_name=saliency_method_name,
            )
            prediction = saliency_method(
                model,
                saliency_input,
                item=item,
                item_index=item_index,
                target_map=target,
                target_class=target_class,
            )
            if _save_cached_saliency(cache_settings, cache_key, prediction):
                cache_writes += 1
        prediction_map = _prepare_prediction_map(prediction, target.shape)

        row: dict[str, Any] = {
            "image_id": item.get("image_id", f"item_{item_index:04d}"),
            "image_path": item.get("image_path", ""),
        }
        metric_context = _build_metric_context(
            item=item,
            item_index=item_index,
            target_shape=target.shape,
            shuffled_pool=shuffled_pool,
            settings=metric_context_settings,
        )
        for metric_name, metric_fn in metric_fns.items():
            row[metric_name] = metric_fn(prediction_map, target, metric_context)
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
        cache_hits=cache_hits,
        cache_writes=cache_writes,
    )
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def _build_metric_functions(metric_names: list[str], config: dict[str, Any]) -> dict[str, MetricFn]:
    controls = config.get("metric_controls", {})
    seed = int(controls.get("seed", config.get("seed", 0)))
    auc_borji_splits = int(controls.get("auc_borji_splits", 100))
    shuffled_auc_splits = int(controls.get("shuffled_auc_splits", 100))
    emd_downsample = controls.get("emd_downsample", 32)
    registry: dict[str, MetricFn] = {
        "nss": lambda prediction, target, _context: nss(prediction, target),
        "auc_judd": lambda prediction, target, _context: auc_judd(prediction, target),
        "auc_borji": lambda prediction, target, context: auc_borji(
            prediction,
            target,
            positive_fixations=context.get("positive_fixations"),
            splits=auc_borji_splits,
            seed=seed + int(context.get("item_index", 0)),
        ),
        "shuffled_auc": lambda prediction, target, context: shuffled_auc(
            prediction,
            target,
            context.get("negative_fixations", np.zeros((0, 2), dtype=np.int64)),
            positive_fixations=context.get("positive_fixations"),
            splits=shuffled_auc_splits,
            seed=seed + int(context.get("item_index", 0)),
        ),
        "cc": lambda prediction, target, _context: cc(prediction, target),
        "similarity": lambda prediction, target, _context: similarity(prediction, target),
        "kl": lambda prediction, target, _context: kl_divergence(target, prediction),
        "kl_divergence": lambda prediction, target, _context: kl_divergence(target, prediction),
        "emd": lambda prediction, target, _context: emd_2d(
            target,
            prediction,
            downsample=None if emd_downsample is None else int(emd_downsample),
        ),
        "emd_2d": lambda prediction, target, _context: emd_2d(
            target,
            prediction,
            downsample=None if emd_downsample is None else int(emd_downsample),
        ),
        "mae": lambda prediction, target, _context: mean_absolute_error(prediction, target),
        "pearson": lambda prediction, target, _context: pearson_correlation(prediction, target),
    }
    missing = [name for name in metric_names if name not in registry]
    if missing:
        raise KeyError(f"Unsupported saliency metrics: {missing}")
    return {name: registry[name] for name in metric_names}


def _build_metric_context(
    item: dict[str, Any],
    item_index: int,
    target_shape: tuple[int, int],
    shuffled_pool: dict[str, Any] | None,
    settings: dict[str, Any],
) -> MetricContext:
    image_id = str(item.get("image_id", f"item_{item_index:04d}"))
    positive_fixations = _sample_coords(
        _fixation_coords_for_item(item, target_shape),
        max_points=int(settings["max_positive_fixations_per_image"]),
        seed=int(settings["seed"]) + item_index,
    )
    context: MetricContext = {
        "item": item,
        "item_index": item_index,
        "image_id": image_id,
        "positive_fixations": positive_fixations,
    }
    if shuffled_pool is not None:
        context["negative_fixations"] = _negative_fixations_for_item(shuffled_pool, image_id)
    return context


def _build_metric_context_settings(config: dict[str, Any]) -> dict[str, Any]:
    controls = config.get("metric_controls", {})
    return {
        "seed": int(controls.get("seed", config.get("seed", 0))),
        "max_positive_fixations_per_image": int(
            controls.get("max_positive_fixations_per_image", 256)
        ),
    }


def _build_shuffled_fixation_pool(config: dict[str, Any]) -> dict[str, Any]:
    controls = config.get("metric_controls", {})
    seed = int(controls.get("seed", config.get("seed", 0)))
    max_points_per_image = int(controls.get("shuffled_auc_pool_points_per_image", 256))
    rng = np.random.default_rng(seed)
    coords_by_image: list[np.ndarray] = []
    image_ids: list[str] = []

    for item_index, item in enumerate(build_dataset(config)):
        target = _as_2d_array(item.get("fixation_map"))
        coords = _fixation_coords_for_item(item, target.shape)
        if coords.size == 0:
            continue
        if max_points_per_image > 0:
            coords = _sample_coords(coords, max_points=max_points_per_image, rng=rng)
        coords_by_image.append(coords.astype(np.int64, copy=False))
        image_ids.extend([str(item.get("image_id", f"item_{item_index:04d}"))] * coords.shape[0])

    if not coords_by_image:
        return {
            "coords": np.zeros((0, 2), dtype=np.int64),
            "image_ids": np.asarray([], dtype=object),
        }
    return {
        "coords": np.concatenate(coords_by_image, axis=0),
        "image_ids": np.asarray(image_ids, dtype=object),
    }


def _negative_fixations_for_item(pool: dict[str, Any], image_id: str) -> np.ndarray:
    coords = pool.get("coords")
    image_ids = pool.get("image_ids")
    if coords is None or image_ids is None or len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(coords)[np.asarray(image_ids, dtype=object) != image_id]


def _fixation_coords_for_item(item: dict[str, Any], target_shape: tuple[int, int]) -> np.ndarray:
    points = item.get("fixation_points")
    if points is not None:
        point_array = np.asarray(_to_numpy(points), dtype=np.float32)
        if point_array.size:
            return _xy_points_to_yx_coords(point_array, target_shape)
    fixation_map = _as_2d_array(item.get("fixation_map"))
    return np.argwhere(fixation_map > 0).astype(np.int64)


def _sample_coords(
    coords: np.ndarray,
    *,
    max_points: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if max_points <= 0 or coords.shape[0] <= max_points:
        return coords
    generator = rng if rng is not None else np.random.default_rng(seed)
    selected = generator.choice(coords.shape[0], size=max_points, replace=False)
    return coords[selected]


def _xy_points_to_yx_coords(points: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if point_array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    coords = np.rint(point_array[:, [1, 0]]).astype(np.int64)
    height, width = target_shape
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 1] >= 0)
        & (coords[:, 0] < height)
        & (coords[:, 1] < width)
    )
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(coords[valid], axis=0)


def _prepare_saliency_input(
    image: Any,
    config: dict[str, Any],
    device: str,
    method_name: str,
) -> Any:
    if _saliency_requires_torch(method_name):
        return preprocess_image_for_model(image, config=config, device=device)
    return image


def _saliency_requires_torch(method_name: str | None) -> bool:
    return str(method_name) in {
        "vanilla_gradient",
        "integrated_gradients",
        "gradcam",
        "attention_rollout",
        "rollout",
    }


def _saliency_is_model_independent(method_name: str | None) -> bool:
    return str(method_name) in {"center_bias", "random_saliency"}


def _move_model_to_device(model: Any, device: str) -> None:
    module = getattr(model, "model", model)
    to = getattr(module, "to", None)
    if callable(to):
        to(device)
    eval_fn = getattr(module, "eval", None)
    if callable(eval_fn):
        eval_fn()


def _build_cache_settings(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_config = config.get("cache", {})
    enabled = bool(cache_config.get("enabled", False))
    cache_dir_config = Path(str(cache_config.get("dir", "saliency_maps")))
    cache_dir = (
        cache_dir_config
        if cache_dir_config.is_absolute()
        else output_dir / cache_dir_config
    )
    if enabled:
        ensure_dir(cache_dir)
    return {
        "enabled": enabled,
        "reuse": bool(cache_config.get("reuse", True)),
        "dir": cache_dir,
    }


def _build_saliency_cache_key(
    config: dict[str, Any],
    item: dict[str, Any],
    target_shape: tuple[int, int],
) -> dict[str, Any]:
    dataset_config = config.get("dataset", {})
    return {
        "dataset": dataset_config.get("label") or dataset_config.get("name"),
        "split": dataset_config.get("split"),
        "image_id": item.get("image_id"),
        "image_path": item.get("image_path", ""),
        "target_shape": list(target_shape),
        "model": config.get("model", {}),
        "saliency": config.get("saliency", {}),
        "preprocessing": config.get("preprocessing", {}),
    }


def _load_cached_saliency(
    cache_settings: dict[str, Any],
    cache_key: dict[str, Any],
) -> np.ndarray | None:
    if not cache_settings["enabled"] or not cache_settings["reuse"]:
        return None
    map_path, metadata_path = _cache_paths(cache_settings["dir"], cache_key)
    if not map_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if metadata.get("cache_key") != cache_key:
        return None
    return np.load(map_path)


def _save_cached_saliency(
    cache_settings: dict[str, Any],
    cache_key: dict[str, Any],
    prediction: Any,
) -> bool:
    if not cache_settings["enabled"]:
        return False
    map_path, metadata_path = _cache_paths(cache_settings["dir"], cache_key)
    np.save(map_path, _to_numpy(prediction).astype(np.float32))
    metadata_path.write_text(
        json.dumps({"cache_key": cache_key}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return True


def _cache_paths(cache_dir: Path, cache_key: dict[str, Any]) -> tuple[Path, Path]:
    encoded = json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    image_id = _safe_cache_stem(str(cache_key.get("image_id") or "item"))
    stem = f"{image_id}_{digest}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _safe_cache_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem[:80] or "item"


def _saliency_family(method_name: Any) -> str:
    if method_name in {"vanilla_gradient", "integrated_gradients"}:
        return "evidence_sensitivity"
    if method_name == "gradcam":
        return "class_localization"
    if method_name in {"attention_rollout", "rollout"}:
        return "internal_routing"
    if method_name in {"center_bias", "random_saliency"}:
        return "baseline"
    return "unknown"


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
    cache_hits: int = 0,
    cache_writes: int = 0,
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
        "dataset": config.get("dataset", {}).get("label") or config.get("dataset", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "saliency_method": config.get("saliency", {}).get("method"),
        "saliency_family": _saliency_family(config.get("saliency", {}).get("method")),
        "cache_hits": int(cache_hits),
        "cache_writes": int(cache_writes),
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
