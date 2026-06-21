"""Model-independent saliency baselines."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hma.datasets.fixation_parsers import load_observer_fixations
from hma.metrics.saliency_metrics import simple_center_bias_map
from hma.datasets.fixation_utils import points_to_fixation_map
from hma.saliency.postprocess import postprocess_saliency_map


def center_bias_saliency(
    _model_wrapper: Any,
    images: Any,
    target_map: Any | None = None,
    sigma: float | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Return a Gaussian center-bias map at the target saliency shape."""
    height, width = _infer_map_shape(images, target_map)
    return simple_center_bias_map(height, width, sigma=sigma)


def random_saliency(
    _model_wrapper: Any,
    images: Any,
    target_map: Any | None = None,
    seed: int = 0,
    item_index: int = 0,
    item: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Return a deterministic random saliency map for an item."""
    height, width = _infer_map_shape(images, target_map)
    image_id = "" if item is None else str(item.get("image_id", ""))
    rng = np.random.default_rng(_stable_seed(seed, item_index, image_id))
    return postprocess_saliency_map(rng.random((height, width), dtype=np.float32))


def coco_search18_task_prior_saliency(
    _model_wrapper: Any,
    images: Any,
    target_map: Any | None = None,
    item: dict[str, Any] | None = None,
    prior: "COCOSearch18TaskPrior | None" = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Return a COCO-Search18 target/task-conditioned train-split fixation prior."""
    if prior is None:
        raise ValueError("coco_search18_task_prior requires a prebuilt prior")
    height, width = _infer_map_shape(images, target_map)
    metadata = {} if item is None else dict(item.get("metadata", {}) or {})
    prediction = prior.map_for(
        target_category=str(metadata.get("target_category", "")),
        task=str(metadata.get("task", "")),
    )
    if prediction.shape != (height, width):
        prediction = postprocess_saliency_map(prediction, target_shape=(height, width))
    return prediction


def empirical_spatial_prior_saliency(
    _model_wrapper: Any,
    images: Any,
    target_map: Any | None = None,
    prior: "EmpiricalSpatialPrior | None" = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Return a train-split empirical fixation prior."""
    if prior is None:
        raise ValueError("empirical_spatial_prior requires a prebuilt prior")
    height, width = _infer_map_shape(images, target_map)
    prediction = prior.map
    if prediction.shape != (height, width):
        prediction = postprocess_saliency_map(prediction, target_shape=(height, width))
    return prediction


@dataclass(frozen=True)
class EmpiricalSpatialPrior:
    """Dataset-level spatial prior built only from allowed training rows."""

    map: np.ndarray
    image_size: tuple[int, int]
    source_split: str
    excluded_image_ids: tuple[str, ...] = ()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        split: str = "train",
        image_size: tuple[int, int] = (224, 224),
        fixation_sigma: float = 10.0,
        exclude_image_ids: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
    ) -> "EmpiricalSpatialPrior":
        excluded = {str(value) for value in (exclude_image_ids or [])}
        point_sets: list[np.ndarray] = []
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = {"split", "width", "height"} - fieldnames
            has_inline_points = "fixation_points" in fieldnames
            has_points_path = "fixation_points_path" in fieldnames
            if not has_inline_points and not has_points_path:
                missing.add("fixation_points or fixation_points_path")
            if missing:
                raise ValueError(
                    f"Empirical spatial prior manifest missing columns: {sorted(missing)}"
                )
            has_image_id = "image_id" in fieldnames
            dataset_name = _infer_manifest_dataset(path)
            dataset_root = _infer_manifest_dataset_root(path, dataset_name)
            for row in reader:
                if row.get("split") != split:
                    continue
                if has_image_id and str(row.get("image_id", "")) in excluded:
                    continue
                points = _load_manifest_points(
                    row,
                    manifest_path=path,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                )
                if points.size == 0:
                    continue
                point_sets.append(
                    _scale_points(
                        points,
                        width=_optional_float(row.get("width")),
                        height=_optional_float(row.get("height")),
                        image_size=image_size,
                    )
                )

        if point_sets:
            prior_map = points_to_fixation_map(
                np.concatenate(point_sets, axis=0),
                height=image_size[0],
                width=image_size[1],
                sigma=fixation_sigma,
            )
        else:
            prior_map = simple_center_bias_map(
                image_size[0],
                image_size[1],
                sigma=fixation_sigma,
            )
        return cls(
            map=prior_map,
            image_size=image_size,
            source_split=str(split),
            excluded_image_ids=tuple(sorted(excluded)),
        )


@dataclass(frozen=True)
class COCOSearch18TaskPrior:
    """Target/task-conditioned spatial prior built from COCO-Search18 training rows."""

    maps: dict[tuple[str, str], np.ndarray]
    image_size: tuple[int, int]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        split: str = "train",
        image_size: tuple[int, int] = (224, 224),
        fixation_sigma: float = 10.0,
    ) -> "COCOSearch18TaskPrior":
        buckets: dict[tuple[str, str], list[np.ndarray]] = {}
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {
                "split",
                "width",
                "height",
                "target_category",
                "task",
                "fixation_points",
            } - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"COCO-Search18 prior manifest missing columns: {sorted(missing)}"
                )
            for row in reader:
                if row.get("split") != split:
                    continue
                points = _parse_manifest_points(row.get("fixation_points", ""))
                if points.size == 0:
                    continue
                scaled = _scale_points(
                    points,
                    width=_optional_float(row.get("width")),
                    height=_optional_float(row.get("height")),
                    image_size=image_size,
                )
                target = str(row.get("target_category", ""))
                task = str(row.get("task", ""))
                for key in ((target, task), (target, "*"), ("*", task), ("*", "*")):
                    buckets.setdefault(key, []).append(scaled)

        maps = {
            key: points_to_fixation_map(
                np.concatenate(point_sets, axis=0),
                height=image_size[0],
                width=image_size[1],
                sigma=fixation_sigma,
            )
            for key, point_sets in buckets.items()
            if point_sets
        }
        if not maps:
            maps[("*", "*")] = simple_center_bias_map(
                image_size[0],
                image_size[1],
                sigma=fixation_sigma,
            )
        return cls(maps=maps, image_size=image_size)

    def map_for(self, *, target_category: str, task: str) -> np.ndarray:
        for key in (
            (target_category, task),
            (target_category, "*"),
            ("*", task),
            ("*", "*"),
        ):
            if key in self.maps:
                return self.maps[key]
        return simple_center_bias_map(*self.image_size)


def _infer_map_shape(images: Any, target_map: Any | None) -> tuple[int, int]:
    if target_map is not None:
        array = np.asarray(target_map)
        if array.ndim == 2:
            return int(array.shape[0]), int(array.shape[1])
    shape = tuple(getattr(images, "shape", np.asarray(images).shape))
    if len(shape) < 2:
        raise ValueError("Cannot infer saliency map shape from image input")
    return int(shape[-2]), int(shape[-1])


def _stable_seed(seed: int, item_index: int, image_id: str) -> int:
    payload = f"{int(seed)}:{int(item_index)}:{image_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _parse_manifest_points(raw_points: str) -> np.ndarray:
    if raw_points == "":
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(json.loads(raw_points), dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points.reshape(-1, 2)


def _load_manifest_points(
    row: dict[str, str],
    *,
    manifest_path: Path,
    dataset_root: Path | None,
    dataset_name: str | None,
) -> np.ndarray:
    inline_points = _parse_manifest_points(row.get("fixation_points", ""))
    if inline_points.size:
        return inline_points
    points_path = row.get("fixation_points_path", "")
    if not points_path:
        return np.zeros((0, 2), dtype=np.float32)
    resolved = _resolve_manifest_asset_path(
        points_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    if not resolved.is_file():
        return np.zeros((0, 2), dtype=np.float32)
    observers = load_observer_fixations(resolved, dataset=dataset_name)
    if not observers:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(observers, axis=0).astype(np.float32, copy=False)


def _resolve_manifest_asset_path(
    value: str,
    *,
    manifest_path: Path,
    dataset_root: Path | None,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if dataset_root is not None:
        rooted = (dataset_root / candidate).resolve()
        if rooted.is_file():
            return rooted
    return (manifest_path.parent / candidate).resolve()


def _infer_manifest_dataset(path: Path) -> str | None:
    name = path.name.lower()
    if "salicon" in name:
        return "salicon"
    if "cat2000" in name:
        return "cat2000"
    return None


def _infer_manifest_dataset_root(path: Path, dataset_name: str | None) -> Path | None:
    if dataset_name is None:
        return None
    root_name = {
        "salicon": "SALICON",
        "cat2000": "CAT2000",
    }.get(dataset_name)
    if root_name is None:
        return None
    for parent in (path.parent, *path.parents):
        data_root = parent / "data" / "raw" / root_name
        if data_root.exists():
            return data_root.resolve()
    return None


def _scale_points(
    points: np.ndarray,
    *,
    width: float | None,
    height: float | None,
    image_size: tuple[int, int],
) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if width and height:
        scaled[:, 0] *= image_size[1] / float(width)
        scaled[:, 1] *= image_size[0] / float(height)
    return scaled


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
