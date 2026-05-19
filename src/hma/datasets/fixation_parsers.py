"""Dataset-specific fixation-location parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_observer_fixations(path: str | Path, dataset: str | None = None) -> list[np.ndarray]:
    """Load per-observer fixation points in (x, y) order from a known dataset file."""
    file_path = Path(path)
    dataset_name = (dataset or _infer_dataset_from_path(file_path)).lower()
    if dataset_name == "salicon":
        return load_salicon_observer_fixations(file_path)
    if dataset_name == "cat2000":
        return load_cat2000_observer_fixations(file_path)
    raise ValueError(f"Unsupported fixation dataset: {dataset}")


def load_salicon_observer_fixations(path: str | Path) -> list[np.ndarray]:
    """Load SALICON gaze[*].fixations arrays as per-observer (x, y) points."""
    payload = _loadmat(path)
    gaze = payload.get("gaze")
    if gaze is None:
        return []

    observers = []
    for observer in np.asarray(gaze, dtype=object).ravel():
        fixations = _mat_struct_field(observer, "fixations")
        points = _as_xy_points(fixations)
        if points.size:
            observers.append(points)
    return observers


def load_cat2000_observer_fixations(path: str | Path) -> list[np.ndarray]:
    """Load CAT2000 fixLocs masks as one observer/control point set."""
    payload = _loadmat(path)
    fix_locs = payload.get("fixLocs")
    if fix_locs is None:
        return []

    coords_yx = np.argwhere(np.asarray(fix_locs) > 0)
    if coords_yx.size == 0:
        return []
    coords_xy = coords_yx[:, [1, 0]].astype(np.float32, copy=False)
    return [coords_xy]


def _loadmat(path: str | Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "Parsing .mat fixation files requires scipy. Install the neural extra "
            'with `pip install -e ".[neural]"`.'
        ) from exc
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _mat_struct_field(value: Any, field: str) -> Any:
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, np.void) and field in value.dtype.names:
        return value[field]
    if isinstance(value, dict):
        return value.get(field)
    return None


def _as_xy_points(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0, 2), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        array = array.reshape(-1, 2)
    except ValueError:
        return np.zeros((0, 2), dtype=np.float32)
    finite = np.isfinite(array).all(axis=1)
    return array[finite].astype(np.float32, copy=False)


def _infer_dataset_from_path(path: Path) -> str:
    lowered_parts = {part.lower() for part in path.parts}
    if "salicon" in lowered_parts:
        return "salicon"
    if "cat2000" in lowered_parts:
        return "cat2000"
    return ""
