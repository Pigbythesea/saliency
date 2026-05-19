"""Utilities for fixation point processing."""

from __future__ import annotations

import numpy as np

from hma.saliency.postprocess import normalize_saliency_map


def points_to_fixation_map(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    """Convert Nx2 fixation points in (x, y) order into a Gaussian map."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    fixation_map = np.zeros((height, width), dtype=np.float32)
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.size == 0:
        return fixation_map
    point_array = point_array.reshape(-1, 2)

    fast_map = _points_to_fixation_map_fast(point_array, height, width, sigma)
    if fast_map is not None:
        return fast_map

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    for x, y in point_array:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        squared_distance = (x_grid - x) ** 2 + (y_grid - y) ** 2
        fixation_map += np.exp(-squared_distance / (2.0 * sigma**2)).astype(
            np.float32
        )

    if float(np.sum(fixation_map)) == 0.0:
        return fixation_map
    return normalize_saliency_map(fixation_map)


def _points_to_fixation_map_fast(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> np.ndarray | None:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return None

    impulses = np.zeros((height, width), dtype=np.float32)
    rounded = np.rint(points).astype(np.int64)
    valid = (
        np.isfinite(points).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] < height)
    )
    if not np.any(valid):
        return impulses
    np.add.at(impulses, (rounded[valid, 1], rounded[valid, 0]), 1.0)
    smoothed = gaussian_filter(impulses, sigma=float(sigma), mode="constant")
    if float(np.sum(smoothed)) == 0.0:
        return impulses
    return normalize_saliency_map(smoothed.astype(np.float32, copy=False))
