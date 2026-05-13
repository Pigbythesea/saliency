"""Model-independent saliency baselines."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from hma.metrics.saliency_metrics import simple_center_bias_map
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
