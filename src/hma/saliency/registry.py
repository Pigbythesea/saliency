"""Saliency method registry."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from hma.saliency.gradcam import gradcam_saliency
from hma.saliency.gradients import vanilla_gradient_saliency
from hma.saliency.integrated_gradients import integrated_gradients_saliency


def build_saliency_method(config: dict[str, Any]) -> Callable[..., Any]:
    """Build a saliency callable from a saliency-only or full experiment config."""
    saliency_config = _extract_saliency_config(config)
    method = saliency_config.get("method") or saliency_config.get("name")
    if method is None:
        raise KeyError("Saliency config must contain 'method' or 'name'")

    if method == "vanilla_gradient":
        return vanilla_gradient_saliency
    if method == "integrated_gradients":
        return partial(
            integrated_gradients_saliency,
            steps=int(saliency_config.get("steps", 16)),
        )
    if method == "gradcam":
        target_layer = saliency_config.get("target_layer")
        if not target_layer:
            raise KeyError("Grad-CAM saliency config requires 'target_layer'")
        return partial(gradcam_saliency, target_layer=str(target_layer))

    raise KeyError(f"Unknown saliency method: {method}")


def _extract_saliency_config(config: dict[str, Any]) -> dict[str, Any]:
    if "saliency" in config and isinstance(config["saliency"], dict):
        return config["saliency"]
    return config
