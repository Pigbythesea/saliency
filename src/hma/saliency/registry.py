"""Saliency method registry."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from hma.saliency.attention_rollout import attention_rollout_saliency
from hma.saliency.baselines import center_bias_saliency, random_saliency
from hma.saliency.gradcam import gradcam_saliency
from hma.saliency.gradients import vanilla_gradient_saliency
from hma.saliency.integrated_gradients import integrated_gradients_saliency
from hma.saliency.occlusion import occlusion_saliency
from hma.saliency.precomputed import precomputed_map_saliency


def build_saliency_method(config: dict[str, Any]) -> Callable[..., Any]:
    """Build a saliency callable from a saliency-only or full experiment config."""
    saliency_config = _extract_saliency_config(config)
    method = saliency_config.get("method") or saliency_config.get("name")
    if method is None:
        raise KeyError("Saliency config must contain 'method' or 'name'")

    if method == "vanilla_gradient":
        return vanilla_gradient_saliency
    if method == "center_bias":
        return partial(
            center_bias_saliency,
            sigma=saliency_config.get("sigma"),
        )
    if method == "random_saliency":
        return partial(
            random_saliency,
            seed=int(saliency_config.get("seed", 0)),
        )
    if method == "dummy_gradient_free":
        return _dummy_gradient_free_saliency
    if method == "integrated_gradients":
        return partial(
            integrated_gradients_saliency,
            steps=int(saliency_config.get("steps", 16)),
        )
    if method == "occlusion":
        return partial(
            occlusion_saliency,
            patch_size=saliency_config.get("patch_size", 32),
            stride=saliency_config.get("stride"),
            baseline_value=float(saliency_config.get("baseline_value", 0.0)),
        )
    if method == "gradcam":
        target_layer = saliency_config.get("target_layer")
        if not target_layer:
            raise KeyError("Grad-CAM saliency config requires 'target_layer'")
        return partial(gradcam_saliency, target_layer=str(target_layer))
    if method in {"attention_rollout", "rollout"}:
        kwargs = {
            "discard_ratio": float(saliency_config.get("discard_ratio", 0.0)),
            "head_fusion": saliency_config.get("head_fusion", "mean"),
            "target_layer": saliency_config.get("target_layer"),
            "grid_size": saliency_config.get("grid_size"),
        }
        return partial(attention_rollout_saliency, **kwargs)
    if method in {"precomputed_map", "deepgaze_precomputed"}:
        return partial(
            precomputed_map_saliency,
            root=saliency_config.get("root"),
            path_key=saliency_config.get("path_key", "precomputed_map_path"),
            metadata_key=saliency_config.get("metadata_key"),
            path_template=saliency_config.get("path_template"),
            npz_key=saliency_config.get("npz_key"),
        )

    raise KeyError(f"Unknown saliency method: {method}")


def _extract_saliency_config(config: dict[str, Any]) -> dict[str, Any]:
    if "saliency" in config and isinstance(config["saliency"], dict):
        return config["saliency"]
    return config


def _dummy_gradient_free_saliency(
    model_wrapper: Any,
    images: Any,
    item_index: int = 0,
    **_kwargs: Any,
) -> Any:
    predict = getattr(model_wrapper, "predict", None)
    if not callable(predict):
        raise TypeError("dummy_gradient_free requires a model with predict(image)")
    return predict(images, item_index=item_index)
