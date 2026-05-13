"""Model registry and builder helpers."""

from __future__ import annotations

from typing import Any

from hma.models.dummy import DummySaliencyModel
from hma.models.timm_wrappers import TimmModelWrapper


def build_model(config: dict[str, Any]) -> Any:
    """Build a model wrapper from a model-only or full experiment config."""
    model_config = _extract_model_config(config)
    model_name = model_config.get("name") or model_config.get("model_name")
    if not model_name:
        raise KeyError("Model config must contain a non-empty 'name'")

    if model_name in {"dummy_saliency", "dummy_vision_encoder"}:
        return DummySaliencyModel.from_config(model_config)

    backend = model_config.get("backend", "timm")
    if backend == "timm":
        kwargs = dict(model_config.get("kwargs", {}))
        return TimmModelWrapper(
            model_name=str(model_name),
            pretrained=bool(model_config.get("pretrained", False)),
            eval_mode=bool(model_config.get("eval_mode", True)),
            features_only=bool(model_config.get("features_only", False)),
            **kwargs,
        )

    raise KeyError(f"Unknown model backend: {backend}")


def _extract_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if "model" in config and isinstance(config["model"], dict):
        return config["model"]
    return config
