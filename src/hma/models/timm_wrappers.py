"""Wrappers for image models provided by timm."""

from __future__ import annotations

import importlib
from typing import Any

from hma.models.base import BaseModelWrapper


class TimmModelWrapper(BaseModelWrapper):
    """Thin wrapper around a timm image model."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        eval_mode: bool = True,
        features_only: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.eval_mode = eval_mode
        self.features_only = features_only

        try:
            timm = importlib.import_module("timm")
        except ImportError as exc:
            raise ImportError(
                "timm is required for TimmModelWrapper. Install model extras with "
                '`pip install -e ".[models]"` or `uv pip install -e ".[models]"`.'
            ) from exc

        create_kwargs = dict(kwargs)
        if features_only:
            create_kwargs["features_only"] = True
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            **create_kwargs,
        )

        if eval_mode and hasattr(self.model, "eval"):
            self.model.eval()

    def forward(self, images: Any) -> Any:
        """Run the wrapped model."""
        return self.model(images)

    def get_features(self, images: Any, layers: list[str] | None = None) -> Any:
        """Return model features when the wrapped model exposes them."""
        if layers and layers != ["embedding"]:
            return self._get_named_layer_features(images, layers)
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        return self.forward(images)

    def get_last_logits(self, images: Any) -> Any:
        """Return final logits from the wrapped model."""
        return self.forward(images)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model name and parameter count."""
        return {
            "model_name": self.model_name,
            "parameter_count": _count_parameters(self.model),
        }

    def _get_named_layer_features(self, images: Any, layers: list[str]) -> dict[str, Any]:
        modules = dict(_named_modules(self.model))
        missing = [layer for layer in layers if layer not in modules]
        if missing:
            available = ", ".join(sorted(modules)[:20])
            raise ValueError(
                f"Unknown timm layer(s): {missing}. Available module names include: {available}"
            )

        features: dict[str, Any] = {}
        handles = []

        def capture(layer_name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                features[layer_name] = output

            return hook

        for layer in layers:
            register = getattr(modules[layer], "register_forward_hook", None)
            if not callable(register):
                raise TypeError(f"Layer '{layer}' does not support forward hooks")
            handles.append(register(capture(layer)))

        try:
            self.forward(images)
        finally:
            for handle in handles:
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()

        missing_outputs = [layer for layer in layers if layer not in features]
        if missing_outputs:
            raise RuntimeError(
                f"Forward hooks did not capture outputs for layer(s): {missing_outputs}"
            )
        return features


def _count_parameters(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0

    total = 0
    for parameter in parameters():
        numel = getattr(parameter, "numel", None)
        if callable(numel):
            total += int(numel())
    return total


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        return [(name, module) for name, module in named_modules() if name]
    return []
