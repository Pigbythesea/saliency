"""Minimal Grad-CAM implementation for CNN-like models."""

from __future__ import annotations

from typing import Any

from hma.saliency.gradients import _normalize_batch_maps, _require_torch, _select_logits


def gradcam_saliency(
    model_wrapper: Any,
    images: Any,
    target_layer: str,
    target_class: int | list[int] | None = None,
) -> Any:
    """Compute Grad-CAM maps for a named CNN-like target layer."""
    torch = _require_torch()
    model = getattr(model_wrapper, "model", None)
    if model is None:
        raise ValueError("Grad-CAM requires model_wrapper.model to access target layers")

    layer = _get_named_layer(model, target_layer)
    activations = {}
    gradients = {}

    def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        activations["value"] = output

    def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        gradients["value"] = grad_output[0]

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        logits = model_wrapper.get_last_logits(images)
        selected = _select_logits(logits, target_class, torch)
        model.zero_grad(set_to_none=True)
        selected.sum().backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError(
            f"Grad-CAM could not capture activations/gradients for layer '{target_layer}'"
        )

    activation = activations["value"]
    gradient = gradients["value"]
    if activation.ndim != 4 or gradient.ndim != 4:
        raise ValueError(
            f"Grad-CAM requires 4D CNN feature maps for layer '{target_layer}'"
        )

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return _normalize_batch_maps(cam.detach(), torch)


def _get_named_layer(model: Any, target_layer: str) -> Any:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise ValueError("Grad-CAM requires model.named_modules() to find target layers")

    for name, module in named_modules():
        if name == target_layer:
            return module

    available = [name for name, _module in named_modules() if name]
    preview = ", ".join(available[:10]) if available else "<none>"
    raise ValueError(
        f"Target layer '{target_layer}' not found for Grad-CAM. "
        f"Available layers include: {preview}"
    )
