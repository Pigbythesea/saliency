"""Integrated gradients saliency."""

from __future__ import annotations

from typing import Any

from hma.saliency.gradients import _normalize_batch_maps, _require_torch, _select_logits


def integrated_gradients_saliency(
    model_wrapper: Any,
    images: Any,
    target_class: int | list[int] | None = None,
    steps: int = 16,
) -> Any:
    """Compute simple integrated gradients saliency maps as Bx1xHxW tensors."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    torch = _require_torch()
    inputs = images.detach()
    baseline = torch.zeros_like(inputs)
    total_gradients = torch.zeros_like(inputs)

    for alpha in torch.linspace(0.0, 1.0, steps, device=inputs.device):
        scaled = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        logits = model_wrapper.get_last_logits(scaled)
        selected = _select_logits(logits, target_class, torch)
        gradients = torch.autograd.grad(selected.sum(), scaled)[0]
        total_gradients += gradients.detach()

    average_gradients = total_gradients / float(steps)
    integrated = (inputs - baseline) * average_gradients
    saliency = integrated.detach().abs().amax(dim=1, keepdim=True)
    return _normalize_batch_maps(saliency, torch)
