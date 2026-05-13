"""Gradient-based saliency methods."""

from __future__ import annotations

from typing import Any


def vanilla_gradient_saliency(
    model_wrapper: Any,
    images: Any,
    target_class: int | list[int] | None = None,
) -> Any:
    """Compute vanilla input-gradient saliency maps as Bx1xHxW tensors."""
    torch = _require_torch()
    inputs = images.detach().clone().requires_grad_(True)
    logits = model_wrapper.get_last_logits(inputs)
    selected = _select_logits(logits, target_class, torch)
    selected.sum().backward()

    if inputs.grad is None:
        raise RuntimeError("No input gradients were produced for saliency computation")

    saliency = inputs.grad.detach().abs().amax(dim=1, keepdim=True)
    return _normalize_batch_maps(saliency, torch)


def _select_logits(logits: Any, target_class: int | list[int] | None, torch: Any) -> Any:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape BxC, got {tuple(logits.shape)}")

    batch_size = logits.shape[0]
    if target_class is None:
        targets = logits.detach().argmax(dim=1)
    elif isinstance(target_class, int):
        targets = torch.full(
            (batch_size,),
            target_class,
            dtype=torch.long,
            device=logits.device,
        )
    else:
        targets = torch.as_tensor(target_class, dtype=torch.long, device=logits.device)
        if targets.numel() != batch_size:
            raise ValueError("target_class list length must match batch size")

    return logits[torch.arange(batch_size, device=logits.device), targets]


def _normalize_batch_maps(maps: Any, torch: Any, eps: float = 1e-8) -> Any:
    flattened = maps.flatten(start_dim=1)
    minimum = flattened.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flattened.max(dim=1).values.view(-1, 1, 1, 1)
    normalized = (maps - minimum) / (maximum - minimum).clamp_min(eps)
    normalized = torch.where(
        torch.isfinite(normalized),
        normalized,
        torch.zeros_like(normalized),
    )
    return normalized.clamp_min(0.0).clamp_max(1.0)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for saliency extraction. Install saliency extras with "
            '`pip install -e ".[saliency]"` or `uv pip install -e ".[saliency]"`.'
        ) from exc
    return torch
