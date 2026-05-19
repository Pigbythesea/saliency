"""Perturbation-based occlusion saliency."""

from __future__ import annotations

from typing import Any

from hma.saliency.gradients import _normalize_batch_maps, _require_torch, _select_logits


def occlusion_saliency(
    model_wrapper: Any,
    images: Any,
    target_class: int | list[int] | None = None,
    patch_size: int | tuple[int, int] = 32,
    stride: int | tuple[int, int] | None = None,
    baseline_value: float = 0.0,
    **_kwargs: Any,
) -> Any:
    """Compute output-drop saliency by sliding an occluding patch over the input."""
    torch = _require_torch()
    if images.ndim != 4:
        raise ValueError(f"Expected input tensor with shape BxCxHxW, got {tuple(images.shape)}")

    patch_h, patch_w = _as_hw(patch_size, "patch_size")
    stride_h, stride_w = _as_hw(stride or patch_size, "stride")
    if patch_h <= 0 or patch_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("patch_size and stride must contain positive integers")

    inputs = images.detach()
    height, width = int(inputs.shape[-2]), int(inputs.shape[-1])
    saliency = torch.zeros(
        (inputs.shape[0], 1, height, width),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    counts = torch.zeros_like(saliency)

    with torch.no_grad():
        original_logits = model_wrapper.get_last_logits(inputs)
        resolved_target = _resolve_target_class(original_logits, target_class)
        original_scores = _select_logits(original_logits, resolved_target, torch).view(-1, 1, 1, 1)

        for top in _patch_starts(height, patch_h, stride_h):
            bottom = min(top + patch_h, height)
            for left in _patch_starts(width, patch_w, stride_w):
                right = min(left + patch_w, width)
                occluded = inputs.clone()
                occluded[:, :, top:bottom, left:right] = float(baseline_value)
                occluded_scores = _select_logits(
                    model_wrapper.get_last_logits(occluded),
                    resolved_target,
                    torch,
                ).view(-1, 1, 1, 1)
                score_drop = (original_scores - occluded_scores).clamp_min(0.0)
                saliency[:, :, top:bottom, left:right] += score_drop
                counts[:, :, top:bottom, left:right] += 1.0

    averaged = saliency / counts.clamp_min(1.0)
    return _normalize_batch_maps(averaged, torch)


def _as_hw(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or length-2 sequence")
    return int(value[0]), int(value[1])


def _resolve_target_class(logits: Any, target_class: int | list[int] | None) -> int | list[int]:
    if target_class is not None:
        return target_class
    return [int(value) for value in logits.detach().argmax(dim=1).cpu().tolist()]


def _patch_starts(size: int, patch: int, stride: int) -> list[int]:
    if patch >= size:
        return [0]
    starts = list(range(0, max(1, size - patch + 1), stride))
    final = size - patch
    if starts[-1] != final:
        starts.append(final)
    return starts
