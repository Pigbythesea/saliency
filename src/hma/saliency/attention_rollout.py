"""Attention rollout for ViT-like models."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from hma.saliency.postprocess import postprocess_saliency_map


class UnsupportedModelError(RuntimeError):
    """Raised when attention rollout cannot capture attention matrices."""


def compute_attention_rollout(
    attention_tensors: list[Any],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Compute cumulative attention rollout from a list of attention tensors."""
    if not attention_tensors:
        raise ValueError("attention_tensors must contain at least one tensor")
    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must be in [0, 1)")

    rollout = None
    for attention in attention_tensors:
        fused = _fuse_attention_heads(_to_numpy(attention), head_fusion=head_fusion)
        if discard_ratio > 0.0:
            fused = _discard_low_attention(fused, discard_ratio)
        fused = _add_residual_and_normalize(fused)
        rollout = fused if rollout is None else np.matmul(fused, rollout)
    return rollout.astype(np.float32)


def cls_to_patch_relevance(rollout: Any) -> np.ndarray:
    """Extract CLS-token relevance to patch tokens."""
    array = np.asarray(rollout, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected rollout shape BxTxT, got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError("Rollout must include a CLS token and at least one patch token")
    return array[:, 0, 1:].astype(np.float32)


def patch_relevance_to_grid(
    patch_relevance: Any,
    grid_size: tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    """Convert patch relevance from BxP to BxHxW patch grid."""
    relevance = np.asarray(patch_relevance, dtype=np.float32)
    if relevance.ndim != 2:
        raise ValueError(f"Expected patch relevance shape BxP, got {relevance.shape}")

    patch_count = relevance.shape[1]
    if grid_size is None:
        side = int(np.sqrt(patch_count))
        if side * side != patch_count:
            raise ValueError(
                "Patch count is not square; provide explicit grid_size for rollout"
            )
        grid_height, grid_width = side, side
    else:
        if len(grid_size) != 2:
            raise ValueError("grid_size must be a length-2 sequence")
        grid_height, grid_width = int(grid_size[0]), int(grid_size[1])
        if grid_height * grid_width != patch_count:
            raise ValueError(
                f"grid_size {grid_size} does not match {patch_count} patch tokens"
            )

    return relevance.reshape(relevance.shape[0], grid_height, grid_width)


def attention_rollout_to_saliency_map(
    attention_tensors: list[Any],
    image_shape: tuple[int, int] | list[int],
    grid_size: tuple[int, int] | list[int] | None = None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Convert attention tensors to Bx1xHxW saliency maps."""
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a length-2 (height, width) sequence")
    target_shape = (int(image_shape[0]), int(image_shape[1]))
    rollout = compute_attention_rollout(
        attention_tensors,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
    )
    relevance = cls_to_patch_relevance(rollout)
    grids = patch_relevance_to_grid(relevance, grid_size=grid_size)
    maps = [
        postprocess_saliency_map(grid, target_shape=target_shape)
        for grid in grids
    ]
    return np.stack(maps, axis=0)[:, None, :, :].astype(np.float32)


def attention_rollout_saliency(
    model_wrapper: Any,
    images: Any,
    target_class: int | list[int] | None = None,
    target_layer: str | None = None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    grid_size: tuple[int, int] | list[int] | None = None,
    **_kwargs: Any,
) -> Any:
    """Compute attention rollout saliency for wrappers exposing attention matrices."""
    del target_class
    attention_tensors = _get_attention_tensors(model_wrapper, images, target_layer)
    image_shape = _infer_image_shape(images)
    maps = attention_rollout_to_saliency_map(
        attention_tensors,
        image_shape=image_shape,
        grid_size=grid_size,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
    )
    return _to_like_input_type(maps, images)


def _get_attention_tensors(
    model_wrapper: Any,
    images: Any,
    target_layer: str | None,
) -> list[Any]:
    get_attention = getattr(model_wrapper, "get_attention_matrices", None)
    if callable(get_attention):
        attention = get_attention(images)
        if attention:
            return list(attention)

    model = getattr(model_wrapper, "model", None)
    if model is None:
        raise UnsupportedModelError(
            "Attention rollout requires get_attention_matrices(images) or "
            "model_wrapper.model with compatible attention modules."
        )

    captured: list[Any] = []
    hooks = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise UnsupportedModelError(
            "Attention rollout requires model.named_modules() or "
            "get_attention_matrices(images)."
        )

    for name, module in named_modules():
        if target_layer is not None and name != target_layer:
            continue
        if target_layer is None and "attn" not in name.lower():
            continue
        hook = module.register_forward_hook(
            partial(_capture_attention_hook, captured=captured)
        )
        hooks.append(hook)

    if not hooks:
        raise UnsupportedModelError(
            "No compatible attention modules found for attention rollout. "
            "Expose get_attention_matrices(images) or choose a valid target_layer."
        )

    restored_fused_attention = _disable_fused_attention(model)
    try:
        forward = getattr(model_wrapper, "get_last_logits", None)
        if callable(forward):
            forward(images)
        else:
            model(images)
    finally:
        _restore_fused_attention(restored_fused_attention)
        for hook in hooks:
            hook.remove()

    for _name, module in named_modules():
        if target_layer is not None and _name != target_layer:
            continue
        for attr in ("last_attn", "attention_weights", "attn_weights"):
            value = getattr(module, attr, None)
            if value is not None:
                captured.append(value)

    if not captured:
        raise UnsupportedModelError(
            "Model did not expose attention matrices during rollout capture. "
            "Use a wrapper with get_attention_matrices(images) or modules that "
            "return/store attention weights."
        )
    return captured


def _capture_attention_hook(
    _module: Any,
    _inputs: Any,
    output: Any,
    captured: list[Any],
) -> None:
    candidate = output
    if isinstance(output, tuple):
        candidate = next(
            (item for item in output if _looks_like_attention(item)),
            None,
        )
    if _looks_like_attention(candidate):
        captured.append(candidate)


def _disable_fused_attention(model: Any) -> list[tuple[Any, Any]]:
    restored = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return restored
    for _name, module in named_modules():
        if hasattr(module, "fused_attn"):
            restored.append((module, getattr(module, "fused_attn")))
            setattr(module, "fused_attn", False)
    return restored


def _restore_fused_attention(restored: list[tuple[Any, Any]]) -> None:
    for module, value in restored:
        setattr(module, "fused_attn", value)


def _looks_like_attention(value: Any) -> bool:
    try:
        shape = tuple(value.shape)
    except AttributeError:
        return False
    return len(shape) in {3, 4} and shape[-1] == shape[-2]


def _fuse_attention_heads(attention: np.ndarray, head_fusion: str) -> np.ndarray:
    array = np.asarray(attention, dtype=np.float32)
    if array.ndim == 3:
        fused = array
    elif array.ndim == 4:
        if head_fusion == "mean":
            fused = array.mean(axis=1)
        elif head_fusion == "max":
            fused = array.max(axis=1)
        elif head_fusion == "min":
            fused = array.min(axis=1)
        else:
            raise ValueError("head_fusion must be one of: mean, max, min")
    else:
        raise ValueError(f"Expected attention tensor BxHxTxT or BxTxT, got {array.shape}")

    if fused.shape[-1] != fused.shape[-2]:
        raise ValueError(f"Attention tensor must be square over tokens, got {fused.shape}")
    return fused


def _discard_low_attention(attention: np.ndarray, discard_ratio: float) -> np.ndarray:
    if discard_ratio <= 0.0:
        return attention

    batch, tokens, _tokens = attention.shape
    flattened = attention.reshape(batch, -1)
    keep = np.ones_like(flattened, dtype=bool)
    cls_indices = np.arange(tokens)
    keep[:, cls_indices] = True
    discard_count = int(flattened.shape[1] * discard_ratio)
    if discard_count <= 0:
        return attention

    for index in range(batch):
        order = np.argsort(flattened[index])
        keep[index, order[:discard_count]] = False
    discarded = flattened * keep
    return discarded.reshape(attention.shape)


def _add_residual_and_normalize(attention: np.ndarray) -> np.ndarray:
    tokens = attention.shape[-1]
    identity = np.eye(tokens, dtype=np.float32)[None, :, :]
    residual = attention + identity
    row_sums = residual.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    return residual / row_sums


def _infer_image_shape(images: Any) -> tuple[int, int]:
    shape = tuple(getattr(images, "shape", ()))
    if len(shape) < 2:
        raise ValueError("images must expose shape ending in height, width")
    return int(shape[-2]), int(shape[-1])


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_like_input_type(maps: np.ndarray, images: Any) -> Any:
    try:
        import torch
    except ImportError:
        return maps

    if isinstance(images, torch.Tensor):
        return torch.as_tensor(maps, dtype=images.dtype, device=images.device)
    return maps
