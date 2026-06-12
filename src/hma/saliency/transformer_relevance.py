"""Gradient-weighted transformer relevance maps for ViT-like timm models."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

from hma.saliency.attention_rollout import UnsupportedModelError
from hma.saliency.gradients import _normalize_batch_maps, _require_torch, _select_logits


def transformer_relevance_saliency(
    model_wrapper: Any,
    images: Any,
    target_class: int | list[int] | None = None,
    target_layer: str | None = None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    grid_size: tuple[int, int] | list[int] | None = None,
    **_kwargs: Any,
) -> Any:
    """Compute Chefer-style gradient-weighted attention relevance maps.

    This is a scoped transformer attribution control: it uses attention tensors and
    their target-logit gradients, but it is not a full AttnLRP implementation.
    """
    torch = _require_torch()
    if not hasattr(images, "shape"):
        raise TypeError("transformer_relevance requires tensor images with shape BxCxHxW")
    if len(tuple(images.shape)) != 4:
        raise ValueError(f"Expected tensor images with shape BxCxHxW, got {tuple(images.shape)}")
    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must be in [0, 1)")

    model = getattr(model_wrapper, "model", None)
    if model is None:
        raise UnsupportedModelError(
            "Transformer relevance requires model_wrapper.model with compatible "
            "ViT attention modules."
        )

    captured: list[Any] = []
    patched = _patch_attention_modules(model, captured, target_layer=target_layer)
    try:
        zero_grad = getattr(model, "zero_grad", None)
        if callable(zero_grad):
            zero_grad(set_to_none=True)

        with torch.enable_grad():
            forward = getattr(model_wrapper, "get_last_logits", None)
            logits = forward(images) if callable(forward) else model(images)
            selected = _select_logits(logits, target_class, torch)
            selected.sum().backward()

        if not captured:
            raise UnsupportedModelError(
                "No compatible attention tensors were captured for transformer relevance."
            )
        maps = transformer_relevance_to_saliency_map(
            captured,
            image_shape=(int(images.shape[-2]), int(images.shape[-1])),
            grid_size=grid_size,
            discard_ratio=discard_ratio,
            head_fusion=head_fusion,
        )
        return maps.to(device=images.device, dtype=images.dtype)
    finally:
        _restore_attention_modules(patched)


def transformer_relevance_to_saliency_map(
    attention_tensors: list[Any],
    image_shape: tuple[int, int] | list[int],
    grid_size: tuple[int, int] | list[int] | None = None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> Any:
    """Convert captured attention tensors and gradients into Bx1xHxW maps."""
    torch = _require_torch()
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a length-2 (height, width) sequence")
    if not attention_tensors:
        raise ValueError("attention_tensors must contain at least one tensor")

    rollout = compute_transformer_relevance_rollout(
        attention_tensors,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
    )
    patch_relevance = rollout[:, 0, 1:].detach()
    grid_height, grid_width = _resolve_grid_size(
        patch_count=int(patch_relevance.shape[1]),
        grid_size=grid_size,
    )
    grids = patch_relevance.reshape(-1, 1, grid_height, grid_width)
    maps = torch.nn.functional.interpolate(
        grids,
        size=(int(image_shape[0]), int(image_shape[1])),
        mode="bilinear",
        align_corners=False,
    )
    return _normalize_batch_maps(maps, torch)


def compute_transformer_relevance_rollout(
    attention_tensors: list[Any],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> Any:
    """Roll gradient-weighted attention relevance across transformer layers."""
    torch = _require_torch()
    if not attention_tensors:
        raise ValueError("attention_tensors must contain at least one tensor")
    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must be in [0, 1)")

    rollout = None
    for attention in attention_tensors:
        gradient = getattr(attention, "grad", None)
        if gradient is None:
            raise RuntimeError(
                "Transformer relevance requires attention gradients; run backward "
                "on the selected target logit before converting maps."
            )
        weighted = torch.relu(attention.detach() * gradient.detach())
        fused = _fuse_attention_heads(weighted, head_fusion=head_fusion)
        if discard_ratio > 0.0:
            fused = _discard_low_attention(fused, discard_ratio=discard_ratio)
        fused = _add_residual_and_normalize(fused)
        rollout = fused if rollout is None else torch.bmm(fused, rollout)
    return rollout


def _patch_attention_modules(
    model: Any,
    captured: list[Any],
    *,
    target_layer: str | None,
) -> list[tuple[Any, Callable[..., Any]]]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise UnsupportedModelError(
            "Transformer relevance requires model.named_modules() to find attention layers."
        )

    patched: list[tuple[Any, Callable[..., Any]]] = []
    for name, module in named_modules():
        if target_layer is not None and name != target_layer:
            continue
        if target_layer is None and "attn" not in name.lower():
            continue
        if not _is_supported_timm_attention(module):
            continue
        original = module.forward
        module.forward = MethodType(_make_attention_forward(captured), module)
        patched.append((module, original))

    if not patched:
        detail = f" for target_layer={target_layer!r}" if target_layer else ""
        raise UnsupportedModelError(
            "No compatible timm ViT attention modules found"
            f"{detail}. Expected modules with qkv, num_heads, head_dim, q_norm, "
            "k_norm, attn_drop, norm, proj, and proj_drop."
        )
    return patched


def _restore_attention_modules(patched: list[tuple[Any, Callable[..., Any]]]) -> None:
    for module, original in patched:
        module.forward = original


def _make_attention_forward(captured: list[Any]) -> Callable[..., Any]:
    def patched_forward(
        self: Any,
        x: Any,
        attn_mask: Any | None = None,
        is_causal: bool = False,
    ) -> Any:
        return _forward_attention_with_capture(
            self,
            x,
            captured=captured,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

    return patched_forward


def _forward_attention_with_capture(
    module: Any,
    x: Any,
    *,
    captured: list[Any],
    attn_mask: Any | None,
    is_causal: bool,
) -> Any:
    torch = _require_torch()
    batch, tokens, channels = x.shape
    qkv = module.qkv(x).reshape(
        batch,
        tokens,
        3,
        module.num_heads,
        module.head_dim,
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    q = q * module.scale
    attention = q @ k.transpose(-2, -1)
    attention = _apply_optional_attention_mask(
        attention,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    attention = attention.softmax(dim=-1)
    attention = module.attn_drop(attention)
    attention.retain_grad()
    captured.append(attention)

    output = attention @ v
    output = output.transpose(1, 2).reshape(
        batch,
        tokens,
        int(getattr(module, "attn_dim", channels)),
    )
    output = module.norm(output)
    output = module.proj(output)
    output = module.proj_drop(output)
    return output


def _apply_optional_attention_mask(
    attention: Any,
    *,
    attn_mask: Any | None,
    is_causal: bool,
) -> Any:
    if attn_mask is None and not is_causal:
        return attention

    try:
        from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask
    except ImportError as exc:  # pragma: no cover - only triggered without timm
        raise UnsupportedModelError(
            "Transformer relevance requires timm attention mask helpers for masked "
            "or causal attention."
        ) from exc

    bias = resolve_self_attn_mask(attention.shape[-1], attention, attn_mask, is_causal)
    return maybe_add_mask(attention, bias)


def _is_supported_timm_attention(module: Any) -> bool:
    required = (
        "qkv",
        "num_heads",
        "head_dim",
        "q_norm",
        "k_norm",
        "attn_drop",
        "norm",
        "proj",
        "proj_drop",
    )
    return all(hasattr(module, attr) for attr in required)


def _fuse_attention_heads(attention: Any, *, head_fusion: str) -> Any:
    if attention.ndim == 3:
        fused = attention
    elif attention.ndim == 4:
        if head_fusion == "mean":
            fused = attention.mean(dim=1)
        elif head_fusion == "max":
            fused = attention.max(dim=1).values
        elif head_fusion == "min":
            fused = attention.min(dim=1).values
        else:
            raise ValueError("head_fusion must be one of: mean, max, min")
    else:
        raise ValueError(
            f"Expected attention tensor BxHxTxT or BxTxT, got {tuple(attention.shape)}"
        )

    if fused.shape[-1] != fused.shape[-2]:
        raise ValueError(f"Attention tensor must be square over tokens, got {tuple(fused.shape)}")
    return fused


def _discard_low_attention(attention: Any, *, discard_ratio: float) -> Any:
    if discard_ratio <= 0.0:
        return attention

    torch = _require_torch()
    batch, tokens, _tokens = attention.shape
    flattened = attention.reshape(batch, -1)
    keep = torch.ones_like(flattened, dtype=torch.bool)
    cls_indices = torch.arange(tokens, device=attention.device)
    keep[:, cls_indices] = True
    discard_count = int(flattened.shape[1] * discard_ratio)
    if discard_count <= 0:
        return attention

    for index in range(batch):
        order = torch.argsort(flattened[index])
        keep[index, order[:discard_count]] = False
    discarded = flattened * keep.to(dtype=flattened.dtype)
    return discarded.reshape(attention.shape)


def _add_residual_and_normalize(attention: Any) -> Any:
    torch = _require_torch()
    tokens = attention.shape[-1]
    identity = torch.eye(
        tokens,
        dtype=attention.dtype,
        device=attention.device,
    ).unsqueeze(0)
    residual = attention + identity
    row_sums = residual.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return residual / row_sums


def _resolve_grid_size(
    *,
    patch_count: int,
    grid_size: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if grid_size is None:
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise UnsupportedModelError(
                "Patch count is not square; provide explicit grid_size for "
                "transformer_relevance."
            )
        return side, side

    if len(grid_size) != 2:
        raise ValueError("grid_size must be a length-2 sequence")
    height, width = int(grid_size[0]), int(grid_size[1])
    if height * width != patch_count:
        raise ValueError(f"grid_size {grid_size} does not match {patch_count} patch tokens")
    return height, width
