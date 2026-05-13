"""Config-driven image preprocessing for torch vision models."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from hma.utils.device import resolve_device


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image_for_model(
    image: Any,
    config: dict[str, Any] | None = None,
    device: str | None = None,
) -> Any:
    """Convert a PIL/NumPy/tensor image to a normalized BxCxHxW torch tensor."""
    torch = _require_torch()
    preprocessing = dict((config or {}).get("preprocessing", {}))
    target_size = _parse_input_size(preprocessing.get("input_size", [224, 224]))
    resolved_device = resolve_device(device or (config or {}).get("device", "auto"))

    tensor = _image_to_tensor(image, torch)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got {tensor.shape}")

    tensor = tensor.to(device=resolved_device, dtype=torch.float32)
    if target_size is not None and tuple(tensor.shape[-2:]) != target_size:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    mean = _parse_normalization_values(preprocessing.get("mean", "imagenet"), IMAGENET_MEAN)
    std = _parse_normalization_values(preprocessing.get("std", "imagenet"), IMAGENET_STD)
    if mean is not None and std is not None:
        mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
        std_tensor = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor.clamp_min(1e-8)

    return tensor


def _image_to_tensor(image: Any, torch: Any) -> Any:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.moveaxis(array, -1, 0))

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        if tensor.max().detach().item() > 1.0:
            tensor = tensor / 255.0
        return _ensure_channel_first(tensor, torch)

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = np.moveaxis(array, -1, 0)
    elif array.ndim == 3 and array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif array.ndim == 3 and array.shape[0] != 3:
        raise ValueError(f"Expected HxWxC or CxHxW image array, got {array.shape}")

    tensor = torch.as_tensor(array, dtype=torch.float32)
    if float(np.nanmax(array)) > 1.0:
        tensor = tensor / 255.0
    return tensor


def _ensure_channel_first(tensor: Any, torch: Any) -> Any:
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        if tensor.shape[0] == 3:
            return tensor
        if tensor.shape[-1] == 1:
            return tensor.permute(2, 0, 1).repeat(3, 1, 1)
        if tensor.shape[-1] == 3:
            return tensor.permute(2, 0, 1)
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor.repeat(1, 3, 1, 1)
        if tensor.shape[1] == 3:
            return tensor
        if tensor.shape[-1] == 1:
            return tensor.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
        if tensor.shape[-1] == 3:
            return tensor.permute(0, 3, 1, 2)
    raise ValueError(f"Expected image tensor in CHW/HWC/BCHW/BHWC format, got {tuple(tensor.shape)}")


def _parse_input_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError("preprocessing.input_size must be an int or length-2 sequence")
    height, width = value
    return (int(height), int(width))


def _parse_normalization_values(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if value is None or value == "none":
        return None
    if value == "imagenet":
        return default
    if len(value) != 3:
        raise ValueError("preprocessing mean/std values must have three channels")
    return tuple(float(channel) for channel in value)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for model preprocessing. Install model extras with "
            '`pip install -e ".[models]"` or `uv pip install -e ".[models]"`.'
        ) from exc
    return torch
