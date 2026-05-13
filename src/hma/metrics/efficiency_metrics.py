"""Model efficiency profiling helpers."""

from __future__ import annotations

import importlib
import statistics
import time
import warnings
from typing import Any

from hma.utils.device import resolve_device


def count_parameters(model: Any) -> int:
    """Count model parameters for modules exposing parameters()."""
    module = _unwrap_model(model)
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0

    total = 0
    for parameter in parameters():
        numel = getattr(parameter, "numel", None)
        if callable(numel):
            total += int(numel())
    return total


def estimate_model_size_mb(model: Any) -> float:
    """Estimate parameter storage size in megabytes."""
    module = _unwrap_model(model)
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0.0

    total_bytes = 0
    for parameter in parameters():
        numel = getattr(parameter, "numel", None)
        if not callable(numel):
            continue
        element_size = getattr(parameter, "element_size", None)
        bytes_per_element = int(element_size()) if callable(element_size) else 4
        total_bytes += int(numel()) * bytes_per_element
    return total_bytes / (1024.0 * 1024.0)


def measure_latency(
    model: Any,
    input_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float]:
    """Measure simple forward-pass latency in milliseconds."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")

    torch = _require_torch()
    resolved_device = resolve_device(device)
    module = _unwrap_model(model)
    if hasattr(module, "to"):
        module = module.to(resolved_device)
    if hasattr(module, "eval"):
        module.eval()

    inputs = torch.randn(tuple(input_shape), device=resolved_device)
    forward = _get_forward_callable(model, module)

    with torch.no_grad():
        for _ in range(warmup):
            forward(inputs)
        _synchronize_if_needed(torch, resolved_device)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            forward(inputs)
            _synchronize_if_needed(torch, resolved_device)
            timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_mean_ms": float(statistics.mean(timings)),
        "latency_median_ms": float(statistics.median(timings)),
        "latency_min_ms": float(min(timings)),
        "latency_max_ms": float(max(timings)),
    }


def estimate_flops(
    model: Any,
    input_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
) -> int | None:
    """Best-effort FLOPs estimate using optional fvcore or ptflops."""
    module = _unwrap_model(model)
    resolved_device = resolve_device(device)
    torch = _import_optional("torch")
    if torch is None:
        warnings.warn("FLOPs estimation requires torch; returning None", stacklevel=2)
        return None

    if hasattr(module, "to"):
        module = module.to(resolved_device)
    if hasattr(module, "eval"):
        module.eval()
    inputs = torch.randn(tuple(input_shape), device=resolved_device)

    fvcore = _import_optional("fvcore.nn")
    if fvcore is not None and hasattr(fvcore, "FlopCountAnalysis"):
        try:
            return int(fvcore.FlopCountAnalysis(module, inputs).total())
        except Exception as exc:  # pragma: no cover - optional dependency path
            warnings.warn(f"fvcore FLOPs estimation failed: {exc}", stacklevel=2)

    ptflops = _import_optional("ptflops")
    if ptflops is not None and hasattr(ptflops, "get_model_complexity_info"):
        try:
            input_res = tuple(input_shape[1:])
            flops, _params = ptflops.get_model_complexity_info(
                module,
                input_res,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            return int(flops)
        except Exception as exc:  # pragma: no cover - optional dependency path
            warnings.warn(f"ptflops FLOPs estimation failed: {exc}", stacklevel=2)

    warnings.warn(
        "FLOPs estimation requires optional dependency fvcore or ptflops; returning None",
        stacklevel=2,
    )
    return None


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _get_forward_callable(model: Any, module: Any) -> Any:
    forward = getattr(model, "forward", None)
    if callable(forward):
        return forward
    return module


def _synchronize_if_needed(torch: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_torch() -> Any:
    torch = _import_optional("torch")
    if torch is None:
        raise ImportError(
            "torch is required for latency measurement. Install saliency extras with "
            '`pip install -e ".[saliency]"` or `uv pip install -e ".[saliency]"`.'
        )
    return torch


def _import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
