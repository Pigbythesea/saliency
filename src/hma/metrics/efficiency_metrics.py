"""Model efficiency profiling helpers."""

from __future__ import annotations

import importlib
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from hma.utils.device import resolve_device


@dataclass(frozen=True)
class SequentialCostRecord:
    """Total per-image/task resource use for sequential or adaptive models."""

    model_id: str
    image_id: str
    task_id: str
    comparability_group: str
    fixation_count: int
    scanpath_length: float
    recurrent_steps: int
    diffusion_steps: int
    selected_glimpses: int
    stopped: bool
    stop_step: int | None
    high_resolution_sampled_area: float
    image_area: float
    high_resolution_sampled_fraction: float
    total_cost: float
    total_cost_unit: str
    total_latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "hma.efficiency.sequential_cost.v1",
            "model_id": self.model_id,
            "image_id": self.image_id,
            "task_id": self.task_id,
            "comparability_group": self.comparability_group,
            "fixation_count": self.fixation_count,
            "scanpath_length": self.scanpath_length,
            "recurrent_steps": self.recurrent_steps,
            "diffusion_steps": self.diffusion_steps,
            "selected_glimpses": self.selected_glimpses,
            "stopped": self.stopped,
            "stop_step": self.stop_step,
            "high_resolution_sampled_area": self.high_resolution_sampled_area,
            "image_area": self.image_area,
            "high_resolution_sampled_fraction": (
                self.high_resolution_sampled_fraction
            ),
            "total_cost_per_image_task": self.total_cost,
            "total_cost_unit": self.total_cost_unit,
            "total_latency_ms": self.total_latency_ms,
        }


def build_sequential_cost_record(
    *,
    model_id: str,
    image_id: str,
    task_id: str = "",
    comparability_group: str,
    fixations: list[tuple[float, float]] | None = None,
    recurrent_steps: int = 0,
    diffusion_steps: int = 0,
    selected_glimpses: int = 0,
    stopped: bool = False,
    stop_step: int | None = None,
    high_resolution_sampled_area: float = 0.0,
    image_area: float,
    cost_components: dict[str, float],
    total_cost_unit: str,
    total_latency_ms: float | None = None,
) -> SequentialCostRecord:
    """Validate and combine same-unit cost components for one image/task."""
    if image_area <= 0.0:
        raise ValueError("image_area must be positive")
    if not total_cost_unit:
        raise ValueError("total_cost_unit must be explicit")
    counts = {
        "recurrent_steps": recurrent_steps,
        "diffusion_steps": diffusion_steps,
        "selected_glimpses": selected_glimpses,
    }
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("step and glimpse counts must be nonnegative")
    if stop_step is not None and int(stop_step) < 0:
        raise ValueError("stop_step must be nonnegative")
    high_resolution_sampled_area = float(high_resolution_sampled_area)
    if high_resolution_sampled_area < 0.0:
        raise ValueError("high_resolution_sampled_area must be nonnegative")
    components = {str(key): float(value) for key, value in cost_components.items()}
    if not components or any(not np.isfinite(value) or value < 0.0 for value in components.values()):
        raise ValueError("cost_components must contain finite nonnegative values")
    points = np.asarray(fixations or [], dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("fixations must be a finite sequence of x/y pairs")
    scanpath_length = (
        float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if len(points) > 1
        else 0.0
    )
    latency = None if total_latency_ms is None else float(total_latency_ms)
    if latency is not None and (not np.isfinite(latency) or latency < 0.0):
        raise ValueError("total_latency_ms must be finite and nonnegative")
    return SequentialCostRecord(
        model_id=str(model_id),
        image_id=str(image_id),
        task_id=str(task_id),
        comparability_group=str(comparability_group),
        fixation_count=int(len(points)),
        scanpath_length=scanpath_length,
        recurrent_steps=int(recurrent_steps),
        diffusion_steps=int(diffusion_steps),
        selected_glimpses=int(selected_glimpses),
        stopped=bool(stopped),
        stop_step=int(stop_step) if stop_step is not None else None,
        high_resolution_sampled_area=high_resolution_sampled_area,
        image_area=float(image_area),
        high_resolution_sampled_fraction=high_resolution_sampled_area
        / float(image_area),
        total_cost=float(sum(components.values())),
        total_cost_unit=str(total_cost_unit),
        total_latency_ms=latency,
    )


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
