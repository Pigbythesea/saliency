"""Device selection helpers."""

from __future__ import annotations

import importlib
from typing import Any


def resolve_device(device: str | None = None) -> str:
    """Resolve device strings, using GPU by default when available."""
    requested = (device or "auto").lower()
    if requested not in {"auto", "gpu"}:
        return requested

    torch = _import_optional("torch")
    if torch is None:
        return "cpu"
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def _import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
