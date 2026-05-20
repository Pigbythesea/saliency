"""Precomputed saliency-map loader for reference baselines."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def precomputed_map_saliency(
    _model_wrapper: Any,
    _images: Any,
    *,
    item: dict[str, Any],
    root: str | Path | None = None,
    path_key: str = "precomputed_map_path",
    metadata_key: str | None = None,
    path_template: str | None = None,
    npz_key: str | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Return a saliency map loaded from a configured path."""
    map_path = _resolve_map_path(
        item,
        root=root,
        path_key=path_key,
        metadata_key=metadata_key,
        path_template=path_template,
    )
    if not map_path.is_file():
        raise FileNotFoundError(f"Precomputed saliency map not found: {map_path}")
    return _load_map(map_path, npz_key=npz_key)


def _resolve_map_path(
    item: dict[str, Any],
    *,
    root: str | Path | None,
    path_key: str,
    metadata_key: str | None,
    path_template: str | None,
) -> Path:
    raw_path = item.get(path_key)
    metadata = item.get("metadata", {})
    if raw_path is None and isinstance(metadata, dict):
        raw_path = metadata.get(metadata_key or path_key)
    if raw_path is None and path_template is not None:
        raw_path = path_template.format(
            image_id=item.get("image_id", ""),
            image_path=item.get("image_path", ""),
            map_key=precomputed_map_key(item),
        )
    if raw_path is None or str(raw_path) == "":
        raise KeyError(
            "Precomputed saliency requires a map path from item, metadata, "
            "or saliency.path_template"
        )

    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if root is not None:
        return (Path(root).expanduser().resolve() / candidate).resolve()
    return candidate.resolve()


def _load_map(path: Path, *, npz_key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".npz":
        payload = np.load(path)
        key = npz_key or ("saliency" if "saliency" in payload else payload.files[0])
        return np.asarray(payload[key], dtype=np.float32)
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def precomputed_map_key(item: dict[str, Any] | str | Path) -> str:
    """Return a stable, filesystem-safe key for a precomputed map."""
    if isinstance(item, dict):
        source = str(item.get("image_path") or item.get("image_id") or "item")
    else:
        source = str(item)
    normalized = source.replace("\\", "/")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    if len(stem) > 80:
        stem = stem[-80:].strip("._")
    return f"{stem or 'item'}_{digest}"
