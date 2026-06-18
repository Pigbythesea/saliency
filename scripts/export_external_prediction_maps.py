"""Export external artifact map-like outputs as precomputed saliency maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.artifacts import load_external_arrays  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


def export_prediction_maps(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    category: str | None = None,
    key: str | None = None,
    output_size: tuple[int, int] = (224, 224),
) -> Path:
    selected_category, image_ids, arrays, manifest = _load_selected_arrays(
        artifact_dir,
        category=category,
        key=key,
    )
    selected_key = _select_key(arrays, preferred=key, category=selected_category)
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    expected = {f"{map_key}.npy" for map_key in image_ids}
    for stale in output_root.glob("*.npy"):
        if stale.name not in expected:
            stale.unlink()
    summaries = []
    values = arrays[selected_key]
    for index, map_key in enumerate(image_ids):
        prediction = _as_map(values[index], output_size=output_size)
        np.save(output_root / f"{map_key}.npy", prediction)
        summaries.append(
            {
                "map_key": map_key,
                "minimum": float(np.min(prediction)),
                "maximum": float(np.max(prediction)),
                "mean": float(np.mean(prediction)),
            }
        )
    metadata = {
        "schema_version": "hma.paper1.clean_prediction_maps.v1",
        "model_id": manifest.get("model_id", ""),
        "artifact_dir": str(resolve_path(artifact_dir)),
        "category": selected_category,
        "source_key": selected_key,
        "output_size": list(output_size),
        "num_maps": len(image_ids),
        "summaries": summaries,
    }
    path = output_root / "prediction_map_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_selected_arrays(
    artifact_dir: str | Path,
    *,
    category: str | None,
    key: str | None,
) -> tuple[str, list[str], dict[str, np.ndarray], dict[str, Any]]:
    categories = [category] if category else ["outputs", "resource_allocation"]
    failures: list[str] = []
    for candidate in categories:
        if candidate is None:
            continue
        try:
            image_ids, arrays, manifest = load_external_arrays(
                artifact_dir,
                category=candidate,
                keys=[key] if key else None,
            )
        except Exception as exc:
            failures.append(f"{candidate}:{type(exc).__name__}:{exc}")
            continue
        if arrays:
            return candidate, image_ids, arrays, manifest
        failures.append(f"{candidate}:no_arrays")
    raise ValueError("No map-like external arrays available: " + " | ".join(failures))


def _select_key(
    arrays: dict[str, np.ndarray],
    *,
    preferred: str | None,
    category: str,
) -> str:
    if preferred:
        if preferred not in arrays:
            raise KeyError(f"Requested external array key is missing: {preferred}")
        return preferred
    if category == "outputs":
        non_logits = [key for key in arrays if key != "logits"]
        if non_logits:
            return sorted(non_logits)[0]
    resource_priority = [
        key
        for prefix in ("prediction_masks.stage_", "token_source_assignments.", "full_token_mask")
        for key in arrays
        if key == prefix or key.startswith(prefix)
    ]
    if resource_priority:
        return sorted(resource_priority)[-1]
    return sorted(arrays)[0]


def _as_map(value: Any, *, output_size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 0:
        raise ValueError("External output cannot be converted to a map scalar")
    if array.ndim == 1:
        array = _square_grid(array)
    elif array.ndim == 3:
        if array.shape[0] in {1, 3}:
            array = np.mean(array, axis=0)
        elif array.shape[-1] in {1, 3}:
            array = np.mean(array, axis=-1)
        else:
            raise ValueError(f"Unsupported map tensor shape: {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"External output cannot be converted to 2D map: {array.shape}")
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if array.shape != output_size:
        array = np.asarray(
            Image.fromarray(array, mode="F").resize(
                (int(output_size[1]), int(output_size[0])),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    return array


def _square_grid(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size > 1 and _is_square(vector.size - 1):
        vector = vector[1:]
    side = int(math.isqrt(vector.size))
    if side * side != vector.size:
        raise ValueError(f"Vector output is not a square patch grid: {vector.size}")
    return vector.reshape(side, side)


def _is_square(value: int) -> bool:
    if value < 0:
        return False
    side = int(math.isqrt(value))
    return side * side == value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", choices=["outputs", "resource_allocation"])
    parser.add_argument("--key")
    parser.add_argument("--output-height", type=int, default=224)
    parser.add_argument("--output-width", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = export_prediction_maps(
        args.artifact,
        args.output_dir,
        category=args.category,
        key=args.key,
        output_size=(args.output_height, args.output_width),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
