"""Convert operational token decisions into precomputed behavioral maps."""

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

from hma.external.artifacts import load_external_arrays
from hma.utils.paths import resolve_path


def export_routing_maps(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    output_size: tuple[int, int] = (224, 224),
) -> Path:
    image_ids, resources, manifest = load_external_arrays(
        artifact_dir,
        category="resource_allocation",
    )
    source_key, map_kind = _select_operational_source(resources)
    source = resources[source_key]
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_filenames = {f"{map_key}.npy" for map_key in image_ids}
    stale_map_files = [
        path
        for path in output_root.glob("*.npy")
        if path.name not in expected_filenames
    ]
    for path in stale_map_files:
        path.unlink()
    summaries = []
    for index, map_key in enumerate(image_ids):
        routing = _routing_vector(source[index], map_kind=map_kind)
        grid = _square_grid(routing)
        resized = np.asarray(
            Image.fromarray(grid.astype(np.float32), mode="F").resize(
                (int(output_size[1]), int(output_size[0])),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        np.save(output_root / f"{map_key}.npy", resized)
        summaries.append(
            {
                "map_key": map_key,
                "minimum": float(np.min(resized)),
                "maximum": float(np.max(resized)),
                "mean": float(np.mean(resized)),
            }
        )
    metadata = {
        "schema_version": "hma.external.routing_maps.v2",
        "model_id": manifest["model_id"],
        "artifact_dir": str(resolve_path(artifact_dir)),
        "resource_key": source_key,
        "map_kind": map_kind,
        "output_size": list(output_size),
        "num_maps": len(image_ids),
        "stale_map_files_removed": len(stale_map_files),
        "map_definition": _map_definition(map_kind),
        "summaries": summaries,
    }
    path = output_root / "routing_map_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _select_operational_source(
    resources: dict[str, np.ndarray],
) -> tuple[str, str]:
    masks = [
        key for key in resources if key.startswith("prediction_masks.stage_")
    ]
    if masks:
        return max(masks, key=_numeric_suffix), "retained_token_mask"
    sources = [
        key for key in resources if key.startswith("token_source_assignments.")
    ]
    if sources:
        return max(sources, key=_numeric_suffix), "inverse_merge_group_size"
    if "full_token_mask" in resources:
        return "full_token_mask", "full_token_control"
    raise ValueError(
        "Artifact has no retained-token mask, merge assignment, or full-token control"
    )


def _routing_vector(values: np.ndarray, *, map_kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if map_kind in {"retained_token_mask", "full_token_control"}:
        return array.reshape(-1)
    if map_kind == "inverse_merge_group_size":
        if array.ndim != 2:
            raise ValueError(
                f"ToMe source assignment must be a matrix, got {array.shape}"
            )
        source = array
        if source.shape[1] > 1 and _is_square(source.shape[1] - 1):
            source = source[:, 1:]
        group_sizes = np.sum(source, axis=1, keepdims=True)
        inverse_sizes = np.divide(
            1.0,
            group_sizes,
            out=np.zeros_like(group_sizes),
            where=group_sizes > 0,
        )
        allocation = np.sum(source * inverse_sizes, axis=0)
        return allocation.reshape(-1)
    raise ValueError(f"Unknown routing map kind: {map_kind}")


def _square_grid(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size > 1 and _is_square(vector.size - 1):
        vector = vector[1:]
    side = int(math.isqrt(vector.size))
    if side * side != vector.size:
        raise ValueError(f"Routing vector is not a square patch grid: {vector.size}")
    return vector.reshape(side, side)


def _is_square(value: int) -> bool:
    if value < 0:
        return False
    side = int(math.isqrt(value))
    return side * side == value


def _numeric_suffix(value: str) -> int:
    try:
        return int(value.rsplit(".", 1)[-1].replace("stage_", ""))
    except ValueError:
        return -1


def _map_definition(map_kind: str) -> str:
    if map_kind == "retained_token_mask":
        return "Binary final-stage DynamicViT retention in original patch coordinates."
    if map_kind == "inverse_merge_group_size":
        return (
            "Per-original-patch allocation equal to the inverse size of its final "
            "ToMe merge group; larger values indicate less merging."
        )
    return "Uniform full-token allocation for the paired DeiT-S control."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-height", type=int, default=224)
    parser.add_argument("--output-width", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = export_routing_maps(
        args.artifact,
        args.output_dir,
        output_size=(args.output_height, args.output_width),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
