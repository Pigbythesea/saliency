"""Compute best-effort inter-observer saliency controls from manifest fixations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hma.datasets.fixation_utils import points_to_fixation_map
from hma.datasets.fixation_parsers import load_observer_fixations
from hma.metrics.saliency_metrics import auc_judd, nss
from hma.saliency.postprocess import postprocess_saliency_map
from hma.utils.paths import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize inter-observer controls.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=None, help="Dataset root for relative paths.")
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "salicon", "cat2000", "coco_search18"],
    )
    parser.add_argument("--image-size", default="224,224", help="height,width")
    parser.add_argument("--fixation-sigma", type=float, default=10.0)
    parser.add_argument("--max-observers-per-image", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = summarize_observer_controls(
        args.manifest,
        root=args.root,
        dataset=args.dataset,
        image_size=_parse_image_size(args.image_size),
        fixation_sigma=args.fixation_sigma,
        max_observers_per_image=args.max_observers_per_image,
    )
    output = Path(args.output)
    ensure_dir(output.parent)
    _write_rows(output, rows)
    print(f"Observer controls: {output.resolve()} ({len(rows)} rows)")


def summarize_observer_controls(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    dataset: str = "auto",
    image_size: tuple[int, int],
    fixation_sigma: float = 10.0,
    max_observers_per_image: int | None = 10,
) -> list[dict[str, Any]]:
    """Return per-image observer-control NSS/AUC rows when fixations are parseable."""
    records = _load_manifest_records(manifest_path)
    root_path = Path(root).expanduser().resolve() if root is not None else None
    mat_rows = _summarize_mat_observer_controls(
        records,
        root_path=root_path,
        dataset=dataset,
        image_size=image_size,
        fixation_sigma=fixation_sigma,
        max_observers_per_image=max_observers_per_image,
    )
    inline_rows = _summarize_inline_observer_controls(
        records,
        image_size=image_size,
        fixation_sigma=fixation_sigma,
        max_observers_per_image=max_observers_per_image,
    )
    return [*mat_rows, *inline_rows]


def _summarize_inline_observer_controls(
    records: list[dict[str, str]],
    *,
    image_size: tuple[int, int],
    fixation_sigma: float,
    max_observers_per_image: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        points = _parse_manifest_fixation_points(record)
        if points.size:
            grouped[str(record.get("image_id", ""))].append(
                {
                    **record,
                    "points": _scale_inline_points(
                        points,
                        record=record,
                        image_size=image_size,
                    ),
                }
            )

    output = []
    height, width = image_size
    for image_id, image_records in sorted(grouped.items()):
        if max_observers_per_image is not None and max_observers_per_image > 0:
            image_records = image_records[:max_observers_per_image]
        if len(image_records) < 2:
            continue
        maps = [
            points_to_fixation_map(
                record["points"],
                height=height,
                width=width,
                sigma=fixation_sigma,
            )
            for record in image_records
        ]
        for index, record in enumerate(image_records):
            target = maps[index]
            prediction = np.mean(
                [other_map for other_index, other_map in enumerate(maps) if other_index != index],
                axis=0,
            )
            output.append(
                {
                    "image_id": image_id,
                    "subject_id": record.get("subject_id", ""),
                    "trial_id": record.get("trial_id", ""),
                    "inter_observer_nss": nss(prediction, target),
                    "inter_observer_auc": auc_judd(prediction, target),
                    "num_observers": len(image_records),
                    "control_type": "leave_one_observer_out",
                    "source": "inline_fixation_points",
                }
            )
    return output


def _summarize_mat_observer_controls(
    records: list[dict[str, str]],
    *,
    root_path: Path | None,
    dataset: str,
    image_size: tuple[int, int],
    fixation_sigma: float,
    max_observers_per_image: int | None,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        if record.get("fixation_points"):
            continue
        raw_path = record.get("fixation_points_path", "")
        if not raw_path:
            continue
        fixation_path = _resolve_manifest_path(raw_path, root_path)
        if not fixation_path.is_file():
            continue
        dataset_name = _resolve_dataset_name(dataset, record, fixation_path)
        observers = _load_scaled_observers(
            fixation_path,
            dataset=dataset_name,
            record=record,
            image_size=image_size,
        )
        if max_observers_per_image is not None and max_observers_per_image > 0:
            observers = observers[:max_observers_per_image]
        if not observers:
            continue

        output.extend(
            _leave_one_observer_rows(
                record,
                observers,
                image_size=image_size,
                fixation_sigma=fixation_sigma,
                source=str(fixation_path),
            )
        )
        if len(observers) == 1 and record.get("fixation_map_path"):
            reference_row = _single_location_reference_row(
                record,
                observers[0],
                root_path=root_path,
                image_size=image_size,
                fixation_sigma=fixation_sigma,
                source=str(fixation_path),
            )
            if reference_row is not None:
                output.append(reference_row)
    return output


def _leave_one_observer_rows(
    record: dict[str, str],
    observers: list[np.ndarray],
    *,
    image_size: tuple[int, int],
    fixation_sigma: float,
    source: str,
) -> list[dict[str, Any]]:
    if len(observers) < 2:
        return []
    maps = [
        points_to_fixation_map(points, height=image_size[0], width=image_size[1], sigma=fixation_sigma)
        for points in observers
    ]
    rows = []
    for index, target in enumerate(maps):
        prediction = np.mean(
            [other_map for other_index, other_map in enumerate(maps) if other_index != index],
            axis=0,
        )
        rows.append(
            {
                "image_id": record.get("image_id", ""),
                "subject_id": f"observer_{index}",
                "trial_id": record.get("trial_id", ""),
                "inter_observer_nss": nss(prediction, target),
                "inter_observer_auc": auc_judd(prediction, target),
                "num_observers": len(observers),
                "control_type": "leave_one_observer_out",
                "source": source,
            }
        )
    return rows


def _single_location_reference_row(
    record: dict[str, str],
    points: np.ndarray,
    *,
    root_path: Path | None,
    image_size: tuple[int, int],
    fixation_sigma: float,
    source: str,
) -> dict[str, Any] | None:
    target_path = _resolve_manifest_path(record["fixation_map_path"], root_path)
    if not target_path.is_file():
        return None
    with Image.open(target_path) as image:
        target = postprocess_saliency_map(
            np.asarray(image.convert("L"), dtype=np.float32),
            target_shape=image_size,
        )
    prediction = points_to_fixation_map(
        points,
        height=image_size[0],
        width=image_size[1],
        sigma=fixation_sigma,
    )
    return {
        "image_id": record.get("image_id", ""),
        "subject_id": "aggregate_fixation_locs",
        "trial_id": record.get("trial_id", ""),
        "inter_observer_nss": nss(prediction, target),
        "inter_observer_auc": auc_judd(prediction, target),
        "num_observers": 1,
        "control_type": "single_fixationloc_reference",
        "source": source,
    }


def _load_scaled_observers(
    path: Path,
    *,
    dataset: str,
    record: dict[str, str],
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    observers = load_observer_fixations(path, dataset=dataset)
    if not observers:
        return []
    original_width = _optional_float(record.get("width"))
    original_height = _optional_float(record.get("height"))
    if original_width is None or original_height is None:
        return observers
    scale_x = image_size[1] / original_width
    scale_y = image_size[0] / original_height
    scaled = []
    for points in observers:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        point_array[:, 0] *= scale_x
        point_array[:, 1] *= scale_y
        scaled.append(point_array)
    return scaled


def _resolve_manifest_path(path: str, root_path: Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if root_path is not None:
        return (root_path / candidate).resolve()
    return candidate.resolve()


def _resolve_dataset_name(dataset: str, record: dict[str, str], path: Path) -> str:
    if dataset != "auto":
        return dataset
    if "category" in record:
        return "cat2000"
    lowered_parts = {part.lower() for part in path.parts}
    if "salicon" in lowered_parts:
        return "salicon"
    if "cat2000" in lowered_parts:
        return "cat2000"
    return "coco_search18"


def _load_manifest_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_manifest_fixation_points(record: dict[str, str]) -> np.ndarray:
    raw_points = record.get("fixation_points", "")
    if raw_points:
        try:
            points = np.asarray(json.loads(raw_points), dtype=np.float32).reshape(-1, 2)
        except (json.JSONDecodeError, ValueError):
            return np.zeros((0, 2), dtype=np.float32)
        return points
    return np.zeros((0, 2), dtype=np.float32)


def _scale_inline_points(
    points: np.ndarray,
    *,
    record: dict[str, str],
    image_size: tuple[int, int],
) -> np.ndarray:
    original_width = _optional_float(record.get("width"))
    original_height = _optional_float(record.get("height"))
    if original_width is None or original_height is None:
        return points
    scaled = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    scaled[:, 0] *= image_size[1] / original_width
    scaled[:, 1] *= image_size[0] / original_height
    return scaled


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_id",
        "subject_id",
        "trial_id",
        "inter_observer_nss",
        "inter_observer_auc",
        "num_observers",
        "control_type",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_image_size(raw: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--image-size must be height,width")
    return parts[0], parts[1]


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
