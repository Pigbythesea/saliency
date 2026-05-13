"""Compute best-effort inter-observer saliency controls from manifest fixations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hma.datasets.fixation_utils import points_to_fixation_map
from hma.metrics.saliency_metrics import auc_judd, nss
from hma.utils.paths import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize inter-observer controls.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", default="224,224", help="height,width")
    parser.add_argument("--fixation-sigma", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = summarize_observer_controls(
        args.manifest,
        image_size=_parse_image_size(args.image_size),
        fixation_sigma=args.fixation_sigma,
    )
    output = Path(args.output)
    ensure_dir(output.parent)
    _write_rows(output, rows)
    print(f"Observer controls: {output.resolve()} ({len(rows)} rows)")


def summarize_observer_controls(
    manifest_path: str | Path,
    *,
    image_size: tuple[int, int],
    fixation_sigma: float = 10.0,
) -> list[dict[str, Any]]:
    """Return per-image inter-observer NSS/AUC rows when fixation points are parseable."""
    records = _load_manifest_records(manifest_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        points = _parse_manifest_fixation_points(record)
        if points.size:
            grouped[str(record.get("image_id", ""))].append({**record, "points": points})

    output = []
    height, width = image_size
    for image_id, image_records in sorted(grouped.items()):
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
                }
            )
    return output


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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_id",
        "subject_id",
        "trial_id",
        "inter_observer_nss",
        "inter_observer_auc",
        "num_observers",
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


if __name__ == "__main__":
    main()
