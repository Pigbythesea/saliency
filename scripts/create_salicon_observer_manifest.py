"""Create a SALICON worker-level fixation manifest from official JSON annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "image_id",
    "image_path",
    "fixation_map_path",
    "fixation_points_path",
    "split",
    "width",
    "height",
    "fixation_points",
    "subject_id",
    "trial_id",
]


def create_salicon_observer_manifest(
    *,
    base_manifest: str | Path,
    annotation_jsons: list[str | Path],
    output_manifest: str | Path,
) -> dict[str, int | str]:
    """Write one manifest row per SALICON JSON worker annotation."""
    base_rows = _load_base_rows(base_manifest)
    rows: list[dict[str, str]] = []
    annotations_found = 0
    annotations_matched = 0

    for annotation_json in annotation_jsons:
        path = Path(annotation_json).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        image_lookup = _build_image_lookup(payload.get("images", []))
        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"{path} does not contain a list-valued annotations field")

        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            annotations_found += 1
            image_id = _resolve_image_id(annotation, image_lookup)
            base_row = base_rows.get(image_id)
            if base_row is None:
                continue
            points = _normalize_fixations(annotation.get("fixations"))
            if not points:
                continue
            annotations_matched += 1
            rows.append(
                {
                    **base_row,
                    "fixation_points": json.dumps(points),
                    "subject_id": str(annotation.get("worker_id", "")),
                    "trial_id": str(annotation.get("id", "")),
                }
            )

    output = Path(output_manifest).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "base_manifest": str(Path(base_manifest).expanduser().resolve()),
        "output_manifest": str(output),
        "annotation_files": len(annotation_jsons),
        "annotations_found": annotations_found,
        "annotations_matched": annotations_matched,
        "rows_written": len(rows),
    }


def _load_base_rows(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        image_id = row["image_id"]
        output[image_id] = {
            "image_id": image_id,
            "image_path": row.get("image_path", ""),
            "fixation_map_path": row.get("fixation_map_path", ""),
            "fixation_points_path": row.get("fixation_points_path", ""),
            "split": row.get("split", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
        }
    return output


def _build_image_lookup(images: Any) -> dict[int, str]:
    lookup = {}
    if not isinstance(images, list):
        return lookup
    for image in images:
        if not isinstance(image, dict) or "id" not in image:
            continue
        file_name = str(image.get("file_name") or image.get("name") or "")
        if not file_name:
            continue
        lookup[int(image["id"])] = Path(file_name).stem
    return lookup


def _resolve_image_id(annotation: dict[str, Any], image_lookup: dict[int, str]) -> str:
    raw_image_id = annotation.get("image_id")
    if isinstance(raw_image_id, int) and raw_image_id in image_lookup:
        return image_lookup[raw_image_id]
    if isinstance(raw_image_id, str):
        return Path(raw_image_id).stem
    if raw_image_id is not None:
        return f"COCO_train2014_{int(raw_image_id):012d}"
    return ""


def _normalize_fixations(value: Any) -> list[list[float]]:
    points = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        x, y = point[0], point[1]
        if _is_number(x) and _is_number(y):
            points.append([float(x), float(y)])
    return points


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create SALICON worker-level observer manifest from JSON annotations."
    )
    parser.add_argument("--base-manifest", default="data/manifests/salicon_manifest.csv")
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="data/manifests/salicon_observer_annotations_manifest.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = create_salicon_observer_manifest(
        base_manifest=args.base_manifest,
        annotation_jsons=args.annotations,
        output_manifest=args.output,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
