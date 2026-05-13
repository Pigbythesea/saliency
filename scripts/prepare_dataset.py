"""Prepare dataset manifests for HMA datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image


MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "fixation_map_path",
    "split",
    "width",
    "height",
]
CAT2000_MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "fixation_map_path",
    "category",
    "split",
    "width",
    "height",
]
COCO_SEARCH18_MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "split",
    "width",
    "height",
    "target_category",
    "task",
    "fixation_points",
    "subject_id",
    "trial_id",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAP_PATH_TERMS = {"fixation", "fixations", "saliency", "maps", "map", "density"}
GENERIC_CATEGORY_PARTS = {
    "image",
    "images",
    "stimuli",
    "stimulus",
    "maps",
    "map",
    "fixation",
    "fixations",
    "saliency",
    "density",
    "groundtruth",
    "ground_truth",
}
SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def build_salicon_manifest(
    root: str | Path,
    output_path: str | Path = "data/manifests/salicon_manifest.csv",
) -> dict[str, int | Path]:
    """Scan a SALICON root directory and write a portable manifest CSV."""
    root_path = Path(root).expanduser().resolve()
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()

    image_paths, map_paths = _collect_salicon_files(root_path)
    map_lookup = _build_map_lookup(map_paths)
    rows = []

    for image_path in sorted(image_paths):
        map_path = map_lookup.get(_normalized_stem(image_path))
        if map_path is None:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": _portable_relative_path(image_path, root_path),
                "fixation_map_path": _portable_relative_path(map_path, root_path),
                "split": _infer_split(image_path),
                "width": width,
                "height": height,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "root": root_path,
        "output_path": output,
        "images_found": len(image_paths),
        "maps_found": len(map_paths),
        "rows_written": len(rows),
    }


def build_cat2000_manifest(
    root: str | Path,
    output_path: str | Path = "data/manifests/cat2000_manifest.csv",
) -> dict[str, int | Path]:
    """Scan a CAT2000 root directory and write a portable manifest CSV."""
    root_path = Path(root).expanduser().resolve()
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()

    image_paths, map_paths = _collect_salicon_files(root_path)
    map_lookup = _build_categorized_map_lookup(map_paths, root_path)
    rows = []

    for image_path in sorted(image_paths):
        category = _infer_category(image_path.relative_to(root_path))
        map_path = map_lookup.get((category, _normalized_stem(image_path)))
        if map_path is None:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": _portable_relative_path(image_path, root_path),
                "fixation_map_path": _portable_relative_path(map_path, root_path),
                "category": category,
                "split": _infer_split(image_path),
                "width": width,
                "height": height,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAT2000_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "root": root_path,
        "output_path": output,
        "images_found": len(image_paths),
        "maps_found": len(map_paths),
        "rows_written": len(rows),
    }


def build_coco_search18_manifest(
    root: str | Path,
    annotations_path: str | Path,
    output_path: str | Path = "data/manifests/coco_search18_manifest.csv",
) -> dict[str, int | Path]:
    """Convert COCO-Search18-style JSON annotations into a portable manifest."""
    root_path = Path(root).expanduser().resolve()
    annotations = _load_annotation_records(annotations_path)
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()

    rows = []
    for annotation in annotations:
        image_path = _resolve_annotation_image_path(root_path, annotation)
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        image_id = str(
            _first_present(annotation, ["image_id", "image", "file_name"])
            or image_path.stem
        )
        rows.append(
            {
                "image_id": image_id,
                "image_path": _portable_relative_path(image_path, root_path),
                "split": str(annotation.get("split", "train")),
                "width": width,
                "height": height,
                "target_category": str(
                    _first_present(annotation, ["target_category", "target", "category"])
                    or ""
                ),
                "task": str(annotation.get("task", "search")),
                "fixation_points": json.dumps(_extract_fixation_points(annotation)),
                "subject_id": _first_present(annotation, ["subject_id", "subject"]),
                "trial_id": _first_present(annotation, ["trial_id", "trial"]),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COCO_SEARCH18_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "root": root_path,
        "annotations_path": Path(annotations_path).expanduser().resolve(),
        "output_path": output,
        "annotations_found": len(annotations),
        "rows_written": len(rows),
    }


def _collect_salicon_files(root: Path) -> tuple[list[Path], list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    image_paths: list[Path] = []
    map_paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(root)
        if _is_map_path(relative_path):
            map_paths.append(path)
        else:
            image_paths.append(path)
    return image_paths, map_paths


def _build_map_lookup(map_paths: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(map_paths, key=_map_priority):
        lookup.setdefault(_normalized_stem(path), path)
    return lookup


def _build_categorized_map_lookup(
    map_paths: list[Path],
    root: Path,
) -> dict[tuple[str, str], Path]:
    lookup: dict[tuple[str, str], Path] = {}
    for path in sorted(map_paths, key=_map_priority):
        category = _infer_category(path.relative_to(root))
        lookup.setdefault((category, _normalized_stem(path)), path)
    return lookup


def _is_map_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    stem_tokens = set(re.split(r"[_\-\s]+", path.stem.lower()))
    return bool((parts | stem_tokens) & MAP_PATH_TERMS)


def _map_priority(path: Path) -> tuple[int, str]:
    lowered_parts = [part.lower() for part in path.parts]
    if any(part in {"fixation", "fixations"} for part in lowered_parts):
        return (0, str(path))
    if any(part in {"saliency", "density"} for part in lowered_parts):
        return (1, str(path))
    return (2, str(path))


def _normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_fixation", "_fixations", "_saliency", "_density", "_map"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return re.sub(r"[^a-z0-9]+", "", stem)


def _infer_split(path: Path) -> str:
    for part in path.parts:
        split = SPLIT_ALIASES.get(part.lower())
        if split is not None:
            return split
    return "unknown"


def _infer_category(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        lowered = part.lower()
        if lowered in SPLIT_ALIASES or lowered in GENERIC_CATEGORY_PARTS:
            continue
        return part
    return "unknown"


def _portable_relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _load_annotation_records(annotations_path: str | Path) -> list[dict]:
    path = Path(annotations_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in ("annotations", "trials", "data"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        if records is None:
            raise ValueError("Annotation JSON must contain annotations, trials, or data")
    else:
        raise TypeError("Annotation JSON must be a list or dictionary")

    return [record for record in records if isinstance(record, dict)]


def _resolve_annotation_image_path(root: Path, annotation: dict) -> Path:
    raw_path = _first_present(annotation, ["image_path", "file_name", "image"])
    if raw_path is None:
        image_id = _first_present(annotation, ["image_id"])
        raw_path = image_id if image_id is not None else ""
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _extract_fixation_points(annotation: dict) -> list[list[float]]:
    for key in ("fixation_points", "fixations"):
        points = annotation.get(key)
        if points is None:
            continue
        if isinstance(points, list):
            return _normalize_point_list(points)

    x_values = annotation.get("X") or annotation.get("x")
    y_values = annotation.get("Y") or annotation.get("y")
    if isinstance(x_values, list) and isinstance(y_values, list):
        return [
            [float(x), float(y)]
            for x, y in zip(x_values, y_values)
            if _is_number(x) and _is_number(y)
        ]
    return []


def _normalize_point_list(points: list) -> list[list[float]]:
    normalized = []
    for point in points:
        if isinstance(point, dict):
            x = point.get("x", point.get("X"))
            y = point.get("y", point.get("Y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        if _is_number(x) and _is_number(y):
            normalized.append([float(x), float(y)])
    return normalized


def _first_present(data: dict, keys: list[str]) -> object | None:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HMA dataset manifests.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["salicon", "cat2000", "coco_search18"],
    )
    parser.add_argument("--root", required=True, help="Path to the dataset root.")
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to COCO-Search18 annotation JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output manifest CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset == "salicon":
        output = args.output or "data/manifests/salicon_manifest.csv"
        summary = build_salicon_manifest(args.root, output)
        print(
            "SALICON manifest written to "
            f"{summary['output_path']} "
            f"({summary['rows_written']} rows, "
            f"{summary['images_found']} images, "
            f"{summary['maps_found']} maps)"
        )
    elif args.dataset == "cat2000":
        output = args.output or "data/manifests/cat2000_manifest.csv"
        summary = build_cat2000_manifest(args.root, output)
        print(
            "CAT2000 manifest written to "
            f"{summary['output_path']} "
            f"({summary['rows_written']} rows, "
            f"{summary['images_found']} images, "
            f"{summary['maps_found']} maps)"
        )
    elif args.dataset == "coco_search18":
        if args.annotations is None:
            raise SystemExit("--annotations is required for --dataset coco_search18")
        output = args.output or "data/manifests/coco_search18_manifest.csv"
        summary = build_coco_search18_manifest(args.root, args.annotations, output)
        print(
            "COCO-Search18 manifest written to "
            f"{summary['output_path']} "
            f"({summary['rows_written']} rows, "
            f"{summary['annotations_found']} annotations)"
        )


if __name__ == "__main__":
    main()
