"""Prepare dataset manifests for HMA datasets."""

from __future__ import annotations

import argparse
import csv
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HMA dataset manifests.")
    parser.add_argument("--dataset", required=True, choices=["salicon", "cat2000"])
    parser.add_argument("--root", required=True, help="Path to the dataset root.")
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


if __name__ == "__main__":
    main()
