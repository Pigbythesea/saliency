"""Validate NSD / Algonauts-style neural manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "split",
    "subject_id",
    "roi",
    "roi_response_path",
    "roi_responses",
}


def validate_neural_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
    split: str | None = None,
    subject_id: str | None = None,
    roi: str | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Validate neural manifest files and return a compact summary."""
    manifest = Path(manifest_path).expanduser().resolve()
    root_path = Path(root).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = _load_rows(manifest)
    columns = set(rows[0].keys()) if rows else set()
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    selected = []
    split_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    roi_counts: Counter[str] = Counter()
    response_shape: tuple[int, ...] | None = None

    for record in rows:
        split_counts[record["split"]] += 1
        if split is not None and record["split"] != split:
            continue
        if subject_id is not None and record["subject_id"] != subject_id:
            continue
        if roi is not None and record["roi"] != roi:
            continue
        if max_items is not None and len(selected) >= int(max_items):
            continue

        image_path = _resolve_path(record["image_path"], root_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for {record['image_id']}: {image_path}")

        response = _load_response(record, root_path)
        if response is None:
            raise ValueError(
                f"Row {record['image_id']} must provide roi_response_path or roi_responses"
            )
        flat_response = np.asarray(response, dtype=np.float32).ravel()
        if response_shape is None:
            response_shape = flat_response.shape
        elif flat_response.shape != response_shape:
            raise ValueError(
                "All selected response vectors must have matching shapes: "
                f"expected {response_shape}, got {flat_response.shape} for {record['image_id']}"
            )

        subject_counts[record["subject_id"]] += 1
        roi_counts[record["roi"]] += 1
        selected.append(record)

    if not selected:
        raise ValueError("No rows matched the requested manifest filters")

    return {
        "manifest": str(manifest),
        "root": str(root_path),
        "selected_rows": len(selected),
        "total_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "selected_subject_counts": dict(sorted(subject_counts.items())),
        "selected_roi_counts": dict(sorted(roi_counts.items())),
        "response_shape": list(response_shape or ()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an HMA neural manifest.")
    parser.add_argument("--manifest", required=True, help="CSV manifest to validate.")
    parser.add_argument("--root", default=".", help="Root for relative image/response paths.")
    parser.add_argument("--split", help="Optional split filter.")
    parser.add_argument("--subject-id", help="Optional subject filter.")
    parser.add_argument("--roi", help="Optional ROI filter.")
    parser.add_argument("--max-items", type=int, help="Maximum selected rows to validate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_neural_manifest(
        args.manifest,
        root=args.root,
        split=args.split,
        subject_id=args.subject_id,
        roi=args.roi,
        max_items=args.max_items,
    )
    print(json.dumps(summary, indent=2))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_response(record: dict[str, str], root: Path) -> np.ndarray | None:
    response_path = record.get("roi_response_path", "")
    if response_path:
        path = _resolve_path(response_path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Response file not found for {record['image_id']}: {path}")
        if path.suffix == ".npz":
            data = np.load(path)
            key = "responses" if "responses" in data else data.files[0]
            return np.asarray(data[key], dtype=np.float32)
        return np.asarray(np.load(path), dtype=np.float32)

    inline = record.get("roi_responses", "")
    if inline:
        return np.asarray(json.loads(inline), dtype=np.float32)
    return None


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


if __name__ == "__main__":
    main()
