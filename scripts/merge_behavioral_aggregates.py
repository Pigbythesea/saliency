"""Merge behavioral aggregate CSVs while preserving the frozen base table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


DEDUPLICATION_KEYS = [
    "dataset",
    "model",
    "saliency_method",
    "saliency_family",
    "metric",
]


def merge_behavioral_aggregates(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    deduplication_keys: Iterable[str] = DEDUPLICATION_KEYS,
) -> Path:
    """Merge aggregate CSVs, letting later inputs replace exact duplicate keys."""
    input_paths = [Path(path) for path in inputs]
    if not input_paths:
        raise ValueError("At least one input CSV is required")

    fieldnames: list[str] = []
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    key_fields = list(deduplication_keys)
    for path in input_paths:
        rows, columns = _read_csv(path)
        for column in columns:
            if column not in fieldnames:
                fieldnames.append(column)
        for row in rows:
            key = tuple(str(row.get(field, "")) for field in key_fields)
            merged[key] = row

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in merged.values():
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge behavioral aggregate CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = merge_behavioral_aggregates(args.inputs, args.output)
    print(output.resolve())


if __name__ == "__main__":
    main()
