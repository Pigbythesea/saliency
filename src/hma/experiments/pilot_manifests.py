"""Utilities for deterministic pilot manifest generation."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from hma.utils.paths import ensure_dir


def create_pilot_manifest(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 500,
    split: str | None = None,
    stratify_column: str | None = None,
    seed: int = 123,
) -> dict[str, Any]:
    """Create a deterministic subset manifest while preserving source columns."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    rows, fieldnames = _load_rows(source)
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]

    if stratify_column:
        sampled = _stratified_sample(rows, stratify_column, max_rows, seed)
    else:
        sampled = _random_sample(rows, max_rows, seed)

    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    return {
        "input_path": source.resolve(),
        "output_path": output.resolve(),
        "source_rows": len(rows),
        "rows_written": len(sampled),
        "split": split,
        "stratify_column": stratify_column,
        "seed": seed,
    }


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _random_sample(
    rows: list[dict[str, str]],
    max_rows: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    row_copy = list(rows)
    rng.shuffle(row_copy)
    return row_copy[: min(max_rows, len(row_copy))]


def _stratified_sample(
    rows: list[dict[str, str]],
    stratify_column: str,
    max_rows: int,
    seed: int,
) -> list[dict[str, str]]:
    if not rows:
        return []
    if stratify_column not in rows[0]:
        raise KeyError(f"Manifest does not contain stratify column: {stratify_column}")

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(stratify_column, "")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    group_keys = sorted(groups)
    target = min(max_rows, len(rows))
    selected: list[dict[str, str]] = []

    while len(selected) < target:
        added = False
        for key in group_keys:
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop())
                added = True
        if not added:
            break

    rng.shuffle(selected)
    return selected
