"""Aggregate benchmark result CSV files."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hma.utils.paths import ensure_dir


IDENTIFIER_COLUMNS = {"image_id", "image_path"}
AGGREGATE_FIELDNAMES = [
    "dataset",
    "model",
    "saliency_method",
    "metric",
    "n",
    "mean",
    "std",
    "stderr",
    "ci95_low",
    "ci95_high",
]


def find_per_image_csvs(paths: Iterable[str | Path]) -> list[Path]:
    """Return per-image metric CSVs from explicit files or result directories."""
    result_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("per_image_metrics.csv"))
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"Result path does not exist: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result_paths.append(resolved)

    return result_paths


def load_per_image_records(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load one wide per-image metrics CSV into long metric records."""
    path = Path(csv_path)
    metadata = _load_sibling_metadata(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return records
        metric_columns = [
            field for field in reader.fieldnames if field not in IDENTIFIER_COLUMNS
        ]
        for row in reader:
            for metric in metric_columns:
                value = _parse_finite_float(row.get(metric))
                if value is None:
                    continue
                records.append(
                    {
                        "dataset": metadata["dataset"],
                        "model": metadata["model"],
                        "saliency_method": metadata["saliency_method"],
                        "experiment": metadata["experiment"],
                        "metric": metric,
                        "value": value,
                        "image_id": row.get("image_id", ""),
                        "image_path": row.get("image_path", ""),
                        "source_csv": str(path),
                    }
                )

    return records


def aggregate_result_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Aggregate one or more result files/directories into metric summary rows."""
    records: list[dict[str, Any]] = []
    for csv_path in find_per_image_csvs(paths):
        records.extend(load_per_image_records(csv_path))
    return aggregate_records(records)


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate long metric records by dataset/model/saliency method/metric."""
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("dataset", "unknown")),
            str(record.get("model", "unknown")),
            str(record.get("saliency_method", "unknown")),
            str(record.get("metric", "unknown")),
        )
        value = _parse_finite_float(record.get("value"))
        if value is not None:
            grouped[key].append(value)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        n = int(values.size)
        mean = float(np.mean(values)) if n else 0.0
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        stderr = float(std / math.sqrt(n)) if n else 0.0
        margin = 1.96 * stderr
        dataset, model, saliency_method, metric = key
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "saliency_method": saliency_method,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "stderr": stderr,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }
        )

    return rows


def save_aggregate_table(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Save aggregate rows to CSV and return the resolved output path."""
    output_path = Path(path).expanduser()
    ensure_dir(output_path.parent)
    row_list = list(rows)
    fieldnames = list(AGGREGATE_FIELDNAMES)
    extra_fields = sorted(
        {
            key
            for row in row_list
            for key in row.keys()
            if key not in AGGREGATE_FIELDNAMES
        }
    )
    fieldnames.extend(extra_fields)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)

    return output_path.resolve()


def load_aggregate_table(path: str | Path) -> list[dict[str, Any]]:
    """Load an aggregate CSV table."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_sibling_metadata(csv_path: Path) -> dict[str, str]:
    metadata = {
        "dataset": "unknown",
        "model": "unknown",
        "saliency_method": "unknown",
        "experiment": "unknown",
    }
    json_path = csv_path.parent / "aggregate_metrics.json"
    if not json_path.is_file():
        return metadata

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return metadata

    for key in metadata:
        value = payload.get(key)
        if value is not None:
            metadata[key] = str(value)
    return metadata


def _parse_finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
