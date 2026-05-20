"""Merge model efficiency CSVs by model name."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path("outputs/neural_roi_summary/model_efficiency_with_ssl.csv")


def merge_efficiency_profiles(
    inputs: Iterable[str | Path],
    output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Merge efficiency rows, with later inputs replacing earlier model rows."""
    rows_by_model: dict[str, dict[str, Any]] = {}
    fieldnames: list[str] = []
    for input_path in inputs:
        path = Path(input_path)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for fieldname in reader.fieldnames or []:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            for row in reader:
                model = str(row.get("model_name") or row.get("model") or "")
                if not model:
                    continue
                rows_by_model[model] = row

    if "model_name" not in fieldnames:
        fieldnames.insert(0, "model_name")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for model in sorted(rows_by_model):
            writer.writerow(rows_by_model[model])
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge model efficiency CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input efficiency CSV files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Merged output CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = merge_efficiency_profiles(args.inputs, args.output)
    print(output)


if __name__ == "__main__":
    main()
