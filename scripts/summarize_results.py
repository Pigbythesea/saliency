"""Write compact summary tables for an aggregate HMA results CSV."""

from __future__ import annotations

import argparse

from hma.experiments.summarize_results import summarize_aggregate_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize aggregate HMA results.")
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--efficiency-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = summarize_aggregate_results(
        args.aggregate_csv,
        args.output_dir,
        efficiency_csv=args.efficiency_csv,
    )
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
