"""Create deterministic pilot manifests for the first real model matrix."""

from __future__ import annotations

import argparse

from hma.experiments.pilot_manifests import create_pilot_manifest


DEFAULT_JOBS = [
    {
        "input_path": "data/manifests/salicon_manifest.csv",
        "output_path": "data/manifests/pilot/salicon_pilot500_manifest.csv",
        "split": "val",
        "stratify_column": None,
    },
    {
        "input_path": "data/manifests/cat2000_manifest.csv",
        "output_path": "data/manifests/pilot/cat2000_pilot500_manifest.csv",
        "split": "train",
        "stratify_column": "category",
    },
    {
        "input_path": "data/manifests/coco_search18_manifest.csv",
        "output_path": "data/manifests/pilot/coco_search18_pilot500_manifest.csv",
        "split": "val",
        "stratify_column": "target_category",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create HMA pilot manifests.")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for job in DEFAULT_JOBS:
        summary = create_pilot_manifest(
            job["input_path"],
            job["output_path"],
            max_rows=args.max_rows,
            split=job["split"],
            stratify_column=job["stratify_column"],
            seed=args.seed,
        )
        print(
            f"{summary['output_path']}: {summary['rows_written']} rows "
            f"from {summary['source_rows']} source rows"
        )


if __name__ == "__main__":
    main()
