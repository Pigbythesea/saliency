"""Create deterministic scaled manifests for the V2 static benchmark."""

from __future__ import annotations

from hma.experiments.pilot_manifests import create_pilot_manifest


SEED = 123
MAX_ROWS = 2000

MANIFEST_JOBS = [
    {
        "input": "data/manifests/salicon_manifest.csv",
        "output": "data/manifests/v2/salicon_static2000_manifest.csv",
        "split": "val",
        "stratify": None,
    },
    {
        "input": "data/manifests/cat2000_manifest.csv",
        "output": "data/manifests/v2/cat2000_static2000_manifest.csv",
        "split": "train",
        "stratify": "category",
    },
    {
        "input": "data/manifests/coco_search18_manifest.csv",
        "output": "data/manifests/v2/coco_search18_static2000_manifest.csv",
        "split": "val",
        "stratify": "target_category",
    },
]


def main() -> None:
    for job in MANIFEST_JOBS:
        summary = create_pilot_manifest(
            job["input"],
            job["output"],
            max_rows=MAX_ROWS,
            split=job["split"],
            stratify_column=job["stratify"],
            seed=SEED,
        )
        print(
            f"{summary['output_path']}: {summary['rows_written']} rows "
            f"from {summary['source_rows']} source rows"
        )


if __name__ == "__main__":
    main()
