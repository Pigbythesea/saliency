"""Restore older matched-panel DINOv2 cells into repository-local outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


CELL_NAMES = [
    f"vit_small_patch14_dinov2_{roi}_flatten_pca_validation_selection_full"
    for roi in ("v1", "v2", "v3", "hv4")
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(
    source_root: str | Path = "C:/saliency_outputs/neural_subj01_full",
    destination_root: str | Path = "outputs/neural_subj01_full",
    manifest_path: str | Path = (
        "outputs/methodology_trace/dinov2_provenance_hashes.csv"
    ),
) -> Path:
    source = Path(source_root)
    destination = Path(destination_root)
    rows: list[dict[str, object]] = []
    for cell_name in CELL_NAMES:
        source_cell = source / cell_name
        destination_cell = destination / cell_name
        if not source_cell.is_dir():
            raise FileNotFoundError(f"Missing DINOv2 source cell: {source_cell}")
        destination_cell.mkdir(parents=True, exist_ok=True)
        source_files = sorted(path for path in source_cell.iterdir() if path.is_file())
        for source_path in source_files:
            destination_path = destination_cell / source_path.name
            shutil.copy2(source_path, destination_path)
            source_hash = _sha256(source_path)
            destination_hash = _sha256(destination_path)
            rows.append(
                {
                    "cell": cell_name,
                    "file": source_path.name,
                    "source_path": source_path.as_posix(),
                    "destination_path": destination_path.as_posix(),
                    "size_bytes": source_path.stat().st_size,
                    "sha256": source_hash,
                    "verified": source_hash == destination_hash,
                }
            )
    if not rows or not all(bool(row["verified"]) for row in rows):
        raise ValueError("DINOv2 provenance restoration hash verification failed")
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="C:/saliency_outputs/neural_subj01_full",
    )
    parser.add_argument(
        "--destination-root",
        default="outputs/neural_subj01_full",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/methodology_trace/dinov2_provenance_hashes.csv",
    )
    args = parser.parse_args()
    print(restore(args.source_root, args.destination_root, args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
