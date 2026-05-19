"""Create an HMA neural manifest from Algonauts 2023 subject data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "split",
    "subject_id",
    "roi",
    "roi_response_path",
    "roi_responses",
]


def create_algonauts_manifest(
    *,
    root: str | Path,
    subject: str,
    hemisphere: str = "lh",
    num_vertices: int = 512,
    roi: str | None = None,
    output_manifest: str | Path = "data/manifests/nsd_algonauts_manifest.csv",
    max_items: int | None = None,
    overwrite_responses: bool = False,
) -> dict[str, str | int]:
    """Write per-image response files and a manifest for one Algonauts subject."""
    root_path = Path(root).expanduser().resolve()
    subject_dir = root_path / subject
    image_dir = subject_dir / "training_split" / "training_images"
    fmri_path = (
        subject_dir
        / "training_split"
        / "training_fmri"
        / f"{hemisphere}_training_fmri.npy"
    )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")
    if not fmri_path.is_file():
        raise FileNotFoundError(f"Training fMRI file not found: {fmri_path}")

    images = sorted(image_dir.glob("train-*_nsd-*.png"))
    responses = np.load(fmri_path, mmap_mode="r")
    if responses.ndim != 2:
        raise ValueError(f"Expected a 2D fMRI array, got shape {responses.shape}")
    if len(images) != responses.shape[0]:
        raise ValueError(
            "Training image count does not match fMRI rows: "
            f"{len(images)} images vs {responses.shape[0]} rows"
        )
    if num_vertices < 1 or num_vertices > responses.shape[1]:
        raise ValueError(
            f"num_vertices must be between 1 and {responses.shape[1]}, got {num_vertices}"
        )

    selected_images = images[: int(max_items)] if max_items is not None else images
    roi_label = roi or f"all_{hemisphere}_{num_vertices}"
    response_dir = subject_dir / "responses" / roi_label
    response_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_manifest).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for row_index, image_path in enumerate(selected_images):
        response_path = response_dir / f"{image_path.stem}.npy"
        if overwrite_responses or not response_path.is_file():
            np.save(response_path, np.asarray(responses[row_index, :num_vertices], dtype=np.float32))
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": _relative_posix(image_path, root_path),
                "split": "train",
                "subject_id": subject,
                "roi": roi_label,
                "roi_response_path": _relative_posix(response_path, root_path),
                "roi_responses": "",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "manifest": str(output_path),
        "root": str(root_path),
        "subject": subject,
        "hemisphere": hemisphere,
        "roi": roi_label,
        "rows": len(rows),
        "response_dim": num_vertices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an HMA manifest from Algonauts 2023 subject data."
    )
    parser.add_argument("--root", default="data/raw/nsd_algonauts")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--hemisphere", choices=["lh", "rh"], default="lh")
    parser.add_argument("--num-vertices", type=int, default=512)
    parser.add_argument("--roi", help="Manifest ROI label. Defaults to all_<hemisphere>_<num_vertices>.")
    parser.add_argument(
        "--output-manifest",
        default="data/manifests/nsd_algonauts_manifest.csv",
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--overwrite-responses", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = create_algonauts_manifest(
        root=args.root,
        subject=args.subject,
        hemisphere=args.hemisphere,
        num_vertices=args.num_vertices,
        roi=args.roi,
        output_manifest=args.output_manifest,
        max_items=args.max_items,
        overwrite_responses=args.overwrite_responses,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


if __name__ == "__main__":
    main()
