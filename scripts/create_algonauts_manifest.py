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
    hemispheres: list[str] | None = None,
    num_vertices: int = 512,
    roi: str | None = None,
    roi_class: str | None = None,
    roi_names: list[str] | None = None,
    combine_hemispheres: bool = False,
    output_manifest: str | Path = "data/manifests/nsd_algonauts_manifest.csv",
    max_items: int | None = None,
    overwrite_responses: bool = False,
) -> dict[str, str | int]:
    """Write per-image response files and a manifest for one Algonauts subject."""
    root_path = Path(root).expanduser().resolve()
    subject_dir = root_path / subject
    image_dir = subject_dir / "training_split" / "training_images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")
    if roi_class or roi_names:
        return _create_roi_manifest(
            root_path=root_path,
            subject=subject,
            hemispheres=hemispheres or [hemisphere],
            roi_class=roi_class or "prf-visualrois",
            roi_names=roi_names or [],
            combine_hemispheres=combine_hemispheres,
            output_manifest=output_manifest,
            max_items=max_items,
            overwrite_responses=overwrite_responses,
        )

    fmri_path = (
        subject_dir
        / "training_split"
        / "training_fmri"
        / f"{hemisphere}_training_fmri.npy"
    )
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


def resolve_prf_visual_roi_labels(roi_name: str) -> set[str]:
    """Return Algonauts PRF visual ROI labels for a combined ROI name."""
    normalized = roi_name.strip()
    if normalized == "V1":
        return {"V1v", "V1d"}
    if normalized == "V2":
        return {"V2v", "V2d"}
    if normalized == "V3":
        return {"V3v", "V3d"}
    if normalized == "hV4":
        return {"hV4"}
    return {normalized}


def _create_roi_manifest(
    *,
    root_path: Path,
    subject: str,
    hemispheres: list[str],
    roi_class: str,
    roi_names: list[str],
    combine_hemispheres: bool,
    output_manifest: str | Path,
    max_items: int | None,
    overwrite_responses: bool,
) -> dict[str, str | int]:
    if not roi_names:
        raise ValueError("ROI manifest mode requires at least one ROI name")
    if not combine_hemispheres and len(hemispheres) > 1:
        raise ValueError("Multiple hemispheres require --combine-hemispheres")

    subject_dir = root_path / subject
    image_dir = subject_dir / "training_split" / "training_images"
    images = sorted(image_dir.glob("train-*_nsd-*.png"))
    selected_images = images[: int(max_items)] if max_items is not None else images
    output_path = Path(output_manifest).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_path = subject_dir / "roi_masks" / f"mapping_{roi_class}.npy"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"ROI mapping not found: {mapping_path}")
    label_to_value = _load_label_to_value(mapping_path)

    hemisphere_data = {}
    for hemi in hemispheres:
        fmri_path = subject_dir / "training_split" / "training_fmri" / f"{hemi}_training_fmri.npy"
        mask_path = subject_dir / "roi_masks" / f"{hemi}.{roi_class}_challenge_space.npy"
        if not fmri_path.is_file():
            raise FileNotFoundError(f"Training fMRI file not found: {fmri_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"ROI mask not found: {mask_path}")
        responses = np.load(fmri_path, mmap_mode="r")
        mask = np.asarray(np.load(mask_path), dtype=np.int64)
        if len(images) != responses.shape[0]:
            raise ValueError(
                f"{hemi} image count does not match fMRI rows: "
                f"{len(images)} images vs {responses.shape[0]} rows"
            )
        if mask.shape[0] != responses.shape[1]:
            raise ValueError(
                f"{hemi} mask length does not match fMRI vertices: "
                f"{mask.shape[0]} mask values vs {responses.shape[1]} vertices"
            )
        hemisphere_data[hemi] = {"responses": responses, "mask": mask}

    rows: list[dict[str, str]] = []
    roi_dims: dict[str, int] = {}
    for roi_name in roi_names:
        label_values = _roi_label_values(roi_name, label_to_value)
        selected_columns = {
            hemi: np.flatnonzero(np.isin(data["mask"], label_values))
            for hemi, data in hemisphere_data.items()
        }
        selected_columns = {
            hemi: columns for hemi, columns in selected_columns.items() if columns.size > 0
        }
        response_dim = int(sum(columns.size for columns in selected_columns.values()))
        if response_dim == 0:
            raise ValueError(f"ROI '{roi_name}' has zero selected vertices")
        roi_dims[roi_name] = response_dim
        response_dir = subject_dir / "responses" / roi_name
        response_dir.mkdir(parents=True, exist_ok=True)

        for row_index, image_path in enumerate(selected_images):
            response_path = response_dir / f"{image_path.stem}.npy"
            if overwrite_responses or not response_path.is_file():
                parts = [
                    np.asarray(
                        hemisphere_data[hemi]["responses"][row_index, columns],
                        dtype=np.float32,
                    )
                    for hemi, columns in selected_columns.items()
                ]
                np.save(response_path, np.concatenate(parts, axis=0))
            rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": _relative_posix(image_path, root_path),
                    "split": "train",
                    "subject_id": subject,
                    "roi": roi_name,
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
        "hemispheres": ",".join(hemispheres),
        "roi_class": roi_class,
        "roi": ",".join(roi_names),
        "rows": len(rows),
        "response_dim": ",".join(f"{roi}:{roi_dims[roi]}" for roi in roi_names),
    }


def _load_label_to_value(mapping_path: Path) -> dict[str, int]:
    mapping = np.load(mapping_path, allow_pickle=True).item()
    return {str(label): int(value) for value, label in mapping.items() if int(value) != 0}


def _roi_label_values(roi_name: str, label_to_value: dict[str, int]) -> list[int]:
    labels = resolve_prf_visual_roi_labels(roi_name)
    missing = sorted(label for label in labels if label not in label_to_value)
    if missing:
        raise ValueError(f"Unknown ROI label(s) for {roi_name}: {missing}")
    return [label_to_value[label] for label in sorted(labels)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an HMA manifest from Algonauts 2023 subject data."
    )
    parser.add_argument("--root", default="data/raw/nsd_algonauts")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--hemisphere", choices=["lh", "rh"], default="lh")
    parser.add_argument("--hemispheres", nargs="+", choices=["lh", "rh"])
    parser.add_argument("--num-vertices", type=int, default=512)
    parser.add_argument("--roi", help="Manifest ROI label. Defaults to all_<hemisphere>_<num_vertices>.")
    parser.add_argument("--roi-class", help="ROI mask class, for example prf-visualrois.")
    parser.add_argument("--roi-names", nargs="+", help="ROI names to export, for example V1 V2 V3 hV4.")
    parser.add_argument("--combine-hemispheres", action="store_true")
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
        hemispheres=args.hemispheres,
        num_vertices=args.num_vertices,
        roi=args.roi,
        roi_class=args.roi_class,
        roi_names=args.roi_names,
        combine_hemispheres=args.combine_hemispheres,
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
