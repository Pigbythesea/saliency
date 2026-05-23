"""Create ROI noise-ceiling sidecars from NSD ncsnr files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


ROI_NAMES = ["V1", "V2", "V3", "hV4"]
HEMISPHERES = ["lh", "rh"]
NOISE_COLUMNS = ["noise_ceiling_path", "noise_ceiling_values", "noise_ceiling_source"]


def create_nsd_noise_ceiling_manifest(
    *,
    algonauts_root: str | Path = "data/raw/nsd_algonauts",
    nsd_full_root: str | Path = "data/raw/nsd_full",
    subject: str = "subj01",
    manifest_path: str | Path = "data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv",
    output_manifest: str | Path | None = None,
    roi_names: list[str] | None = None,
    n_trials: int = 3,
) -> dict[str, Any]:
    """Write ROI-level noise-ceiling vectors and add them to a manifest."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    algonauts_root_path = Path(algonauts_root).expanduser().resolve()
    nsd_full_root_path = Path(nsd_full_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve() if output_manifest else manifest
    rois = roi_names or list(ROI_NAMES)

    subject_dir = algonauts_root_path / subject
    mask_dir = subject_dir / "roi_masks"
    output_dir = subject_dir / "noise_ceilings"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_label_to_value(mask_dir / "mapping_prf-visualrois.npy")
    challenge_noise_by_hemi = {
        hemi: _load_challenge_noise(
            ncsnr_path=(
                nsd_full_root_path
                / subject
                / "fsaverage"
                / "betas_fithrf_GLMdenoise_RR"
                / f"{hemi}.ncsnr.mgh"
            ),
            all_vertices_path=mask_dir / f"{hemi}.all-vertices_fsaverage_space.npy",
            roi_mask_path=mask_dir / f"{hemi}.prf-visualrois_challenge_space.npy",
            n_trials=n_trials,
        )
        for hemi in HEMISPHERES
    }

    written: dict[str, Path] = {}
    summary_rows = []
    for roi in rois:
        label_values = _roi_label_values(roi, mapping)
        parts = []
        for hemi in HEMISPHERES:
            challenge_noise, roi_mask = challenge_noise_by_hemi[hemi]
            columns = np.flatnonzero(np.isin(roi_mask, label_values))
            if columns.size:
                parts.append(challenge_noise[columns])
        if not parts:
            raise ValueError(f"ROI '{roi}' has zero selected noise-ceiling vertices")
        values = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        _validate_response_dim(subject_dir / "responses" / roi, values, roi)
        output_path = output_dir / f"{roi}.npy"
        np.save(output_path, values)
        written[roi] = output_path
        summary_rows.append(
            {
                "roi": roi,
                "path": _relative_posix(output_path, algonauts_root_path),
                "num_targets": int(values.size),
                "n_trials": int(n_trials),
                "mean_noise_ceiling": float(np.mean(values)),
                "min_noise_ceiling": float(np.min(values)),
                "max_noise_ceiling": float(np.max(values)),
            }
        )

    summary_path = output_dir / "noise_ceiling_summary.csv"
    _write_rows(
        summary_path,
        summary_rows,
        [
            "roi",
            "path",
            "num_targets",
            "n_trials",
            "mean_noise_ceiling",
            "min_noise_ceiling",
            "max_noise_ceiling",
        ],
    )
    rows, fieldnames = _load_manifest(manifest)
    source = f"nsd_ncsnr_mgh_n_trials_{n_trials}"
    updated = _attach_noise_ceilings_to_manifest(
        rows,
        root=algonauts_root_path,
        written=written,
        source=source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, updated, [*fieldnames, *[c for c in NOISE_COLUMNS if c not in fieldnames]])
    return {
        "manifest": str(output),
        "noise_ceiling_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "rois": ",".join(rois),
        "n_trials": int(n_trials),
        "rows": len(updated),
    }


def _load_challenge_noise(
    *,
    ncsnr_path: Path,
    all_vertices_path: Path,
    roi_mask_path: Path,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    ncsnr = _load_mgh_vector(ncsnr_path)
    all_vertices = np.asarray(np.load(all_vertices_path), dtype=bool).ravel()
    roi_mask = np.asarray(np.load(roi_mask_path), dtype=np.int64).ravel()
    if ncsnr.size != all_vertices.size:
        raise ValueError(
            f"{ncsnr_path.name} has {ncsnr.size} values but "
            f"{all_vertices_path.name} has {all_vertices.size}"
        )
    if int(all_vertices.sum()) != roi_mask.size:
        raise ValueError(
            f"{all_vertices_path.name} selects {int(all_vertices.sum())} vertices "
            f"but {roi_mask_path.name} has {roi_mask.size}"
        )
    challenge_ncsnr = ncsnr[all_vertices]
    return _noise_ceiling_from_ncsnr(challenge_ncsnr, n_trials=n_trials), roi_mask


def _load_mgh_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"NSD ncsnr file not found: {path}")
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to read NSD .mgh files. Install it with "
            "`.\\.venv\\Scripts\\python.exe -m pip install nibabel`."
        ) from exc
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64).ravel()


def _noise_ceiling_from_ncsnr(ncsnr: np.ndarray, *, n_trials: int) -> np.ndarray:
    squared = np.asarray(ncsnr, dtype=np.float64) ** 2
    ceiling = squared / (squared + (1.0 / float(n_trials)))
    return np.clip(np.nan_to_num(ceiling, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _load_label_to_value(mapping_path: Path) -> dict[str, int]:
    mapping = np.load(mapping_path, allow_pickle=True).item()
    return {str(label): int(value) for value, label in mapping.items() if int(value) != 0}


def _resolve_prf_visual_roi_labels(roi_name: str) -> set[str]:
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


def _roi_label_values(roi_name: str, label_to_value: dict[str, int]) -> list[int]:
    labels = _resolve_prf_visual_roi_labels(roi_name)
    missing = sorted(label for label in labels if label not in label_to_value)
    if missing:
        raise ValueError(f"Unknown ROI label(s) for {roi_name}: {missing}")
    return [label_to_value[label] for label in sorted(labels)]


def _validate_response_dim(response_dir: Path, values: np.ndarray, roi: str) -> None:
    response_files = sorted(response_dir.glob("*.npy"))
    if not response_files:
        return
    response = np.load(response_files[0])
    if response.size != values.size:
        raise ValueError(
            f"ROI '{roi}' noise ceiling has {values.size} targets but "
            f"{response_files[0]} has {response.size} responses"
        )


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"Manifest has no header: {path}")
    return rows, fieldnames


def _attach_noise_ceilings_to_manifest(
    rows: list[dict[str, str]],
    *,
    root: Path,
    written: dict[str, Path],
    source: str,
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        updated = dict(row)
        roi = str(updated.get("roi", ""))
        path = written.get(roi)
        if path is not None:
            updated["noise_ceiling_path"] = _relative_posix(path, root)
            updated["noise_ceiling_values"] = ""
            updated["noise_ceiling_source"] = source
        else:
            updated.setdefault("noise_ceiling_path", "")
            updated.setdefault("noise_ceiling_values", "")
            updated.setdefault("noise_ceiling_source", "")
        output.append(updated)
    return output


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create ROI noise ceilings from NSD ncsnr files and update a manifest."
    )
    parser.add_argument("--algonauts-root", default="data/raw/nsd_algonauts")
    parser.add_argument("--nsd-full-root", default="data/raw/nsd_full")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv",
    )
    parser.add_argument("--output-manifest")
    parser.add_argument("--roi-names", nargs="+", default=ROI_NAMES)
    parser.add_argument("--n-trials", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_nsd_noise_ceiling_manifest(
        algonauts_root=args.algonauts_root,
        nsd_full_root=args.nsd_full_root,
        subject=args.subject,
        manifest_path=args.manifest_path,
        output_manifest=args.output_manifest,
        roi_names=args.roi_names,
        n_trials=args.n_trials,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
