"""Compute Paper 1 V1 subject-robustness geometry scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hma.utils.config import load_yaml, save_yaml
from scripts.compute_matched_geometry import compute_matched_geometry


DEFAULT_CONFIG = Path("configs/paper1_experiment_v1.yaml")


def compute_paper1_v1_subject_robustness_geometry(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    methods_override: list[str] | None = None,
    models: list[str] | None = None,
    rois: list[str] | None = None,
    subset_sizes_override: list[int] | None = None,
    subset_seeds_override: list[int] | None = None,
    skip_existing: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Compute geometry for every confirmatory subject output directory."""
    config = load_yaml(config_path)
    scope_paths = write_subject_geometry_scope_configs(config)
    if progress:
        _print_progress(
            f"subject robustness geometry: starting {len(scope_paths)} subject scopes"
        )
    written: list[Path] = []
    for index, scope_path in enumerate(scope_paths, start=1):
        subject_config = load_yaml(scope_path)
        subject = str(subject_config["discovery_matrix"].get("subject_id", scope_path.stem))
        if progress:
            _print_progress(
                f"subject robustness geometry: subject {index}/{len(scope_paths)} {subject} start"
            )
        subject_written = (
            compute_matched_geometry(
                scope_path,
                methods_override=methods_override,
                models=models,
                rois=rois,
                subset_sizes_override=subset_sizes_override,
                subset_seeds_override=subset_seeds_override,
                skip_existing=skip_existing,
                progress=progress,
                progress_label=f"subject {subject}",
            )
        )
        written.extend(subject_written)
        if progress:
            _print_progress(
                f"subject robustness geometry: subject {index}/{len(scope_paths)} "
                f"{subject} done outputs={len(subject_written)}"
            )
    if progress:
        _print_progress(
            f"subject robustness geometry: finished outputs={len(written)}"
        )
    return written


def write_subject_geometry_scope_configs(config: dict[str, Any]) -> list[Path]:
    """Write per-subject temporary V1-style scope configs for geometry."""
    confirmatory = _confirmatory_matrix(config)
    summary_dir = Path(confirmatory["summary_output_dir"])
    scope_dir = summary_dir / "subject_robustness_geometry_scopes"
    scope_dir.mkdir(parents=True, exist_ok=True)

    subjects = [str(subject) for subject in confirmatory["subjects"]]
    rois = [str(roi) for roi in confirmatory["rois"]]
    paths: list[Path] = []
    for subject in subjects:
        subject_config = {
            "discovery_matrix": {
                "subject_id": subject,
                "output_root": str(confirmatory["output_root_template"]).format(subject=subject),
                "config_root": str(Path(confirmatory["config_root"]) / subject),
                "summary_output_dir": str(summary_dir / f"subject_robustness_{subject}"),
                "models": list(confirmatory["reduced_models"]),
                "roi_groups": {
                    "prf_visualrois": {
                        "roi_class": confirmatory.get("roi_class", "prf-visualrois"),
                        "manifest_path": str(confirmatory["manifest_path_template"]).format(
                            subject=subject
                        ),
                        "rois": rois,
                    }
                },
                "expected_cells": {
                    "models": len(confirmatory["reduced_models"]),
                    "rois": len(rois),
                    "model_roi_cells": len(confirmatory["reduced_models"]) * len(rois),
                },
                # Confirmatory Algonauts subjects have subject-specific image
                # availability. Do not carry subj01's 9841-image expectation
                # into geometry validation or summary completeness checks.
                "max_items": int(confirmatory.get("max_items") or 0),
            },
            "encoding": dict(config["encoding"]),
            "geometry": dict(config["geometry"]),
        }
        path = scope_dir / f"{subject}.yaml"
        save_yaml(subject_config, path)
        paths.append(path)
    return paths


def _confirmatory_matrix(config: dict[str, Any]) -> dict[str, Any]:
    confirmatory = config.get("confirmatory_matrix")
    if not isinstance(confirmatory, dict):
        raise ValueError("Paper 1 V1 config must contain confirmatory_matrix")
    return confirmatory


def _print_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--rois", nargs="+")
    parser.add_argument("--subset-sizes", nargs="+", type=int)
    parser.add_argument("--subset-seeds", nargs="+", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = compute_paper1_v1_subject_robustness_geometry(
        args.config,
        methods_override=args.methods,
        models=args.models,
        rois=args.rois,
        subset_sizes_override=args.subset_sizes,
        subset_seeds_override=args.subset_seeds,
        skip_existing=args.skip_existing,
        progress=not args.no_progress,
    )
    print(f"Wrote {len(written)} subject-robustness geometry score files")


if __name__ == "__main__":
    main()
