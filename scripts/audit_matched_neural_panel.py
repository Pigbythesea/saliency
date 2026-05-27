"""Audit matched full-image-count neural panel artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from hma.utils.config import load_yaml


EXPECTED_MODELS = [
    "resnet50",
    "convnext_tiny",
    "deit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
]
ROI_SLUGS = [("V1", "v1"), ("V2", "v2"), ("V3", "v3"), ("hV4", "hv4")]
REQUIRED_FILES = [
    "encoding_scores.csv",
    "encoding_target_scores.csv",
    "metadata.json",
    "feature_reduction_metadata.json",
]
OPTIONAL_FILES = ["selection_candidates.csv", "selection_artifact.json"]


def audit_matched_neural_panel(
    *,
    config_dir: str | Path = "configs/experiments/neural_subj01_full",
    output_csv: str | Path = "outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv",
    skip_csv: str | Path | None = None,
    models: list[str] | None = None,
    suffix: str = "flatten_pca_validation_selection_full",
) -> Path:
    """Write an artifact audit CSV for the matched full-image neural panel."""
    config_root = Path(config_dir)
    skip_reasons = _load_skip_reasons(skip_csv)
    rows: list[dict[str, Any]] = []
    for model in models or EXPECTED_MODELS:
        for roi, slug in ROI_SLUGS:
            run_name = f"{model}_{slug}_{suffix}"
            config_path = config_root / f"{run_name}.yaml"
            skip_reason = skip_reasons.get((model, roi), "")
            output_dir = _output_dir_from_config(config_path)
            missing_required = [
                filename
                for filename in REQUIRED_FILES
                if output_dir is None or not (output_dir / filename).is_file()
            ]
            optional_present = [
                filename
                for filename in OPTIONAL_FILES
                if output_dir is not None and (output_dir / filename).is_file()
            ]
            if skip_reason:
                status = "explicitly_skipped"
            elif output_dir is None or not output_dir.exists():
                status = "missing"
            elif missing_required:
                status = "incomplete"
            else:
                status = _complete_status(output_dir)
            rows.append(
                {
                    "model": model,
                    "roi": roi,
                    "run_name": run_name,
                    "status": status,
                    "skip_reason": skip_reason,
                    "config_path": str(config_path),
                    "config_exists": str(config_path.is_file()).lower(),
                    "output_dir": "" if output_dir is None else str(output_dir),
                    "missing_required_files": " ".join(missing_required),
                    "optional_files_present": " ".join(optional_present),
                    "feature_reduction": _metadata_value(output_dir, "feature_reduction"),
                    "num_items": _metadata_value(output_dir, "num_items"),
                    "selected_layer": _metadata_value(output_dir, "selected_layer"),
                    "selected_ridge_alpha": _metadata_value(output_dir, "selected_ridge_alpha"),
                }
            )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _load_skip_reasons(path: str | Path | None) -> dict[tuple[str, str], str]:
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (str(row.get("model", "")), str(row.get("roi", ""))): str(
                row.get("skip_reason", "")
            )
            for row in rows
            if row.get("model") and row.get("roi") and row.get("skip_reason")
        }


def _output_dir_from_config(config_path: Path) -> Path | None:
    if not config_path.is_file():
        return None
    config = load_yaml(config_path)
    output_dir = config.get("output", {}).get("dir", "")
    return Path(output_dir) if output_dir else None


def _metadata_value(output_dir: Path | None, key: str) -> Any:
    if output_dir is None:
        return ""
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return ""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "metadata_unreadable"
    return metadata.get(key, "")


def _complete_status(output_dir: Path) -> str:
    feature_reduction = str(_metadata_value(output_dir, "feature_reduction"))
    num_items = str(_metadata_value(output_dir, "num_items"))
    if feature_reduction != "flatten_pca" or num_items != "9841":
        return "incomplete"
    return "complete"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit matched full-image-count flatten_pca neural panel artifacts."
    )
    parser.add_argument("--config-dir", default="configs/experiments/neural_subj01_full")
    parser.add_argument(
        "--output-csv",
        default="outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv",
    )
    parser.add_argument("--skip-csv", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--suffix", default="flatten_pca_validation_selection_full")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = audit_matched_neural_panel(
        config_dir=args.config_dir,
        output_csv=args.output_csv,
        skip_csv=args.skip_csv,
        models=args.models,
        suffix=args.suffix,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
