"""Generate and audit Paper 1 V1 ROI-expanded neural configs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hma.utils.config import load_yaml
from scripts.create_algonauts_manifest import create_algonauts_manifest
from scripts.create_neural_roi500_configs import (
    create_matched_full_subject_flatten_pca_configs,
)


DEFAULT_CONFIG_PATH = Path("configs/paper1_experiment_v1.yaml")
DEFAULT_AUDIT_PATH = Path("outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv")
STREAM_ROI_SLUGS = {
    "midventral": "midventral",
    "midlateral": "midlateral",
    "midparietal": "midparietal",
    "ventral": "ventral",
    "lateral": "lateral",
    "parietal": "parietal",
}
PRF_ROI_SLUGS = {
    "V1": "v1",
    "V2": "v2",
    "V3": "v3",
    "hV4": "hv4",
}
AUDIT_COLUMNS = ["check", "status", "detail"]


def generate_paper1_v1_roi_expanded_readiness(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overwrite_responses: bool = False,
    write_audit: bool = True,
) -> dict[str, Any]:
    """Generate the stream manifest, V1 neural configs, and readiness audit."""
    config = load_yaml(config_path)
    discovery = config["discovery_matrix"]
    encoding = config["encoding"]
    subject = discovery["subject_id"]
    roi_groups = discovery["roi_groups"]
    stream_group = roi_groups["streams"]
    stream_manifest_path = Path(stream_group["manifest_path"])

    create_algonauts_manifest(
        root="data/raw/nsd_algonauts",
        subject=subject,
        hemispheres=["lh", "rh"],
        roi_class=stream_group["roi_class"],
        roi_names=list(stream_group["rois"]),
        combine_hemispheres=True,
        output_manifest=stream_manifest_path,
        max_items=int(discovery["max_items"]),
        overwrite_responses=overwrite_responses,
        attach_noise_ceilings=False,
    )

    config_root = Path(discovery["config_root"])
    output_root = discovery["output_root"]
    models = list(discovery["models"])
    pca_components = int(encoding["pca_components"])
    ridge_alphas = [float(value) for value in encoding["ridge_alphas"]]
    validation_fraction = float(encoding["validation_fraction"])
    selection_primary_score = encoding["selection_primary_score"]

    written_configs: list[Path] = []
    for group_name, group in roi_groups.items():
        rois = list(group["rois"])
        roi_slugs = _roi_slugs_for_group(group_name, rois)
        written_configs.extend(
            create_matched_full_subject_flatten_pca_configs(
                output_dir=config_root,
                manifest_path=group["manifest_path"],
                output_root=output_root,
                models=models,
                rois=rois,
                max_items=int(discovery["max_items"]),
                pca_components=pca_components,
                roi_slugs=roi_slugs,
                ridge_alphas=ridge_alphas,
                validation_fraction=validation_fraction,
                selection_primary_score=selection_primary_score,
            )
        )

    audit_rows = audit_paper1_v1_roi_expanded_readiness(
        config=config,
        config_paths=written_configs,
    )
    if write_audit:
        audit_path = Path(discovery["summary_output_dir"]) / DEFAULT_AUDIT_PATH.name
        write_audit_rows(audit_path, audit_rows)
    return {
        "stream_manifest": stream_manifest_path,
        "config_paths": written_configs,
        "audit_rows": audit_rows,
    }


def audit_paper1_v1_roi_expanded_readiness(
    *,
    config: dict[str, Any],
    config_paths: list[Path] | None = None,
) -> list[dict[str, str]]:
    """Return pass/fail audit rows for the Paper 1 V1 readiness artifacts."""
    discovery = config["discovery_matrix"]
    roi_groups = discovery["roi_groups"]
    expected_models = list(discovery["models"])
    expected_rois = [
        roi for group in roi_groups.values() for roi in group["rois"]
    ]
    expected_cells = int(discovery["expected_cells"]["model_roi_cells"])
    config_root = Path(discovery["config_root"])
    summary_rows: list[dict[str, str]] = []

    stream_rois = list(roi_groups["streams"]["rois"])
    stream_manifest = Path(roi_groups["streams"]["manifest_path"])
    prf_manifest = Path(roi_groups["prf_visualrois"]["manifest_path"])
    _add_check(
        summary_rows,
        "stream_manifest_exists",
        stream_manifest.is_file(),
        str(stream_manifest),
    )
    stream_rows = _read_manifest_rows(stream_manifest) if stream_manifest.is_file() else []
    prf_rows = _read_manifest_rows(prf_manifest) if prf_manifest.is_file() else []

    stream_roi_set = sorted({row["roi"] for row in stream_rows})
    _add_check(
        summary_rows,
        "stream_rois_present",
        stream_roi_set == sorted(stream_rois),
        ",".join(stream_roi_set),
    )
    roi_dims = _response_dims_by_roi(stream_rows)
    _add_check(
        summary_rows,
        "stream_roi_response_dims_nonzero",
        all(roi_dims.get(roi, 0) > 0 for roi in stream_rois),
        ",".join(f"{roi}:{roi_dims.get(roi, 0)}" for roi in stream_rois),
    )
    _add_check(
        summary_rows,
        "stream_prf_image_order_match",
        _image_order(stream_rows, stream_rois[0]) == _image_order(prf_rows, "V1"),
        f"stream={len(_image_order(stream_rows, stream_rois[0]))};prf={len(_image_order(prf_rows, 'V1'))}",
    )

    paths = config_paths if config_paths is not None else sorted(config_root.glob("*.yaml"))
    configs = [load_yaml(path) for path in paths if path.is_file()]
    _add_check(
        summary_rows,
        "expected_config_count",
        len(paths) == expected_cells,
        f"found={len(paths)};expected={expected_cells}",
    )

    found_models = sorted({cfg["model"]["name"] for cfg in configs})
    found_rois = sorted({cfg["dataset"]["roi"] for cfg in configs})
    _add_check(
        summary_rows,
        "model_coverage",
        found_models == sorted(expected_models),
        ",".join(found_models),
    )
    _add_check(
        summary_rows,
        "roi_coverage",
        found_rois == sorted(expected_rois),
        ",".join(found_rois),
    )
    _add_check(
        summary_rows,
        "config_manifest_paths_match_roi_groups",
        _config_manifest_paths_match(configs, roi_groups),
        "manifest paths match configured ROI groups",
    )
    _add_check(
        summary_rows,
        "config_output_dirs_match_v1_root",
        all(str(cfg["output"]["dir"]).startswith(discovery["output_root"] + "/") for cfg in configs),
        discovery["output_root"],
    )
    _add_check(
        summary_rows,
        "encoding_settings_match_v1_config",
        _encoding_settings_match(configs, config["encoding"], int(discovery["max_items"])),
        "flatten_pca validation-selection settings",
    )
    return summary_rows


def write_audit_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _roi_slugs_for_group(group_name: str, rois: list[str]) -> dict[str, str]:
    if group_name == "prf_visualrois":
        return {roi: PRF_ROI_SLUGS[roi] for roi in rois}
    if group_name == "streams":
        return {roi: STREAM_ROI_SLUGS[roi] for roi in rois}
    raise ValueError(f"Unsupported Paper 1 V1 ROI group: {group_name}")


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _response_dims_by_roi(rows: list[dict[str, str]]) -> dict[str, int]:
    dims: dict[str, int] = {}
    root = Path("data/raw/nsd_algonauts")
    for row in rows:
        roi = row["roi"]
        if roi in dims:
            continue
        response = np.load(root / row["roi_response_path"], mmap_mode="r")
        dims[roi] = int(response.shape[0]) if response.ndim > 0 else int(response.size)
    return dims


def _image_order(rows: list[dict[str, str]], roi: str) -> list[str]:
    return [row["image_id"] for row in rows if row["roi"] == roi]


def _config_manifest_paths_match(configs: list[dict[str, Any]], roi_groups: dict[str, Any]) -> bool:
    manifest_by_roi = {
        roi: group["manifest_path"]
        for group in roi_groups.values()
        for roi in group["rois"]
    }
    return all(
        cfg["dataset"]["manifest_path"] == manifest_by_roi.get(cfg["dataset"]["roi"])
        for cfg in configs
    )


def _encoding_settings_match(
    configs: list[dict[str, Any]],
    encoding: dict[str, Any],
    max_items: int,
) -> bool:
    expected_alphas = [float(value) for value in encoding["ridge_alphas"]]
    for cfg in configs:
        neural = cfg["neural"]
        if cfg["dataset"]["max_items"] != max_items:
            return False
        if neural["feature_reduction"] != "flatten_pca":
            return False
        if int(neural["pca_components"]) != int(encoding["pca_components"]):
            return False
        if neural["pca_solver"] != encoding["pca_solver"]:
            return False
        if bool(neural["pca_whiten"]) != bool(encoding["pca_whiten"]):
            return False
        if [float(value) for value in neural["ridge_alphas"]] != expected_alphas:
            return False
        if float(neural["validation_fraction"]) != float(encoding["validation_fraction"]):
            return False
        if neural["selection"]["primary_score"] != encoding["selection_primary_score"]:
            return False
        if neural["rsa"]["enabled"] is not False:
            return False
    return True


def _add_check(rows: list[dict[str, str]], check: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "check": check,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Paper 1 V1 ROI-expanded manifest/config readiness artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overwrite-responses", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_paper1_v1_roi_expanded_readiness(
        config_path=args.config,
        overwrite_responses=args.overwrite_responses,
    )
    print(result["stream_manifest"])
    for path in result["config_paths"]:
        print(path)


if __name__ == "__main__":
    main()
