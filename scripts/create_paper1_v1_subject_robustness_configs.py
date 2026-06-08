"""Generate and audit Paper 1 V1 subject-robustness neural configs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hma.utils.config import load_yaml
from scripts.create_paper1_v1_roi_expanded_configs import PRF_ROI_SLUGS
from scripts.create_neural_roi500_configs import (
    create_matched_full_subject_flatten_pca_configs,
)


DEFAULT_CONFIG_PATH = Path("configs/paper1_experiment_v1.yaml")
DEFAULT_AUDIT_PATH = Path(
    "outputs/paper1_experiment_v1/summary/subject_robustness_artifact_audit.csv"
)
AUDIT_COLUMNS = ["check", "status", "detail"]


def generate_paper1_v1_subject_robustness_configs(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    write_audit: bool = True,
) -> dict[str, Any]:
    """Generate confirmatory subject-robustness configs and audit rows."""
    config = load_yaml(config_path)
    confirmatory = _confirmatory_matrix(config)
    encoding = config["encoding"]
    subjects = [str(subject) for subject in confirmatory["subjects"]]
    models = [str(model) for model in confirmatory["reduced_models"]]
    rois = [str(roi) for roi in confirmatory["rois"]]
    config_root = Path(confirmatory["config_root"])
    output_root_template = str(confirmatory["output_root_template"])
    manifest_template = str(confirmatory["manifest_path_template"])

    pca_components = int(encoding["pca_components"])
    ridge_alphas = [float(value) for value in encoding["ridge_alphas"]]
    validation_fraction = float(encoding["validation_fraction"])
    selection_primary_score = str(encoding["selection_primary_score"])

    config_paths: list[Path] = []
    for subject in subjects:
        subject_config_root = config_root / subject
        manifest_path = manifest_template.format(subject=subject)
        output_root = output_root_template.format(subject=subject)
        config_paths.extend(
            create_matched_full_subject_flatten_pca_configs(
                output_dir=subject_config_root,
                manifest_path=manifest_path,
                output_root=output_root,
                models=models,
                rois=rois,
                max_items=int(confirmatory.get("max_items") or config["discovery_matrix"]["max_items"]),
                pca_components=pca_components,
                roi_slugs=PRF_ROI_SLUGS,
                ridge_alphas=ridge_alphas,
                validation_fraction=validation_fraction,
                selection_primary_score=selection_primary_score,
                subject_id=subject,
            )
        )

    audit_rows = audit_paper1_v1_subject_robustness_configs(
        config=config,
        config_paths=config_paths,
    )
    if write_audit:
        audit_path = Path(confirmatory["summary_output_dir"]) / DEFAULT_AUDIT_PATH.name
        write_audit_rows(audit_path, audit_rows)
    return {"config_paths": config_paths, "audit_rows": audit_rows}


def audit_paper1_v1_subject_robustness_configs(
    *,
    config: dict[str, Any],
    config_paths: list[Path] | None = None,
) -> list[dict[str, str]]:
    """Return pass/fail audit rows for subject-robustness readiness."""
    confirmatory = _confirmatory_matrix(config)
    subjects = [str(subject) for subject in confirmatory["subjects"]]
    models = [str(model) for model in confirmatory["reduced_models"]]
    rois = [str(roi) for roi in confirmatory["rois"]]
    expected = confirmatory.get("expected_cells", {})
    expected_total = int(
        expected.get("subject_model_roi_cells") or len(subjects) * len(models) * len(rois)
    )
    expected_per_subject = int(expected.get("cells_per_subject") or len(models) * len(rois))
    config_root = Path(confirmatory["config_root"])
    manifest_template = str(confirmatory["manifest_path_template"])

    paths = config_paths if config_paths is not None else sorted(config_root.glob("*/*.yaml"))
    configs = [load_yaml(path) for path in paths if path.is_file()]
    cells = {
        (
            str(cfg["dataset"]["subject_id"]),
            str(cfg["model"]["name"]),
            str(cfg["dataset"]["roi"]),
        )
        for cfg in configs
    }
    expected_cells = {
        (subject, model, roi)
        for subject in subjects
        for model in models
        for roi in rois
    }
    rows: list[dict[str, str]] = []

    _add_check(
        rows,
        "subject_robustness_expected_config_count",
        len(paths) == expected_total,
        f"found={len(paths)};expected={expected_total}",
    )
    for subject in subjects:
        subject_paths = [path for path in paths if path.parent.name == subject]
        _add_check(
            rows,
            f"subject_robustness_{subject}_config_count",
            len(subject_paths) == expected_per_subject,
            f"found={len(subject_paths)};expected={expected_per_subject}",
        )
        manifest_path = Path(manifest_template.format(subject=subject))
        _add_check(
            rows,
            f"subject_robustness_{subject}_manifest_exists",
            manifest_path.is_file(),
            str(manifest_path),
        )

    found_subjects = sorted({subject for subject, _model, _roi in cells})
    found_models = sorted({model for _subject, model, _roi in cells})
    found_rois = sorted({roi for _subject, _model, roi in cells})
    _add_check(
        rows,
        "subject_robustness_subject_coverage",
        found_subjects == sorted(subjects),
        ",".join(found_subjects),
    )
    _add_check(
        rows,
        "subject_robustness_model_coverage",
        found_models == sorted(models),
        ",".join(found_models),
    )
    _add_check(
        rows,
        "subject_robustness_roi_coverage",
        found_rois == sorted(rois),
        ",".join(found_rois),
    )
    _add_check(
        rows,
        "subject_robustness_prf_only_scope",
        set(found_rois).issubset(set(PRF_ROI_SLUGS)),
        ",".join(found_rois),
    )
    missing_cells = sorted(expected_cells - cells)
    unexpected_cells = sorted(cells - expected_cells)
    _add_check(
        rows,
        "subject_robustness_exact_cell_coverage",
        not missing_cells and not unexpected_cells,
        (
            f"missing={len(missing_cells)};unexpected={len(unexpected_cells)}"
            + ("" if not missing_cells else f";first_missing={missing_cells[0]}")
        ),
    )
    _add_check(
        rows,
        "subject_robustness_manifest_paths_match_subjects",
        all(
            cfg["dataset"]["manifest_path"]
            == manifest_template.format(subject=cfg["dataset"]["subject_id"])
            for cfg in configs
        ),
        "subject-specific PRF manifests",
    )
    _add_check(
        rows,
        "subject_robustness_output_dirs_match_subjects",
        all(
            str(cfg["output"]["dir"]).startswith(
                str(confirmatory["output_root_template"]).format(
                    subject=cfg["dataset"]["subject_id"]
                )
                + "/"
            )
            for cfg in configs
        ),
        str(confirmatory["output_root_template"]),
    )
    _add_check(
        rows,
        "subject_robustness_encoding_settings_match_v1",
        _encoding_settings_match(configs, config["encoding"], int(config["discovery_matrix"]["max_items"])),
        "flatten_pca validation-selection settings",
    )
    return rows


def write_audit_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _confirmatory_matrix(config: dict[str, Any]) -> dict[str, Any]:
    confirmatory = config.get("confirmatory_matrix")
    if not isinstance(confirmatory, dict):
        raise ValueError("Paper 1 V1 config must contain confirmatory_matrix")
    return confirmatory


def _encoding_settings_match(
    configs: list[dict[str, Any]],
    encoding: dict[str, Any],
    max_items: int,
) -> bool:
    return all(
        cfg["dataset"]["max_items"] == max_items
        and cfg["neural"]["feature_reduction"] == encoding["method"]
        and cfg["neural"]["pca_components"] == encoding["pca_components"]
        and cfg["neural"]["selection"]["enabled"] is True
        and cfg["neural"]["selection"]["primary_score"] == encoding["selection_primary_score"]
        and cfg["neural"]["validation_fraction"] == encoding["validation_fraction"]
        and cfg["neural"]["rsa"]["enabled"] is False
        for cfg in configs
    )


def _add_check(rows: list[dict[str, str]], check: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "check": check,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_paper1_v1_subject_robustness_configs(config_path=args.config)
    print(f"Wrote {len(result['config_paths'])} subject-robustness configs")
    for path in result["config_paths"]:
        print(path)


if __name__ == "__main__":
    main()
