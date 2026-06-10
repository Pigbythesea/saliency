"""Summarize Paper 1 V1 subject-robustness outputs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hma.experiments import summarize_neural_roi_results
from hma.utils.config import load_yaml
from scripts.compute_paper1_v1_subject_robustness_geometry import (
    write_subject_geometry_scope_configs,
)


DEFAULT_CONFIG = Path("configs/paper1_experiment_v1.yaml")
DINOV2_MODEL = "vit_small_patch14_dinov2"
RESNET_MODEL = "resnet50"
COMPARISON_LABEL = f"{DINOV2_MODEL}_minus_{RESNET_MODEL}"
BOOTSTRAP_SEED = 123
BOOTSTRAP_RESAMPLES = 2000
DECISION_FIELDNAMES = [
    "subject_id",
    "expected_encoding_models",
    "completed_encoding_models",
    "expected_geometry_models",
    "completed_geometry_models",
    "baseline_encoding_leader",
    "subject_encoding_leader",
    "encoding_leader_matches_subj01",
    "baseline_cka_leader",
    "subject_cka_leader",
    "cka_leader_matches_subj01",
    "decision_label",
    "detail",
]
ENCODING_MARGIN_FIELDNAMES = [
    "subject_id",
    "roi",
    "comparison",
    "target_score_field",
    "uncertainty_scope",
    "n_paired_targets",
    "num_rois",
    "mean_margin",
    "median_margin",
    "positive_target_count",
    "positive_target_fraction",
    "bootstrap_resamples",
    "bootstrap_seed",
    "ci95_low",
    "ci95_high",
    "support_label",
    "detail",
]
GEOMETRY_MARGIN_FIELDNAMES = [
    "subject_id",
    "roi",
    "geometry_method",
    "geometry_method_family",
    "comparison",
    "n_paired_rows",
    "num_subjects",
    "num_rois",
    "mean_margin",
    "median_margin",
    "min_margin",
    "max_margin",
    "positive_count",
    "positive_fraction",
    "support_label",
    "detail",
]
UNCERTAINTY_DECISION_FIELDNAMES = [
    *DECISION_FIELDNAMES,
    "encoding_margin_label",
    "encoding_mean_margin",
    "encoding_ci95_low",
    "encoding_ci95_high",
    "encoding_n_paired_targets",
    "geometry_cka_margin_label",
    "geometry_cka_mean_margin",
    "geometry_subset_rsa_supported_methods",
    "geometry_subset_rsa_tested_methods",
    "geometry_subset_rsa_support_fraction",
    "uncertainty_decision_label",
    "uncertainty_detail",
]


def summarize_paper1_v1_subject_robustness_results(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """Write per-subject summaries, combined tables, and robustness decisions."""
    config = load_yaml(config_path)
    confirmatory = _confirmatory_matrix(config)
    summary_dir = Path(confirmatory["summary_output_dir"])
    summary_dir.mkdir(parents=True, exist_ok=True)
    scope_paths = write_subject_geometry_scope_configs(config)

    behavioral_csv = config.get("baseline_inputs", {}).get("behavioral_csv")
    if behavioral_csv and not Path(behavioral_csv).is_file():
        behavioral_csv = None

    per_subject_outputs: dict[str, dict[str, Path]] = {}
    for scope_path in scope_paths:
        subject_config = load_yaml(scope_path)
        discovery = subject_config["discovery_matrix"]
        subject = str(discovery["subject_id"])
        config_root = Path(discovery["config_root"])
        output_root = Path(discovery["output_root"])
        input_dirs = [output_root / path.stem for path in sorted(config_root.glob("*.yaml"))]
        per_subject_outputs[subject] = summarize_neural_roi_results(
            input_dirs,
            discovery["summary_output_dir"],
            behavioral_csv=behavioral_csv,
            scope_config=scope_path,
        )

    encoding_rows = _collect_subject_rows(
        per_subject_outputs,
        "roi_expanded_encoding_model_rankings",
    )
    geometry_rows = _collect_subject_rows(
        per_subject_outputs,
        "roi_expanded_geometry_model_rankings",
    )
    sensitivity_rows = _collect_subject_rows(
        per_subject_outputs,
        "roi_expanded_geometry_method_sensitivity_decisions",
    )
    encoding_target_rows = _collect_subject_rows(
        per_subject_outputs,
        "combined_encoding_target_scores",
    )
    combined_geometry_rows = _collect_subject_rows(
        per_subject_outputs,
        "combined_geometry_scores",
    )
    decision_rows = subject_robustness_decision_rows(
        config=config,
        encoding_rows=encoding_rows,
        geometry_rows=geometry_rows,
    )
    subjects = [str(subject) for subject in confirmatory["subjects"]]
    rois = [str(roi) for roi in confirmatory["rois"]]
    encoding_margin_rows = subject_robustness_encoding_margin_uncertainty_rows(
        encoding_target_rows,
        subjects=subjects,
        rois=rois,
    )
    geometry_margin_rows = subject_robustness_geometry_margin_summary_rows(
        combined_geometry_rows,
        subjects=subjects,
        rois=rois,
    )
    uncertainty_decision_rows = subject_robustness_uncertainty_decision_rows(
        decision_rows=decision_rows,
        encoding_margin_rows=encoding_margin_rows,
        geometry_margin_rows=geometry_margin_rows,
        subjects=subjects,
    )

    outputs = {
        "subject_robustness_encoding_model_rankings": (
            summary_dir / "subject_robustness_encoding_model_rankings.csv"
        ),
        "subject_robustness_geometry_model_rankings": (
            summary_dir / "subject_robustness_geometry_model_rankings.csv"
        ),
        "subject_robustness_geometry_method_sensitivity_decisions": (
            summary_dir / "subject_robustness_geometry_method_sensitivity_decisions.csv"
        ),
        "subject_robustness_decisions": summary_dir / "subject_robustness_decisions.csv",
        "subject_robustness_encoding_margin_uncertainty": (
            summary_dir / "subject_robustness_encoding_margin_uncertainty.csv"
        ),
        "subject_robustness_geometry_margin_summary": (
            summary_dir / "subject_robustness_geometry_margin_summary.csv"
        ),
        "subject_robustness_uncertainty_decisions": (
            summary_dir / "subject_robustness_uncertainty_decisions.csv"
        ),
    }
    _write_rows(outputs["subject_robustness_encoding_model_rankings"], encoding_rows)
    _write_rows(outputs["subject_robustness_geometry_model_rankings"], geometry_rows)
    _write_rows(
        outputs["subject_robustness_geometry_method_sensitivity_decisions"],
        sensitivity_rows,
    )
    _write_rows(
        outputs["subject_robustness_decisions"],
        decision_rows,
        fieldnames=DECISION_FIELDNAMES,
    )
    _write_rows(
        outputs["subject_robustness_encoding_margin_uncertainty"],
        encoding_margin_rows,
        fieldnames=ENCODING_MARGIN_FIELDNAMES,
    )
    _write_rows(
        outputs["subject_robustness_geometry_margin_summary"],
        geometry_margin_rows,
        fieldnames=GEOMETRY_MARGIN_FIELDNAMES,
    )
    _write_rows(
        outputs["subject_robustness_uncertainty_decisions"],
        uncertainty_decision_rows,
        fieldnames=UNCERTAINTY_DECISION_FIELDNAMES,
    )
    return outputs


def subject_robustness_encoding_margin_uncertainty_rows(
    target_rows: list[dict[str, str]],
    *,
    subjects: list[str],
    rois: list[str],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, str]]:
    """Write DINOv2-minus-ResNet paired target-level encoding margins."""
    paired_by_group: dict[tuple[str, str], list[float]] = {}
    target_pairs: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in target_rows:
        model = row.get("model", "")
        if model not in {DINOV2_MODEL, RESNET_MODEL}:
            continue
        if not _valid_target_row(row):
            continue
        score = _optional_float(row.get("pearson_r"))
        if score is None:
            continue
        key = (str(row.get("subject_id", "")), str(row.get("roi", "")), str(row.get("target_index", "")))
        target_pairs.setdefault(key, {})[model] = score

    for (subject, roi, _target), scores in target_pairs.items():
        if DINOV2_MODEL not in scores or RESNET_MODEL not in scores:
            continue
        paired_by_group.setdefault((subject, roi), []).append(
            scores[DINOV2_MODEL] - scores[RESNET_MODEL]
        )

    rows: list[dict[str, str]] = []
    all_margins: list[float] = []
    for subject in subjects:
        subject_margins: list[float] = []
        present_rois = 0
        for roi in rois:
            margins = paired_by_group.get((subject, roi), [])
            if margins:
                present_rois += 1
            subject_margins.extend(margins)
            rows.append(
                _encoding_margin_row(
                    subject,
                    roi,
                    margins,
                    num_rois=1 if margins else 0,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
            )
        all_margins.extend(subject_margins)
        rows.append(
            _encoding_margin_row(
                subject,
                "mean_prf_visualrois",
                subject_margins,
                num_rois=present_rois,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
        )

    rows.append(
        _encoding_margin_row(
            "all_confirmatory_subjects",
            "all_prf_visualrois",
            all_margins,
            num_rois=len({(subject, roi) for subject in subjects for roi in rois if paired_by_group.get((subject, roi))}),
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
    )
    return rows


def subject_robustness_geometry_margin_summary_rows(
    geometry_rows: list[dict[str, str]],
    *,
    subjects: list[str],
    rois: list[str],
) -> list[dict[str, str]]:
    """Write deterministic DINOv2-minus-ResNet geometry margins."""
    score_pairs: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in geometry_rows:
        model = row.get("model", "")
        if model not in {DINOV2_MODEL, RESNET_MODEL}:
            continue
        if not _valid_geometry_row(row):
            continue
        score = _optional_float(row.get("score"))
        if score is None:
            continue
        key = (
            str(row.get("subject_id", "")),
            str(row.get("roi", "")),
            str(row.get("geometry_method", "")),
        )
        score_pairs.setdefault(key, {})[model] = score

    margins_by_subject_method: dict[tuple[str, str], list[float]] = {}
    margins_by_method: dict[str, list[float]] = {}
    method_names = sorted({key[2] for key in score_pairs})
    rows: list[dict[str, str]] = []
    for subject in subjects:
        for roi in rois:
            for method in method_names:
                pair = score_pairs.get((subject, roi, method), {})
                margins: list[float] = []
                if DINOV2_MODEL in pair and RESNET_MODEL in pair:
                    margin = pair[DINOV2_MODEL] - pair[RESNET_MODEL]
                    margins.append(margin)
                    margins_by_subject_method.setdefault((subject, method), []).append(margin)
                    margins_by_method.setdefault(method, []).append(margin)
                rows.append(
                    _geometry_margin_row(
                        subject,
                        roi,
                        method,
                        margins,
                        num_subjects=1 if margins else 0,
                        num_rois=1 if margins else 0,
                    )
                )

        for method in method_names:
            margins = margins_by_subject_method.get((subject, method), [])
            rows.append(
                _geometry_margin_row(
                    subject,
                    "mean_prf_visualrois",
                    method,
                    margins,
                    num_subjects=1 if margins else 0,
                    num_rois=len(margins),
                )
            )

    for method in method_names:
        margins = margins_by_method.get(method, [])
        rows.append(
            _geometry_margin_row(
                "all_confirmatory_subjects",
                "all_prf_visualrois",
                method,
                margins,
                num_subjects=len(
                    {
                        subject
                        for subject in subjects
                        if margins_by_subject_method.get((subject, method))
                    }
                ),
                num_rois=len(margins),
            )
        )
    return rows


def subject_robustness_uncertainty_decision_rows(
    *,
    decision_rows: list[dict[str, str]],
    encoding_margin_rows: list[dict[str, str]],
    geometry_margin_rows: list[dict[str, str]],
    subjects: list[str],
) -> list[dict[str, str]]:
    """Combine rank-only decisions with margin uncertainty decisions."""
    encoding_by_subject = {
        row.get("subject_id", ""): row
        for row in encoding_margin_rows
        if row.get("roi") in {"mean_prf_visualrois", "all_prf_visualrois"}
    }
    cka_by_subject = {
        row.get("subject_id", ""): row
        for row in geometry_margin_rows
        if row.get("roi") in {"mean_prf_visualrois", "all_prf_visualrois"}
        and row.get("geometry_method") == "linear_cka_full9841"
    }
    subset_by_subject: dict[str, list[dict[str, str]]] = {}
    for row in geometry_margin_rows:
        if row.get("roi") not in {"mean_prf_visualrois", "all_prf_visualrois"}:
            continue
        if not row.get("geometry_method", "").startswith("subset_rsa"):
            continue
        subset_by_subject.setdefault(row.get("subject_id", ""), []).append(row)

    rank_by_subject = {row.get("subject_id", ""): row for row in decision_rows}
    subject_final_labels: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for subject in [*subjects, "all_confirmatory_subjects"]:
        base = dict(rank_by_subject.get(subject, {"subject_id": subject}))
        encoding = encoding_by_subject.get(subject, {})
        cka = cka_by_subject.get(subject, {})
        subset_rows = subset_by_subject.get(subject, [])
        subset_tested = len(subset_rows)
        subset_supported = sum(
            1 for row in subset_rows if row.get("support_label") == "dinov2_supported"
        )
        subset_fraction = subset_supported / subset_tested if subset_tested else None
        final_label = _uncertainty_final_label(
            encoding_label=encoding.get("support_label", ""),
            cka_label=cka.get("support_label", ""),
            subject=subject,
            subject_final_labels=subject_final_labels,
        )
        if subject != "all_confirmatory_subjects":
            subject_final_labels[subject] = final_label
        elif final_label != "uncertainty_incomplete":
            subject_labels = [subject_final_labels.get(item, "") for item in subjects]
            if len(set(subject_labels)) > 1 and cka.get("support_label") == "dinov2_supported":
                final_label = "geometry_replicated_encoding_ambiguous"
        base.update(
            {
                "encoding_margin_label": encoding.get("support_label", ""),
                "encoding_mean_margin": encoding.get("mean_margin", ""),
                "encoding_ci95_low": encoding.get("ci95_low", ""),
                "encoding_ci95_high": encoding.get("ci95_high", ""),
                "encoding_n_paired_targets": encoding.get("n_paired_targets", ""),
                "geometry_cka_margin_label": cka.get("support_label", ""),
                "geometry_cka_mean_margin": cka.get("mean_margin", ""),
                "geometry_subset_rsa_supported_methods": str(subset_supported),
                "geometry_subset_rsa_tested_methods": str(subset_tested),
                "geometry_subset_rsa_support_fraction": _format_float(subset_fraction),
                "uncertainty_decision_label": final_label,
                "uncertainty_detail": _uncertainty_detail(
                    encoding.get("support_label", ""),
                    cka.get("support_label", ""),
                    subset_supported,
                    subset_tested,
                ),
            }
        )
        rows.append(base)
    return rows


def _encoding_margin_row(
    subject: str,
    roi: str,
    margins: list[float],
    *,
    num_rois: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, str]:
    if not margins:
        return {
            "subject_id": subject,
            "roi": roi,
            "comparison": COMPARISON_LABEL,
            "target_score_field": "pearson_r",
            "uncertainty_scope": "target_level_bootstrap",
            "n_paired_targets": "0",
            "num_rois": str(num_rois),
            "mean_margin": "",
            "median_margin": "",
            "positive_target_count": "0",
            "positive_target_fraction": "",
            "bootstrap_resamples": str(bootstrap_resamples),
            "bootstrap_seed": str(bootstrap_seed),
            "ci95_low": "",
            "ci95_high": "",
            "support_label": "incomplete",
            "detail": "missing paired DINOv2 and ResNet target rows",
        }

    ci_low, ci_high = _bootstrap_mean_ci(
        margins,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + _stable_seed(subject, roi),
    )
    mean_margin = _mean(margins)
    positive_count = sum(1 for margin in margins if margin > 0.0)
    if mean_margin > 0.0 and ci_low > 0.0:
        label = "dinov2_supported"
        detail = "mean target margin is positive and bootstrap CI excludes zero"
    elif mean_margin < 0.0 and ci_high < 0.0:
        label = "resnet50_supported"
        detail = "mean target margin is negative and bootstrap CI excludes zero"
    else:
        label = "ambiguous"
        detail = "target-level bootstrap CI overlaps zero"
    return {
        "subject_id": subject,
        "roi": roi,
        "comparison": COMPARISON_LABEL,
        "target_score_field": "pearson_r",
        "uncertainty_scope": "target_level_bootstrap",
        "n_paired_targets": str(len(margins)),
        "num_rois": str(num_rois),
        "mean_margin": _format_float(mean_margin),
        "median_margin": _format_float(_median(margins)),
        "positive_target_count": str(positive_count),
        "positive_target_fraction": _format_float(positive_count / len(margins)),
        "bootstrap_resamples": str(bootstrap_resamples),
        "bootstrap_seed": str(bootstrap_seed),
        "ci95_low": _format_float(ci_low),
        "ci95_high": _format_float(ci_high),
        "support_label": label,
        "detail": detail,
    }


def _geometry_margin_row(
    subject: str,
    roi: str,
    method: str,
    margins: list[float],
    *,
    num_subjects: int,
    num_rois: int,
) -> dict[str, str]:
    if not margins:
        return {
            "subject_id": subject,
            "roi": roi,
            "geometry_method": method,
            "geometry_method_family": _geometry_method_family(method),
            "comparison": COMPARISON_LABEL,
            "n_paired_rows": "0",
            "num_subjects": str(num_subjects),
            "num_rois": str(num_rois),
            "mean_margin": "",
            "median_margin": "",
            "min_margin": "",
            "max_margin": "",
            "positive_count": "0",
            "positive_fraction": "",
            "support_label": "incomplete",
            "detail": "missing paired DINOv2 and ResNet geometry rows",
        }

    mean_margin = _mean(margins)
    positive_count = sum(1 for margin in margins if margin > 0.0)
    positive_fraction = positive_count / len(margins)
    if mean_margin > 0.0:
        label = "dinov2_supported"
        detail = "mean DINOv2-minus-ResNet geometry margin is positive"
    elif mean_margin < 0.0:
        label = "resnet50_supported"
        detail = "mean DINOv2-minus-ResNet geometry margin is negative"
    else:
        label = "ambiguous"
        detail = "mean DINOv2-minus-ResNet geometry margin is zero"
    return {
        "subject_id": subject,
        "roi": roi,
        "geometry_method": method,
        "geometry_method_family": _geometry_method_family(method),
        "comparison": COMPARISON_LABEL,
        "n_paired_rows": str(len(margins)),
        "num_subjects": str(num_subjects),
        "num_rois": str(num_rois),
        "mean_margin": _format_float(mean_margin),
        "median_margin": _format_float(_median(margins)),
        "min_margin": _format_float(min(margins)),
        "max_margin": _format_float(max(margins)),
        "positive_count": str(positive_count),
        "positive_fraction": _format_float(positive_fraction),
        "support_label": label,
        "detail": detail,
    }


def _uncertainty_final_label(
    *,
    encoding_label: str,
    cka_label: str,
    subject: str,
    subject_final_labels: dict[str, str],
) -> str:
    if encoding_label in {"", "incomplete"} or cka_label in {"", "incomplete"}:
        return "uncertainty_incomplete"
    if cka_label != "dinov2_supported":
        return "geometry_not_replicated_encoding_ambiguous"
    if subject == "all_confirmatory_subjects" and subject_final_labels:
        labels = set(subject_final_labels.values())
        if len(labels) == 1:
            return next(iter(labels))
        return "geometry_replicated_encoding_ambiguous"
    if encoding_label == "dinov2_supported":
        return "geometry_replicated_encoding_supported"
    if encoding_label == "resnet50_supported":
        return "geometry_replicated_encoding_resnet_supported"
    return "geometry_replicated_encoding_ambiguous"


def _uncertainty_detail(
    encoding_label: str,
    cka_label: str,
    subset_supported: int,
    subset_tested: int,
) -> str:
    if encoding_label in {"", "incomplete"} or cka_label in {"", "incomplete"}:
        return "missing required paired encoding or CKA margin rows"
    return (
        f"encoding={encoding_label}; cka={cka_label}; "
        f"subset_rsa_supported={subset_supported}/{subset_tested}"
    )


def subject_robustness_decision_rows(
    *,
    config: dict[str, Any],
    encoding_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compare confirmatory subjects against the frozen subj01 V1 leaders."""
    confirmatory = _confirmatory_matrix(config)
    subjects = [str(subject) for subject in confirmatory["subjects"]]
    expected_models = len(confirmatory["reduced_models"])
    baseline_encoding = _leader_from_csv(
        Path(config["discovery_matrix"]["summary_output_dir"])
        / "roi_expanded_encoding_model_rankings.csv",
        rank_key="rank_mean_noise_normalized",
        fallback_rank_key="rank_mean_encoding",
    )
    baseline_cka = _leader_from_csv(
        Path(config["discovery_matrix"]["summary_output_dir"])
        / "roi_expanded_geometry_model_rankings.csv",
        rank_key="rank_mean_geometry",
        method="linear_cka_full9841",
    )

    rows: list[dict[str, str]] = []
    labels: list[str] = []
    for subject in subjects:
        subject_encoding = [
            row for row in encoding_rows if row.get("subject_id") == subject
        ]
        subject_geometry = [
            row
            for row in geometry_rows
            if row.get("subject_id") == subject
            and row.get("geometry_method") == "linear_cka_full9841"
        ]
        subject_encoding_leader = _leader_from_rows(
            subject_encoding,
            rank_key="rank_mean_noise_normalized",
            fallback_rank_key="rank_mean_encoding",
        )
        subject_cka_leader = _leader_from_rows(
            subject_geometry,
            rank_key="rank_mean_geometry",
        )
        completed_encoding_models = len({row.get("model", "") for row in subject_encoding})
        completed_geometry_models = len({row.get("model", "") for row in subject_geometry})
        encoding_match = bool(
            baseline_encoding
            and subject_encoding_leader
            and baseline_encoding == subject_encoding_leader
        )
        cka_match = bool(
            baseline_cka and subject_cka_leader and baseline_cka == subject_cka_leader
        )
        if (
            not baseline_encoding
            or not baseline_cka
            or completed_encoding_models < expected_models
            or completed_geometry_models < expected_models
        ):
            label = "incomplete"
            detail = "missing baseline or subject encoding/geometry model coverage"
        elif encoding_match and cka_match:
            label = "replicated"
            detail = "encoding and full-image CKA leaders match subj01"
        elif encoding_match or cka_match:
            label = "partial"
            detail = "only one of encoding or full-image CKA leaders matches subj01"
        else:
            label = "failed"
            detail = "encoding and full-image CKA leaders both differ from subj01"
        labels.append(label)
        rows.append(
            {
                "subject_id": subject,
                "expected_encoding_models": str(expected_models),
                "completed_encoding_models": str(completed_encoding_models),
                "expected_geometry_models": str(expected_models),
                "completed_geometry_models": str(completed_geometry_models),
                "baseline_encoding_leader": baseline_encoding,
                "subject_encoding_leader": subject_encoding_leader,
                "encoding_leader_matches_subj01": str(encoding_match).lower(),
                "baseline_cka_leader": baseline_cka,
                "subject_cka_leader": subject_cka_leader,
                "cka_leader_matches_subj01": str(cka_match).lower(),
                "decision_label": label,
                "detail": detail,
            }
        )

    rows.append(_overall_decision_row(labels, expected_models))
    return rows


def _overall_decision_row(labels: list[str], expected_models: int) -> dict[str, str]:
    if not labels or any(label == "incomplete" for label in labels):
        label = "incomplete"
    elif all(label == "replicated" for label in labels):
        label = "replicated"
    elif any(label in {"replicated", "partial"} for label in labels):
        label = "partial"
    else:
        label = "failed"
    return {
        "subject_id": "all_confirmatory_subjects",
        "expected_encoding_models": str(expected_models),
        "completed_encoding_models": "",
        "expected_geometry_models": str(expected_models),
        "completed_geometry_models": "",
        "baseline_encoding_leader": "",
        "subject_encoding_leader": "",
        "encoding_leader_matches_subj01": "",
        "baseline_cka_leader": "",
        "subject_cka_leader": "",
        "cka_leader_matches_subj01": "",
        "decision_label": label,
        "detail": "aggregate over subject-level decisions",
    }


def _collect_subject_rows(
    per_subject_outputs: dict[str, dict[str, Path]],
    output_key: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject, outputs in sorted(per_subject_outputs.items()):
        path = outputs.get(output_key)
        if path is None or not path.is_file():
            continue
        for row in _read_rows(path):
            updated = dict(row)
            updated["subject_id"] = subject
            rows.append(updated)
    return rows


def _leader_from_csv(
    path: Path,
    *,
    rank_key: str,
    fallback_rank_key: str | None = None,
    method: str | None = None,
) -> str:
    if not path.is_file():
        return ""
    return _leader_from_rows(
        _read_rows(path),
        rank_key=rank_key,
        fallback_rank_key=fallback_rank_key,
        method=method,
    )


def _leader_from_rows(
    rows: list[dict[str, str]],
    *,
    rank_key: str,
    fallback_rank_key: str | None = None,
    method: str | None = None,
) -> str:
    candidates = rows
    if method is not None:
        candidates = [row for row in candidates if row.get("geometry_method") == method]
    candidates = [row for row in candidates if row.get("model")]
    if not candidates:
        return ""
    ranking_key = rank_key
    if not any(_is_rank_one(row.get(ranking_key, "")) for row in candidates):
        ranking_key = fallback_rank_key or rank_key
    rank_one = [row for row in candidates if _is_rank_one(row.get(ranking_key, ""))]
    if rank_one:
        return sorted(str(row["model"]) for row in rank_one)[0]
    return sorted(candidates, key=lambda row: _float_or_inf(row.get(ranking_key, "")))[0][
        "model"
    ]


def _is_rank_one(value: str) -> bool:
    try:
        return int(float(value)) == 1
    except (TypeError, ValueError):
        return False


def _float_or_inf(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _valid_target_row(row: dict[str, str]) -> bool:
    return (
        str(row.get("valid_prediction_variance", "true")).lower() != "false"
        and str(row.get("valid_target_variance", "true")).lower() != "false"
    )


def _valid_geometry_row(row: dict[str, str]) -> bool:
    valid = str(row.get("valid", "true")).lower()
    status = str(row.get("status", "ok")).lower()
    return valid != "false" and status not in {"error", "failed", "invalid"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _bootstrap_mean_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or resamples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _index in range(n)) / n)
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return means[low_index], means[high_index]


def _stable_seed(*parts: str) -> int:
    total = 0
    for part in parts:
        for index, char in enumerate(part):
            total += (index + 1) * ord(char)
    return total


def _format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def _geometry_method_family(method: str) -> str:
    if method == "linear_cka_full9841":
        return "linear_cka"
    if method.startswith("subset_rsa"):
        return "subset_rsa"
    return "other"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path,
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in names} for row in rows])


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for preferred in ["subject_id", "model", "geometry_method"]:
        if any(preferred in row for row in rows):
            names.append(preferred)
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names or ["subject_id"]


def _confirmatory_matrix(config: dict[str, Any]) -> dict[str, Any]:
    confirmatory = config.get("confirmatory_matrix")
    if not isinstance(confirmatory, dict):
        raise ValueError("Paper 1 V1 config must contain confirmatory_matrix")
    return confirmatory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = summarize_paper1_v1_subject_robustness_results(args.config)
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
