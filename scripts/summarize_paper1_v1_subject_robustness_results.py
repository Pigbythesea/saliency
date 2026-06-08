"""Summarize Paper 1 V1 subject-robustness outputs."""

from __future__ import annotations

import argparse
import csv
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
    decision_rows = subject_robustness_decision_rows(
        config=config,
        encoding_rows=encoding_rows,
        geometry_rows=geometry_rows,
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
    return outputs


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
