"""Compare single-layer and multi-layer learned-readout smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TARGET_FILE = "encoding_target_scores.csv"
SCORE_FILE = "encoding_scores.csv"
SUMMARY_FIELDS = [
    "baseline_label",
    "candidate_label",
    "baseline_dir",
    "candidate_dir",
    "single_dir",
    "multi_dir",
    "n_targets",
    "valid_noise_ceiling_targets",
    "raw_mean_single",
    "raw_mean_multi",
    "raw_mean_delta",
    "noise_normalized_mean_single",
    "noise_normalized_mean_multi",
    "noise_normalized_mean_delta",
    "raw_improved_targets",
    "raw_improved_fraction",
    "noise_normalized_improved_targets",
    "noise_normalized_improved_fraction",
    "recommendation",
    "rationale",
]
DELTA_FIELDS = [
    "target_index",
    "single_pearson_r",
    "multi_pearson_r",
    "pearson_r_delta",
    "single_noise_normalized_score",
    "multi_noise_normalized_score",
    "noise_normalized_delta",
    "noise_ceiling",
    "valid_noise_ceiling",
]


def compare_learned_readout_smokes(
    single_dir: str | Path,
    multi_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_label: str = "single",
    candidate_label: str = "multi",
) -> dict[str, Path]:
    """Write per-target and summary comparisons for two learned-readout smoke runs."""
    single_path = Path(single_dir)
    multi_path = Path(multi_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    single_targets = _read_targets(single_path / TARGET_FILE)
    multi_targets = _read_targets(multi_path / TARGET_FILE)
    target_rows = _target_delta_rows(single_targets, multi_targets)
    single_score = _read_single_row(single_path / SCORE_FILE)
    multi_score = _read_single_row(multi_path / SCORE_FILE)
    summary = _summary_row(
        single_dir=single_path,
        multi_dir=multi_path,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        target_rows=target_rows,
        single_score=single_score,
        multi_score=multi_score,
    )

    deltas_path = output_path / "target_deltas.csv"
    summary_csv_path = output_path / "summary.csv"
    summary_json_path = output_path / "summary.json"
    readme_path = output_path / "README.md"
    _write_rows(deltas_path, target_rows, DELTA_FIELDS)
    _write_rows(summary_csv_path, [summary], SUMMARY_FIELDS)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(readme_path, summary)
    return {
        "target_deltas": deltas_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "readme": readme_path,
    }


def _read_targets(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    by_target: dict[str, dict[str, str]] = {}
    for row in rows:
        target = str(row.get("target_index", ""))
        if not target:
            raise ValueError(f"Missing target_index in {path}")
        by_target[target] = row
    return by_target


def _target_delta_rows(
    single_rows: dict[str, dict[str, str]],
    multi_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if set(single_rows) != set(multi_rows):
        missing_single = sorted(set(multi_rows) - set(single_rows))
        missing_multi = sorted(set(single_rows) - set(multi_rows))
        raise ValueError(
            "Target-score files must contain the same target_index values: "
            f"missing_single={missing_single[:5]}, missing_multi={missing_multi[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for target in sorted(single_rows, key=lambda value: int(value)):
        single = single_rows[target]
        multi = multi_rows[target]
        single_raw = _float(single.get("pearson_r"))
        multi_raw = _float(multi.get("pearson_r"))
        single_norm = _optional_float(single.get("noise_normalized_score"))
        multi_norm = _optional_float(multi.get("noise_normalized_score"))
        rows.append(
            {
                "target_index": target,
                "single_pearson_r": single_raw,
                "multi_pearson_r": multi_raw,
                "pearson_r_delta": multi_raw - single_raw,
                "single_noise_normalized_score": "" if single_norm is None else single_norm,
                "multi_noise_normalized_score": "" if multi_norm is None else multi_norm,
                "noise_normalized_delta": (
                    "" if single_norm is None or multi_norm is None else multi_norm - single_norm
                ),
                "noise_ceiling": single.get("noise_ceiling", ""),
                "valid_noise_ceiling": single.get("valid_noise_ceiling", ""),
            }
        )
    return rows


def _summary_row(
    *,
    single_dir: Path,
    multi_dir: Path,
    baseline_label: str,
    candidate_label: str,
    target_rows: list[dict[str, Any]],
    single_score: dict[str, str],
    multi_score: dict[str, str],
) -> dict[str, Any]:
    raw_deltas = [_float(row["pearson_r_delta"]) for row in target_rows]
    normalized_deltas = [
        _float(row["noise_normalized_delta"])
        for row in target_rows
        if row["noise_normalized_delta"] != ""
    ]
    valid_normalized_count = len(normalized_deltas)
    raw_mean_delta = _float(multi_score.get("mean_score")) - _float(single_score.get("mean_score"))
    normalized_mean_delta = _float(
        multi_score.get("mean_noise_normalized_score")
    ) - _float(single_score.get("mean_noise_normalized_score"))
    raw_improved = sum(1 for value in raw_deltas if value > 0.0)
    normalized_improved = sum(1 for value in normalized_deltas if value > 0.0)
    recommendation, rationale = _recommendation(
        raw_mean_delta=raw_mean_delta,
        normalized_mean_delta=normalized_mean_delta,
        raw_improved_fraction=raw_improved / len(raw_deltas) if raw_deltas else 0.0,
        normalized_improved_fraction=(
            normalized_improved / valid_normalized_count
            if valid_normalized_count
            else 0.0
        ),
        candidate_label=candidate_label,
        candidate_dir=multi_dir,
    )
    return {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_dir": str(single_dir),
        "candidate_dir": str(multi_dir),
        "single_dir": str(single_dir),
        "multi_dir": str(multi_dir),
        "n_targets": len(target_rows),
        "valid_noise_ceiling_targets": valid_normalized_count,
        "raw_mean_single": _float(single_score.get("mean_score")),
        "raw_mean_multi": _float(multi_score.get("mean_score")),
        "raw_mean_delta": raw_mean_delta,
        "noise_normalized_mean_single": _float(
            single_score.get("mean_noise_normalized_score")
        ),
        "noise_normalized_mean_multi": _float(
            multi_score.get("mean_noise_normalized_score")
        ),
        "noise_normalized_mean_delta": normalized_mean_delta,
        "raw_improved_targets": raw_improved,
        "raw_improved_fraction": raw_improved / len(raw_deltas) if raw_deltas else 0.0,
        "noise_normalized_improved_targets": normalized_improved,
        "noise_normalized_improved_fraction": (
            normalized_improved / valid_normalized_count
            if valid_normalized_count
            else 0.0
        ),
        "recommendation": recommendation,
        "rationale": rationale,
    }


def _recommendation(
    *,
    raw_mean_delta: float,
    normalized_mean_delta: float,
    raw_improved_fraction: float,
    normalized_improved_fraction: float,
    candidate_label: str,
    candidate_dir: Path,
) -> tuple[str, str]:
    positive_label, negative_label = _recommendation_labels(candidate_label, candidate_dir)
    if (
        raw_mean_delta >= 0.0
        and normalized_mean_delta > 0.0
        and raw_improved_fraction >= 0.5
        and normalized_improved_fraction >= 0.5
    ):
        return (
            positive_label,
            "Candidate improves both aggregate metrics and at least half of targets.",
        )
    if normalized_mean_delta > 0.0 and normalized_improved_fraction >= 0.5:
        return (
            "inconclusive_do_not_prioritize_full_run",
            "Noise-normalized aggregate improves, but raw aggregate or raw target coverage does not.",
        )
    return (
        negative_label,
        "Candidate smoke does not show enough target-level support to justify a full run now.",
    )


def _recommendation_labels(candidate_label: str, candidate_dir: Path) -> tuple[str, str]:
    text = f"{candidate_label} {candidate_dir}".lower()
    if "voxel" in text:
        return "run_full_v1_voxel_specific", "freeze_current_dinov2_protocol"
    if "multi" in text:
        return "run_full_v1_multilayer", "move_to_voxel_specific_readout"
    safe_label = "_".join(candidate_label.lower().split()) or "candidate"
    return f"run_full_{safe_label}", "freeze_current_protocol"


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned Readout Smoke Comparison",
        "",
        "This compares a baseline learned-readout smoke against a candidate learned-readout smoke.",
        "",
        "## Summary",
        "",
        f"- Baseline: `{summary['baseline_label']}`.",
        f"- Candidate: `{summary['candidate_label']}`.",
        f"- Targets compared: {summary['n_targets']}.",
        f"- Valid noise-ceiling targets: {summary['valid_noise_ceiling_targets']}.",
        f"- Raw mean delta: {_format_float(summary['raw_mean_delta'])}.",
        f"- Noise-normalized mean delta: {_format_float(summary['noise_normalized_mean_delta'])}.",
        f"- Raw improved targets: {summary['raw_improved_targets']} ({_format_fraction(summary['raw_improved_fraction'])}).",
        "- Noise-normalized improved targets: "
        f"{summary['noise_normalized_improved_targets']} "
        f"({_format_fraction(summary['noise_normalized_improved_fraction'])}).",
        f"- Recommendation: `{summary['recommendation']}`.",
        f"- Rationale: {summary['rationale']}",
        "",
        "## Files",
        "",
        "- `target_deltas.csv`: per-target single versus multi-layer deltas.",
        "- `summary.csv`: one-row machine-readable summary.",
        "- `summary.json`: JSON copy of the summary.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_single_row(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return parsed


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _format_float(value: Any) -> str:
    return f"{_float(value):.6g}"


def _format_fraction(value: Any) -> str:
    return f"{100.0 * _float(value):.1f}%"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare learned-readout smoke outputs.")
    parser.add_argument(
        "--single-dir",
        default="outputs/neural_large_smoke/vit_small_patch14_dinov2_v1_learned_spatial_readout_smoke",
    )
    parser.add_argument(
        "--multi-dir",
        default="outputs/neural_large_smoke/vit_small_patch14_dinov2_v1_multilayer_learned_spatial_readout_smoke",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/neural_large_smoke/vit_small_patch14_dinov2_v1_multilayer_vs_single_smoke_comparison",
    )
    parser.add_argument("--baseline-label", default="single")
    parser.add_argument("--candidate-label", default="multi")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = compare_learned_readout_smokes(
        args.single_dir,
        args.multi_dir,
        args.output_dir,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
