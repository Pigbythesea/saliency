"""Verify cached Matrix V2 PCA analyses against accepted uncached outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.experiments.neural_alignment import run_neural_alignment
from hma.utils.config import load_yaml, save_yaml
from hma.utils.paths import resolve_path


MODELS = (
    "deit_small_static",
    "dynamicvit_deit_small_keep_0_7",
    "tome_deit_small_r13",
)
ROIS = ("v1", "ventral", "lateral", "parietal")


def audit_pca_cache_equivalence(
    *,
    config_root: str | Path = "configs/experiments/paper1_matrix_v2/scientific64",
    output_root: str | Path = "outputs/paper1_matrix_v2/cache_equivalence/scientific64",
    tolerance: float = 1e-7,
) -> Path:
    config_dir = resolve_path(config_root)
    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for model_id in MODELS:
        for roi in ROIS:
            source_config = (
                config_dir / f"{model_id}_{roi}_scientific64.yaml"
            )
            config = load_yaml(source_config)
            baseline_dir = resolve_path(config["output"]["dir"])
            cached_dir = root / "analyses" / model_id / roi
            config["external_artifact"]["feature_cache_dir"] = str(
                root / "cache" / model_id / "raw_features"
            )
            config["neural"]["pca_cache_dir"] = str(
                root / "cache" / model_id / "pca"
            )
            config["output"]["dir"] = str(cached_dir)
            generated_config = root / "configs" / source_config.name
            save_yaml(config, generated_config)
            run_neural_alignment(generated_config)
            comparison = _compare_analysis_dirs(
                baseline_dir,
                cached_dir,
                tolerance=tolerance,
            )
            results.append(
                {
                    "model_id": model_id,
                    "roi": roi,
                    **comparison,
                }
            )
    summary = {
        "schema_version": "hma.matrix_v2.pca_cache_equivalence.v1",
        "tolerance": tolerance,
        "all_equivalent": all(row["equivalent"] for row in results),
        "results": results,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not summary["all_equivalent"]:
        raise RuntimeError(f"PCA cache equivalence failed; inspect {summary_path}")
    return summary_path


def _compare_analysis_dirs(
    baseline_dir: Path,
    cached_dir: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    baseline_selection = json.loads(
        (baseline_dir / "selection_artifact.json").read_text(encoding="utf-8")
    )
    cached_selection = json.loads(
        (cached_dir / "selection_artifact.json").read_text(encoding="utf-8")
    )
    checks = {
        "selected_candidate": _candidate_scientific_fields(
            baseline_selection["selected_candidate"]
        )
        == _candidate_scientific_fields(
            cached_selection["selected_candidate"]
        ),
        "selected_alpha": _values_close(
            baseline_selection["selected_alpha"],
            cached_selection["selected_alpha"],
            tolerance=tolerance,
        ),
        "selection_score": _values_close(
            baseline_selection["selection_score"],
            cached_selection["selection_score"],
            tolerance=tolerance,
        ),
        "encoding_scores": _csv_numeric_equivalent(
            baseline_dir / "encoding_scores.csv",
            cached_dir / "encoding_scores.csv",
            tolerance=tolerance,
        ),
        "geometry_scores": _csv_numeric_equivalent(
            baseline_dir / "geometry_scores.csv",
            cached_dir / "geometry_scores.csv",
            tolerance=tolerance,
        ),
    }
    reduction = json.loads(
        (cached_dir / "feature_reduction_metadata.json").read_text(encoding="utf-8")
    )
    checks["final_reduction_cache_recorded"] = "cache_hit" in reduction["layers"][0]
    return {
        "equivalent": all(checks.values()),
        "checks": checks,
    }


def _candidate_scientific_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"pca_cache_dir", "pca_cache_source"}
    }


def _csv_numeric_equivalent(
    baseline_path: Path,
    cached_path: Path,
    *,
    tolerance: float,
) -> bool:
    baseline = _read_csv(baseline_path)
    cached = _read_csv(cached_path)
    if len(baseline) != len(cached):
        return False
    for baseline_row, cached_row in zip(baseline, cached):
        if baseline_row.keys() != cached_row.keys():
            return False
        for key in baseline_row:
            left = baseline_row[key]
            right = cached_row[key]
            if left == right:
                continue
            if not _values_close(left, right, tolerance=tolerance):
                return False
    return True


def _values_close(left: Any, right: Any, *, tolerance: float) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=tolerance, abs_tol=tolerance)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root",
        default="configs/experiments/paper1_matrix_v2/scientific64",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/paper1_matrix_v2/cache_equivalence/scientific64",
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = audit_pca_cache_equivalence(
        config_root=args.config_root,
        output_root=args.output_root,
        tolerance=args.tolerance,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
