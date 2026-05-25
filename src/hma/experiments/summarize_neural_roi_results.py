"""Summary tables for neural ROI alignment runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from hma.utils.paths import ensure_dir


ENCODING_FILE = "encoding_scores.csv"
ENCODING_TARGET_FILE = "encoding_target_scores.csv"
RSA_FILE = "rsa_scores.csv"
METADATA_FILE = "metadata.json"
STATIC_SUFFIX = "_static2000"
BRIDGE_METHODS = {"vanilla_gradient", "gradcam", "attention_rollout"}
LOWER_IS_BETTER_METRICS = {"kl", "emd", "emd_2d", "mae", "mse", "rmse", "loss"}


def summarize_neural_roi_results(
    input_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    behavioral_csv: str | Path | None = None,
    efficiency_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Combine neural ROI outputs and write compact summary tables."""
    dirs = [Path(path).expanduser().resolve() for path in input_dirs]
    if not dirs:
        raise ValueError("At least one neural output directory is required")

    output = ensure_dir(output_dir)
    encoding_rows, encoding_target_rows, rsa_rows = _load_neural_rows(dirs)
    _annotate_target_noise_validity(encoding_target_rows)
    encoding_rows = _attach_noise_normalized_aggregates(
        encoding_rows,
        encoding_target_rows,
    )
    best_rows = _best_layer_rows(encoding_rows, rsa_rows)

    outputs = {
        "combined_encoding_scores": output / "combined_encoding_scores.csv",
        "combined_rsa_scores": output / "combined_rsa_scores.csv",
        "best_layers_by_roi": output / "best_layers_by_roi.csv",
        "best_encoding_by_model_roi": output / "best_encoding_by_model_roi.csv",
        "best_rsa_by_model_roi": output / "best_rsa_by_model_roi.csv",
        "paper_model_roi_winners": output / "paper_model_roi_winners.csv",
        "neural_model_rankings": output / "neural_model_rankings.csv",
        "learned_readout_vs_flatten_pca": output / "learned_readout_vs_flatten_pca.csv",
        "summary_note": output / "neural_roi_summary.md",
        "multimodel_interpretation_note": output / "multimodel_interpretation_note.md",
    }
    best_encoding_rows = [row for row in best_rows if row.get("score_type") == "encoding"]
    best_rsa_rows = [row for row in best_rows if row.get("score_type") == "rsa"]
    paper_winner_rows = _paper_model_roi_winners(best_encoding_rows, best_rsa_rows)
    neural_ranking_rows = _neural_model_rankings(best_rows, efficiency_csv)
    learned_readout_comparison_rows = _learned_readout_vs_flatten_pca_rows(encoding_rows)
    _write_rows(outputs["combined_encoding_scores"], encoding_rows, ENCODING_FIELDNAMES)
    if encoding_target_rows:
        outputs["combined_encoding_target_scores"] = output / "combined_encoding_target_scores.csv"
        _write_rows(
            outputs["combined_encoding_target_scores"],
            encoding_target_rows,
            ENCODING_TARGET_FIELDNAMES,
        )
    _write_rows(outputs["combined_rsa_scores"], rsa_rows, RSA_FIELDNAMES)
    _write_rows(outputs["best_layers_by_roi"], best_rows, BEST_LAYER_FIELDNAMES)
    _write_rows(
        outputs["best_encoding_by_model_roi"],
        best_encoding_rows,
        BEST_LAYER_FIELDNAMES,
    )
    _write_rows(
        outputs["best_rsa_by_model_roi"],
        best_rsa_rows,
        BEST_LAYER_FIELDNAMES,
    )
    _write_rows(outputs["paper_model_roi_winners"], paper_winner_rows, PAPER_WINNER_FIELDNAMES)
    _write_rows(outputs["neural_model_rankings"], neural_ranking_rows, NEURAL_RANKING_FIELDNAMES)
    _write_rows(
        outputs["learned_readout_vs_flatten_pca"],
        learned_readout_comparison_rows,
        LEARNED_READOUT_COMPARISON_FIELDNAMES,
    )

    bridge_rows: list[dict[str, Any]] = []
    behavior_neural_alignment_rows: list[dict[str, Any]] = []
    leader_overlap_rows: list[dict[str, Any]] = []
    if behavioral_csv is not None:
        bridge_rows = _behavior_neural_bridge(behavioral_csv, best_rows)
        behavior_neural_alignment_rows = _behavior_neural_alignment_summary(
            bridge_rows,
            neural_ranking_rows,
        )
        leader_overlap_rows = _behavior_neural_leader_overlap(
            bridge_rows,
            neural_ranking_rows,
        )
        outputs["behavior_neural_bridge"] = output / "behavior_neural_bridge.csv"
        outputs["behavior_neural_model_summary"] = output / "behavior_neural_model_summary.csv"
        outputs["behavior_neural_alignment_summary"] = (
            output / "behavior_neural_alignment_summary.csv"
        )
        outputs["behavior_neural_leader_overlap"] = output / "behavior_neural_leader_overlap.csv"
        _write_rows(outputs["behavior_neural_bridge"], bridge_rows, BRIDGE_FIELDNAMES)
        _write_rows(
            outputs["behavior_neural_model_summary"],
            _behavior_neural_model_summary(bridge_rows),
            MODEL_SUMMARY_FIELDNAMES,
        )
        _write_rows(
            outputs["behavior_neural_alignment_summary"],
            behavior_neural_alignment_rows,
            BEHAVIOR_NEURAL_ALIGNMENT_FIELDNAMES,
        )
        _write_rows(
            outputs["behavior_neural_leader_overlap"],
            leader_overlap_rows,
            LEADER_OVERLAP_FIELDNAMES,
        )

    if efficiency_csv is not None:
        outputs["alignment_per_efficiency"] = output / "alignment_per_efficiency.csv"
        _write_rows(
            outputs["alignment_per_efficiency"],
            _alignment_per_efficiency(best_rows, efficiency_csv),
            EFFICIENCY_FIELDNAMES,
        )

    _write_summary_note(
        outputs["summary_note"],
        encoding_rows=encoding_rows,
        encoding_target_rows=encoding_target_rows,
        rsa_rows=rsa_rows,
        best_rows=best_rows,
        bridge_rows=bridge_rows,
        learned_readout_comparison_rows=learned_readout_comparison_rows,
        input_dirs=dirs,
        behavioral_csv=behavioral_csv,
        efficiency_csv=efficiency_csv,
    )
    candidate_inventory = output / "ssl_multimodal_candidate_inventory.csv"
    if candidate_inventory.is_file():
        outputs["ssl_multimodal_candidate_inventory"] = candidate_inventory
    _write_multimodel_interpretation_note(
        outputs["multimodel_interpretation_note"],
        neural_ranking_rows=neural_ranking_rows,
        paper_winner_rows=paper_winner_rows,
        behavior_neural_alignment_rows=behavior_neural_alignment_rows,
        leader_overlap_rows=leader_overlap_rows,
        candidate_inventory=candidate_inventory if candidate_inventory.is_file() else None,
        behavioral_csv=behavioral_csv,
        efficiency_csv=efficiency_csv,
    )
    return outputs


def _load_neural_rows(
    dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    encoding_rows: list[dict[str, Any]] = []
    encoding_target_rows: list[dict[str, Any]] = []
    rsa_rows: list[dict[str, Any]] = []
    for directory in dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Neural output directory not found: {directory}")
        metadata = _load_metadata(directory / METADATA_FILE)
        encoding_path = directory / ENCODING_FILE
        if not encoding_path.is_file():
            raise FileNotFoundError(f"Encoding scores not found: {encoding_path}")
        encoding_rows.extend(_load_csv_with_context(encoding_path, directory, metadata))

        encoding_target_path = directory / ENCODING_TARGET_FILE
        if encoding_target_path.is_file():
            encoding_target_rows.extend(
                _load_csv_with_context(encoding_target_path, directory, metadata)
            )

        rsa_path = directory / RSA_FILE
        if rsa_path.is_file():
            rsa_rows.extend(_load_csv_with_context(rsa_path, directory, metadata))
    return encoding_rows, encoding_target_rows, rsa_rows


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_csv_with_context(
    path: Path,
    source_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _load_csv_rows(path)
    for row in rows:
        row["source_dir"] = str(source_dir)
        row["metadata_num_items"] = str(metadata.get("num_items", ""))
        row["metadata_config_path"] = str(metadata.get("config_path", ""))
    return rows


def _best_layer_rows(
    encoding_rows: list[dict[str, Any]],
    rsa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _best_encoding_by_group(
        encoding_rows,
        keys=["model", "subject_id", "roi", "metric"],
    ):
        primary_score, score_type = _primary_encoding_score(group)
        rows.append(
            {
                "score_type": "encoding",
                "model": group.get("model", ""),
                "subject_id": group.get("subject_id", ""),
                "roi": group.get("roi", ""),
                "layer": group.get("layer", ""),
                "metric": group.get("metric", ""),
                "compare_method": "",
                "score": primary_score,
                "encoding_score_type": score_type,
                "raw_score": group.get("mean_score", ""),
                "median_score": group.get("median_score", ""),
                "std_score": group.get("std_score", ""),
                "mean_noise_normalized_score": group.get("mean_noise_normalized_score", ""),
                "median_noise_normalized_score": group.get("median_noise_normalized_score", ""),
                "valid_noise_ceiling_targets": group.get("valid_noise_ceiling_targets", ""),
                "zero_noise_ceiling_targets": group.get("zero_noise_ceiling_targets", ""),
                "invalid_noise_ceiling_targets": group.get("invalid_noise_ceiling_targets", ""),
                "n_train": group.get("n_train", ""),
                "n_test": group.get("n_test", ""),
                "n_items": "",
                "num_targets": group.get("num_targets", ""),
                "dataset": group.get("dataset", ""),
                "source_dir": group.get("source_dir", ""),
            }
        )

    for group in _best_by_group(
        rsa_rows,
        keys=["model", "subject_id", "roi", "compare_method"],
        score_key="score",
    ):
        rows.append(
            {
                "score_type": "rsa",
                "model": group.get("model", ""),
                "subject_id": group.get("subject_id", ""),
                "roi": group.get("roi", ""),
                "layer": group.get("layer", ""),
                "metric": "",
                "compare_method": group.get("compare_method", ""),
                "score": group.get("score", ""),
                "median_score": "",
                "std_score": "",
                "n_train": "",
                "n_test": "",
                "n_items": group.get("n_items", ""),
                "num_targets": "",
                "dataset": group.get("dataset", ""),
                "source_dir": group.get("source_dir", ""),
            }
        )
    rows.sort(key=lambda row: (row["model"], row["subject_id"], row["roi"], row["score_type"]))
    return rows


def _best_by_group(
    rows: Iterable[dict[str, Any]],
    *,
    keys: list[str],
    score_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)

    best_rows = []
    for key in sorted(grouped):
        candidates = grouped[key]
        best_rows.append(
            sorted(
                candidates,
                key=lambda row: _float(row.get(score_key)),
                reverse=True,
            )[0]
        )
    return best_rows


def _best_encoding_by_group(
    rows: Iterable[dict[str, Any]],
    *,
    keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)

    best_rows = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if any(_optional_float(row.get("mean_noise_normalized_score")) is not None for row in candidates):
            score_key = "mean_noise_normalized_score"
        else:
            score_key = "mean_score"
        best_rows.append(
            sorted(
                candidates,
                key=lambda row: _float(row.get(score_key)),
                reverse=True,
            )[0]
        )
    return best_rows


def _primary_encoding_score(row: dict[str, Any]) -> tuple[Any, str]:
    normalized = row.get("mean_noise_normalized_score", "")
    if _optional_float(normalized) is not None:
        return normalized, "noise_normalized"
    return row.get("mean_score", ""), "raw"


def _learned_readout_vs_flatten_pca_rows(
    encoding_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in encoding_rows:
        feature_reduction = str(row.get("feature_reduction", ""))
        if feature_reduction not in {"flatten_pca", "learned_spatial_readout"}:
            continue
        key = (
            str(row.get("model", "")),
            str(row.get("subject_id", "")),
            str(row.get("roi", "")),
            str(row.get("metric", "")),
        )
        by_key.setdefault(key, {})[feature_reduction] = row

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        pair = by_key[key]
        flatten = pair.get("flatten_pca")
        learned = pair.get("learned_spatial_readout")
        if flatten is None or learned is None:
            continue
        flatten_raw = _optional_float(flatten.get("mean_score"))
        learned_raw = _optional_float(learned.get("mean_score"))
        flatten_normalized = _optional_float(flatten.get("mean_noise_normalized_score"))
        learned_normalized = _optional_float(learned.get("mean_noise_normalized_score"))
        rows.append(
            {
                "model": key[0],
                "subject_id": key[1],
                "roi": key[2],
                "metric": key[3],
                "flatten_pca_layer": flatten.get("layer", ""),
                "learned_readout_layer": learned.get("layer", ""),
                "flatten_pca_raw_score": flatten.get("mean_score", ""),
                "learned_readout_raw_score": learned.get("mean_score", ""),
                "raw_delta": (
                    learned_raw - flatten_raw
                    if learned_raw is not None and flatten_raw is not None
                    else ""
                ),
                "flatten_pca_noise_normalized_score": flatten.get(
                    "mean_noise_normalized_score",
                    "",
                ),
                "learned_readout_noise_normalized_score": learned.get(
                    "mean_noise_normalized_score",
                    "",
                ),
                "noise_normalized_delta": (
                    learned_normalized - flatten_normalized
                    if learned_normalized is not None and flatten_normalized is not None
                    else ""
                ),
                "flatten_pca_valid_noise_ceiling_targets": flatten.get(
                    "valid_noise_ceiling_targets",
                    "",
                ),
                "learned_readout_valid_noise_ceiling_targets": learned.get(
                    "valid_noise_ceiling_targets",
                    "",
                ),
                "flatten_pca_zero_noise_ceiling_targets": flatten.get(
                    "zero_noise_ceiling_targets",
                    "",
                ),
                "learned_readout_zero_noise_ceiling_targets": learned.get(
                    "zero_noise_ceiling_targets",
                    "",
                ),
                "flatten_pca_invalid_noise_ceiling_targets": flatten.get(
                    "invalid_noise_ceiling_targets",
                    "",
                ),
                "learned_readout_invalid_noise_ceiling_targets": learned.get(
                    "invalid_noise_ceiling_targets",
                    "",
                ),
                "flatten_pca_n_train": flatten.get("n_train", ""),
                "learned_readout_n_train": learned.get("n_train", ""),
                "flatten_pca_n_test": flatten.get("n_test", ""),
                "learned_readout_n_test": learned.get("n_test", ""),
                "flatten_pca_source_dir": flatten.get("source_dir", ""),
                "learned_readout_source_dir": learned.get("source_dir", ""),
            }
        )
    return rows


def _attach_noise_normalized_aggregates(
    encoding_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not target_rows:
        return encoding_rows

    grouped_targets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in target_rows:
        grouped_targets.setdefault(_encoding_layer_key(row), []).append(row)

    enriched: list[dict[str, Any]] = []
    for row in encoding_rows:
        updated = dict(row)
        target_group = grouped_targets.get(_encoding_layer_key(row), [])
        if target_group:
            updated.update(_noise_normalized_summary(target_group))
        else:
            updated.setdefault("mean_noise_normalized_score", "")
            updated.setdefault("median_noise_normalized_score", "")
            updated.setdefault("valid_noise_ceiling_targets", "")
            updated.setdefault("zero_noise_ceiling_targets", "")
            updated.setdefault("invalid_noise_ceiling_targets", "")
        enriched.append(updated)
    return enriched


def _encoding_layer_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("source_dir", "")),
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("subject_id", "")),
        str(row.get("roi", "")),
        str(row.get("layer", "")),
        str(row.get("metric", "")),
    )


def _annotate_target_noise_validity(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if "valid_noise_ceiling" in row and str(row.get("valid_noise_ceiling", "")):
            continue
        ceiling = _optional_float(row.get("noise_ceiling"))
        valid = ceiling is not None and ceiling > 0.0
        row["valid_noise_ceiling"] = str(valid).lower()


def _noise_normalized_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_scores: list[float] = []
    zero_count = 0
    invalid_count = 0
    for row in rows:
        ceiling = _optional_float(row.get("noise_ceiling"))
        score = _optional_float(row.get("noise_normalized_score"))
        if ceiling is None or ceiling < 0.0:
            invalid_count += 1
            continue
        if ceiling == 0.0:
            zero_count += 1
            continue
        if score is None:
            invalid_count += 1
            continue
        valid_scores.append(score)
    return {
        "mean_noise_normalized_score": _mean(valid_scores) if valid_scores else "",
        "median_noise_normalized_score": _median(valid_scores) if valid_scores else "",
        "valid_noise_ceiling_targets": len(valid_scores),
        "zero_noise_ceiling_targets": zero_count,
        "invalid_noise_ceiling_targets": invalid_count,
    }


def _behavior_neural_bridge(
    behavioral_csv: str | Path,
    best_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    neural_models = {str(row.get("model", "")) for row in best_rows if row.get("model")}
    behavioral_rows = [
        row
        for row in _load_csv_rows(Path(behavioral_csv))
        if row.get("model") in neural_models
        and str(row.get("dataset", "")).endswith(STATIC_SUFFIX)
        and row.get("saliency_method") in BRIDGE_METHODS
    ]
    encoding_best = {
        (row["model"], row["subject_id"], row["roi"]): row
        for row in best_rows
        if row.get("score_type") == "encoding"
    }
    rsa_best = {
        (row["model"], row["subject_id"], row["roi"]): row
        for row in best_rows
        if row.get("score_type") == "rsa"
    }
    keys = sorted(set(encoding_best) | set(rsa_best))

    bridge_rows = []
    for behavior in behavioral_rows:
        for key in keys:
            if key[0] != behavior.get("model"):
                continue
            encoding = encoding_best.get(key, {})
            rsa = rsa_best.get(key, {})
            bridge_rows.append(
                {
                    "behavior_dataset": behavior.get("dataset", ""),
                    "behavior_metric": behavior.get("metric", ""),
                    "behavior_model": behavior.get("model", ""),
                    "behavior_saliency_method": behavior.get("saliency_method", ""),
                    "behavior_saliency_family": behavior.get("saliency_family", ""),
                    "behavior_n": behavior.get("n", ""),
                    "behavior_mean": behavior.get("mean", ""),
                    "behavior_ci95_low": behavior.get("ci95_low", ""),
                    "behavior_ci95_high": behavior.get("ci95_high", ""),
                    "neural_model": key[0],
                    "subject_id": key[1],
                    "roi": key[2],
                    "best_encoding_layer": encoding.get("layer", ""),
                    "best_encoding_metric": encoding.get("metric", ""),
                    "best_encoding_score": encoding.get("score", ""),
                    "best_encoding_n_train": encoding.get("n_train", ""),
                    "best_encoding_n_test": encoding.get("n_test", ""),
                    "best_rsa_layer": rsa.get("layer", ""),
                    "best_rsa_method": rsa.get("compare_method", ""),
                    "best_rsa_score": rsa.get("score", ""),
                    "best_rsa_n_items": rsa.get("n_items", ""),
                }
            )
    bridge_rows.sort(
        key=lambda row: (
            row["behavior_dataset"],
            row["behavior_metric"],
            row["behavior_saliency_method"],
            row["roi"],
        )
    )
    return bridge_rows


def _behavior_neural_model_summary(bridge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in bridge_rows:
        key = (
            str(row.get("neural_model", "")),
            str(row.get("behavior_dataset", "")),
            str(row.get("behavior_metric", "")),
            str(row.get("behavior_saliency_method", "")),
            str(row.get("behavior_saliency_family", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        encoding_scores = [_float(row.get("best_encoding_score")) for row in rows]
        rsa_scores = [_float(row.get("best_rsa_score")) for row in rows]
        summary_rows.append(
            {
                "model": key[0],
                "behavior_dataset": key[1],
                "behavior_metric": key[2],
                "behavior_saliency_method": key[3],
                "behavior_saliency_family": key[4],
                "num_rois": len({row.get("roi", "") for row in rows}),
                "behavior_mean": rows[0].get("behavior_mean", ""),
                "mean_best_encoding_score_across_rois": _mean(encoding_scores),
                "mean_best_rsa_score_across_rois": _mean(rsa_scores),
            }
        )
    return summary_rows


def _paper_model_roi_winners(
    best_encoding_rows: list[dict[str, Any]],
    best_rsa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join best encoding and RSA rows into one paper-facing row per model/ROI."""
    encoding_by_key = {
        (
            str(row.get("model", "")),
            str(row.get("subject_id", "")),
            str(row.get("roi", "")),
        ): row
        for row in best_encoding_rows
    }
    rsa_by_key = {
        (
            str(row.get("model", "")),
            str(row.get("subject_id", "")),
            str(row.get("roi", "")),
        ): row
        for row in best_rsa_rows
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(encoding_by_key) | set(rsa_by_key)):
        encoding = encoding_by_key.get(key, {})
        rsa = rsa_by_key.get(key, {})
        rows.append(
            {
                "model": key[0],
                "subject_id": key[1],
                "roi": key[2],
                "best_encoding_layer": encoding.get("layer", ""),
                "best_encoding_metric": encoding.get("metric", ""),
                "best_encoding_score": encoding.get("score", ""),
                "best_encoding_score_type": encoding.get("encoding_score_type", ""),
                "best_encoding_raw_score": encoding.get("raw_score", ""),
                "best_encoding_median_score": encoding.get("median_score", ""),
                "best_encoding_std_score": encoding.get("std_score", ""),
                "best_encoding_mean_noise_normalized_score": encoding.get(
                    "mean_noise_normalized_score",
                    "",
                ),
                "best_encoding_median_noise_normalized_score": encoding.get(
                    "median_noise_normalized_score",
                    "",
                ),
                "valid_noise_ceiling_targets": encoding.get("valid_noise_ceiling_targets", ""),
                "zero_noise_ceiling_targets": encoding.get("zero_noise_ceiling_targets", ""),
                "invalid_noise_ceiling_targets": encoding.get(
                    "invalid_noise_ceiling_targets",
                    "",
                ),
                "best_encoding_n_train": encoding.get("n_train", ""),
                "best_encoding_n_test": encoding.get("n_test", ""),
                "best_rsa_layer": rsa.get("layer", ""),
                "best_rsa_method": rsa.get("compare_method", ""),
                "best_rsa_score": rsa.get("score", ""),
                "best_rsa_n_items": rsa.get("n_items", ""),
                "encoding_source_dir": encoding.get("source_dir", ""),
                "rsa_source_dir": rsa.get("source_dir", ""),
            }
        )
    return rows


def _neural_model_rankings(
    best_rows: list[dict[str, Any]],
    efficiency_csv: str | Path | None,
) -> list[dict[str, Any]]:
    """Aggregate best neural rows by model and add rank columns."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in best_rows:
        model = str(row.get("model", ""))
        if model:
            grouped.setdefault(model, []).append(row)

    efficiency_by_model: dict[str, dict[str, str]] = {}
    if efficiency_csv is not None:
        efficiency_by_model = {
            str(row.get("model_name") or row.get("model")): row
            for row in _load_csv_rows(efficiency_csv)
            if row.get("model_name") or row.get("model")
        }

    rows: list[dict[str, Any]] = []
    for model in sorted(grouped):
        model_rows = grouped[model]
        encoding_scores = [
            _float(row.get("raw_score") or row.get("score"))
            for row in model_rows
            if row.get("score_type") == "encoding"
        ]
        normalized_scores = [
            _float(row.get("mean_noise_normalized_score"))
            for row in model_rows
            if row.get("score_type") == "encoding"
            and _optional_float(row.get("mean_noise_normalized_score")) is not None
        ]
        rsa_scores = [
            _float(row.get("score"))
            for row in model_rows
            if row.get("score_type") == "rsa"
        ]
        mean_encoding = _mean(encoding_scores)
        mean_normalized = _mean(normalized_scores) if normalized_scores else ""
        mean_rsa = _mean(rsa_scores)
        valid_targets = sum(
            int(_float(row.get("valid_noise_ceiling_targets")))
            for row in model_rows
            if row.get("score_type") == "encoding"
        )
        zero_targets = sum(
            int(_float(row.get("zero_noise_ceiling_targets")))
            for row in model_rows
            if row.get("score_type") == "encoding"
        )
        invalid_targets = sum(
            int(_float(row.get("invalid_noise_ceiling_targets")))
            for row in model_rows
            if row.get("score_type") == "encoding"
        )
        efficiency = efficiency_by_model.get(model, {})
        latency = _optional_float(efficiency.get("latency_mean_ms"))
        params = _optional_float(efficiency.get("parameter_count"))
        size = _optional_float(efficiency.get("model_size_mb"))
        flops = _optional_float(efficiency.get("flops"))
        rows.append(
            {
                "model": model,
                "num_encoding_rois": len(encoding_scores),
                "num_rsa_rois": len(rsa_scores),
                "mean_encoding_score": mean_encoding,
                "mean_noise_normalized_score": mean_normalized,
                "mean_noise_normalized_score_x100": (
                    mean_normalized * 100.0 if normalized_scores else ""
                ),
                "valid_noise_ceiling_targets": valid_targets,
                "zero_noise_ceiling_targets": zero_targets,
                "invalid_noise_ceiling_targets": invalid_targets,
                "mean_rsa_score": mean_rsa,
                "latency_mean_ms": efficiency.get("latency_mean_ms", ""),
                "parameter_count": efficiency.get("parameter_count", ""),
                "model_size_mb": efficiency.get("model_size_mb", ""),
                "flops": efficiency.get("flops", ""),
                "encoding_score_per_latency_mean_ms": (
                    mean_encoding / latency if latency else ""
                ),
                "rsa_score_per_latency_mean_ms": mean_rsa / latency if latency else "",
                "encoding_score_per_million_parameters": (
                    mean_encoding / (params / 1_000_000.0) if params else ""
                ),
                "rsa_score_per_million_parameters": (
                    mean_rsa / (params / 1_000_000.0) if params else ""
                ),
                "encoding_score_per_model_size_mb": mean_encoding / size if size else "",
                "rsa_score_per_model_size_mb": mean_rsa / size if size else "",
                "encoding_score_per_gflop": (
                    mean_encoding / (flops / 1_000_000_000.0) if flops else ""
                ),
                "rsa_score_per_gflop": mean_rsa / (flops / 1_000_000_000.0) if flops else "",
            }
        )

    _add_rank_column(rows, "mean_encoding_score", "rank_mean_encoding")
    _add_rank_column(rows, "mean_noise_normalized_score", "rank_mean_noise_normalized")
    _add_rank_column(rows, "mean_rsa_score", "rank_mean_rsa")
    _add_rank_column(
        rows,
        "encoding_score_per_latency_mean_ms",
        "rank_encoding_per_latency",
    )
    _add_rank_column(rows, "rsa_score_per_latency_mean_ms", "rank_rsa_per_latency")
    _add_rank_column(
        rows,
        "encoding_score_per_million_parameters",
        "rank_encoding_per_million_parameters",
    )
    _add_rank_column(
        rows,
        "rsa_score_per_million_parameters",
        "rank_rsa_per_million_parameters",
    )
    if any(row.get("rank_mean_noise_normalized") for row in rows):
        rows.sort(key=lambda row: int(row.get("rank_mean_noise_normalized") or 10**9))
    else:
        rows.sort(key=lambda row: int(row.get("rank_mean_encoding") or 10**9))
    return rows


def _behavior_neural_alignment_summary(
    bridge_rows: list[dict[str, Any]],
    neural_ranking_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize model-matched behavioral rows with model-level neural aggregates."""
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in bridge_rows:
        key = (
            str(row.get("neural_model", "")),
            str(row.get("behavior_dataset", "")),
            str(row.get("behavior_metric", "")),
            str(row.get("behavior_saliency_method", "")),
            str(row.get("behavior_saliency_family", "")),
        )
        grouped.setdefault(key, []).append(row)

    rankings_by_model = {str(row.get("model", "")): row for row in neural_ranking_rows}
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bridge_group = grouped[key]
        ranking = rankings_by_model.get(key[0], {})
        behavior_metric = key[2]
        rows.append(
            {
                "model": key[0],
                "behavior_dataset": key[1],
                "behavior_metric": behavior_metric,
                "behavior_metric_direction": _metric_direction(behavior_metric),
                "behavior_saliency_method": key[3],
                "behavior_saliency_family": key[4],
                "num_rois": len({row.get("roi", "") for row in bridge_group}),
                "behavior_mean": bridge_group[0].get("behavior_mean", ""),
                "mean_encoding_score": ranking.get("mean_encoding_score", ""),
                "mean_noise_normalized_score": ranking.get("mean_noise_normalized_score", ""),
                "mean_rsa_score": ranking.get("mean_rsa_score", ""),
                "rank_mean_encoding": ranking.get("rank_mean_encoding", ""),
                "rank_mean_noise_normalized": ranking.get(
                    "rank_mean_noise_normalized",
                    "",
                ),
                "rank_mean_rsa": ranking.get("rank_mean_rsa", ""),
                "rank_encoding_per_latency": ranking.get("rank_encoding_per_latency", ""),
                "rank_rsa_per_latency": ranking.get("rank_rsa_per_latency", ""),
                "interpretation_scope": "descriptive_one_subject_roi500",
            }
        )
    return rows


def _behavior_neural_leader_overlap(
    bridge_rows: list[dict[str, Any]],
    neural_ranking_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report whether behavioral winners match neural model-level winners."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in bridge_rows:
        key = (
            str(row.get("behavior_dataset", "")),
            str(row.get("behavior_metric", "")),
            str(row.get("behavior_saliency_method", "")),
            str(row.get("behavior_saliency_family", "")),
        )
        grouped.setdefault(key, []).append(row)

    encoding_leader_key = (
        "mean_noise_normalized_score"
        if any(_optional_float(row.get("mean_noise_normalized_score")) is not None for row in neural_ranking_rows)
        else "mean_encoding_score"
    )
    encoding_leader = _leader_by_score(neural_ranking_rows, encoding_leader_key)
    rsa_leader = _leader_by_score(neural_ranking_rows, "mean_rsa_score")
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates_by_model: dict[str, dict[str, Any]] = {}
        for row in grouped[key]:
            model = str(row.get("neural_model", ""))
            if model and model not in candidates_by_model:
                candidates_by_model[model] = row
        behavior_metric = key[1]
        behavioral_leader = _behavioral_leader(
            list(candidates_by_model.values()),
            behavior_metric,
        )
        behavioral_model = str(behavioral_leader.get("neural_model", ""))
        rows.append(
            {
                "behavior_dataset": key[0],
                "behavior_metric": behavior_metric,
                "behavior_metric_direction": _metric_direction(behavior_metric),
                "behavior_saliency_method": key[2],
                "behavior_saliency_family": key[3],
                "behavior_leader_model": behavioral_model,
                "behavior_leader_mean": behavioral_leader.get("behavior_mean", ""),
                "neural_encoding_leader_model": encoding_leader.get("model", ""),
                "neural_encoding_leader_mean_score": encoding_leader.get(
                    encoding_leader_key,
                    "",
                ),
                "matches_encoding_leader": str(
                    bool(behavioral_model)
                    and behavioral_model == encoding_leader.get("model", "")
                ).lower(),
                "neural_rsa_leader_model": rsa_leader.get("model", ""),
                "neural_rsa_leader_mean_score": rsa_leader.get("mean_rsa_score", ""),
                "matches_rsa_leader": str(
                    bool(behavioral_model) and behavioral_model == rsa_leader.get("model", "")
                ).lower(),
                "interpretation_scope": "descriptive_one_subject_roi500",
            }
        )
    return rows


def _alignment_per_efficiency(
    best_rows: list[dict[str, Any]],
    efficiency_csv: str | Path,
) -> list[dict[str, Any]]:
    efficiency_by_model = {
        str(row.get("model_name") or row.get("model")): row
        for row in _load_csv_rows(efficiency_csv)
        if row.get("model_name") or row.get("model")
    }
    output: list[dict[str, Any]] = []
    for row in best_rows:
        model = str(row.get("model", ""))
        efficiency = efficiency_by_model.get(model)
        if efficiency is None:
            continue
        score = _float(row.get("score"))
        latency = _optional_float(efficiency.get("latency_mean_ms"))
        params = _optional_float(efficiency.get("parameter_count"))
        size = _optional_float(efficiency.get("model_size_mb"))
        flops = _optional_float(efficiency.get("flops"))
        output.append(
            {
                **row,
                "latency_mean_ms": efficiency.get("latency_mean_ms", ""),
                "parameter_count": efficiency.get("parameter_count", ""),
                "model_size_mb": efficiency.get("model_size_mb", ""),
                "flops": efficiency.get("flops", ""),
                "score_per_latency_mean_ms": score / latency if latency else "",
                "score_per_million_parameters": score / (params / 1_000_000.0) if params else "",
                "score_per_model_size_mb": score / size if size else "",
                "score_per_gflop": score / (flops / 1_000_000_000.0) if flops else "",
            }
        )
    return output


def _add_rank_column(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> None:
    ranked = [
        row
        for row in rows
        if _optional_float(row.get(score_key)) is not None
    ]
    ranked.sort(key=lambda row: _float(row.get(score_key)), reverse=True)
    previous_score: float | None = None
    previous_rank = 0
    for index, row in enumerate(ranked, start=1):
        score = _float(row.get(score_key))
        rank = previous_rank if previous_score == score else index
        row[rank_key] = rank
        previous_score = score
        previous_rank = rank
    for row in rows:
        row.setdefault(rank_key, "")


def _leader_by_score(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    candidates = [row for row in rows if _optional_float(row.get(score_key)) is not None]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda row: _float(row.get(score_key)), reverse=True)[0]


def _behavioral_leader(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    candidates = [row for row in rows if _optional_float(row.get("behavior_mean")) is not None]
    if not candidates:
        return {}
    reverse = _metric_higher_is_better(metric)
    return sorted(candidates, key=lambda row: _float(row.get("behavior_mean")), reverse=reverse)[0]


def _metric_higher_is_better(metric: str) -> bool:
    return metric.lower() not in LOWER_IS_BETTER_METRICS


def _metric_direction(metric: str) -> str:
    return "higher_is_better" if _metric_higher_is_better(metric) else "lower_is_better"


def _write_summary_note(
    path: Path,
    *,
    encoding_rows: list[dict[str, Any]],
    encoding_target_rows: list[dict[str, Any]],
    rsa_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    learned_readout_comparison_rows: list[dict[str, Any]],
    input_dirs: list[Path],
    behavioral_csv: str | Path | None,
    efficiency_csv: str | Path | None,
) -> None:
    encoding_best = [row for row in best_rows if row.get("score_type") == "encoding"]
    rsa_best = [row for row in best_rows if row.get("score_type") == "rsa"]
    lines = [
        "# Neural ROI Summary",
        "",
        "This note is generated from neural ROI alignment outputs.",
        "",
        "## Scope",
        "",
        f"- Input directories: {len(input_dirs)}.",
        f"- Encoding rows: {len(encoding_rows)}.",
        f"- Encoding target rows: {len(encoding_target_rows)}.",
        f"- RSA rows: {len(rsa_rows)}.",
        f"- Behavioral bridge CSV: {behavioral_csv if behavioral_csv else 'not provided'}.",
        f"- Efficiency CSV: {efficiency_csv if efficiency_csv else 'not provided'}.",
        f"- Benchmark-style encoding scope: {_benchmark_target_scope_note(encoding_target_rows)}.",
        "",
        "## Best Encoding Layers",
        "",
    ]
    if encoding_best:
        for row in sorted(encoding_best, key=lambda item: (item["roi"], item["metric"])):
            lines.append(
                "- "
                f"{row['model']} {row['subject_id']} {row['roi']} "
                f"{row['metric']}: {row['layer']} score={_float(row['score']):.6g}."
                f" score_type={row.get('encoding_score_type', 'raw')}."
            )
    else:
        lines.append("- No encoding best-layer rows are available.")

    lines.extend(["", "## Best RSA Layers", ""])
    if rsa_best:
        for row in sorted(rsa_best, key=lambda item: (item["roi"], item["compare_method"])):
            lines.append(
                "- "
                f"{row['model']} {row['subject_id']} {row['roi']} "
                f"{row['compare_method']}: {row['layer']} score={_float(row['score']):.6g}."
            )
    else:
        lines.append("- No RSA best-layer rows are available.")

    lines.extend(["", "## Behavior-Neural Bridge", ""])
    if bridge_rows:
        lines.append(
            "- Descriptive bridge rows were generated for matching static2000 behavioral "
            "models and neural ROI outputs."
        )
    else:
        lines.append("- No behavior-neural bridge rows were generated.")
    lines.append(
        "- Do not interpret bridge rows as cross-model correlations until neural outputs "
        "exist for multiple model families."
    )
    lines.extend(["", "## Learned Readout Versus Flatten PCA", ""])
    if learned_readout_comparison_rows:
        improved = sum(
            1
            for row in learned_readout_comparison_rows
            if _optional_float(row.get("raw_delta")) is not None
            and _float(row.get("raw_delta")) > 0.0
        )
        lines.append(
            "- Matched learned-readout and `flatten_pca` comparison rows: "
            f"{len(learned_readout_comparison_rows)}; raw-score improvements: "
            f"{improved}/{len(learned_readout_comparison_rows)}."
        )
    else:
        lines.append("- No matched learned-readout and `flatten_pca` comparison rows are available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_target_scope_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "not available; input directories do not include per-target benchmark scores"
    scope_counts = _status_counts(rows, "metric_scope")
    scopes = set(scope_counts)
    if scopes == {"benchmark_style_noise_normalized"}:
        return "noise-normalized"
    if "benchmark_style_noise_normalized" in scopes:
        counts = ", ".join(
            f"{scope}={count}" for scope, count in sorted(scope_counts.items())
        )
        return f"mixed target-level scope ({counts})"
    return "non-noise-normalized"


def _write_multimodel_interpretation_note(
    path: Path,
    *,
    neural_ranking_rows: list[dict[str, Any]],
    paper_winner_rows: list[dict[str, Any]],
    behavior_neural_alignment_rows: list[dict[str, Any]],
    leader_overlap_rows: list[dict[str, Any]],
    candidate_inventory: Path | None,
    behavioral_csv: str | Path | None,
    efficiency_csv: str | Path | None,
) -> None:
    normalized_encoding = _leader_by_score(
        neural_ranking_rows,
        "mean_noise_normalized_score",
    )
    raw_encoding = _leader_by_score(neural_ranking_rows, "mean_encoding_score")
    raw_rsa = _leader_by_score(neural_ranking_rows, "mean_rsa_score")
    latency_encoding = _leader_by_score(
        neural_ranking_rows,
        "encoding_score_per_latency_mean_ms",
    )
    latency_rsa = _leader_by_score(neural_ranking_rows, "rsa_score_per_latency_mean_ms")
    overlap_matches_encoding = sum(
        1 for row in leader_overlap_rows if row.get("matches_encoding_leader") == "true"
    )
    overlap_matches_rsa = sum(
        1 for row in leader_overlap_rows if row.get("matches_rsa_leader") == "true"
    )

    lines = [
        "# Multimodel Behavior-Neural Interpretation Note",
        "",
        "This note is generated from the current static2000 behavioral summaries and "
        "multi-model ROI500 neural summaries.",
        "",
        "## Scope",
        "",
        f"- Model/ROI winner rows: {len(paper_winner_rows)}.",
        f"- Model ranking rows: {len(neural_ranking_rows)}.",
        f"- Behavior-neural alignment rows: {len(behavior_neural_alignment_rows)}.",
        f"- Behavioral CSV: {behavioral_csv if behavioral_csv else 'not provided'}.",
        f"- Efficiency CSV: {efficiency_csv if efficiency_csv else 'not provided'}.",
        "- Interpretation boundary: descriptive only; one subject, ROI500 subset, and "
        "frozen static2000 behavioral rows.",
        "",
        "## Neural Ranking",
        "",
    ]
    if normalized_encoding:
        lines.append(
            "- Strongest mean ROI500 noise-normalized encoding model: "
            f"{normalized_encoding.get('model')} "
            f"({_float(normalized_encoding.get('mean_noise_normalized_score')):.6g}; "
            f"x100={_float(normalized_encoding.get('mean_noise_normalized_score_x100')):.6g})."
        )
    if raw_encoding:
        lines.append(
            "- Strongest mean ROI500 raw encoding model: "
            f"{raw_encoding.get('model')} "
            f"({ _float(raw_encoding.get('mean_encoding_score')):.6g})."
        )
    if raw_rsa:
        lines.append(
            "- Strongest mean ROI500 RSA model: "
            f"{raw_rsa.get('model')} ({_float(raw_rsa.get('mean_rsa_score')):.6g})."
        )
    if not raw_encoding and not raw_rsa:
        lines.append("- No neural ranking rows are available.")

    lines.extend(["", "## Efficiency-Normalized Ranking", ""])
    if latency_encoding:
        lines.append(
            "- Best encoding per latency: "
            f"{latency_encoding.get('model')} "
            f"({_float(latency_encoding.get('encoding_score_per_latency_mean_ms')):.6g})."
        )
    if latency_rsa:
        lines.append(
            "- Best RSA per latency: "
            f"{latency_rsa.get('model')} "
            f"({_float(latency_rsa.get('rsa_score_per_latency_mean_ms')):.6g})."
        )
    if not latency_encoding and not latency_rsa:
        lines.append("- Efficiency CSV was not provided or did not match current models.")

    lines.extend(["", "## Behavioral Saliency Ranking", ""])
    if leader_overlap_rows:
        lines.append(
            "- Behavioral leaders are computed within each static2000 dataset, metric, "
            "saliency method, and saliency family among models with matching neural outputs."
        )
        lines.append(
            "- Leader overlap counts: "
            f"{overlap_matches_encoding}/{len(leader_overlap_rows)} match the raw "
            f"encoding leader; {overlap_matches_rsa}/{len(leader_overlap_rows)} match "
            "the raw RSA leader."
        )
    else:
        lines.append("- Behavioral bridge rows were not generated.")

    lines.extend(["", "## Bridge Interpretation", ""])
    lines.append(
        "- Bridge tables are descriptive joins, not causal tests or cross-model "
        "correlation claims."
    )
    lines.append(
        "- Use `behavior_neural_alignment_summary.csv` for paper-style side-by-side "
        "behavioral and neural rows."
    )
    lines.append(
        "- Use `behavior_neural_leader_overlap.csv` for the compact leader-match check."
    )

    lines.extend(["", "## SSL And Multimodal Candidate Prep", ""])
    if candidate_inventory is not None:
        candidate_rows = _load_csv_rows(candidate_inventory)
        compatible = [
            row for row in candidate_rows if row.get("wrapper_compatible") == "true"
        ]
        complete = [
            row for row in candidate_rows if row.get("pretrained_weights_run") == "true"
        ]
        status_counts = _status_counts(candidate_rows, "pretrained_run_status")
        lines.append(f"- Candidate inventory CSV: {candidate_inventory}.")
        lines.append(
            f"- Dry-inspected compatible candidates: {len(compatible)}/"
            f"{len(candidate_rows)}."
        )
        lines.append(f"- Pretrained debug runs complete: {len(complete)}.")
        if status_counts:
            lines.append(
                "- Pretrained status counts: "
                + ", ".join(
                    f"{status}={count}" for status, count in sorted(status_counts.items())
                )
                + "."
            )
    else:
        lines.append("- Candidate inventory CSV was not present when this note was generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(field) or "not_run")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path,
    rows: Iterable[dict[str, Any]],
    default_fieldnames: list[str],
) -> None:
    row_list = list(rows)
    fieldnames = list(default_fieldnames)
    for row in row_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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


ENCODING_FIELDNAMES = [
    "dataset",
    "model",
    "subject_id",
    "roi",
    "layer",
    "metric",
    "metric_scope",
    "n_train",
    "n_test",
    "num_targets",
    "mean_score",
    "median_score",
    "std_score",
    "mean_r2_score_from_r",
    "mean_noise_normalized_score",
    "median_noise_normalized_score",
    "valid_noise_ceiling_targets",
    "zero_noise_ceiling_targets",
    "invalid_noise_ceiling_targets",
    "selected_ridge_alpha",
    "alpha_selection_mode",
    "split_seed",
    "feature_reduction",
    "source_dir",
    "metadata_num_items",
    "metadata_config_path",
]

ENCODING_TARGET_FIELDNAMES = [
    "dataset",
    "model",
    "subject_id",
    "roi",
    "layer",
    "metric",
    "metric_scope",
    "target_index",
    "pearson_r",
    "r2_score_from_r",
    "prediction_r2",
    "noise_ceiling",
    "noise_normalized_score",
    "valid_noise_ceiling",
    "valid_prediction_variance",
    "valid_target_variance",
    "n_train",
    "n_test",
    "selected_ridge_alpha",
    "alpha_selection_mode",
    "split_seed",
    "feature_reduction",
    "source_dir",
    "metadata_num_items",
    "metadata_config_path",
]

RSA_FIELDNAMES = [
    "dataset",
    "model",
    "subject_id",
    "roi",
    "layer",
    "model_rdm_metric",
    "response_rdm_metric",
    "compare_method",
    "n_items",
    "score",
    "source_dir",
    "metadata_num_items",
    "metadata_config_path",
]

BEST_LAYER_FIELDNAMES = [
    "score_type",
    "model",
    "subject_id",
    "roi",
    "layer",
    "metric",
    "compare_method",
    "score",
    "encoding_score_type",
    "raw_score",
    "median_score",
    "std_score",
    "mean_noise_normalized_score",
    "median_noise_normalized_score",
    "valid_noise_ceiling_targets",
    "zero_noise_ceiling_targets",
    "invalid_noise_ceiling_targets",
    "n_train",
    "n_test",
    "n_items",
    "num_targets",
    "dataset",
    "source_dir",
]

BRIDGE_FIELDNAMES = [
    "behavior_dataset",
    "behavior_metric",
    "behavior_model",
    "behavior_saliency_method",
    "behavior_saliency_family",
    "behavior_n",
    "behavior_mean",
    "behavior_ci95_low",
    "behavior_ci95_high",
    "neural_model",
    "subject_id",
    "roi",
    "best_encoding_layer",
    "best_encoding_metric",
    "best_encoding_score",
    "best_encoding_n_train",
    "best_encoding_n_test",
    "best_rsa_layer",
    "best_rsa_method",
    "best_rsa_score",
    "best_rsa_n_items",
]

MODEL_SUMMARY_FIELDNAMES = [
    "model",
    "behavior_dataset",
    "behavior_metric",
    "behavior_saliency_method",
    "behavior_saliency_family",
    "num_rois",
    "behavior_mean",
    "mean_best_encoding_score_across_rois",
    "mean_best_rsa_score_across_rois",
]

PAPER_WINNER_FIELDNAMES = [
    "model",
    "subject_id",
    "roi",
    "best_encoding_layer",
    "best_encoding_metric",
    "best_encoding_score",
    "best_encoding_score_type",
    "best_encoding_raw_score",
    "best_encoding_median_score",
    "best_encoding_std_score",
    "best_encoding_mean_noise_normalized_score",
    "best_encoding_median_noise_normalized_score",
    "valid_noise_ceiling_targets",
    "zero_noise_ceiling_targets",
    "invalid_noise_ceiling_targets",
    "best_encoding_n_train",
    "best_encoding_n_test",
    "best_rsa_layer",
    "best_rsa_method",
    "best_rsa_score",
    "best_rsa_n_items",
    "encoding_source_dir",
    "rsa_source_dir",
]

NEURAL_RANKING_FIELDNAMES = [
    "model",
    "num_encoding_rois",
    "num_rsa_rois",
    "mean_encoding_score",
    "rank_mean_encoding",
    "mean_noise_normalized_score",
    "mean_noise_normalized_score_x100",
    "rank_mean_noise_normalized",
    "valid_noise_ceiling_targets",
    "zero_noise_ceiling_targets",
    "invalid_noise_ceiling_targets",
    "mean_rsa_score",
    "rank_mean_rsa",
    "latency_mean_ms",
    "parameter_count",
    "model_size_mb",
    "flops",
    "encoding_score_per_latency_mean_ms",
    "rank_encoding_per_latency",
    "rsa_score_per_latency_mean_ms",
    "rank_rsa_per_latency",
    "encoding_score_per_million_parameters",
    "rank_encoding_per_million_parameters",
    "rsa_score_per_million_parameters",
    "rank_rsa_per_million_parameters",
    "encoding_score_per_model_size_mb",
    "rsa_score_per_model_size_mb",
    "encoding_score_per_gflop",
    "rsa_score_per_gflop",
]

LEARNED_READOUT_COMPARISON_FIELDNAMES = [
    "model",
    "subject_id",
    "roi",
    "metric",
    "flatten_pca_layer",
    "learned_readout_layer",
    "flatten_pca_raw_score",
    "learned_readout_raw_score",
    "raw_delta",
    "flatten_pca_noise_normalized_score",
    "learned_readout_noise_normalized_score",
    "noise_normalized_delta",
    "flatten_pca_valid_noise_ceiling_targets",
    "learned_readout_valid_noise_ceiling_targets",
    "flatten_pca_zero_noise_ceiling_targets",
    "learned_readout_zero_noise_ceiling_targets",
    "flatten_pca_invalid_noise_ceiling_targets",
    "learned_readout_invalid_noise_ceiling_targets",
    "flatten_pca_n_train",
    "learned_readout_n_train",
    "flatten_pca_n_test",
    "learned_readout_n_test",
    "flatten_pca_source_dir",
    "learned_readout_source_dir",
]

BEHAVIOR_NEURAL_ALIGNMENT_FIELDNAMES = [
    "model",
    "behavior_dataset",
    "behavior_metric",
    "behavior_metric_direction",
    "behavior_saliency_method",
    "behavior_saliency_family",
    "num_rois",
    "behavior_mean",
    "mean_encoding_score",
    "mean_noise_normalized_score",
    "mean_rsa_score",
    "rank_mean_encoding",
    "rank_mean_noise_normalized",
    "rank_mean_rsa",
    "rank_encoding_per_latency",
    "rank_rsa_per_latency",
    "interpretation_scope",
]

LEADER_OVERLAP_FIELDNAMES = [
    "behavior_dataset",
    "behavior_metric",
    "behavior_metric_direction",
    "behavior_saliency_method",
    "behavior_saliency_family",
    "behavior_leader_model",
    "behavior_leader_mean",
    "neural_encoding_leader_model",
    "neural_encoding_leader_mean_score",
    "matches_encoding_leader",
    "neural_rsa_leader_model",
    "neural_rsa_leader_mean_score",
    "matches_rsa_leader",
    "interpretation_scope",
]

EFFICIENCY_FIELDNAMES = [
    *BEST_LAYER_FIELDNAMES,
    "latency_mean_ms",
    "parameter_count",
    "model_size_mb",
    "flops",
    "score_per_latency_mean_ms",
    "score_per_million_parameters",
    "score_per_model_size_mb",
    "score_per_gflop",
]
