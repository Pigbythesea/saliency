"""Create paper-style figures and tables for the current HMA milestone."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

from hma.utils.paths import ensure_dir


DEFAULT_BEHAVIORAL_CSV = Path("outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv")
DEFAULT_NEURAL_DIR = Path("outputs/neural_roi_summary")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_inspection_v1")
LOWER_IS_BETTER = {"kl", "emd", "emd_2d", "mae", "mse", "rmse", "loss"}
DATASET_LABELS = {
    "salicon_static2000": "SALICON",
    "cat2000_static2000": "CAT2000",
    "coco_search18_static2000": "COCO-Search18",
}
MODEL_LABELS = {
    "center_bias_baseline": "Center bias",
    "random_baseline": "Random",
    "deepgaze_reference": "DeepGaze IIE",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T",
    "deit_small_patch16_224": "DeiT-S/16",
    "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
    "vit_small_patch14_dinov2": "DINOv2 ViT-S/14",
    "vit_base_patch16_clip_224": "CLIP ViT-B/16",
    "resnet50_clip": "CLIP ResNet-50",
}
METHOD_LABELS = {
    "center_bias": "Center bias",
    "random_saliency": "Random",
    "deepgaze_precomputed": "DeepGaze",
    "vanilla_gradient": "Gradient",
    "gradcam": "Grad-CAM",
    "attention_rollout": "Attention rollout",
    "integrated_gradients": "Integrated gradients",
    "occlusion": "Occlusion",
}
FAMILY_COLORS = {
    "baseline": "#4b5563",
    "reference": "#0f766e",
    "evidence_sensitivity": "#2563eb",
    "class_localization": "#dc2626",
    "internal_routing": "#f59e0b",
    "perturbation": "#7c3aed",
}
MODEL_COLORS = {
    "resnet50": "#64748b",
    "convnext_tiny": "#0891b2",
    "deit_small_patch16_224": "#c2410c",
    "vit_base_patch16_224": "#7c3aed",
    "vit_small_patch14_dinov2": "#16a34a",
    "vit_base_patch16_clip_224": "#db2777",
    "resnet50_clip": "#0f766e",
}
ROI_ORDER = ["V1", "V2", "V3", "hV4"]
LITERATURE_SANITY_RANGES = {
    "cat2000_static2000": {
        "benchmark": "MIT/Tuebingen CAT2000",
        "center_bias_nss": "2.087",
        "deepgaze_iie_nss": "2.112",
        "note": "Free-viewing reference range; local subsets should preserve ordering scale.",
    },
    "coco_search18_static2000": {
        "benchmark": "COCO-Search18 task-driven search",
        "center_bias_nss": "not directly comparable",
        "deepgaze_iie_nss": "not directly comparable",
        "note": "Task-trained CNN reported NSS 4.64, AUC-Judd 0.95, sAUC 0.84, CC 0.72.",
    },
    "salicon_static2000": {
        "benchmark": "SALICON / MIT-style free-viewing",
        "center_bias_nss": "dataset dependent",
        "deepgaze_iie_nss": "1.996 on SALICON test",
        "note": "DeepGaze-class models should exceed generic center bias under point-NSS.",
    },
}


def create_paper_inspection_pack(
    *,
    behavioral_csv: str | Path = DEFAULT_BEHAVIORAL_CSV,
    neural_dir: str | Path = DEFAULT_NEURAL_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write paper-style figures, compact tables, and a report README."""
    behavioral_path = Path(behavioral_csv)
    neural_root = Path(neural_dir)
    output_root = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_root / "figures")
    tables_dir = ensure_dir(output_root / "tables")

    behavioral_rows = _load_csv_rows(behavioral_path)
    model_rankings = _load_csv_rows(neural_root / "neural_model_rankings.csv")
    matched_panel_rankings = _load_optional_csv_rows(
        neural_root / "matched_full_panel_model_rankings.csv"
    )
    roi_winners = _load_csv_rows(neural_root / "paper_model_roi_winners.csv")
    overlap_rows = _load_csv_rows(neural_root / "behavior_neural_leader_overlap.csv")
    matched_cross_level_correlation_rows = _load_optional_csv_rows(
        neural_root / "matched_cross_level_correlations.csv"
    )
    matched_geometry_model_ranking_rows = _load_optional_csv_rows(
        neural_root / "matched_geometry_model_rankings.csv"
    )
    candidate_rows = _load_csv_rows(neural_root / "ssl_multimodal_candidate_inventory.csv")
    learned_readout_comparison_rows = _load_optional_csv_rows(
        neural_root / "learned_readout_vs_flatten_pca.csv"
    )

    static_nss_rows = _static_metric_rows(behavioral_rows, "nss")
    behavior_table = _top_behavior_rows(static_nss_rows, limit_per_dataset=8)
    neural_table = _neural_ranking_table(model_rankings)
    matched_panel_table = _neural_ranking_table(matched_panel_rankings)
    roi_table = _roi_winner_table(roi_winners)
    overlap_table = _overlap_summary_table(overlap_rows)
    candidate_table = _candidate_table(candidate_rows)
    sanity_table = _benchmark_sanity_table(behavioral_rows)
    learned_readout_table = _learned_readout_comparison_table(
        learned_readout_comparison_rows
    )
    matched_cross_level_table = _matched_cross_level_correlation_table(
        matched_cross_level_correlation_rows
    )
    matched_geometry_table = _matched_geometry_ranking_table(
        matched_geometry_model_ranking_rows
    )

    outputs: dict[str, Path] = {}
    outputs["behavior_table_csv"] = _write_rows(
        tables_dir / "table1_behavior_static2000_nss_top.csv",
        behavior_table,
        list(behavior_table[0]) if behavior_table else [],
    )
    outputs["behavior_table_md"] = _write_markdown_table(
        tables_dir / "table1_behavior_static2000_nss_top.md",
        behavior_table,
    )
    outputs["neural_table_csv"] = _write_rows(
        tables_dir / "table2_neural_model_rankings.csv",
        neural_table,
        list(neural_table[0]) if neural_table else [],
    )
    outputs["neural_table_md"] = _write_markdown_table(
        tables_dir / "table2_neural_model_rankings.md",
        neural_table,
    )
    outputs["matched_panel_table_csv"] = _write_rows(
        tables_dir / "table8_matched_full_panel_model_rankings.csv",
        matched_panel_table,
        list(matched_panel_table[0]) if matched_panel_table else [],
    )
    outputs["matched_panel_table_md"] = _write_markdown_table(
        tables_dir / "table8_matched_full_panel_model_rankings.md",
        matched_panel_table,
    )
    outputs["roi_table_csv"] = _write_rows(
        tables_dir / "table3_roi_winners.csv",
        roi_table,
        list(roi_table[0]) if roi_table else [],
    )
    outputs["roi_table_md"] = _write_markdown_table(
        tables_dir / "table3_roi_winners.md",
        roi_table,
    )
    outputs["overlap_table_csv"] = _write_rows(
        tables_dir / "table4_behavior_neural_overlap_summary.csv",
        overlap_table,
        list(overlap_table[0]) if overlap_table else [],
    )
    outputs["overlap_table_md"] = _write_markdown_table(
        tables_dir / "table4_behavior_neural_overlap_summary.md",
        overlap_table,
    )
    outputs["candidate_table_csv"] = _write_rows(
        tables_dir / "table5_ssl_multimodal_candidates.csv",
        candidate_table,
        list(candidate_table[0]) if candidate_table else [],
    )
    outputs["candidate_table_md"] = _write_markdown_table(
        tables_dir / "table5_ssl_multimodal_candidates.md",
        candidate_table,
    )
    outputs["sanity_table_csv"] = _write_rows(
        tables_dir / "table6_benchmark_sanity_ranges.csv",
        sanity_table,
        list(sanity_table[0]) if sanity_table else [],
    )
    outputs["sanity_table_md"] = _write_markdown_table(
        tables_dir / "table6_benchmark_sanity_ranges.md",
        sanity_table,
    )
    outputs["learned_readout_comparison_csv"] = _write_rows(
        tables_dir / "table7_learned_readout_vs_flatten_pca.csv",
        learned_readout_table,
        list(learned_readout_table[0]) if learned_readout_table else [],
    )
    outputs["learned_readout_comparison_md"] = _write_markdown_table(
        tables_dir / "table7_learned_readout_vs_flatten_pca.md",
        learned_readout_table,
    )
    outputs["matched_cross_level_correlations_csv"] = _write_rows(
        tables_dir / "table9_matched_cross_level_correlations.csv",
        matched_cross_level_table,
        list(matched_cross_level_table[0]) if matched_cross_level_table else [],
    )
    outputs["matched_cross_level_correlations_md"] = _write_markdown_table(
        tables_dir / "table9_matched_cross_level_correlations.md",
        matched_cross_level_table,
    )
    outputs["matched_geometry_rankings_csv"] = _write_rows(
        tables_dir / "table10_matched_geometry_model_rankings.csv",
        matched_geometry_table,
        list(matched_geometry_table[0]) if matched_geometry_table else [],
    )
    outputs["matched_geometry_rankings_md"] = _write_markdown_table(
        tables_dir / "table10_matched_geometry_model_rankings.md",
        matched_geometry_table,
    )

    outputs.update(
        _plot_behavior_nss(static_nss_rows, figures_dir / "figure1_behavior_static2000_nss")
    )
    outputs.update(
        _plot_neural_rankings(model_rankings, figures_dir / "figure2_neural_model_rankings")
    )
    outputs.update(_plot_roi_heatmaps(roi_winners, figures_dir / "figure3_roi_heatmaps"))
    outputs.update(
        _plot_overlap_summary(
            overlap_table,
            figures_dir / "figure4_behavior_neural_leader_overlap",
        )
    )
    if matched_cross_level_table:
        outputs.update(
            _plot_matched_cross_level_correlations(
                matched_cross_level_table,
                figures_dir / "figure5_matched_cross_level_correlations",
            )
        )
    outputs["readme"] = _write_readme(
        output_root / "README.md",
        behavior_table=behavior_table,
        neural_table=neural_table,
        matched_panel_table=matched_panel_table,
        overlap_table=overlap_table,
        candidate_table=candidate_table,
        sanity_table=sanity_table,
        learned_readout_table=learned_readout_table,
        matched_cross_level_table=matched_cross_level_table,
        matched_geometry_table=matched_geometry_table,
        outputs=outputs,
        behavioral_csv=behavioral_path,
    )
    return outputs


def _top_behavior_rows(rows: list[dict[str, str]], *, limit_per_dataset: int) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "dataset")
    table_rows: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        ranked = sorted(grouped[dataset], key=lambda row: _float(row["mean"]), reverse=True)
        for rank, row in enumerate(ranked[:limit_per_dataset], start=1):
            table_rows.append(
                {
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "rank": rank,
                    "model": MODEL_LABELS.get(row["model"], row["model"]),
                    "saliency_method": METHOD_LABELS.get(
                        row["saliency_method"],
                        row["saliency_method"],
                    ),
                    "saliency_family": row.get("saliency_family", ""),
                    "fixation_protocol": row.get("fixation_protocol", "unknown"),
                    "nss_mean": _fmt(row.get("mean")),
                    "ci95": f"[{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}]",
                    "n": row.get("n", ""),
                }
            )
    return table_rows


def _neural_ranking_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    use_normalized = any(row.get("rank_mean_noise_normalized") for row in rows)
    rank_key = "rank_mean_noise_normalized" if use_normalized else "rank_mean_encoding"
    ranked = sorted(rows, key=lambda row: int(float(row.get(rank_key) or 999)))
    return [
        {
            "model": MODEL_LABELS.get(row["model"], row["model"]),
            "mean_noise_normalized": _fmt(row.get("mean_noise_normalized_score")),
            "mean_noise_normalized_x100": _fmt(row.get("mean_noise_normalized_score_x100")),
            "noise_normalized_rank": row.get("rank_mean_noise_normalized", ""),
            "mean_encoding": _fmt(row.get("mean_encoding_score")),
            "encoding_rank": row.get("rank_mean_encoding", ""),
            "mean_rsa": _fmt(row.get("mean_rsa_score")),
            "rsa_rank": row.get("rank_mean_rsa", ""),
            "valid_noise_ceiling_targets": row.get("valid_noise_ceiling_targets", ""),
            "zero_noise_ceiling_targets": row.get("zero_noise_ceiling_targets", ""),
            "invalid_noise_ceiling_targets": row.get("invalid_noise_ceiling_targets", ""),
            "encoding_per_latency_rank": row.get("rank_encoding_per_latency", ""),
            "rsa_per_latency_rank": row.get("rank_rsa_per_latency", ""),
            "latency_ms": _fmt(row.get("latency_mean_ms")),
            "params_m": _fmt(_float(row.get("parameter_count")) / 1_000_000.0),
        }
        for row in ranked
    ]


def _benchmark_sanity_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("saliency_method", ""),
            row.get("metric", ""),
        ): row
        for row in rows
    }
    table: list[dict[str, Any]] = []
    for dataset, reference in LITERATURE_SANITY_RANGES.items():
        center = by_key.get((dataset, "center_bias_baseline", "center_bias", "nss"), {})
        deepgaze = by_key.get((dataset, "deepgaze_reference", "deepgaze_precomputed", "nss"), {})
        protocol = (
            deepgaze.get("fixation_protocol")
            or center.get("fixation_protocol")
            or "unknown"
        )
        table.append(
            {
                "dataset": DATASET_LABELS.get(dataset, dataset),
                "fixation_protocol": protocol,
                "local_center_bias_nss": _fmt(center.get("mean")) if center else "",
                "local_deepgaze_iie_nss": _fmt(deepgaze.get("mean")) if deepgaze else "",
                "literature_center_bias_nss": reference["center_bias_nss"],
                "literature_deepgaze_iie_nss": reference["deepgaze_iie_nss"],
                "benchmark": reference["benchmark"],
                "note": reference["note"],
            }
        )
    return table


def _roi_winner_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            MODEL_LABELS.get(row["model"], row["model"]),
            ROI_ORDER.index(row["roi"]) if row["roi"] in ROI_ORDER else 99,
        ),
    )
    return [
        {
            "model": MODEL_LABELS.get(row["model"], row["model"]),
            "roi": row["roi"],
            "encoding_layer": row.get("best_encoding_layer", ""),
            "encoding_score_type": row.get("best_encoding_score_type", ""),
            "noise_normalized_encoding": _fmt(
                row.get("best_encoding_mean_noise_normalized_score")
            ),
            "raw_encoding": _fmt(row.get("best_encoding_raw_score")),
            "encoding_score": _fmt(row.get("best_encoding_score")),
            "rsa_layer": row.get("best_rsa_layer", ""),
            "rsa_score": _fmt(row.get("best_rsa_score")),
        }
        for row in ordered
    ]


def _overlap_summary_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = _group_by(rows, "behavior_saliency_family")
    table: list[dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        count = len(family_rows)
        encoding_matches = sum(row.get("matches_encoding_leader") == "true" for row in family_rows)
        rsa_matches = sum(row.get("matches_rsa_leader") == "true" for row in family_rows)
        table.append(
            {
                "saliency_family": family,
                "groups": count,
                "encoding_leader_matches": encoding_matches,
                "encoding_match_rate": _fmt(encoding_matches / count if count else 0.0),
                "rsa_leader_matches": rsa_matches,
                "rsa_match_rate": _fmt(rsa_matches / count if count else 0.0),
            }
        )
    total = len(rows)
    if total:
        encoding_total = sum(row.get("matches_encoding_leader") == "true" for row in rows)
        rsa_total = sum(row.get("matches_rsa_leader") == "true" for row in rows)
        table.append(
            {
                "saliency_family": "all",
                "groups": total,
                "encoding_leader_matches": encoding_total,
                "encoding_match_rate": _fmt(encoding_total / total),
                "rsa_leader_matches": rsa_total,
                "rsa_match_rate": _fmt(rsa_total / total),
            }
        )
    return table


def _candidate_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "family": row.get("family", ""),
            "model": row.get("model_name", ""),
            "available": row.get("available_in_timm", ""),
            "verified_layers": row.get("verified_layers", ""),
            "pretrained_run": row.get("pretrained_weights_run", ""),
            "pretrained_status": row.get("pretrained_run_status", ""),
            "pretrained_weight_status": row.get("pretrained_weight_status", ""),
            "debug_config": Path(row.get("debug_config_path", "")).name
            if row.get("debug_config_path")
            else "",
            "pretrained_debug_config": Path(row.get("pretrained_debug_config_path", "")).name
            if row.get("pretrained_debug_config_path")
            else "",
        }
        for row in rows
    ]


def _learned_readout_comparison_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: ROI_ORDER.index(row["roi"]) if row.get("roi") in ROI_ORDER else 99,
    )
    return [
        {
            "model": MODEL_LABELS.get(row.get("model", ""), row.get("model", "")),
            "roi": row.get("roi", ""),
            "flatten_pca_layer": row.get("flatten_pca_layer", ""),
            "learned_readout_layer": row.get("learned_readout_layer", ""),
            "flatten_pca_raw": _fmt(row.get("flatten_pca_raw_score")),
            "learned_readout_raw": _fmt(row.get("learned_readout_raw_score")),
            "raw_delta": _fmt(row.get("raw_delta")),
            "flatten_pca_noise_normalized": _fmt(
                row.get("flatten_pca_noise_normalized_score")
            ),
            "learned_readout_noise_normalized": _fmt(
                row.get("learned_readout_noise_normalized_score")
            ),
            "noise_normalized_delta": _fmt(row.get("noise_normalized_delta")),
            "valid_targets": row.get("learned_readout_valid_noise_ceiling_targets", ""),
            "zero_ceiling_targets": row.get("learned_readout_zero_noise_ceiling_targets", ""),
        }
        for row in ordered
    ]


def _matched_cross_level_correlation_table(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("status") != "complete",
            DATASET_LABELS.get(row.get("behavior_dataset", ""), row.get("behavior_dataset", "")),
            row.get("behavior_metric", ""),
            row.get("behavior_saliency_method", ""),
            row.get("neural_scope", ""),
            ROI_ORDER.index(row["roi_or_mean"])
            if row.get("roi_or_mean") in ROI_ORDER
            else 99,
            row.get("roi_or_mean", ""),
        ),
    )
    return [
        {
            "dataset": DATASET_LABELS.get(row.get("behavior_dataset", ""), row.get("behavior_dataset", "")),
            "metric": row.get("behavior_metric", ""),
            "direction": row.get("behavior_metric_direction", ""),
            "saliency_method": METHOD_LABELS.get(
                row.get("behavior_saliency_method", ""),
                row.get("behavior_saliency_method", ""),
            ),
            "saliency_family": row.get("behavior_saliency_family", ""),
            "roi_or_mean": row.get("roi_or_mean", ""),
            "n_models": row.get("n_models", ""),
            "status": row.get("status", ""),
            "spearman_noise_normalized": _fmt(
                row.get("spearman_behavior_vs_noise_normalized")
            )
            if row.get("spearman_behavior_vs_noise_normalized")
            else "",
            "spearman_raw_encoding": _fmt(row.get("spearman_behavior_vs_raw_encoding"))
            if row.get("spearman_behavior_vs_raw_encoding")
            else "",
            "ols_noise_normalized_slope": _fmt(row.get("ols_noise_normalized_slope"))
            if row.get("ols_noise_normalized_slope")
            else "",
            "ols_noise_normalized_r2": _fmt(row.get("ols_noise_normalized_r2"))
            if row.get("ols_noise_normalized_r2")
            else "",
            "ols_raw_encoding_slope": _fmt(row.get("ols_raw_encoding_slope"))
            if row.get("ols_raw_encoding_slope")
            else "",
            "ols_raw_encoding_r2": _fmt(row.get("ols_raw_encoding_r2"))
            if row.get("ols_raw_encoding_r2")
            else "",
            "geometry_method": row.get("geometry_method", ""),
            "spearman_geometry": _fmt(row.get("spearman_behavior_vs_geometry"))
            if row.get("spearman_behavior_vs_geometry")
            else "",
            "ols_geometry_r2": _fmt(row.get("ols_geometry_r2"))
            if row.get("ols_geometry_r2")
            else "",
            "spearman_encoding_geometry": _fmt(row.get("spearman_encoding_vs_geometry"))
            if row.get("spearman_encoding_vs_geometry")
            else "",
        }
        for row in ordered
    ]


def _matched_geometry_ranking_table(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("geometry_method", ""),
            int(row.get("rank_mean_geometry") or 999),
            row.get("model", ""),
        ),
    )
    return [
        {
            "model": MODEL_LABELS.get(row.get("model", ""), row.get("model", "")),
            "geometry_method": row.get("geometry_method", ""),
            "num_rois": row.get("num_geometry_rois", ""),
            "mean_geometry_score": _fmt(row.get("mean_geometry_score")),
            "geometry_rank": row.get("rank_mean_geometry", ""),
            "rois": row.get("rois", ""),
            "scope": row.get("interpretation_scope", ""),
        }
        for row in ordered
    ]


def _plot_behavior_nss(rows: list[dict[str, str]], output_stem: Path) -> dict[str, Path]:
    plt = _pyplot()
    _set_style(plt)
    grouped = _group_by(rows, "dataset")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13.2, 5.2), sharex=False)
    for ax, dataset in zip(axes, sorted(grouped)):
        ranked = sorted(grouped[dataset], key=lambda row: _float(row["mean"]), reverse=True)[:8]
        ranked.reverse()
        labels = [
            f"{MODEL_LABELS.get(row['model'], row['model'])}\n{METHOD_LABELS.get(row['saliency_method'], row['saliency_method'])}"
            for row in ranked
        ]
        values = [_float(row["mean"]) for row in ranked]
        colors = [FAMILY_COLORS.get(row.get("saliency_family", ""), "#6b7280") for row in ranked]
        positions = range(len(ranked))
        ax.barh(list(positions), values, color=colors, height=0.72)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), loc="left", fontweight="bold")
        ax.set_xlabel("NSS")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for y, value in zip(positions, values):
            ax.text(value + 0.015, y, _fmt(value), va="center", fontsize=8)
    fig.suptitle("Static2000 Behavioral Alignment: Top NSS Rows", fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Higher NSS is better. Rows include baselines, references, and model saliency where available.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return _save_figure(fig, output_stem, plt, "behavior_nss")


def _plot_neural_rankings(rows: list[dict[str, str]], output_stem: Path) -> dict[str, Path]:
    plt = _pyplot()
    _set_style(plt)
    primary_field = (
        "mean_noise_normalized_score"
        if any(row.get("mean_noise_normalized_score") for row in rows)
        else "mean_encoding_score"
    )
    ranked = sorted(rows, key=lambda row: _float(row.get(primary_field)), reverse=True)
    labels = [MODEL_LABELS.get(row["model"], row["model"]) for row in ranked]
    colors = [MODEL_COLORS.get(row["model"], "#6b7280") for row in ranked]
    metrics = [
        (primary_field, "Mean noise-normalized encoding" if primary_field == "mean_noise_normalized_score" else "Mean encoding"),
        ("mean_encoding_score", "Mean raw encoding"),
        ("mean_rsa_score", "Mean RSA"),
        ("rsa_score_per_latency_mean_ms", "RSA / ms"),
    ]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.8, 7.2))
    for ax, (field, title) in zip(axes.ravel(), metrics):
        values = [_float(row.get(field)) for row in ranked]
        positions = range(len(ranked))
        ax.bar(list(positions), values, color=colors, width=0.7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for x, value in zip(positions, values):
            ax.text(x, value, _fmt(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Current Neural Alignment Across Models", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(fig, output_stem, plt, "neural_rankings")


def _plot_roi_heatmaps(rows: list[dict[str, str]], output_stem: Path) -> dict[str, Path]:
    plt = _pyplot()
    _set_style(plt)
    models = sorted({row["model"] for row in rows}, key=lambda model: MODEL_LABELS.get(model, model))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.2, 5.4))
    encoding_field = (
        "best_encoding_mean_noise_normalized_score"
        if any(row.get("best_encoding_mean_noise_normalized_score") for row in rows)
        else "best_encoding_score"
    )
    encoding_title = (
        "Best noise-normalized encoding"
        if encoding_field == "best_encoding_mean_noise_normalized_score"
        else "Best encoding score"
    )
    for ax, field, title in [
        (axes[0], encoding_field, encoding_title),
        (axes[1], "best_rsa_score", "Best RSA score"),
    ]:
        matrix = []
        for model in models:
            model_rows = {row["roi"]: row for row in rows if row["model"] == model}
            matrix.append([_float(model_rows.get(roi, {}).get(field)) for roi in ROI_ORDER])
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ROI_ORDER)))
        ax.set_xticklabels(ROI_ORDER)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
        ax.set_title(title, loc="left", fontweight="bold")
        for y, model in enumerate(models):
            for x, roi in enumerate(ROI_ORDER):
                value = matrix[y][x]
                layer_field = "best_encoding_layer" if field == encoding_field else "best_rsa_layer"
                layer = next(
                    (
                        row.get(layer_field, "")
                        for row in rows
                        if row["model"] == model and row["roi"] == roi
                    ),
                    "",
                )
                ax.text(x, y, f"{_fmt(value)}\n{layer}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle("Best Layer Per Model And PRF Visual ROI", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(fig, output_stem, plt, "roi_heatmaps")


def _plot_overlap_summary(rows: list[dict[str, Any]], output_stem: Path) -> dict[str, Path]:
    plt = _pyplot()
    _set_style(plt)
    display_rows = [row for row in rows if row["saliency_family"] != "all"]
    labels = [str(row["saliency_family"]).replace("_", "\n") for row in display_rows]
    encoding = [_float(row["encoding_match_rate"]) for row in display_rows]
    rsa = [_float(row["rsa_match_rate"]) for row in display_rows]
    x_positions = list(range(len(display_rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar([x - width / 2 for x in x_positions], encoding, width=width, color="#c2410c", label="Encoding leader")
    ax.bar([x + width / 2 for x in x_positions], rsa, width=width, color="#7c3aed", label="RSA leader")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Leader match rate")
    ax.set_title("Behavioral Leaders vs Neural Leaders", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    for x, value in zip([x - width / 2 for x in x_positions], encoding):
        ax.text(x, value + 0.025, _fmt(value), ha="center", fontsize=8)
    for x, value in zip([x + width / 2 for x in x_positions], rsa):
        ax.text(x, value + 0.025, _fmt(value), ha="center", fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Bridge is descriptive: one subject, mixed neural scope, static2000 behavioral rows.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save_figure(fig, output_stem, plt, "leader_overlap")


def _plot_matched_cross_level_correlations(
    rows: list[dict[str, Any]],
    output_stem: Path,
) -> dict[str, Path]:
    display_rows = [
        row
        for row in rows
        if row.get("status") == "complete"
        and row.get("metric") == "nss"
        and row.get("spearman_noise_normalized")
        and row.get("roi_or_mean") in {"V1", "V2", "V3", "hV4", "across_roi_mean"}
    ]
    if not display_rows:
        return {}
    display_rows = display_rows[:30]
    plt = _pyplot()
    _set_style(plt)
    labels = [
        f"{row.get('dataset', '')}\n{row.get('saliency_method', '')}\n{row.get('roi_or_mean', '')}"
        for row in display_rows
    ]
    values = [_float(row.get("spearman_noise_normalized")) for row in display_rows]
    colors = [
        FAMILY_COLORS.get(str(row.get("saliency_family", "")), "#6b7280")
        for row in display_rows
    ]
    positions = list(range(len(display_rows)))
    fig, ax = plt.subplots(figsize=(max(9.0, len(display_rows) * 0.45), 5.2))
    ax.bar(positions, values, color=colors, width=0.72)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Spearman rho")
    ax.set_title(
        "Matched Cross-Level NSS vs Noise-Normalized Encoding",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, value in zip(positions, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.025 if value >= 0 else -0.025
        ax.text(x, value + offset, _fmt(value), ha="center", va=va, fontsize=7)
    fig.text(
        0.01,
        0.01,
        "Model-level correlations across the matched six-model panel; one-subject local neural data.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save_figure(fig, output_stem, plt, "matched_cross_level_correlations")


def _write_readme(
    path: Path,
    *,
    behavior_table: list[dict[str, Any]],
    neural_table: list[dict[str, Any]],
    overlap_table: list[dict[str, Any]],
    candidate_table: list[dict[str, Any]],
    outputs: dict[str, Path],
    behavioral_csv: str | Path | None = None,
    sanity_table: list[dict[str, Any]] | None = None,
    learned_readout_table: list[dict[str, Any]] | None = None,
    matched_panel_table: list[dict[str, Any]] | None = None,
    matched_cross_level_table: list[dict[str, Any]] | None = None,
    matched_geometry_table: list[dict[str, Any]] | None = None,
) -> Path:
    best_behavior = behavior_table[0] if behavior_table else {}
    matched_rows = matched_panel_table or []
    matched_panel_complete = _matched_panel_complete(matched_rows)
    headline_neural_table = matched_rows if matched_panel_complete else neural_table
    normalized_available = any(row.get("noise_normalized_rank") for row in headline_neural_table)
    encoding_rank_key = "noise_normalized_rank" if normalized_available else "encoding_rank"
    best_neural_encoding = sorted(
        headline_neural_table,
        key=lambda row: int(row.get(encoding_rank_key) or 999),
    )[0]
    best_neural_rsa = sorted(neural_table, key=lambda row: int(row["rsa_rank"] or 999))[0]
    all_overlap = next((row for row in overlap_table if row["saliency_family"] == "all"), {})
    pretrained_count = sum(
        1 for row in candidate_table if str(row.get("pretrained_run", "")).lower() == "true"
    )
    status_counts = _candidate_status_counts(candidate_table)
    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    ) or "none"
    learned_readout_rows = learned_readout_table or []
    learned_improvements = sum(
        1 for row in learned_readout_rows if _float(row.get("raw_delta")) > 0.0
    )
    matched_cross_level_rows = matched_cross_level_table or []
    matched_geometry_rows = matched_geometry_table or []
    top_geometry = next(
        (
            row
            for row in matched_geometry_rows
            if row.get("geometry_method") == "linear_cka"
            and row.get("geometry_rank") == "1"
        ),
        matched_geometry_rows[0] if matched_geometry_rows else {},
    )
    complete_cross_level_rows = [
        row for row in matched_cross_level_rows if row.get("status") == "complete"
    ]
    top_cross_level = next(
        (
            row
            for row in complete_cross_level_rows
            if row.get("metric") == "nss"
            and row.get("roi_or_mean") == "across_roi_mean"
            and row.get("spearman_noise_normalized")
        ),
        complete_cross_level_rows[0] if complete_cross_level_rows else {},
    )
    lines = [
        "# Paper Inspection V1",
        "",
        "This directory is a human-inspection pack for the current HMA milestone. It uses existing frozen outputs only and does not run new model experiments.",
        f"Behavioral source CSV: `{behavioral_csv}`.",
        "",
        "## Headline Readout",
        "",
        f"- Top displayed behavioral NSS row: {best_behavior.get('dataset', '')} / {best_behavior.get('model', '')} / {best_behavior.get('saliency_method', '')}, NSS={best_behavior.get('nss_mean', '')}.",
        (
            f"- {'Matched-panel ' if matched_panel_complete else ''}Noise-normalized neural encoding leader: {best_neural_encoding.get('model', '')}, "
            f"mean noise-normalized score={best_neural_encoding.get('mean_noise_normalized', '')} "
            f"(x100={best_neural_encoding.get('mean_noise_normalized_x100', '')})."
            if normalized_available
            else f"- {'Matched-panel ' if matched_panel_complete else ''}Raw neural encoding leader: {best_neural_encoding.get('model', '')}, mean encoding={best_neural_encoding.get('mean_encoding', '')}."
        ),
        f"- Raw neural RSA leader: {best_neural_rsa.get('model', '')}, mean RSA={best_neural_rsa.get('mean_rsa', '')}.",
        (
            "- Matched representational-geometry leader: "
            f"{top_geometry.get('model', '')}, method={top_geometry.get('geometry_method', '')}, "
            f"mean score={top_geometry.get('mean_geometry_score', '')}."
            if matched_geometry_rows
            else "- Matched representational-geometry table is not available in this pack."
        ),
        (
            "- Matched full-image `flatten_pca` panel is complete and used for the encoding headline."
            if matched_panel_complete
            else f"- Matched full-image `flatten_pca` panel rows available: {len(matched_rows)}; mixed-scope ranking remains the headline until all expected model/ROI cells are complete."
        ),
        f"- Overall behavior-to-encoding leader match rate: {all_overlap.get('encoding_match_rate', '')}.",
        f"- Overall behavior-to-RSA leader match rate: {all_overlap.get('rsa_match_rate', '')}.",
        (
            "- Learned spatial readout improves over matched `flatten_pca` rows in "
            f"{learned_improvements}/{len(learned_readout_rows)} PRF ROI comparisons; "
            "these remain one-subject local baselines, not Algonauts leaderboard scores."
            if learned_readout_rows
            else "- Learned spatial readout versus `flatten_pca` comparison table is not available in this pack."
        ),
        (
            "- Matched cross-level correlation headline: "
            f"{top_cross_level.get('dataset', '')} / {top_cross_level.get('metric', '')} / "
            f"{top_cross_level.get('saliency_method', '')} / {top_cross_level.get('roi_or_mean', '')}, "
            f"Spearman rho={top_cross_level.get('spearman_noise_normalized', '')} against noise-normalized encoding "
            f"across n={top_cross_level.get('n_models', '')} matched models."
            if complete_cross_level_rows
            else "- Matched cross-level correlation table is not available in this pack."
        ),
        f"- SSL/multimodal candidates dry-inspected: {len(candidate_table)}; pretrained debug runs complete: {pretrained_count}.",
        f"- SSL/multimodal pretrained status counts: {status_text}.",
        "",
        "## Figures",
        "",
        "- `figures/figure1_behavior_static2000_nss.png`: top static2000 NSS rows by dataset.",
        "- `figures/figure2_neural_model_rankings.png`: noise-normalized encoding, raw encoding, legacy RSA, and latency-normalized RSA scores.",
        "- `figures/figure3_roi_heatmaps.png`: best noise-normalized encoding/RSA layers and scores by model and ROI.",
        "- `figures/figure4_behavior_neural_leader_overlap.png`: descriptive leader-overlap rates.",
        "- `figures/figure5_matched_cross_level_correlations.png`: matched model-level NSS correlations where available.",
        "",
        "## Tables",
        "",
        "- `tables/table1_behavior_static2000_nss_top.md`",
        "- `tables/table2_neural_model_rankings.md`",
        "- `tables/table3_roi_winners.md`",
        "- `tables/table4_behavior_neural_overlap_summary.md`",
        "- `tables/table5_ssl_multimodal_candidates.md`",
        "- `tables/table6_benchmark_sanity_ranges.md`",
        "- `tables/table7_learned_readout_vs_flatten_pca.md`",
        "- `tables/table8_matched_full_panel_model_rankings.md`",
        "- `tables/table9_matched_cross_level_correlations.md`",
        "- `tables/table10_matched_geometry_model_rankings.md`",
        "",
        "## Academic SOTA Context",
        "",
        "- Behavioral free-viewing: current local DeepGaze IIE reference rows are below or near published benchmark ranges, but preserve the expected ordering over center bias. The local CAT2000 DeepGaze IIE NSS is 1.838 versus MIT/Tuebingen CAT2000 DeepGaze IIE NSS 2.5265, and the local SALICON DeepGaze IIE NSS is 1.743 versus the DeepGaze IIE SALICON test NSS 1.996 reported in the ICCV 2021 supplement.",
        "- Behavioral task-driven search: COCO-Search18 rows are not directly comparable to free-viewing saliency SOTA. A COCO-Search18 task-trained CNN report gives NSS 4.64, AUC-Judd 0.95, sAUC 0.84, and CC 0.72; the local DeepGaze IIE row is a free-viewing reference applied to a task-search dataset, not a task-trained SOTA model.",
        "- Neural Algonauts/NSD: the current neural pack is one-subject and internal-split. It now combines ROI500 diagnostic rows with validation-selected full-image-count PRF visual ROI `flatten_pca` rows for `subj01`; the official Algonauts 2023 leaderboard still uses mean noise-normalized encoding accuracy across held-out test vertices, subjects, and hemispheres, so local ROI summaries are not leaderboard-comparable scores.",
        "- References: https://saliency.tuebingen.ai/results.html ; https://openaccess.thecvf.com/content/ICCV2021/supplemental/Linardos_DeepGaze_IIE_Calibrated_ICCV_2021_supplemental.pdf ; https://arxiv.org/abs/2210.15093 ; https://algonautsproject.com/2023/challenge.html",
        "",
        "## Behavioral Metric Boundary",
        "",
        "NSS/AUC rows are benchmark-equivalent only when `fixation_protocol` is `points` or `task_points`. Rows marked `density_fallback` or `unknown` should be treated as diagnostic and not compared to academic SOTA tables.",
        "Old static2000 NSS outputs generated before the point-fixation revision are superseded for scientific interpretation.",
        "",
        "## Interpretation Boundary",
        "",
        "Treat this pack as a paper-style inspection layer, not final evidence. The neural side is one subject with legacy ROI500 RSA diagnostics, validation-selected full-image-count PRF ROI encoding baselines, and matched scalable geometry when `table10` is present. Matched cross-level correlations are model-level summaries over a small six-model panel, not causal tests.",
        "Use matched geometry tables for representational-geometry claims; keep ROI500 RSA as continuity diagnostics only.",
        "Use the matched full-image `flatten_pca` table for cross-model encoding comparisons once all expected model/ROI cells are complete. Use leader-overlap tables only as descriptive continuity diagnostics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    outputs["readme"] = path
    return path


def _candidate_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("pretrained_status") or "not_run")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _matched_panel_complete(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 6:
        return False
    return all(int(float(row.get("valid_noise_ceiling_targets") or 0)) > 0 for row in rows)


def _static_metric_rows(rows: Iterable[dict[str, str]], metric: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("dataset", "").endswith("_static2000")
        and row.get("metric") == metric
        and _is_finite(row.get("mean"))
    ]


def _group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return path
    fieldnames = list(rows[0])
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_figure(fig: Any, output_stem: Path, plt: Any, label: str) -> dict[str, Path]:
    ensure_dir(output_stem.parent)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {f"{label}_png": png_path, f"{label}_pdf": pdf_path}


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_optional_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    return _load_csv_rows(csv_path)


def _fmt(value: Any) -> str:
    number = _float(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _set_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-style figures and tables for current HMA outputs."
    )
    parser.add_argument("--behavioral-csv", default=str(DEFAULT_BEHAVIORAL_CSV))
    parser.add_argument("--neural-dir", default=str(DEFAULT_NEURAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = create_paper_inspection_pack(
        behavioral_csv=args.behavioral_csv,
        neural_dir=args.neural_dir,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
