import csv

from scripts.create_paper_inspection_pack import (
    MODEL_COLORS,
    MODEL_LABELS,
    _candidate_table,
    _cross_axis_decision_table,
    _geometry_agreement_table,
    _learned_readout_comparison_table,
    _load_optional_csv_rows,
    _matched_cross_level_correlation_table,
    _matched_geometry_ranking_table,
    _neural_ranking_table,
    _observer_control_summary_table,
    _sensitivity_table,
    _subject_robustness_interpretation_table,
    _top_behavior_rows,
    _write_readme,
)


def test_candidate_table_includes_pretrained_status_fields():
    rows = _candidate_table(
        [
            {
                "family": "DINOv2",
                "model_name": "vit_small_patch14_dinov2",
                "available_in_timm": "true",
                "verified_layers": "blocks.0 blocks.3",
                "pretrained_weights_run": "true",
                "pretrained_run_status": "complete",
                "pretrained_weight_status": "pretrained_true",
                "debug_config_path": "configs/dry.yaml",
                "pretrained_debug_config_path": "configs/pretrained.yaml",
            }
        ]
    )

    assert rows == [
        {
            "family": "DINOv2",
            "model": "vit_small_patch14_dinov2",
            "available": "true",
            "verified_layers": "blocks.0 blocks.3",
            "pretrained_run": "true",
            "pretrained_status": "complete",
            "pretrained_weight_status": "pretrained_true",
            "debug_config": "dry.yaml",
            "pretrained_debug_config": "pretrained.yaml",
        }
    ]


def test_readme_reports_dynamic_pretrained_counts(tmp_path):
    outputs = {}
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[
            {
                "dataset": "CAT2000",
                "model": "Center bias",
                "saliency_method": "Center bias",
                "nss_mean": "0.519",
            }
        ],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[
            {
                "saliency_family": "all",
                "encoding_match_rate": "0.5",
                "rsa_match_rate": "0.25",
            }
        ],
        candidate_table=[
            {"pretrained_run": "true", "pretrained_status": "complete"},
            {"pretrained_run": "false", "pretrained_status": "not_run"},
        ],
        outputs=outputs,
    )

    text = path.read_text(encoding="utf-8")
    assert "pretrained debug runs complete: 1" in text
    assert "complete=1" in text
    assert "not_run=1" in text
    assert "pretrained weights run: false" not in text


def test_neural_ranking_table_prefers_noise_normalized_rank():
    rows = _neural_ranking_table(
        [
            {
                "model": "raw_leader",
                "mean_encoding_score": "0.9",
                "rank_mean_encoding": "1",
                "mean_noise_normalized_score": "0.1",
                "mean_noise_normalized_score_x100": "10.0",
                "rank_mean_noise_normalized": "2",
                "mean_rsa_score": "0.2",
                "rank_mean_rsa": "1",
            },
            {
                "model": "normalized_leader",
                "mean_encoding_score": "0.2",
                "rank_mean_encoding": "2",
                "mean_noise_normalized_score": "0.8",
                "mean_noise_normalized_score_x100": "80.0",
                "rank_mean_noise_normalized": "1",
                "mean_rsa_score": "0.1",
                "rank_mean_rsa": "2",
            },
        ]
    )

    assert rows[0]["model"] == "normalized_leader"
    assert rows[0]["noise_normalized_rank"] == "1"
    assert rows[0]["mean_encoding"] == "0.200"
    assert rows[0]["mean_rsa"] == "0.100"


def test_readme_reports_noise_normalized_neural_leader_when_available(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "2",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[],
        candidate_table=[],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "Noise-normalized neural encoding leader: DINOv2 ViT-S/14" in text
    assert "Raw neural encoding leader" not in text
    assert "Raw neural RSA leader" in text


def test_readme_uses_complete_matched_panel_for_encoding_headline(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "Mixed Leader",
                "mean_noise_normalized": "0.9",
                "mean_noise_normalized_x100": "90",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.9",
                "mean_rsa": "0.2",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        matched_panel_table=[
            {
                "model": f"Matched {index}",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": str(index),
                "mean_encoding": "0.3",
                "mean_rsa": "",
                "encoding_rank": str(index),
                "rsa_rank": "",
                "valid_noise_ceiling_targets": "10",
            }
            for index in range(1, 7)
        ],
        overlap_table=[],
        candidate_table=[],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "Matched-panel Noise-normalized neural encoding leader: Matched 1" in text
    assert "matched full-image `flatten_pca` panel is complete" in text.lower()


def test_readme_reports_learned_readout_comparison_when_available(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[],
        candidate_table=[],
        learned_readout_table=[
            {"roi": "V1", "raw_delta": "0.05"},
            {"roi": "V2", "raw_delta": "0.02"},
        ],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "Learned spatial readout improves over matched `flatten_pca` rows in 2/2" in text
    assert "not Algonauts leaderboard scores" in text


def test_learned_readout_table_formats_matched_comparison_rows():
    rows = _learned_readout_comparison_table(
        [
            {
                "model": "vit_small_patch14_dinov2",
                "roi": "V1",
                "flatten_pca_layer": "blocks.3",
                "learned_readout_layer": "blocks.6",
                "flatten_pca_raw_score": "0.5",
                "learned_readout_raw_score": "0.7",
                "raw_delta": "0.2",
                "flatten_pca_noise_normalized_score": "0.6",
                "learned_readout_noise_normalized_score": "0.9",
                "noise_normalized_delta": "0.3",
                "learned_readout_valid_noise_ceiling_targets": "3",
                "learned_readout_zero_noise_ceiling_targets": "0",
            }
        ]
    )

    assert rows[0]["model"] == "DINOv2 ViT-S/14"
    assert rows[0]["raw_delta"] == "0.200"
    assert rows[0]["noise_normalized_delta"] == "0.300"


def test_optional_learned_readout_comparison_csv_can_be_missing(tmp_path):
    assert _load_optional_csv_rows(tmp_path / "missing.csv") == []


def test_matched_cross_level_table_formats_rows():
    rows = _matched_cross_level_correlation_table(
        [
            {
                "behavior_dataset": "salicon_static2000",
                "behavior_metric": "nss",
                "behavior_metric_direction": "higher_is_better",
                "behavior_saliency_method": "gradcam",
                "behavior_saliency_family": "class_localization",
                "neural_scope": "matched_full_image_flatten_pca_model_mean",
                "roi_or_mean": "across_roi_mean",
                "n_models": "6",
                "status": "complete",
                "spearman_behavior_vs_noise_normalized": "0.33333",
                "spearman_behavior_vs_raw_encoding": "0.5",
                "ols_noise_normalized_slope": "0.25",
                "ols_noise_normalized_r2": "0.125",
                "ols_raw_encoding_slope": "0.75",
                "ols_raw_encoding_r2": "0.625",
            }
        ]
    )

    assert rows[0]["dataset"] == "SALICON"
    assert rows[0]["saliency_method"] == "Grad-CAM"
    assert rows[0]["spearman_noise_normalized"] == "0.333"
    assert rows[0]["ols_raw_encoding_r2"] == "0.625"


def test_matched_geometry_table_formats_rows():
    rows = _matched_geometry_ranking_table(
        [
            {
                "model": "vit_small_patch14_dinov2",
                "geometry_method": "linear_cka_full9841",
                "num_geometry_rois": "4",
                "mean_geometry_score": "0.321",
                "rank_mean_geometry": "1",
                "rois": "V1;V2;V3;hV4",
                "interpretation_scope": "matched_full_image_flatten_pca_geometry",
            }
        ]
    )

    assert rows[0]["model"] == MODEL_LABELS["vit_small_patch14_dinov2"]
    assert rows[0]["geometry_method"] == "linear_cka_full9841"
    assert rows[0]["mean_geometry_score"] == "0.321"
    assert rows[0]["geometry_rank"] == "1"


def test_geometry_agreement_and_sensitivity_tables_format_rows():
    agreement = _geometry_agreement_table(
        [
            {
                "roi_or_mean": "across_roi_mean",
                "primary_geometry_method": "linear_cka_full9841",
                "sensitivity_geometry_method": "subset_rsa_corr_rdm_spearman_size128_seed123",
                "n_models": "6",
                "spearman_rank_agreement": "0.8",
                "kendall_rank_agreement": "0.6",
                "status": "complete",
            }
        ]
    )
    decisions = _cross_axis_decision_table(
        [
            {
                "behavior_dataset": "salicon_static2000",
                "behavior_metric": "nss",
                "behavior_saliency_method": "vanilla_gradient",
                "roi_or_mean": "across_roi_mean",
                "relationship": "behavior_vs_geometry",
                "baseline_spearman": "0.5",
                "min_leave_one_model_spearman": "0.4",
                "max_leave_one_model_spearman": "0.7",
                "decision_label": "stable_convergence",
            }
        ]
    )
    sensitivity = _sensitivity_table(
        [
            {
                "behavior_dataset": "salicon_static2000",
                "behavior_metric": "nss",
                "behavior_saliency_method": "vanilla_gradient",
                "roi_or_mean": "across_roi_mean",
                "relationship": "behavior_vs_geometry",
                "sensitivity_type": "leave_one_model",
                "omitted_unit": "resnet50",
                "sensitivity_spearman": "0.4",
                "status": "complete",
            }
        ]
    )

    assert agreement[0]["spearman"] == "0.800"
    assert decisions[0]["decision"] == "stable_convergence"
    assert sensitivity[0]["omitted_unit"] == "resnet50"


def test_subject_robustness_interpretation_preserves_geometry_first_claim():
    rows = _subject_robustness_interpretation_table(
        [
            {
                "subject_id": "subj02",
                "subject_encoding_leader": "vit_small_patch14_dinov2",
                "encoding_margin_label": "dinov2_supported",
                "encoding_mean_margin": "0.002",
                "encoding_ci95_low": "0.001",
                "encoding_ci95_high": "0.003",
                "geometry_cka_margin_label": "dinov2_supported",
                "geometry_cka_mean_margin": "0.014",
                "uncertainty_decision_label": "geometry_replicated_encoding_supported",
            },
            {
                "subject_id": "subj04",
                "subject_encoding_leader": "resnet50",
                "encoding_margin_label": "resnet50_supported",
                "encoding_mean_margin": "-0.002",
                "encoding_ci95_low": "-0.003",
                "encoding_ci95_high": "-0.001",
                "geometry_cka_margin_label": "dinov2_supported",
                "geometry_cka_mean_margin": "0.016",
                "uncertainty_decision_label": "geometry_replicated_encoding_resnet_supported",
            },
            {
                "subject_id": "all_confirmatory_subjects",
                "subject_encoding_leader": "vit_small_patch14_dinov2",
                "encoding_margin_label": "dinov2_supported",
                "encoding_mean_margin": "0.001",
                "encoding_ci95_low": "0.001",
                "encoding_ci95_high": "0.002",
                "geometry_cka_margin_label": "dinov2_supported",
                "geometry_cka_mean_margin": "0.016",
                "uncertainty_decision_label": "geometry_replicated_encoding_ambiguous",
            },
        ],
        [
            {"subject_id": "subj04", "roi": "V1", "support_label": "resnet50_supported"},
            {"subject_id": "subj04", "roi": "V2", "support_label": "dinov2_supported"},
        ],
        [
            {
                "subject_id": "subj04",
                "roi": "mean_prf_visualrois",
                "geometry_method": "linear_cka_full9841",
                "support_label": "dinov2_supported",
            },
            {
                "subject_id": "subj04",
                "roi": "mean_prf_visualrois",
                "geometry_method": "subset_rsa_size512",
                "support_label": "dinov2_supported",
            },
        ],
    )

    by_subject = {row["subject_id"]: row for row in rows}
    assert (
        by_subject["subj02"]["uncertainty_decision"]
        == "geometry_replicated_encoding_supported"
    )
    assert by_subject["subj04"]["encoding_leader"] == "ResNet-50"
    assert (
        by_subject["subj04"]["uncertainty_decision"]
        == "geometry_replicated_encoding_resnet_supported"
    )
    assert (
        by_subject["all_confirmatory_subjects"]["uncertainty_decision"]
        == "geometry_replicated_encoding_ambiguous"
    )
    assert "accepted claim" in by_subject["all_confirmatory_subjects"]["accepted_interpretation"]


def test_observer_control_summary_streams_rows_and_separates_regimes(tmp_path):
    salicon = tmp_path / "salicon.csv"
    coco = tmp_path / "coco.csv"
    _write_observer_rows(
        salicon,
        [
            ("img1", "worker1", "0.5", "0.8", "10"),
            ("img1", "worker2", "0.7", "0.9", "10"),
            ("img2", "worker1", "0.9", "0.95", "8"),
        ],
    )
    _write_observer_rows(
        coco,
        [
            ("img3", "subject1", "1.0", "0.91", "2"),
            ("img4", "subject2", "0.8", "0.89", "4"),
        ],
    )

    rows = _observer_control_summary_table(
        salicon_observer_controls=salicon,
        coco_observer_controls=coco,
    )
    by_dataset = {row["dataset"]: row for row in rows}

    assert by_dataset["SALICON"]["viewing_regime"] == "free_viewing"
    assert by_dataset["SALICON"]["row_count"] == "3"
    assert by_dataset["SALICON"]["image_count"] == "2"
    assert by_dataset["SALICON"]["subject_count"] == "2"
    assert by_dataset["SALICON"]["num_observers_min"] == "8"
    assert by_dataset["SALICON"]["num_observers_max"] == "10"
    assert by_dataset["SALICON"]["mean_inter_observer_nss"] == "0.700"
    assert by_dataset["SALICON"]["median_inter_observer_auc"] == "0.900"
    assert by_dataset["COCO-Search18"]["viewing_regime"] == "task_search"
    assert by_dataset["COCO-Search18"]["status"] == "complete"


def test_observer_control_summary_marks_missing_files_incomplete(tmp_path):
    rows = _observer_control_summary_table(
        salicon_observer_controls=tmp_path / "missing_salicon.csv",
        coco_observer_controls=tmp_path / "missing_coco.csv",
    )

    assert {row["status"] for row in rows} == {"incomplete"}
    assert {row["row_count"] for row in rows} == {"0"}


def test_readme_reports_matched_cross_level_and_descriptive_overlap(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[
            {
                "saliency_family": "all",
                "encoding_match_rate": "0.5",
                "rsa_match_rate": "0.0",
            }
        ],
        candidate_table=[],
        matched_cross_level_table=[
            {
                "dataset": "SALICON",
                "metric": "nss",
                "saliency_method": "Grad-CAM",
                "roi_or_mean": "across_roi_mean",
                "spearman_noise_normalized": "0.333",
                "n_models": "6",
                "status": "complete",
            }
        ],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "Matched cross-level correlation headline" in text
    assert "leader-overlap tables only as descriptive continuity diagnostics" in text
    assert "table9_matched_cross_level_correlations.md" in text


def test_readme_reports_subject_robustness_and_observer_context(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[],
        candidate_table=[],
        subject_interpretation_table=[
            {
                "subject_id": "all_confirmatory_subjects",
                "uncertainty_decision": "geometry_replicated_encoding_ambiguous",
                "accepted_interpretation": "accepted claim: geometry replicated, encoding ambiguous",
            }
        ],
        observer_control_table=[
            {
                "dataset": "SALICON",
                "viewing_regime": "free_viewing",
                "status": "complete",
                "row_count": "3",
            },
            {
                "dataset": "COCO-Search18",
                "viewing_regime": "task_search",
                "status": "complete",
                "row_count": "2",
            },
        ],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "geometry-first dissociation / measurement evidence" in text
    assert "not a universal DINOv2 encoding-win narrative" in text
    assert "human/interobserver context, not model performance rows" in text
    assert "SALICON/CAT2000 free-viewing interpretations must remain separate" in text
    assert "table14_subject_robustness_interpretation.md" in text
    assert "table15_observer_control_summary.md" in text
    assert "table16_attribution_family_cross_axis_interpretation.md" in text


def test_readme_reports_attribution_family_interpretation(tmp_path):
    path = _write_readme(
        tmp_path / "README.md",
        behavior_table=[],
        neural_table=[
            {
                "model": "DINOv2 ViT-S/14",
                "mean_noise_normalized": "0.8",
                "mean_noise_normalized_x100": "80",
                "noise_normalized_rank": "1",
                "mean_encoding": "0.3",
                "mean_rsa": "0.1",
                "encoding_rank": "1",
                "rsa_rank": "1",
            }
        ],
        overlap_table=[],
        candidate_table=[],
        attribution_family_interpretation_table=[
            {
                "dataset": "salicon_static2000",
                "saliency_family": "transformer_relevance",
                "behavior_metric_rows": "28",
                "cross_level_complete_rows": "35",
                "paper_interpretation": "transformer_relevance_improves_rollout_behavior_only",
            }
        ],
        outputs={},
    )

    text = path.read_text(encoding="utf-8")
    assert "Attribution-family interpretation" in text
    assert "transformer relevance rows=28" in text
    assert "cross-level complete=35" in text
    assert "table16_attribution_family_cross_axis_interpretation.md" in text


def test_optional_matched_cross_level_csv_can_be_missing(tmp_path):
    assert _load_optional_csv_rows(tmp_path / "missing_cross_level.csv") == []


def test_ssl_model_labels_and_colors_are_defined():
    assert MODEL_LABELS["vit_small_patch14_dinov2"] == "DINOv2 ViT-S/14"
    assert MODEL_LABELS["vit_base_patch16_clip_224"] == "CLIP ViT-B/16"
    assert MODEL_LABELS["resnet50_clip"] == "CLIP ResNet-50"
    assert MODEL_COLORS["vit_small_patch14_dinov2"]
    assert MODEL_COLORS["vit_base_patch16_clip_224"]
    assert MODEL_COLORS["resnet50_clip"]


def test_behavior_table_can_surface_dinov2_merged_behavior_rows():
    rows = _top_behavior_rows(
        [
            {
                "dataset": "salicon_static2000",
                "model": "vit_small_patch14_dinov2",
                "saliency_method": "attention_rollout",
                "saliency_family": "internal_routing",
                "mean": "0.41",
                "ci95_low": "0.4",
                "ci95_high": "0.42",
                "n": "2000",
            },
            {
                "dataset": "salicon_static2000",
                "model": "resnet50",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "mean": "0.34",
                "ci95_low": "0.33",
                "ci95_high": "0.35",
                "n": "2000",
            },
        ],
        limit_per_dataset=8,
    )

    assert rows[0]["model"] == "DINOv2 ViT-S/14"
    assert rows[0]["saliency_method"] == "Attention rollout"


def _write_observer_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "subject_id",
                "trial_id",
                "inter_observer_nss",
                "inter_observer_auc",
                "num_observers",
                "control_type",
                "source",
            ],
        )
        writer.writeheader()
        for image_id, subject_id, nss, auc, num_observers in rows:
            writer.writerow(
                {
                    "image_id": image_id,
                    "subject_id": subject_id,
                    "trial_id": "",
                    "inter_observer_nss": nss,
                    "inter_observer_auc": auc,
                    "num_observers": num_observers,
                    "control_type": "leave_one_observer_out",
                    "source": "test",
                }
            )
