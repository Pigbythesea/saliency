from scripts.create_paper_inspection_pack import (
    MODEL_COLORS,
    MODEL_LABELS,
    _candidate_table,
    _neural_ranking_table,
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
