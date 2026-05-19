import csv
import json

from hma.experiments.summarize_neural_roi_results import summarize_neural_roi_results
from hma.utils.config import load_yaml
from scripts.create_neural_roi500_configs import (
    create_neural_roi500_configs,
    create_ssl_multimodal_debug_configs,
    inspect_ssl_multimodal_candidates,
)


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_neural_output(path, *, roi, model="resnet50", include_rsa=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "config_path": f"configs/experiments/{roi}.yaml",
                "num_items": 500,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        path / "encoding_scores.csv",
        [
            {
                "dataset": f"nsd_{roi}",
                "model": model,
                "subject_id": "subj01",
                "roi": roi,
                "layer": "layer1",
                "metric": "correlation",
                "n_train": 400,
                "n_test": 100,
                "num_targets": 10,
                "mean_score": 0.1,
                "median_score": 0.09,
                "std_score": 0.01,
            },
            {
                "dataset": f"nsd_{roi}",
                "model": model,
                "subject_id": "subj01",
                "roi": roi,
                "layer": "layer2",
                "metric": "correlation",
                "n_train": 400,
                "n_test": 100,
                "num_targets": 10,
                "mean_score": 0.2,
                "median_score": 0.18,
                "std_score": 0.02,
            },
        ],
        [
            "dataset",
            "model",
            "subject_id",
            "roi",
            "layer",
            "metric",
            "n_train",
            "n_test",
            "num_targets",
            "mean_score",
            "median_score",
            "std_score",
        ],
    )
    if include_rsa:
        _write_csv(
            path / "rsa_scores.csv",
            [
                {
                    "dataset": f"nsd_{roi}",
                    "model": model,
                    "subject_id": "subj01",
                    "roi": roi,
                    "layer": "layer1",
                    "model_rdm_metric": "correlation",
                    "response_rdm_metric": "correlation",
                    "compare_method": "spearman",
                    "n_items": 500,
                    "score": 0.05,
                },
                {
                    "dataset": f"nsd_{roi}",
                    "model": model,
                    "subject_id": "subj01",
                    "roi": roi,
                    "layer": "layer3",
                    "model_rdm_metric": "correlation",
                    "response_rdm_metric": "correlation",
                    "compare_method": "spearman",
                    "n_items": 500,
                    "score": 0.07,
                },
            ],
            [
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
            ],
        )


def test_summarize_neural_roi_results_writes_best_layers_and_bridge(tmp_path):
    v1 = tmp_path / "outputs" / "v1"
    v2 = tmp_path / "outputs" / "v2"
    _write_neural_output(v1, roi="V1")
    _write_neural_output(v2, roi="V2", model="deit_small_patch16_224")

    behavioral_csv = tmp_path / "behavior.csv"
    _write_csv(
        behavioral_csv,
        [
            {
                "dataset": "salicon_static2000",
                "model": "resnet50",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 2000,
                "mean": 0.34,
                "ci95_low": 0.33,
                "ci95_high": 0.35,
            },
            {
                "dataset": "cat2000_static2000",
                "model": "deit_small_patch16_224",
                "saliency_method": "attention_rollout",
                "saliency_family": "internal_routing",
                "metric": "cc",
                "n": 2000,
                "mean": 0.12,
                "ci95_low": 0.11,
                "ci95_high": 0.13,
            },
            {
                "dataset": "cat2000_static2000",
                "model": "vit_base_patch16_224",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "n": 500,
                "mean": 0.3,
                "ci95_low": 0.2,
                "ci95_high": 0.4,
            },
        ],
        [
            "dataset",
            "model",
            "saliency_method",
            "saliency_family",
            "metric",
            "n",
            "mean",
            "ci95_low",
            "ci95_high",
        ],
    )

    efficiency_csv = tmp_path / "efficiency.csv"
    _write_csv(
        efficiency_csv,
        [
            {
                "model_name": "resnet50",
                "latency_mean_ms": 5.0,
                "parameter_count": 25000000,
                "model_size_mb": 100.0,
                "flops": 4000000000,
            },
            {
                "model_name": "deit_small_patch16_224",
                "latency_mean_ms": 4.0,
                "parameter_count": 22000000,
                "model_size_mb": 80.0,
                "flops": 4200000000,
            },
        ],
        ["model_name", "latency_mean_ms", "parameter_count", "model_size_mb", "flops"],
    )

    outputs = summarize_neural_roi_results(
        [v1, v2],
        tmp_path / "summary",
        behavioral_csv=behavioral_csv,
        efficiency_csv=efficiency_csv,
    )

    assert outputs["combined_encoding_scores"].is_file()
    assert outputs["combined_rsa_scores"].is_file()
    assert outputs["best_layers_by_roi"].is_file()
    assert outputs["best_encoding_by_model_roi"].is_file()
    assert outputs["best_rsa_by_model_roi"].is_file()
    assert outputs["behavior_neural_bridge"].is_file()
    assert outputs["behavior_neural_model_summary"].is_file()
    assert outputs["paper_model_roi_winners"].is_file()
    assert outputs["neural_model_rankings"].is_file()
    assert outputs["behavior_neural_alignment_summary"].is_file()
    assert outputs["behavior_neural_leader_overlap"].is_file()
    assert outputs["multimodel_interpretation_note"].is_file()
    assert outputs["alignment_per_efficiency"].is_file()
    assert outputs["summary_note"].is_file()

    best_rows = _read_csv(outputs["best_layers_by_roi"])
    encoding_best = [row for row in best_rows if row["score_type"] == "encoding"]
    rsa_best = [row for row in best_rows if row["score_type"] == "rsa"]
    assert {row["roi"] for row in encoding_best} == {"V1", "V2"}
    assert all(row["layer"] == "layer2" for row in encoding_best)
    assert all(row["layer"] == "layer3" for row in rsa_best)

    bridge_rows = _read_csv(outputs["behavior_neural_bridge"])
    assert {row["behavior_saliency_method"] for row in bridge_rows} == {
        "attention_rollout",
        "gradcam",
    }
    assert {row["roi"] for row in bridge_rows} == {"V1", "V2"}
    assert all(row["best_encoding_layer"] == "layer2" for row in bridge_rows)
    assert all(row["best_rsa_layer"] == "layer3" for row in bridge_rows)
    assert {row["neural_model"] for row in bridge_rows} == {
        "resnet50",
        "deit_small_patch16_224",
    }

    efficiency_rows = _read_csv(outputs["alignment_per_efficiency"])
    assert {row["model"] for row in efficiency_rows} == {
        "resnet50",
        "deit_small_patch16_224",
    }
    assert all(row["score_per_latency_mean_ms"] for row in efficiency_rows)

    paper_rows = _read_csv(outputs["paper_model_roi_winners"])
    assert {row["model"] for row in paper_rows} == {
        "resnet50",
        "deit_small_patch16_224",
    }
    assert all(row["best_encoding_layer"] == "layer2" for row in paper_rows)
    assert all(row["best_rsa_layer"] == "layer3" for row in paper_rows)

    ranking_rows = _read_csv(outputs["neural_model_rankings"])
    assert ranking_rows[0]["model"] == "deit_small_patch16_224"
    assert ranking_rows[0]["rank_mean_encoding"] == "1"
    assert ranking_rows[0]["rank_mean_rsa"] == "1"
    assert ranking_rows[0]["rank_encoding_per_latency"] == "1"

    alignment_rows = _read_csv(outputs["behavior_neural_alignment_summary"])
    assert alignment_rows
    assert {row["interpretation_scope"] for row in alignment_rows} == {
        "descriptive_one_subject_roi500"
    }

    overlap_rows = _read_csv(outputs["behavior_neural_leader_overlap"])
    assert overlap_rows
    assert all(row["behavior_metric_direction"] for row in overlap_rows)


def test_summarize_neural_roi_results_tolerates_missing_rsa(tmp_path):
    output_dir = tmp_path / "outputs" / "v1"
    _write_neural_output(output_dir, roi="V1", include_rsa=False)

    outputs = summarize_neural_roi_results([output_dir], tmp_path / "summary")

    assert _read_csv(outputs["combined_encoding_scores"])
    assert _read_csv(outputs["combined_rsa_scores"]) == []
    best_rows = _read_csv(outputs["best_layers_by_roi"])
    assert {row["score_type"] for row in best_rows} == {"encoding"}


def test_create_neural_roi500_configs_writes_expected_defaults(tmp_path):
    written = create_neural_roi500_configs(output_dir=tmp_path / "configs")

    assert len(written) == 16
    configs = {path.name: load_yaml(path) for path in written}
    assert configs["resnet50_v1_500.yaml"]["dataset"]["roi"] == "V1"
    assert configs["convnext_tiny_v2_500.yaml"]["dataset"]["roi"] == "V2"
    assert configs["deit_small_patch16_224_v3_500.yaml"]["dataset"]["roi"] == "V3"
    assert configs["vit_base_patch16_224_hv4_500.yaml"]["dataset"]["roi"] == "hV4"

    for config in configs.values():
        assert config["dataset"]["manifest_path"] == (
            "data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv"
        )
        assert config["dataset"]["max_items"] == 500
        assert config["model"]["backend"] == "timm"
        assert config["model"]["pretrained"] is True
        assert config["model"]["eval_mode"] is True
        assert config["neural"]["layers"]
        assert config["neural"]["feature_reduction"] == "spatial_mean"
        assert config["neural"]["rsa"]["enabled"] is True
        assert config["neural"]["rsa"]["compare_method"] == "spearman"
        assert config["output"]["dir"].startswith("outputs/neural_roi500/")

    assert configs["resnet50_v1_500.yaml"]["neural"]["layers"] == [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]
    assert configs["convnext_tiny_v1_500.yaml"]["neural"]["layers"] == [
        "stages.0",
        "stages.1",
        "stages.2",
        "stages.3",
    ]
    assert configs["deit_small_patch16_224_v1_500.yaml"]["neural"]["layers"] == [
        "blocks.0",
        "blocks.3",
        "blocks.6",
        "blocks.9",
        "blocks.11",
    ]


def test_create_neural_roi500_configs_supports_debug_subset(tmp_path):
    written = create_neural_roi500_configs(
        output_dir=tmp_path / "debug_configs",
        output_root=tmp_path / "debug_outputs",
        models=["convnext_tiny"],
        rois=["V1"],
        max_items=16,
        name_suffix="debug",
    )

    assert [path.name for path in written] == ["convnext_tiny_v1_debug.yaml"]
    config = load_yaml(written[0])
    assert config["dataset"]["max_items"] == 16
    assert config["dataset"]["roi"] == "V1"
    assert config["output"]["dir"].endswith("convnext_tiny_v1_debug")


def test_behavior_neural_leader_overlap_handles_lower_is_better_metric(tmp_path):
    slow = tmp_path / "outputs" / "slow"
    fast = tmp_path / "outputs" / "fast"
    _write_neural_output(slow, roi="V1", model="slow_model")
    _write_neural_output(fast, roi="V1", model="fast_model")

    behavioral_csv = tmp_path / "behavior.csv"
    _write_csv(
        behavioral_csv,
        [
            {
                "dataset": "salicon_static2000",
                "model": "slow_model",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "kl",
                "n": 2000,
                "mean": 2.0,
                "ci95_low": 1.9,
                "ci95_high": 2.1,
            },
            {
                "dataset": "salicon_static2000",
                "model": "fast_model",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "kl",
                "n": 2000,
                "mean": 1.0,
                "ci95_low": 0.9,
                "ci95_high": 1.1,
            },
        ],
        [
            "dataset",
            "model",
            "saliency_method",
            "saliency_family",
            "metric",
            "n",
            "mean",
            "ci95_low",
            "ci95_high",
        ],
    )

    outputs = summarize_neural_roi_results(
        [slow, fast],
        tmp_path / "summary",
        behavioral_csv=behavioral_csv,
    )

    overlap_rows = _read_csv(outputs["behavior_neural_leader_overlap"])
    assert len(overlap_rows) == 1
    assert overlap_rows[0]["behavior_metric_direction"] == "lower_is_better"
    assert overlap_rows[0]["behavior_leader_model"] == "fast_model"


class FakeCandidateModel:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        return [(name, object()) for name in ["", *self.names]]


class FakeTimmCandidates:
    def list_models(self):
        return [
            "vit_small_patch14_dinov2",
            "resnet50_clip",
            "missing_blocks_model",
        ]

    def create_model(self, model_name, pretrained=False):
        assert pretrained is False
        if model_name == "resnet50_clip":
            return FakeCandidateModel(["layer1", "layer2", "layer3", "layer4"])
        if model_name == "vit_small_patch14_dinov2":
            return FakeCandidateModel([f"blocks.{index}" for index in range(12)])
        return FakeCandidateModel(["stem"])


def test_ssl_multimodal_candidate_inventory_and_debug_configs(tmp_path):
    output_csv = tmp_path / "summary" / "ssl_multimodal_candidate_inventory.csv"
    rows = inspect_ssl_multimodal_candidates(
        output_csv=output_csv,
        timm_module=FakeTimmCandidates(),
        candidates=[
            {"model_name": "vit_small_patch14_dinov2", "family": "DINOv2"},
            {"model_name": "resnet50_clip", "family": "CLIP"},
            {"model_name": "not_available", "family": "DINOv3"},
            {"model_name": "missing_blocks_model", "family": "test"},
        ],
    )

    assert output_csv.is_file()
    by_model = {row["model_name"]: row for row in rows}
    assert by_model["vit_small_patch14_dinov2"]["proposed_layers"] == (
        "blocks.0 blocks.3 blocks.6 blocks.9 blocks.11"
    )
    assert by_model["vit_small_patch14_dinov2"]["wrapper_compatible"] == "true"
    assert by_model["resnet50_clip"]["proposed_layers"] == (
        "layer1 layer2 layer3 layer4"
    )
    assert by_model["resnet50_clip"]["wrapper_compatible"] == "true"
    assert by_model["not_available"]["available_in_timm"] == "false"
    assert by_model["missing_blocks_model"]["wrapper_compatible"] == "false"
    assert all(row["pretrained_weights_run"] == "false" for row in rows)

    written = create_ssl_multimodal_debug_configs(
        rows,
        output_dir=tmp_path / "configs",
        output_root=tmp_path / "outputs",
    )
    assert len(written) == 2
    configs = {path.name: load_yaml(path) for path in written}
    assert configs["vit_small_patch14_dinov2_v1_debug.yaml"]["model"]["pretrained"] is False
    assert configs["resnet50_clip_v1_debug.yaml"]["neural"]["layers"] == [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]
