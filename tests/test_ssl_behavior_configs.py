from hma.utils.config import load_yaml
from scripts.create_ssl_behavior_v1_configs import create_ssl_behavior_v1_configs
from scripts.create_transformer_relevance_configs import (
    create_transformer_relevance_configs,
)
from scripts.run_v2_matrix import (
    DEFAULT_RELIABILITY_CHECKS,
    _parse_reliability_check,
)


def test_ssl_behavior_config_generation_writes_expected_matrix(tmp_path):
    datasets = [
        {
            "label": "salicon_pilot500",
            "name": "salicon",
            "root": "data/raw/SALICON",
            "manifest_path": "data/manifests/pilot/salicon_pilot500_manifest.csv",
            "split": "val",
            "image_size": [224, 224],
            "scale": "pilot",
            "extra": {},
        },
        {
            "label": "salicon_static2000",
            "name": "salicon",
            "root": "data/raw/SALICON",
            "manifest_path": "data/manifests/v2/salicon_static2000_manifest.csv",
            "split": "val",
            "image_size": [224, 224],
            "scale": "static2000",
            "extra": {},
        },
    ]

    written = create_ssl_behavior_v1_configs(
        config_root=tmp_path / "configs",
        output_root=tmp_path / "outputs",
        datasets=datasets,
    )

    assert len(written) == 12
    configs = {path.name: load_yaml(path) for path in written}
    run_pairs = {
        (config["model"]["name"], config["saliency"]["method"])
        for config in configs.values()
    }
    assert run_pairs == {
        ("vit_small_patch14_dinov2", "vanilla_gradient"),
        ("vit_small_patch14_dinov2", "attention_rollout"),
        ("vit_base_patch16_clip_224", "vanilla_gradient"),
        ("vit_base_patch16_clip_224", "attention_rollout"),
        ("resnet50_clip", "vanilla_gradient"),
        ("resnet50_clip", "gradcam"),
    }

    dino = configs[
        "salicon_static2000__vit_small_patch14_dinov2_attention_rollout.yaml"
    ]
    assert dino["model"]["pretrained"] is True
    assert dino["preprocessing"]["input_size"] == [518, 518]
    assert dino["metrics"] == [
        "nss",
        "shuffled_auc",
        "auc_borji",
        "auc_judd",
        "cc",
        "similarity",
        "kl",
    ]
    assert dino["output"]["save_visualizations"] is False
    assert "emd" not in dino["metrics"]

    clip = configs["salicon_static2000__vit_base_patch16_clip_224_vanilla_gradient.yaml"]
    assert clip["preprocessing"]["input_size"] == [224, 224]

    gradcam = configs["salicon_pilot500__resnet50_clip_gradcam.yaml"]
    assert gradcam["saliency"]["target_layer"] == "stages.3"
    assert gradcam["output"]["save_visualizations"] is True
    assert "emd" in gradcam["metrics"]


def test_run_v2_matrix_default_reliability_checks_include_ssl_gates():
    assert (
        "salicon_pilot500",
        "vit_small_patch14_dinov2",
        "attention_rollout",
    ) in DEFAULT_RELIABILITY_CHECKS
    assert (
        "salicon_pilot500",
        "vit_base_patch16_clip_224",
        "attention_rollout",
    ) in DEFAULT_RELIABILITY_CHECKS
    assert ("salicon_pilot500", "resnet50_clip", "gradcam") in DEFAULT_RELIABILITY_CHECKS
    assert _parse_reliability_check("dataset:model:method") == (
        "dataset",
        "model",
        "method",
    )


def test_transformer_relevance_config_generation_is_scoped(tmp_path):
    written = create_transformer_relevance_configs(
        debug_config_root=tmp_path / "debug_configs",
        static_config_root=tmp_path / "static_configs",
        debug_output_root=tmp_path / "debug_outputs",
        static_output_root=tmp_path / "static_outputs",
    )

    assert len(written) == 9
    configs = {path.name: load_yaml(path) for path in written}
    assert (
        "salicon_vit_small_patch14_dinov2_transformer_relevance_smoke.yaml"
        in configs
    )

    static_configs = {
        name: config
        for name, config in configs.items()
        if "static2000__" in name
    }
    assert len(static_configs) == 8
    assert {
        config["dataset"]["label"]
        for config in static_configs.values()
    } == {"salicon_static2000", "cat2000_static2000"}
    assert {
        config["model"]["name"]
        for config in static_configs.values()
    } == {
        "vit_small_patch14_dinov2",
        "vit_base_patch16_clip_224",
        "vit_base_patch16_224",
        "deit_small_patch16_224",
    }
    assert all(
        config["saliency"]["method"] == "transformer_relevance"
        for config in configs.values()
    )
    assert all(
        config["output"]["save_visualizations"] is False
        for config in static_configs.values()
    )
