from pathlib import Path

from hma.utils.config import load_experiment_config, load_yaml, merge_dicts, save_yaml
from hma.utils.paths import get_output_dir, resolve_path


def test_yaml_save_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    expected = {
        "seed": 123,
        "dataset": {"name": "dummy_static_saliency", "max_items": 8},
        "metrics": ["mae", "pearson"],
    }

    save_yaml(expected, path)

    assert load_yaml(path) == expected


def test_merge_dicts_preserves_nested_keys_and_replaces_values():
    base = {
        "dataset": {"name": "base_dataset", "root": "data/base", "max_items": None},
        "metrics": ["mae"],
        "device": "cpu",
    }
    override = {
        "dataset": {"max_items": 8},
        "metrics": ["pearson"],
    }

    merged = merge_dicts(base, override)

    assert merged == {
        "dataset": {"name": "base_dataset", "root": "data/base", "max_items": 8},
        "metrics": ["pearson"],
        "device": "cpu",
    }
    assert base["dataset"]["max_items"] is None


def test_load_experiment_config_fills_defaults():
    config = load_experiment_config("configs/experiments/saliency_static_debug.yaml")

    assert config["seed"] == 123
    assert config["device"] == "cpu"
    assert config["dataset"]["name"] == "dummy_static_saliency"
    assert config["dataset"]["root"] == "data/saliency_static"
    assert config["dataset"]["max_items"] == 3
    assert config["dataset"]["image_shape"] == [3, 16, 16]
    assert config["dataset"]["map_shape"] == [16, 16]
    assert config["model"]["name"] == "dummy_vision_encoder"
    assert config["saliency"]["method"] == "dummy_gradient_free"
    assert config["metrics"] == ["nss", "cc", "similarity", "kl"]
    assert config["output"]["dir"] == "outputs/saliency_static_debug"
    assert config["output"]["save_visualizations"] is True
    assert config["output"]["num_visualizations"] == 2


def test_output_dir_creation_and_relative_path_resolution(tmp_path):
    resolved = resolve_path("outputs/debug", base_dir=tmp_path)
    assert resolved == (tmp_path / "outputs" / "debug").resolve()

    output_dir = get_output_dir({"output": {"dir": tmp_path / "created_output"}})

    assert output_dir == (tmp_path / "created_output").resolve()
    assert output_dir.is_dir()
    assert isinstance(output_dir, Path)
