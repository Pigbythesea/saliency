from hma.config import load_yaml_config


def test_load_dummy_experiment_config():
    config = load_yaml_config("configs/experiments/dummy_pipeline.yaml")

    assert config["experiment"]["name"] == "dummy_pipeline"
    assert config["dataset"]["name"] == "dummy_saliency"
    assert config["model"]["name"] == "dummy_saliency"
