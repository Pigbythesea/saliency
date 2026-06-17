from hma.utils.config import load_yaml
from scripts.run_paper1_admission_panel import (
    _executable_model_specs,
    _latent_model_specs,
    _write_preflight,
)


def test_cluster_only_models_are_excluded_from_local_execution():
    config = load_yaml("configs/paper1_admission_panel_v1.yaml")

    executable = {row["model_id"] for row in _executable_model_specs(config)}
    latent = {row["model_id"] for row in _latent_model_specs(config)}

    assert "resnet50" in executable
    assert "resnet50" in latent
    assert "deepgaze_iie" in executable
    assert "deepgaze_iie" not in latent
    assert "dynamicvit_deit_small_keep_0_7" not in executable
    assert "tome_deit_small_r13" not in latent
    assert "hat" not in executable


def test_preflight_removes_cluster_model_from_pending_after_axis_import(tmp_path):
    config = load_yaml("configs/paper1_admission_panel_v1.yaml")
    rows = [
        {
            "model_id": "dynamicvit_deit_small_keep_0_7",
            "family": "DynamicViT",
            "role": "generic_efficient_computation",
            "behavioral_available": "true",
            "latent_neural_available": "true",
            "geometry_available": "true",
            "efficiency_available": "true",
            "axis_count_available": 4,
        },
        {
            "model_id": "tome_deit_small_r13",
            "family": "ToMe",
            "role": "generic_efficient_computation",
            "behavioral_available": "false",
            "latent_neural_available": "false",
            "geometry_available": "false",
            "efficiency_available": "false",
            "axis_count_available": 0,
        },
    ]
    path = tmp_path / "preflight.md"

    _write_preflight(path, rows, config)

    text = path.read_text(encoding="utf-8")
    pending_line = next(
        line for line in text.splitlines() if line.startswith("Cluster-pending")
    )
    assert "dynamicvit_deit_small_keep_0_7" not in pending_line
    assert "tome_deit_small_r13" in pending_line
