import csv
from pathlib import Path

from hma.utils.config import load_yaml


CONFIG_PATH = Path("configs/paper1_experiment_v1.yaml")
SPEC_PATH = Path("docs/paper1_experiment_spec_v1.md")
DECISIONS_PATH = Path("outputs/planning/paper1_experiment_scope_decisions.csv")


def test_paper1_experiment_v1_config_locks_required_scope():
    config = load_yaml(CONFIG_PATH)

    required_keys = {
        "claim",
        "baseline_inputs",
        "discovery_matrix",
        "confirmatory_matrix",
        "behavioral_controls",
        "encoding",
        "geometry",
        "accepted_tables",
        "failure_criteria",
        "implementation_sequence",
        "do_not_do",
    }
    assert required_keys <= set(config)

    discovery = config["discovery_matrix"]
    assert discovery["subject_id"] == "subj01"
    assert discovery["models"] == [
        "resnet50",
        "vit_base_patch16_224",
        "vit_small_patch14_dinov2",
        "vit_base_patch16_clip_224",
    ]

    roi_groups = discovery["roi_groups"]
    assert roi_groups["prf_visualrois"]["rois"] == ["V1", "V2", "V3", "hV4"]
    assert roi_groups["streams"]["rois"] == [
        "midventral",
        "midlateral",
        "midparietal",
        "ventral",
        "lateral",
        "parietal",
    ]
    assert discovery["expected_cells"] == {
        "models": 4,
        "rois": 10,
        "model_roi_cells": 40,
    }

    assert config["encoding"]["method"] == "flatten_pca"
    assert config["encoding"]["validation_selected_layer"] is True
    assert config["geometry"]["methods"] == ["linear_cka_full9841", "subset_rsa"]
    assert config["geometry"]["subset_sizes"] == [512, 1024, 2048]
    assert config["geometry"]["subset_seeds"] == [123, 456, 789]


def test_paper1_experiment_v1_config_declares_required_artifacts_and_exclusions():
    config = load_yaml(CONFIG_PATH)
    accepted_tables = config["accepted_tables"]

    assert accepted_tables
    assert all(path.startswith("outputs/paper1_experiment_v1/summary/") for path in accepted_tables)
    assert "outputs/paper1_experiment_v1/summary/roi_expanded_cross_axis_decisions.csv" in accepted_tables
    assert "outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv" in accepted_tables

    immediate_sequence = " ".join(config["implementation_sequence"]).lower()
    forbidden_immediate_work = [
        "broad model zoo",
        "deepgaze msdb",
        "task-trained coco-search18",
        "paper 2 causal",
        "floc category",
    ]
    for phrase in forbidden_immediate_work:
        assert phrase not in immediate_sequence

    do_not_do = " ".join(config["do_not_do"]).lower()
    for phrase in [
        "broad model zoo",
        "attention rollout",
        "learned spatial readout",
        "floc category rois",
        "paper 2 causal",
    ]:
        assert phrase in do_not_do


def test_paper1_experiment_spec_mentions_config_and_cmd_commands():
    text = SPEC_PATH.read_text(encoding="utf-8")

    assert "configs/paper1_experiment_v1.yaml" in text
    assert "outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_correlations.csv" in text
    assert "```cmd" in text
    assert "powershell" not in text.lower()


def test_scope_decision_table_has_required_rows_and_columns():
    with DECISIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "decision_area",
        "chosen_scope",
        "reason",
        "reviewer_risk_addressed",
        "accepted_artifact",
        "deferred_alternative",
    ]
    assert {row["decision_area"] for row in rows} == {
        "model_panel",
        "subject_scope",
        "roi_scope",
        "behavioral_controls",
        "attribution_families",
        "encoding_method",
        "geometry_methods",
        "uncertainty",
        "failure_criteria",
    }
    assert all(row["accepted_artifact"].startswith("outputs/") for row in rows)
