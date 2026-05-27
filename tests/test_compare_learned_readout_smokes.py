import csv
import json

from scripts.compare_learned_readout_smokes import compare_learned_readout_smokes


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_smoke_output(path, *, mean_score, mean_noise_normalized_score, targets):
    _write_csv(
        path / "encoding_scores.csv",
        [
            {
                "mean_score": mean_score,
                "mean_noise_normalized_score": mean_noise_normalized_score,
            }
        ],
    )
    _write_csv(path / "encoding_target_scores.csv", targets)


def test_compare_learned_readout_smokes_writes_deltas_and_summary(tmp_path):
    single = tmp_path / "single"
    multi = tmp_path / "multi"
    output = tmp_path / "comparison"
    _write_smoke_output(
        single,
        mean_score=0.4,
        mean_noise_normalized_score=0.5,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.2",
                "noise_normalized_score": "0.3",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            },
            {
                "target_index": "1",
                "pearson_r": "0.6",
                "noise_normalized_score": "0.7",
                "noise_ceiling": "0.4",
                "valid_noise_ceiling": "true",
            },
        ],
    )
    _write_smoke_output(
        multi,
        mean_score=0.45,
        mean_noise_normalized_score=0.55,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.3",
                "noise_normalized_score": "0.4",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            },
            {
                "target_index": "1",
                "pearson_r": "0.65",
                "noise_normalized_score": "0.8",
                "noise_ceiling": "0.4",
                "valid_noise_ceiling": "true",
            },
        ],
    )

    outputs = compare_learned_readout_smokes(single, multi, output)

    assert outputs["target_deltas"].is_file()
    assert outputs["summary_json"].is_file()
    assert outputs["readme"].is_file()
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["raw_improved_targets"] == 2
    assert summary["noise_normalized_improved_targets"] == 2
    assert summary["recommendation"] == "run_full_v1_multilayer"
    with outputs["target_deltas"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pearson_r_delta"] == "0.09999999999999998"
    assert rows[1]["noise_normalized_delta"] == "0.10000000000000009"


def test_compare_learned_readout_smokes_marks_inconclusive_when_raw_drops(tmp_path):
    single = tmp_path / "single"
    multi = tmp_path / "multi"
    output = tmp_path / "comparison"
    _write_smoke_output(
        single,
        mean_score=0.4,
        mean_noise_normalized_score=0.5,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.7",
                "noise_normalized_score": "0.2",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            }
        ],
    )
    _write_smoke_output(
        multi,
        mean_score=0.39,
        mean_noise_normalized_score=0.55,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.6",
                "noise_normalized_score": "0.3",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            }
        ],
    )

    outputs = compare_learned_readout_smokes(single, multi, output)

    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["recommendation"] == "inconclusive_do_not_prioritize_full_run"


def test_compare_learned_readout_smokes_uses_voxel_specific_labels(tmp_path):
    single = tmp_path / "single"
    candidate = tmp_path / "vit_small_patch14_dinov2_v1_voxel_specific_spatial_readout_smoke"
    output = tmp_path / "comparison"
    _write_smoke_output(
        single,
        mean_score=0.4,
        mean_noise_normalized_score=0.5,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.2",
                "noise_normalized_score": "0.3",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            },
            {
                "target_index": "1",
                "pearson_r": "0.6",
                "noise_normalized_score": "0.7",
                "noise_ceiling": "0.4",
                "valid_noise_ceiling": "true",
            },
        ],
    )
    _write_smoke_output(
        candidate,
        mean_score=0.45,
        mean_noise_normalized_score=0.55,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.3",
                "noise_normalized_score": "0.4",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            },
            {
                "target_index": "1",
                "pearson_r": "0.65",
                "noise_normalized_score": "0.8",
                "noise_ceiling": "0.4",
                "valid_noise_ceiling": "true",
            },
        ],
    )

    outputs = compare_learned_readout_smokes(
        single,
        candidate,
        output,
        baseline_label="single-layer",
        candidate_label="voxel-specific",
    )

    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["baseline_label"] == "single-layer"
    assert summary["candidate_label"] == "voxel-specific"
    assert summary["baseline_dir"] == str(single)
    assert summary["candidate_dir"] == str(candidate)
    assert summary["recommendation"] == "run_full_v1_voxel_specific"


def test_compare_learned_readout_smokes_freezes_current_for_failed_voxel_candidate(
    tmp_path,
):
    single = tmp_path / "single"
    candidate = tmp_path / "voxel_specific"
    output = tmp_path / "comparison"
    _write_smoke_output(
        single,
        mean_score=0.4,
        mean_noise_normalized_score=0.5,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.7",
                "noise_normalized_score": "0.7",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            }
        ],
    )
    _write_smoke_output(
        candidate,
        mean_score=0.3,
        mean_noise_normalized_score=0.4,
        targets=[
            {
                "target_index": "0",
                "pearson_r": "0.6",
                "noise_normalized_score": "0.6",
                "noise_ceiling": "0.5",
                "valid_noise_ceiling": "true",
            }
        ],
    )

    outputs = compare_learned_readout_smokes(
        single,
        candidate,
        output,
        candidate_label="voxel-specific",
    )

    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["recommendation"] == "freeze_current_dinov2_protocol"
