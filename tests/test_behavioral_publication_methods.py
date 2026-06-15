import numpy as np
import pytest

from hma.behavioral import (
    coco_search18_hierarchical_interval,
    evaluate_conditional_maps,
    evaluate_scanpath,
    image_cluster_bootstrap,
    salicon_hierarchical_interval,
)


def test_image_cluster_bootstrap_is_reproducible_and_clusters_rows():
    rows = [
        {"image_path": "a", "score": 0.0},
        {"image_path": "a", "score": 2.0},
        {"image_path": "b", "score": 4.0},
    ]

    first = image_cluster_bootstrap(rows, value_key="score", resamples=100, seed=7)
    second = image_cluster_bootstrap(rows, value_key="score", resamples=100, seed=7)

    assert first == second
    assert first.estimate == pytest.approx(2.0)
    assert first.uncertainty_unit == "image_path"
    assert first.ci_low <= first.estimate <= first.ci_high


def test_nested_behavioral_bootstraps_report_correct_units():
    salicon = [
        {"image_path": image, "worker_id": worker, "score": value}
        for image, worker, value in [
            ("a", "w1", 1.0),
            ("a", "w2", 2.0),
            ("b", "w1", 3.0),
            ("b", "w2", 4.0),
        ]
    ]
    search = [
        {
            "image_path": image,
            "target_category": task,
            "subject_id": subject,
            "score": value,
        }
        for image, task, subject, value in [
            ("a", "car", "s1", 1.0),
            ("a", "car", "s2", 2.0),
            ("b", "mug", "s1", 3.0),
            ("b", "mug", "s2", 4.0),
        ]
    ]

    salicon_result = salicon_hierarchical_interval(
        salicon, value_key="score", resamples=50
    )
    search_result = coco_search18_hierarchical_interval(
        search, value_key="score", resamples=50
    )

    assert salicon_result.uncertainty_unit == "worker_id_within_image_path"
    assert (
        search_result.uncertainty_unit
        == "subject_id_within_image_path_target_category"
    )
    assert salicon_result.valid_resamples == 50
    assert search_result.valid_resamples == 50


def test_conditional_map_metrics_keep_task_search_separate():
    prediction = np.zeros((3, 3), dtype=np.float64)
    prediction[1, 1] = 1.0
    baseline = np.ones((3, 3), dtype=np.float64)

    result = evaluate_conditional_maps(
        [prediction],
        [(1, 1)],
        regime="task_search",
        baseline_maps=[baseline],
        task_id="car",
    )

    assert result.behavioral_regime == "task_search"
    assert result.behavioral_object == "conditional_next_fixation"
    assert result.task_id == "car"
    assert result.metrics["conditional_nss"] > 0.0
    assert result.metrics["conditional_information_gain_bits"] > 0.0


def test_scanpath_interface_scores_identical_paths_and_target_step():
    fixations = [(0, 0), (5, 5), (9, 9)]

    result = evaluate_scanpath(
        fixations,
        fixations,
        image_shape=(10, 10),
        regime="task_search",
        task_id="target",
        target_bbox=(8, 8, 10, 10),
        seed=123,
    )

    assert result.behavioral_object == "generated_scanpath"
    assert result.metrics["sequence_score"] == pytest.approx(1.0)
    assert result.metrics["scanpath_length_error"] == pytest.approx(0.0)
    assert result.metrics["target_fixation_step"] == 3.0
