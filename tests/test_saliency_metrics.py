import numpy as np
import pytest

from hma.metrics.saliency import mean_absolute_error, pearson_correlation
from hma.metrics.saliency_metrics import (
    auc_borji,
    auc_judd,
    cc,
    information_gain,
    is_constant_map,
    kl_divergence,
    nss,
    probabilistic_log_likelihood,
    shuffled_auc,
    similarity,
)
from hma.saliency.postprocess import normalize_saliency_map


def test_normalize_saliency_map_scales_to_unit_range():
    values = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    normalized = normalize_saliency_map(values)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.shape == values.shape


def test_normalize_constant_saliency_map_returns_zeros():
    values = np.ones((2, 2), dtype=np.float32)
    normalized = normalize_saliency_map(values)

    assert np.allclose(normalized, 0.0)


def test_mean_absolute_error_on_tiny_arrays():
    prediction = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    assert mean_absolute_error(prediction, target) == 0.5


def test_pearson_correlation_handles_matching_and_constant_maps():
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    constant = np.ones((2, 2), dtype=np.float32)

    assert np.isclose(pearson_correlation(values, values), 1.0)
    assert pearson_correlation(values, constant) == 0.0


def test_constant_prediction_metrics_have_defined_behavior():
    prediction = np.full((4, 4), 2.0, dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    target[1, 1] = 1.0
    negatives = np.array([[0, 0], [2, 2], [3, 3]], dtype=np.int64)

    assert is_constant_map(prediction)
    assert auc_judd(prediction, target) == 0.5
    assert auc_borji(prediction, target, splits=5, seed=1) == 0.5
    assert shuffled_auc(
        prediction,
        target,
        negatives,
        splits=5,
        seed=1,
    ) == 0.5
    assert nss(prediction, target) == 0.0
    assert cc(prediction, target) == 0.0
    assert similarity(prediction, target) == pytest.approx(1.0 / 16.0)
    assert kl_divergence(target, prediction) == pytest.approx(np.log(16.0))


def test_near_constant_prediction_is_not_minmax_amplified():
    prediction = np.ones((4, 4), dtype=np.float32)
    prediction[0, 0] += 1e-7
    target = np.zeros((4, 4), dtype=np.float32)
    target[1, 1] = 1.0
    negatives = np.array([[0, 0], [2, 2]], dtype=np.int64)

    assert is_constant_map(prediction)
    assert auc_judd(prediction, target) == 0.5
    assert auc_borji(prediction, target, splits=5, seed=2) == 0.5
    assert shuffled_auc(
        prediction,
        target,
        negatives,
        splits=5,
        seed=2,
    ) == 0.5
    assert nss(prediction, target) == 0.0
    assert cc(prediction, target) == 0.0


def test_probabilistic_log_likelihood_and_information_gain_use_fixation_counts():
    prediction = np.array([[0.7, 0.1], [0.1, 0.1]], dtype=np.float32)
    baseline = np.ones((2, 2), dtype=np.float32)
    target = np.zeros((2, 2), dtype=np.float32)
    target[0, 0] = 1.0
    repeated_fixations = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64)

    log_likelihood = probabilistic_log_likelihood(
        prediction,
        target,
        positive_fixations=repeated_fixations,
    )
    gain = information_gain(
        prediction,
        baseline,
        target,
        positive_fixations=repeated_fixations,
    )

    expected = np.mean(np.log2([0.7, 0.7, 0.1]))
    assert log_likelihood == pytest.approx(expected)
    assert gain == pytest.approx(expected - np.log2(0.25))


def test_probabilistic_log_likelihood_supports_density_targets():
    prediction = np.array([[0.8, 0.1], [0.05, 0.05]], dtype=np.float32)
    target = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    score = probabilistic_log_likelihood(prediction, target)

    assert score == pytest.approx(np.log2(0.8))
