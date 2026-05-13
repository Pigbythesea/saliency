import numpy as np
import pytest

from hma.metrics.saliency_metrics import (
    auc_judd,
    cc,
    kl_divergence,
    normalize_map,
    nss,
    similarity,
    simple_center_bias_map,
)
from hma.saliency.postprocess import postprocess_saliency_map, resize_saliency_map


def test_normalize_map_scales_values_and_handles_constant_maps():
    values = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    normalized = normalize_map(values)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert np.allclose(normalize_map(np.ones((2, 2), dtype=np.float32)), 0.0)


def test_nss_cc_similarity_and_kl_on_synthetic_arrays():
    saliency = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    fixation = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert nss(saliency, fixation) > 1.0
    assert np.isclose(cc(saliency, saliency), 1.0)
    assert np.isclose(similarity(saliency, saliency), 1.0)
    assert np.isclose(kl_divergence(saliency, saliency), 0.0)


def test_auc_judd_returns_finite_unit_interval_score():
    saliency = np.array([[0.1, 0.2], [0.3, 1.0]], dtype=np.float32)
    fixation = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    score = auc_judd(saliency, fixation)

    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "metric",
    [auc_judd, nss, cc, similarity, kl_divergence],
)
def test_pairwise_metrics_raise_on_shape_mismatch(metric):
    first = np.zeros((2, 2), dtype=np.float32)
    second = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        metric(first, second)


@pytest.mark.parametrize(
    "metric",
    [auc_judd, nss, cc, similarity, kl_divergence],
)
def test_metrics_do_not_emit_nan_for_zero_or_constant_maps(metric):
    zero = np.zeros((4, 4), dtype=np.float32)
    constant = np.ones((4, 4), dtype=np.float32)

    assert np.isfinite(metric(zero, zero))
    assert np.isfinite(metric(constant, constant))


def test_center_bias_scores_better_than_random_map_for_center_fixation():
    fixation = np.zeros((21, 21), dtype=np.float32)
    fixation[10, 10] = 1.0
    center_bias = simple_center_bias_map(21, 21)
    random_map = np.random.default_rng(123).random((21, 21), dtype=np.float32)

    assert auc_judd(center_bias, fixation) > auc_judd(random_map, fixation)
    assert nss(center_bias, fixation) > nss(random_map, fixation)


def test_resize_and_postprocess_saliency_map():
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    resized = resize_saliency_map(values, (4, 4))
    processed = postprocess_saliency_map(values, target_shape=(4, 4))

    assert resized.shape == (4, 4)
    assert np.isclose(resized[0, 0], 0.0)
    assert np.isclose(resized[-1, -1], 3.0)
    assert processed.shape == (4, 4)
    assert processed.min() == 0.0
    assert processed.max() == 1.0
