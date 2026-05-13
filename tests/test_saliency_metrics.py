import numpy as np

from hma.metrics.saliency import mean_absolute_error, pearson_correlation
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
