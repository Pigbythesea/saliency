import numpy as np
import pytest

from hma.saliency import (
    UnsupportedModelError,
    attention_rollout_saliency,
    attention_rollout_to_saliency_map,
    build_saliency_method,
    cls_to_patch_relevance,
    compute_attention_rollout,
    patch_relevance_to_grid,
)


class AttentionWrapper:
    def __init__(self, attention_tensors):
        self.attention_tensors = attention_tensors

    def get_attention_matrices(self, _images):
        return self.attention_tensors


def _mock_attention_layers():
    first = np.zeros((1, 2, 5, 5), dtype=np.float32)
    second = np.zeros((1, 2, 5, 5), dtype=np.float32)
    for tensor in (first, second):
        tensor[:] = np.eye(5, dtype=np.float32)
        tensor[:, :, 0, 1:] = 0.25
    return [first, second]


def test_compute_attention_rollout_shape():
    rollout = compute_attention_rollout(_mock_attention_layers())

    assert rollout.shape == (1, 5, 5)
    assert np.isfinite(rollout).all()


def test_cls_to_patch_relevance_shape():
    rollout = compute_attention_rollout(_mock_attention_layers())
    relevance = cls_to_patch_relevance(rollout)

    assert relevance.shape == (1, 4)
    assert np.isfinite(relevance).all()


def test_patch_relevance_to_grid_infers_square_grid():
    relevance = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    grid = patch_relevance_to_grid(relevance)

    assert grid.shape == (1, 2, 2)
    assert np.allclose(grid[0], [[0.1, 0.2], [0.3, 0.4]])


def test_patch_relevance_to_grid_requires_grid_for_non_square_patch_count():
    relevance = np.ones((1, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="Patch count is not square"):
        patch_relevance_to_grid(relevance)


def test_attention_rollout_to_saliency_map_shape_and_range():
    maps = attention_rollout_to_saliency_map(
        _mock_attention_layers(),
        image_shape=(8, 8),
    )

    assert maps.shape == (1, 1, 8, 8)
    assert np.isfinite(maps).all()
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0


def test_attention_rollout_saliency_uses_wrapper_attention_matrices():
    wrapper = AttentionWrapper(_mock_attention_layers())
    images = np.zeros((1, 3, 8, 8), dtype=np.float32)

    maps = attention_rollout_saliency(wrapper, images)

    assert maps.shape == (1, 1, 8, 8)


def test_attention_rollout_unsupported_wrapper_raises_helpful_error():
    images = np.zeros((1, 3, 8, 8), dtype=np.float32)

    with pytest.raises(UnsupportedModelError, match="get_attention_matrices"):
        attention_rollout_saliency(object(), images)


def test_saliency_registry_builds_attention_rollout_callable():
    wrapper = AttentionWrapper(_mock_attention_layers())
    images = np.zeros((1, 3, 8, 8), dtype=np.float32)
    method = build_saliency_method(
        {
            "saliency": {
                "method": "attention_rollout",
                "discard_ratio": 0.0,
                "head_fusion": "mean",
            }
        }
    )

    maps = method(wrapper, images)

    assert maps.shape == (1, 1, 8, 8)
