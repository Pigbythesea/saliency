import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hma.saliency import (  # noqa: E402
    center_bias_saliency,
    build_saliency_method,
    gradcam_saliency,
    integrated_gradients_saliency,
    random_saliency,
    vanilla_gradient_saliency,
)
from hma.metrics.saliency_metrics import cc, nss, simple_center_bias_map  # noqa: E402


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class TinyWrapper:
    def __init__(self):
        self.model = TinyCNN()
        self.model.eval()

    def get_last_logits(self, images):
        return self.model(images)


@pytest.fixture
def tiny_wrapper():
    torch.manual_seed(0)
    return TinyWrapper()


@pytest.fixture
def images():
    torch.manual_seed(1)
    return torch.rand(2, 3, 8, 8)


def _assert_valid_saliency_map(maps):
    assert maps.shape == (2, 1, 8, 8)
    assert torch.isfinite(maps).all()
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0


def test_vanilla_gradient_saliency_shape_and_range(tiny_wrapper, images):
    maps = vanilla_gradient_saliency(tiny_wrapper, images)

    _assert_valid_saliency_map(maps)


def test_integrated_gradients_saliency_shape_and_range(tiny_wrapper, images):
    maps = integrated_gradients_saliency(tiny_wrapper, images, steps=4)

    _assert_valid_saliency_map(maps)


def test_gradcam_saliency_shape_and_range(tiny_wrapper, images):
    maps = gradcam_saliency(tiny_wrapper, images, target_layer="features.0")

    _assert_valid_saliency_map(maps)


def test_gradcam_missing_layer_raises_informative_error(tiny_wrapper, images):
    with pytest.raises(ValueError, match="Target layer 'missing' not found"):
        gradcam_saliency(tiny_wrapper, images, target_layer="missing")


def test_build_saliency_method_returns_configured_callables(tiny_wrapper, images):
    vanilla = build_saliency_method({"saliency": {"method": "vanilla_gradient"}})
    integrated = build_saliency_method(
        {"saliency": {"method": "integrated_gradients", "steps": 4}}
    )
    gradcam = build_saliency_method(
        {"saliency": {"method": "gradcam", "target_layer": "features.0"}}
    )

    _assert_valid_saliency_map(vanilla(tiny_wrapper, images))
    _assert_valid_saliency_map(integrated(tiny_wrapper, images))
    _assert_valid_saliency_map(gradcam(tiny_wrapper, images))


def test_baseline_saliency_methods_are_valid_and_reproducible():
    target = simple_center_bias_map(16, 16)
    image = torch.zeros(1, 3, 16, 16)

    center = center_bias_saliency(None, image, target_map=target)
    random_a = random_saliency(None, image, target_map=target, seed=11, item_index=3)
    random_b = random_saliency(None, image, target_map=target, seed=11, item_index=3)

    assert center.shape == (16, 16)
    assert np.isclose(center.max(), 1.0)
    assert np.allclose(random_a, random_b)
    assert random_a.min() >= 0.0
    assert random_a.max() <= 1.0


def test_center_bias_beats_random_on_synthetic_center_fixation():
    fixation_map = np.zeros((16, 16), dtype=np.float32)
    fixation_map[8, 8] = 1.0
    image = torch.zeros(1, 3, 16, 16)

    center = center_bias_saliency(None, image, target_map=fixation_map)
    random = random_saliency(None, image, target_map=fixation_map, seed=0)

    assert nss(center, fixation_map) > nss(random, fixation_map)
    assert cc(center, simple_center_bias_map(16, 16)) > cc(random, simple_center_bias_map(16, 16))
