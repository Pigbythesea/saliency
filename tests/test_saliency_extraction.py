import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hma.saliency import (  # noqa: E402
    COCOSearch18TaskPrior,
    EmpiricalSpatialPrior,
    center_bias_saliency,
    build_saliency_method,
    coco_search18_task_prior_saliency,
    empirical_spatial_prior_saliency,
    gradcam_saliency,
    integrated_gradients_saliency,
    occlusion_saliency,
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


def test_occlusion_saliency_shape_range_and_determinism(tiny_wrapper, images):
    first = occlusion_saliency(
        tiny_wrapper,
        images,
        patch_size=4,
        stride=4,
        baseline_value=0.0,
    )
    second = occlusion_saliency(
        tiny_wrapper,
        images,
        patch_size=4,
        stride=4,
        baseline_value=0.0,
    )

    _assert_valid_saliency_map(first)
    assert torch.allclose(first, second)


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
    occlusion = build_saliency_method(
        {"saliency": {"method": "occlusion", "patch_size": 4, "stride": 4}}
    )

    _assert_valid_saliency_map(vanilla(tiny_wrapper, images))
    _assert_valid_saliency_map(integrated(tiny_wrapper, images))
    _assert_valid_saliency_map(gradcam(tiny_wrapper, images))
    _assert_valid_saliency_map(occlusion(tiny_wrapper, images))


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


def test_coco_search18_task_prior_uses_target_and_task_conditioning(tmp_path):
    manifest = tmp_path / "coco.csv"
    rows = [
        ("train", "cup", "present", [[2, 2], [3, 2]]),
        ("train", "cup", "absent", [[13, 13], [14, 13]]),
        ("train", "bottle", "present", [[8, 8], [9, 8]]),
        ("val", "cup", "present", [[15, 15]]),
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "width",
                "height",
                "target_category",
                "task",
                "fixation_points",
            ],
        )
        writer.writeheader()
        for split, target, task, points in rows:
            writer.writerow(
                {
                    "split": split,
                    "width": "16",
                    "height": "16",
                    "target_category": target,
                    "task": task,
                    "fixation_points": json.dumps(points),
                }
            )

    prior = COCOSearch18TaskPrior.from_manifest(
        manifest,
        image_size=(16, 16),
        fixation_sigma=1.0,
    )

    cup_present = prior.map_for(target_category="cup", task="present")
    cup_absent = prior.map_for(target_category="cup", task="absent")
    present_y, present_x = np.unravel_index(np.argmax(cup_present), cup_present.shape)
    absent_y, absent_x = np.unravel_index(np.argmax(cup_absent), cup_absent.shape)

    assert present_x <= 3
    assert present_y <= 3
    assert absent_x >= 13
    assert absent_y >= 13

    target = np.zeros((16, 16), dtype=np.float32)
    prediction = coco_search18_task_prior_saliency(
        None,
        torch.zeros(1, 3, 16, 16),
        target_map=target,
        item={"metadata": {"target_category": "cup", "task": "present"}},
        prior=prior,
    )

    assert prediction.shape == (16, 16)
    assert np.isfinite(prediction).all()


def test_empirical_spatial_prior_uses_train_split_and_excludes_eval_ids(tmp_path):
    manifest = tmp_path / "fixations.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "split", "width", "height", "fixation_points"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "train_allowed",
                "split": "train",
                "width": "16",
                "height": "16",
                "fixation_points": json.dumps([[2, 2], [3, 2]]),
            }
        )
        writer.writerow(
            {
                "image_id": "train_excluded",
                "split": "train",
                "width": "16",
                "height": "16",
                "fixation_points": json.dumps([[14, 14], [15, 14]]),
            }
        )
        writer.writerow(
            {
                "image_id": "val_leak",
                "split": "val",
                "width": "16",
                "height": "16",
                "fixation_points": json.dumps([[15, 15]]),
            }
        )

    prior = EmpiricalSpatialPrior.from_manifest(
        manifest,
        image_size=(16, 16),
        fixation_sigma=1.0,
        exclude_image_ids={"train_excluded"},
    )
    y, x = np.unravel_index(np.argmax(prior.map), prior.map.shape)

    assert x <= 3
    assert y <= 3
    assert "train_excluded" in prior.excluded_image_ids

    prediction = empirical_spatial_prior_saliency(
        None,
        torch.zeros(1, 3, 16, 16),
        target_map=np.zeros((16, 16), dtype=np.float32),
        prior=prior,
    )
    assert prediction.shape == (16, 16)
    assert np.isfinite(prediction).all()
