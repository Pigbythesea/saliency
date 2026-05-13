import numpy as np

from hma.datasets import DummySaliencyDataset, build_dataset


EXPECTED_ITEM_KEYS = {
    "image",
    "image_id",
    "image_path",
    "fixation_map",
    "fixation_points",
    "metadata",
}


def test_dummy_dataset_length_respects_max_items():
    dataset = DummySaliencyDataset(num_items=5, max_items=3)

    assert len(dataset) == 3


def test_dummy_dataset_item_uses_standard_keys():
    dataset = DummySaliencyDataset(split="test", num_items=1, seed=12)
    item = dataset[0]

    assert set(item) == EXPECTED_ITEM_KEYS
    assert item["image"].shape == (3, 16, 16)
    assert item["image_id"] == "test_0000"
    assert item["image_path"] == "data/dummy_saliency/test/test_0000.png"
    assert item["fixation_map"].shape == (16, 16)
    assert item["fixation_points"].shape == (8, 2)
    assert item["metadata"] == {
        "dataset": "dummy_saliency",
        "split": "test",
        "index": 0,
    }


def test_dummy_dataset_is_deterministic_under_fixed_seed():
    first = DummySaliencyDataset(num_items=2, seed=99)[1]
    second = DummySaliencyDataset(num_items=2, seed=99)[1]

    assert first["image_id"] == second["image_id"]
    assert first["image_path"] == second["image_path"]
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first["fixation_map"], second["fixation_map"])
    assert np.array_equal(first["fixation_points"], second["fixation_points"])


def test_build_dataset_from_experiment_config_uses_registry():
    dataset = build_dataset(
        {
            "dataset": {
                "name": "dummy_saliency",
                "num_items": 5,
                "max_items": 2,
                "seed": 3,
            }
        }
    )

    assert isinstance(dataset, DummySaliencyDataset)
    assert len(dataset) == 2
