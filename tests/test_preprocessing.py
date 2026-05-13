import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from hma.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image_for_model


def test_preprocess_pil_rgb_to_batched_tensor_with_imagenet_normalization():
    image = Image.new("RGB", (2, 2), color=(255, 255, 255))

    tensor = preprocess_image_for_model(
        image,
        {
            "device": "cpu",
            "preprocessing": {
                "input_size": [4, 4],
                "mean": "imagenet",
                "std": "imagenet",
            },
        },
    )

    expected = torch.tensor(
        [(1.0 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        dtype=torch.float32,
    )
    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.device.type == "cpu"
    assert torch.allclose(tensor[0, :, 0, 0], expected)


def test_preprocess_numpy_and_tensor_inputs_are_consistent_without_normalization():
    array = np.zeros((2, 2, 3), dtype=np.float32)
    array[:, :, 0] = 0.25
    array[:, :, 1] = 0.5
    array[:, :, 2] = 0.75
    config = {
        "device": "cpu",
        "preprocessing": {"input_size": [2, 2], "mean": "none", "std": "none"},
    }

    numpy_tensor = preprocess_image_for_model(array, config)
    torch_tensor = preprocess_image_for_model(torch.from_numpy(array), config)

    assert numpy_tensor.shape == (1, 3, 2, 2)
    assert torch.allclose(numpy_tensor, torch_tensor)
