"""Dataset abstractions and registry."""

from hma.datasets.base import BaseVisionDataset, VisionDatasetItem
from hma.datasets.registry import build_dataset, get_dataset_class, register_dataset

# Import built-in datasets so they register themselves.
from hma.datasets.cat2000 import CAT2000Dataset
from hma.datasets.dummy import DummySaliencyDataset
from hma.datasets.salicon import SALICONDataset

__all__ = [
    "BaseVisionDataset",
    "CAT2000Dataset",
    "DummySaliencyDataset",
    "SALICONDataset",
    "VisionDatasetItem",
    "build_dataset",
    "get_dataset_class",
    "register_dataset",
]
