"""Shared dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


VisionDatasetItem = dict[str, Any]
"""Standard sample format for vision alignment datasets.

Expected keys:
- image: image data as a tensor-like object, NumPy array, or PIL image.
- image_id: stable string identifier for the image/frame.
- image_path: source path string, which may be synthetic for generated data.
- fixation_map: optional saliency/fixation density array or tensor.
- fixation_points: optional Nx2 array of fixation points in (x, y) order.
- metadata: dataset-specific metadata dictionary.
"""


class BaseVisionDataset(ABC):
    """Abstract base class for HMA vision datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples exposed by the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> VisionDatasetItem:
        """Return one sample in the standard HMA dataset item format."""
