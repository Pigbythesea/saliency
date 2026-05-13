"""Deterministic synthetic datasets for tests and pipeline wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator

import numpy as np

from hma.datasets.base import BaseVisionDataset, VisionDatasetItem
from hma.datasets.registry import register_dataset
from hma.saliency.postprocess import normalize_saliency_map


@dataclass(frozen=True)
class DummySaliencyDataset(BaseVisionDataset):
    """Small deterministic saliency dataset used for tests and debug runs."""

    split: str = "val"
    num_items: int = 4
    max_items: int | None = None
    image_shape: tuple[int, int, int] = (3, 16, 16)
    map_shape: tuple[int, int] = (16, 16)
    num_fixation_points: int = 8
    root: str = "data/dummy_saliency"
    seed: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "DummySaliencyDataset":
        return cls(
            split=config.get("split", "val"),
            num_items=int(config.get("num_items", 4)),
            max_items=config.get("max_items"),
            image_shape=tuple(config.get("image_shape", (3, 16, 16))),
            map_shape=tuple(config.get("map_shape", (16, 16))),
            num_fixation_points=int(config.get("num_fixation_points", 8)),
            root=str(config.get("root", "data/dummy_saliency")),
            seed=int(config.get("seed", 0)),
        )

    def __len__(self) -> int:
        if self.max_items is None:
            return self.num_items
        return min(self.num_items, int(self.max_items))

    def __iter__(self) -> Iterator[VisionDatasetItem]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> VisionDatasetItem:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        rng = np.random.default_rng(self.seed + index)
        image = rng.random(self.image_shape, dtype=np.float32)
        fixation_map = normalize_saliency_map(
            rng.random(self.map_shape, dtype=np.float32)
        )
        fixation_points = self._make_fixation_points(rng)
        image_id = f"{self.split}_{index:04d}"
        image_path = str(PurePosixPath(self.root) / self.split / f"{image_id}.png")

        return {
            "image": image,
            "image_id": image_id,
            "image_path": image_path,
            "fixation_map": fixation_map,
            "fixation_points": fixation_points,
            "metadata": {
                "dataset": "dummy_saliency",
                "split": self.split,
                "index": index,
            },
        }

    def _make_fixation_points(self, rng: np.random.Generator) -> np.ndarray:
        height, width = self.map_shape
        xs = rng.integers(0, width, size=self.num_fixation_points, dtype=np.int32)
        ys = rng.integers(0, height, size=self.num_fixation_points, dtype=np.int32)
        return np.stack([xs, ys], axis=1)


register_dataset("dummy_saliency", DummySaliencyDataset)
register_dataset("dummy_static_saliency", DummySaliencyDataset)
