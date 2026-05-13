"""Deterministic fake saliency predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hma.saliency.postprocess import normalize_saliency_map


@dataclass(frozen=True)
class DummySaliencyModel:
    """Generate fake saliency maps from image intensity plus small noise."""

    noise_scale: float = 0.05
    seed: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "DummySaliencyModel":
        return cls(
            noise_scale=float(config.get("noise_scale", 0.05)),
            seed=int(config.get("seed", 0)),
        )

    def predict(self, image: np.ndarray, item_index: int = 0) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError("Expected image with shape (channels, height, width)")

        base_map = image.mean(axis=0)
        rng = np.random.default_rng(self.seed + item_index)
        noise = rng.normal(0.0, self.noise_scale, size=base_map.shape)
        return normalize_saliency_map(base_map + noise)
