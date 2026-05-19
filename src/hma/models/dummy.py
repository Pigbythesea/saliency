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

    def get_features(self, images, layers: list[str] | None = None):
        """Return compact deterministic image statistics for neural smoke tests."""
        requested = layers or ["embedding"]
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None and isinstance(images, torch.Tensor):
            tensor = images.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            flattened = tensor.flatten(start_dim=2)
            channel_mean = flattened.mean(dim=2)
            channel_std = flattened.std(dim=2, unbiased=False)
            global_mean = channel_mean.mean(dim=1, keepdim=True)
            features = torch.cat([channel_mean, channel_std, global_mean], dim=1)
            return {layer: features for layer in requested}

        array = np.asarray(images, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        flattened = array.reshape(array.shape[0], array.shape[1], -1)
        channel_mean = flattened.mean(axis=2)
        channel_std = flattened.std(axis=2)
        global_mean = channel_mean.mean(axis=1, keepdims=True)
        features = np.concatenate([channel_mean, channel_std, global_mean], axis=1)
        return {layer: features for layer in requested}

    def get_last_logits(self, images):
        """Return deterministic pseudo-logits from image statistics."""
        features = self.get_features(images, layers=["logits"])["logits"]
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(features, torch.Tensor):
            return torch.stack(
                [
                    features[:, 0],
                    features[:, 1] if features.shape[1] > 1 else features[:, 0],
                    features.mean(dim=1),
                ],
                dim=1,
            )
        return np.stack(
            [
                features[:, 0],
                features[:, 1] if features.shape[1] > 1 else features[:, 0],
                features.mean(axis=1),
            ],
            axis=1,
        )
