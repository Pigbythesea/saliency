"""Base interfaces for image model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """Abstract wrapper interface for image models."""

    @abstractmethod
    def forward(self, images: Any) -> Any:
        """Run the model forward pass."""

    @abstractmethod
    def get_features(self, images: Any, layers: list[str] | None = None) -> Any:
        """Return intermediate or final feature representations."""

    @abstractmethod
    def get_last_logits(self, images: Any) -> Any:
        """Return the final model logits."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return wrapper metadata including model name and parameter count."""
