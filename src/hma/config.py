"""Backward-compatible configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hma.utils.config import load_yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a plain dictionary."""
    return load_yaml(path)
