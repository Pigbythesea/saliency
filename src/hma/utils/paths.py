"""Path helpers for repository-relative configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Find the repository root by walking upward to pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate project root containing pyproject.toml")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path against a base directory or the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    base = Path(base_dir).expanduser() if base_dir is not None else get_project_root()
    return (base / candidate).resolve()


def get_data_root(config: dict[str, Any]) -> Path:
    """Return the configured dataset root, defaulting to project data/."""
    dataset_config = config.get("dataset", {})
    return resolve_path(dataset_config.get("root", "data"))


def get_output_dir(config: dict[str, Any]) -> Path:
    """Return and create the configured output directory."""
    output_config = config.get("output", {})
    return ensure_dir(resolve_path(output_config.get("dir", "outputs/default")))
