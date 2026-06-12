"""Loading and validation for the Matrix V2 external-model registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hma.utils.config import load_yaml
from hma.utils.paths import resolve_path


REQUIRED_INSTALLATION_STAGES = (
    "source_ready",
    "environment_ready",
    "checkpoint_ready",
    "adapter_ready",
    "smoke_passed",
    "evidence_ready",
)


@dataclass(frozen=True)
class ExternalModelRegistry:
    """Validated external registry plus repository-relative path helpers."""

    path: Path
    payload: dict[str, Any]

    @property
    def workspace(self) -> dict[str, str]:
        return dict(self.payload.get("workspace", {}))

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return dict(self.payload.get("models", {}))

    def resolve_model_id(self, model_id: str) -> str:
        if model_id in self.models:
            return model_id
        for canonical_id, model in self.models.items():
            aliases = [str(alias) for alias in model.get("aliases", [])]
            if model_id in aliases:
                return canonical_id
        available = ", ".join(sorted(self.models))
        raise KeyError(f"Unknown external model '{model_id}'. Available models: {available}")

    def model(self, model_id: str) -> dict[str, Any]:
        canonical_id = self.resolve_model_id(model_id)
        return {"id": canonical_id, **dict(self.models[canonical_id])}

    def workspace_path(self, key: str) -> Path:
        value = self.workspace.get(key)
        if not value:
            raise KeyError(f"External registry workspace is missing '{key}'")
        return resolve_path(value)

    def environment_path(self, model_id: str) -> Path:
        value = self.model(model_id).get("environment")
        if not value:
            raise KeyError(f"External model '{model_id}' has no environment manifest")
        return resolve_path(value)


def load_external_registry(
    path: str | Path = "configs/external_models/registry.yaml",
) -> ExternalModelRegistry:
    """Load the external registry and reject incomplete structural entries."""
    registry_path = resolve_path(path)
    payload = load_yaml(registry_path)
    if payload.get("schema_version") != "hma.external.registry.v1":
        raise ValueError("Unsupported or missing external registry schema_version")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("External registry must define a non-empty models mapping")
    for model_id, model in models.items():
        _validate_model_entry(str(model_id), model)
    return ExternalModelRegistry(path=registry_path, payload=payload)


def _validate_model_entry(model_id: str, model: Any) -> None:
    if not isinstance(model, dict):
        raise TypeError(f"External model '{model_id}' must be a mapping")
    for key in ("track", "role", "source", "environment", "adapter", "checkpoint"):
        if key not in model:
            raise ValueError(f"External model '{model_id}' is missing '{key}'")
    source = model["source"]
    if not isinstance(source, dict):
        raise TypeError(f"External model '{model_id}' source must be a mapping")
    for key in ("repository", "commit", "license"):
        if not source.get(key):
            raise ValueError(f"External model '{model_id}' source is missing '{key}'")
    checkpoint = model["checkpoint"]
    if not isinstance(checkpoint, dict) or "hash_policy" not in checkpoint:
        raise ValueError(
            f"External model '{model_id}' checkpoint must define a hash_policy"
        )
