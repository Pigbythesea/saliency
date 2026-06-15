"""External-model integration for Matrix V2.

Imports stay lazy so the stdlib-only environment setup entry point can inspect
the registry without importing NumPy-backed artifact code.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ExternalArtifactWriter",
    "ExternalModelRegistry",
    "build_certification_records",
    "load_external_arrays",
    "load_external_features",
    "load_external_features_to_memmaps",
    "load_external_registry",
    "load_publication_adapter_registry",
    "validate_external_artifact",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ARTIFACT_SCHEMA_VERSION",
        "ExternalArtifactWriter",
        "load_external_arrays",
        "load_external_features",
        "load_external_features_to_memmaps",
        "validate_external_artifact",
    }:
        from hma.external import artifacts

        return getattr(artifacts, name)
    if name in {"ExternalModelRegistry", "load_external_registry"}:
        from hma.external import registry

        return getattr(registry, name)
    if name in {
        "build_certification_records",
        "load_publication_adapter_registry",
    }:
        from hma.external import certification

        return getattr(certification, name)
    raise AttributeError(name)
