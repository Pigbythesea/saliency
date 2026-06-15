"""Versioned artifact contract shared by isolated model environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hma.external.hashing import sha256_file, sha256_tree


ARTIFACT_SCHEMA_VERSION = "hma.external.artifact.v1"
ARTIFACT_SCOPES = {"full", "resource_only"}
REQUIRED_PROVENANCE_KEYS = {
    "model_id",
    "repository",
    "repository_commit",
    "environment_hash",
    "checkpoint_hash",
    "seed",
    "hardware",
    "preprocessing",
}


class ExternalArtifactWriter:
    """Write chunked feature and mechanism arrays without reducing their shape."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model_id: str,
        provenance: dict[str, Any],
        expected_mechanism_outputs: Iterable[str] = (),
        artifact_scope: str = "full",
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = str(model_id)
        self.provenance = dict(provenance)
        self.provenance.setdefault("model_id", self.model_id)
        missing = REQUIRED_PROVENANCE_KEYS - set(self.provenance)
        if missing:
            raise ValueError(f"Artifact provenance is missing keys: {sorted(missing)}")
        _validate_provenance_values(self.provenance)
        self.expected_mechanism_outputs = sorted(
            str(value) for value in expected_mechanism_outputs
        )
        self.artifact_scope = str(artifact_scope)
        if self.artifact_scope not in ARTIFACT_SCOPES:
            raise ValueError(
                f"artifact_scope must be one of {sorted(ARTIFACT_SCOPES)}"
            )
        self.image_ids: list[str] = []
        self.feature_chunks: dict[str, list[dict[str, Any]]] = {}
        self.output_chunks: dict[str, list[dict[str, Any]]] = {}
        self.resource_chunks: dict[str, list[dict[str, Any]]] = {}
        self.scanpath_records: list[dict[str, Any]] = []
        self.efficiency: dict[str, Any] = {}
        self._chunk_index = 0

    def write_batch(
        self,
        *,
        image_ids: list[str],
        features: dict[str, Any],
        logits: Any | None = None,
        task_outputs: dict[str, Any] | None = None,
        resource_allocation: dict[str, Any] | None = None,
        scanpaths: list[dict[str, Any]] | None = None,
    ) -> None:
        normalized_ids = [str(value) for value in image_ids]
        if not normalized_ids:
            raise ValueError("Artifact batches must contain at least one image")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Image IDs must be unique within each artifact batch")
        overlap = set(normalized_ids) & set(self.image_ids)
        if overlap:
            raise ValueError(f"Artifact image IDs are duplicated across batches: {sorted(overlap)}")
        start = len(self.image_ids)
        stop = start + len(normalized_ids)
        self.image_ids.extend(normalized_ids)

        for layer, values in features.items():
            array = _batch_array(values, len(normalized_ids), f"feature '{layer}'")
            self._write_array_chunk(
                category="features",
                key=str(layer),
                array=array,
                image_ids=normalized_ids,
                start=start,
                stop=stop,
                target=self.feature_chunks,
            )
        if logits is not None:
            array = _batch_array(logits, len(normalized_ids), "logits")
            self._write_array_chunk(
                category="outputs",
                key="logits",
                array=array,
                image_ids=normalized_ids,
                start=start,
                stop=stop,
                target=self.output_chunks,
            )
        for key, values in (task_outputs or {}).items():
            array = _batch_array(values, len(normalized_ids), f"task output '{key}'")
            self._write_array_chunk(
                category="outputs",
                key=str(key),
                array=array,
                image_ids=normalized_ids,
                start=start,
                stop=stop,
                target=self.output_chunks,
            )
        for key, values in (resource_allocation or {}).items():
            array = _batch_array(values, len(normalized_ids), f"resource output '{key}'")
            self._write_array_chunk(
                category="resource",
                key=str(key),
                array=array,
                image_ids=normalized_ids,
                start=start,
                stop=stop,
                target=self.resource_chunks,
            )
        for record in scanpaths or []:
            item = dict(record)
            if str(item.get("image_id", "")) not in normalized_ids:
                raise ValueError("Each scanpath record must identify an image in its batch")
            self.scanpath_records.append(item)
        self._chunk_index += 1

    def set_efficiency(self, metrics: dict[str, Any]) -> None:
        self.efficiency = _jsonable(dict(metrics))

    def finalize(self) -> Path:
        if not self.image_ids:
            raise ValueError("Cannot finalize an empty external artifact")
        image_ids_path = self.output_dir / "image_ids.json"
        image_ids_path.write_text(
            json.dumps(self.image_ids, indent=2),
            encoding="utf-8",
        )
        scanpaths_path: str | None = None
        if self.scanpath_records:
            path = self.output_dir / "scanpaths.jsonl"
            path.write_text(
                "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in self.scanpath_records),
                encoding="utf-8",
            )
            scanpaths_path = path.name
        self.efficiency.setdefault(
            "resource_summary",
            _resource_summary(self.output_dir, self.resource_chunks),
        )
        efficiency_path = self.output_dir / "efficiency.json"
        efficiency_path.write_text(
            json.dumps(self.efficiency, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "artifact_scope": self.artifact_scope,
            "num_images": len(self.image_ids),
            "image_ids_file": image_ids_path.name,
            "features": self.feature_chunks,
            "outputs": self.output_chunks,
            "resource_allocation": self.resource_chunks,
            "expected_mechanism_outputs": self.expected_mechanism_outputs,
            "scanpaths_file": scanpaths_path,
            "efficiency_file": efficiency_path.name,
            "provenance": _jsonable(self.provenance),
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validate_external_artifact(self.output_dir, verify_hashes=True)
        return manifest_path

    def _write_array_chunk(
        self,
        *,
        category: str,
        key: str,
        array: np.ndarray,
        image_ids: list[str],
        start: int,
        stop: int,
        target: dict[str, list[dict[str, Any]]],
    ) -> None:
        safe_key = _safe_name(key)
        relative = Path(category) / safe_key / f"chunk_{self._chunk_index:05d}.npz"
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_ids=np.asarray(image_ids, dtype=str),
            values=np.asarray(array),
        )
        target.setdefault(key, []).append(
            {
                "file": relative.as_posix(),
                "start": start,
                "stop": stop,
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "sha256": sha256_file(path),
            }
        )


def validate_external_artifact(
    artifact_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate ordering, chunk boundaries, hashes, and mechanism-output gates."""
    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External artifact manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported external artifact schema_version")
    artifact_scope = str(manifest.get("artifact_scope", "full"))
    if artifact_scope not in ARTIFACT_SCOPES:
        raise ValueError("Unsupported external artifact artifact_scope")
    image_ids_path = root / str(manifest.get("image_ids_file", "image_ids.json"))
    image_ids = json.loads(image_ids_path.read_text(encoding="utf-8"))
    if len(image_ids) != int(manifest.get("num_images", -1)):
        raise ValueError("External artifact num_images does not match image_ids")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("External artifact image_ids must be unique and ordered")
    provenance = manifest.get("provenance", {})
    missing = REQUIRED_PROVENANCE_KEYS - set(provenance)
    if missing:
        raise ValueError(f"External artifact provenance is missing keys: {sorted(missing)}")
    _validate_provenance_values(provenance)
    for category in ("features", "outputs", "resource_allocation"):
        mapping = manifest.get(category, {})
        if not isinstance(mapping, dict):
            raise TypeError(f"External artifact '{category}' must be a mapping")
        for key, chunks in mapping.items():
            _validate_chunks(
                root,
                chunks,
                image_ids=image_ids,
                label=f"{category}.{key}",
                verify_hashes=verify_hashes,
            )
    expected = set(manifest.get("expected_mechanism_outputs", []))
    actual = set(manifest.get("resource_allocation", {})) | set(
        manifest.get("outputs", {})
    )
    if manifest.get("scanpaths_file"):
        actual.add("scanpaths")
    if expected and not _mechanism_outputs_satisfied(expected, actual):
        missing_outputs = sorted(
            expected_name
            for expected_name in expected
            if not any(
                actual_name == expected_name or actual_name.startswith(expected_name + ".")
                for actual_name in actual
            )
        )
        raise ValueError(
            "External artifact is missing required operational outputs: "
            f"{missing_outputs}"
        )
    if artifact_scope == "full" and not manifest.get("features"):
        raise ValueError("Full external artifacts must contain feature tensors")
    if artifact_scope == "resource_only" and manifest.get("features"):
        raise ValueError("Resource-only external artifacts may not contain features")
    return manifest


def load_external_features(
    artifact_dir: str | Path,
    *,
    layers: Iterable[str] | None = None,
    verify_hashes: bool = True,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Load ordered raw feature tensors for the shared core analysis."""
    root = Path(artifact_dir).expanduser().resolve()
    manifest = validate_external_artifact(root, verify_hashes=verify_hashes)
    image_ids = json.loads(
        (root / str(manifest["image_ids_file"])).read_text(encoding="utf-8")
    )
    requested = list(layers) if layers is not None else list(manifest["features"])
    missing = [layer for layer in requested if layer not in manifest["features"]]
    if missing:
        raise KeyError(f"External artifact is missing requested feature layers: {missing}")
    features: dict[str, np.ndarray] = {}
    for layer in requested:
        arrays = []
        for chunk in manifest["features"][layer]:
            with np.load(root / chunk["file"], allow_pickle=False) as payload:
                arrays.append(np.asarray(payload["values"]))
        features[layer] = np.concatenate(arrays, axis=0)
    return [str(value) for value in image_ids], features, manifest


def load_external_features_to_memmaps(
    artifact_dir: str | Path,
    *,
    layers: Iterable[str],
    storage_dir: str | Path,
    verify_hashes: bool = True,
) -> tuple[list[str], dict[str, np.memmap], dict[str, Any]]:
    """Stream chunked raw features into disk-backed arrays for full-image runs."""
    root = Path(artifact_dir).expanduser().resolve()
    manifest = validate_external_artifact(root, verify_hashes=verify_hashes)
    image_ids = json.loads(
        (root / str(manifest["image_ids_file"])).read_text(encoding="utf-8")
    )
    requested = list(layers)
    missing = [layer for layer in requested if layer not in manifest["features"]]
    if missing:
        raise KeyError(f"External artifact is missing requested feature layers: {missing}")
    destination = Path(storage_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache_index_path = destination / "memmap_index.json"
    cache_key = _memmap_cache_key(root, manifest, requested)
    cache_index = _load_memmap_index(cache_index_path)
    features: dict[str, np.memmap] = {}
    if cache_index.get("cache_key") == cache_key:
        cached = _open_cached_memmaps(
            destination,
            cache_index,
            layers=requested,
            num_images=len(image_ids),
        )
        if cached is not None:
            return [str(value) for value in image_ids], cached, manifest

    cache_layers: dict[str, dict[str, Any]] = {}
    for layer in requested:
        chunks = manifest["features"][layer]
        first_shape = tuple(int(value) for value in chunks[0]["shape"][1:])
        dtype = np.dtype(chunks[0]["dtype"])
        path = destination / f"{_safe_name(layer)}.dat"
        memmap = np.memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(len(image_ids), *first_shape),
        )
        for chunk in chunks:
            with np.load(root / chunk["file"], allow_pickle=False) as payload:
                memmap[int(chunk["start"]) : int(chunk["stop"])] = payload["values"]
        memmap.flush()
        features[layer] = memmap
        cache_layers[layer] = {
            "file": path.name,
            "dtype": str(dtype),
            "shape": [len(image_ids), *first_shape],
        }
    cache_index_path.write_text(
        json.dumps(
            {
                "schema_version": "hma.external.memmap_cache.v1",
                "cache_key": cache_key,
                "layers": cache_layers,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [str(value) for value in image_ids], features, manifest


def load_external_arrays(
    artifact_dir: str | Path,
    *,
    category: str,
    keys: Iterable[str] | None = None,
    verify_hashes: bool = True,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Load ordered output or resource arrays from an external artifact."""
    if category not in {"outputs", "resource_allocation"}:
        raise ValueError("category must be 'outputs' or 'resource_allocation'")
    root = Path(artifact_dir).expanduser().resolve()
    manifest = validate_external_artifact(root, verify_hashes=verify_hashes)
    image_ids = json.loads(
        (root / str(manifest["image_ids_file"])).read_text(encoding="utf-8")
    )
    mapping = manifest[category]
    requested = list(keys) if keys is not None else list(mapping)
    missing = [key for key in requested if key not in mapping]
    if missing:
        raise KeyError(f"External artifact is missing requested {category}: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for key in requested:
        chunks = []
        for chunk in mapping[key]:
            with np.load(root / chunk["file"], allow_pickle=False) as payload:
                chunks.append(np.asarray(payload["values"]))
        arrays[key] = np.concatenate(chunks, axis=0)
    return [str(value) for value in image_ids], arrays, manifest


def _validate_chunks(
    root: Path,
    chunks: Any,
    *,
    image_ids: list[str],
    label: str,
    verify_hashes: bool,
) -> None:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"External artifact '{label}' must contain chunks")
    expected_start = 0
    feature_shape: tuple[int, ...] | None = None
    for chunk in chunks:
        start = int(chunk["start"])
        stop = int(chunk["stop"])
        if start != expected_start or stop <= start:
            raise ValueError(f"External artifact '{label}' has non-contiguous chunks")
        path = root / str(chunk["file"])
        if not path.is_file():
            raise FileNotFoundError(f"External artifact chunk not found: {path}")
        if verify_hashes and sha256_file(path) != chunk["sha256"]:
            raise ValueError(f"External artifact chunk hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            chunk_ids = [str(value) for value in payload["image_ids"].tolist()]
            values = np.asarray(payload["values"])
        if chunk_ids != image_ids[start:stop]:
            raise ValueError(f"External artifact '{label}' image order mismatch")
        if values.shape[0] != stop - start:
            raise ValueError(f"External artifact '{label}' batch dimension mismatch")
        if [int(value) for value in values.shape] != list(chunk["shape"]):
            raise ValueError(f"External artifact '{label}' recorded shape mismatch")
        current_shape = tuple(int(value) for value in values.shape[1:])
        if feature_shape is None:
            feature_shape = current_shape
        elif current_shape != feature_shape:
            raise ValueError(f"External artifact '{label}' changes shape across chunks")
        expected_start = stop
    if expected_start != len(image_ids):
        raise ValueError(f"External artifact '{label}' does not cover every image")


def _mechanism_outputs_satisfied(expected: set[str], actual: set[str]) -> bool:
    return all(
        any(
            name == required
            or name.startswith(required + ".")
            or (
                required in {"stochastic_scanpaths", "generated_scanpaths"}
                and name == "scanpaths"
            )
            for name in actual
        )
        for required in expected
    )


def _memmap_cache_key(
    root: Path,
    manifest: dict[str, Any],
    layers: list[str],
) -> str:
    payload = {
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "layers": layers,
        "feature_chunks": {
            layer: manifest["features"][layer]
            for layer in layers
        },
    }
    return _sha256_json(payload)


def _load_memmap_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _open_cached_memmaps(
    destination: Path,
    cache_index: dict[str, Any],
    *,
    layers: list[str],
    num_images: int,
) -> dict[str, np.memmap] | None:
    cached_layers = cache_index.get("layers")
    if not isinstance(cached_layers, dict):
        return None
    opened: dict[str, np.memmap] = {}
    for layer in layers:
        entry = cached_layers.get(layer)
        if not isinstance(entry, dict):
            return None
        shape = tuple(int(value) for value in entry.get("shape", []))
        if not shape or shape[0] != num_images:
            return None
        dtype = np.dtype(entry.get("dtype"))
        path = destination / str(entry.get("file", ""))
        expected_size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if not path.is_file() or path.stat().st_size != expected_size:
            return None
        opened[layer] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
    return opened


def _sha256_json(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_provenance_values(provenance: dict[str, Any]) -> None:
    for key in (
        "model_id",
        "repository",
        "repository_commit",
        "environment_hash",
        "checkpoint_hash",
    ):
        if not provenance.get(key):
            raise ValueError(f"Artifact provenance '{key}' must be non-empty")
    if provenance["repository_commit"] == "PIN_REQUIRED":
        raise ValueError("Artifact provenance may not use an unresolved source pin")


def _resource_summary(
    root: Path,
    resources: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    scalar_resources = {
        "fixation_count",
        "scanpath_length",
        "recurrent_steps",
        "diffusion_steps",
        "selected_glimpses",
        "stopping_behavior",
        "high_resolution_sampled_area",
        "total_cost_per_image_task",
    }
    for key, chunks in resources.items():
        if not (
            key.startswith("realized_token_counts.")
            or key.startswith("prediction_masks.")
            or key == "full_token_mask"
            or key.startswith("token_source_assignments.")
            or key in scalar_resources
        ):
            continue
        arrays = []
        for chunk in chunks:
            with np.load(root / chunk["file"], allow_pickle=False) as payload:
                arrays.append(np.asarray(payload["values"], dtype=np.float32))
        values = np.concatenate(arrays, axis=0)
        if key.startswith("realized_token_counts."):
            summary[key] = {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        elif key.startswith("prediction_masks.") or key == "full_token_mask":
            summary[key] = {
                "mean_retained_fraction": float(np.mean(values)),
            }
        elif key.startswith("token_source_assignments.") and values.ndim >= 3:
            summary[key] = {
                "mean_realized_tokens": float(values.shape[1]),
                "original_tokens": int(values.shape[2]),
            }
        elif key in scalar_resources:
            summary[key] = {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "total": float(np.sum(values)),
            }
    return summary


def _batch_array(values: Any, batch_size: int, label: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] != batch_size:
        raise ValueError(
            f"{label} must have leading batch dimension {batch_size}, got {array.shape}"
        )
    if array.dtype == object:
        raise TypeError(f"{label} may not use object dtype")
    return array


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip("._") or "unnamed"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
