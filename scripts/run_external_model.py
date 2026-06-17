"""Run one pinned external model and export the shared Matrix V2 artifact."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def _install_pillow_legacy_resampling_compatibility() -> None:
    """Provide Pillow aliases expected by Detectron2 0.6."""
    resampling = getattr(Image, "Resampling", None)
    aliases = {
        "LINEAR": "BILINEAR",
        "CUBIC": "BICUBIC",
        "ANTIALIAS": "LANCZOS",
    }
    for legacy_name, modern_name in aliases.items():
        if legacy_name in vars(Image):
            continue
        value = getattr(resampling, modern_name, None) if resampling is not None else None
        if value is None:
            value = getattr(Image, modern_name, None)
        if value is not None:
            setattr(Image, legacy_name, value)


_install_pillow_legacy_resampling_compatibility()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.adapters import build_adapter, hardware_metadata
from hma.external.artifacts import ExternalArtifactWriter, sha256_file, sha256_tree
from hma.external.registry import load_external_registry
from hma.saliency.precomputed import precomputed_map_key
from hma.utils.paths import resolve_path


def run_external_model(
    model_id: str,
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path = "configs/external_models/registry.yaml",
    image_root: str | Path = "data/raw/nsd_algonauts",
    split: str | None = None,
    subject_id: str | None = None,
    roi: str | None = None,
    max_items: int | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 123,
    artifact_scope: str = "full",
    artifact_key: str = "image_id",
    profile_efficiency: bool = True,
) -> Path:
    if artifact_key not in {"image_id", "map_key"}:
        raise ValueError(f"Unsupported artifact key: {artifact_key}")
    registry = load_external_registry(registry_path)
    model = registry.model(model_id)
    canonical_id = str(model["id"])
    source_dir = registry.workspace_path("sources") / canonical_id
    checkpoint_path = _checkpoint_path(registry, model)
    rows = _load_image_rows(
        manifest_path,
        image_root=image_root,
        split=split,
        subject_id=subject_id,
        roi=roi,
        max_items=max_items,
        artifact_key=artifact_key,
    )
    adapter = build_adapter(
        str(model["adapter"]),
        model_id=canonical_id,
        model_config=model,
        source_dir=source_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )
    checkpoint_hash = _checkpoint_hash(canonical_id, checkpoint_path)
    environment_lock = resolve_path(
        f"configs/external_models/environment_locks/{canonical_id}.txt"
    )
    if not environment_lock.is_file():
        raise FileNotFoundError(
            f"External environment lock is missing: {environment_lock}"
        )
    provenance = {
        "model_id": canonical_id,
        "repository": model["source"]["repository"],
        "repository_commit": model["source"]["commit"],
        "environment_hash": sha256_file(environment_lock),
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "hardware": hardware_metadata(adapter.torch),
        "preprocessing": model.get("preprocessing", {}),
        "adapter": model["adapter"],
        "artifact_key": artifact_key,
        "source_license": model["source"]["license"],
    }
    writer = ExternalArtifactWriter(
        resolve_path(output_dir),
        model_id=canonical_id,
        provenance=provenance,
        expected_mechanism_outputs=model.get("mechanism_outputs", []),
        artifact_scope=artifact_scope,
    )
    first_images: list[Image.Image] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        image_ids = [row[artifact_key] for row in batch]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        if len(first_images) < 16:
            remaining = 16 - len(first_images)
            first_images.extend(image.copy() for image in images[:remaining])
        output = adapter.run_batch(images, image_ids)
        writer.write_batch(
            image_ids=image_ids,
            features=output.features if artifact_scope == "full" else {},
            logits=output.logits,
            task_outputs=output.task_outputs,
            resource_allocation=output.resource_allocation,
            scanpaths=output.scanpaths,
        )
        for image in images:
            image.close()
    if profile_efficiency:
        writer.set_efficiency(adapter.profile_efficiency(first_images[:16]))
    for image in first_images:
        image.close()
    return writer.finalize()


def smoke_external_model(
    model_id: str,
    *,
    registry_path: str | Path = "configs/external_models/registry.yaml",
    device: str = "cuda",
    seed: int = 123,
) -> dict[str, Any]:
    registry = load_external_registry(registry_path)
    model = registry.model(model_id)
    canonical_id = str(model["id"])
    adapter = build_adapter(
        str(model["adapter"]),
        model_id=canonical_id,
        model_config=model,
        source_dir=registry.workspace_path("sources") / canonical_id,
        checkpoint_path=_checkpoint_path(registry, model),
        device=device,
        seed=seed,
    )
    return adapter.smoke()


def _load_image_rows(
    manifest_path: str | Path,
    *,
    image_root: str | Path,
    split: str | None,
    subject_id: str | None,
    roi: str | None,
    max_items: int | None,
    artifact_key: str = "image_id",
) -> list[dict[str, str]]:
    if artifact_key not in {"image_id", "map_key"}:
        raise ValueError(f"Unsupported artifact key: {artifact_key}")
    path = resolve_path(manifest_path)
    root = resolve_path(image_root)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"image_id", "image_path"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Image manifest is missing columns: {sorted(missing)}")
        for record in reader:
            if split is not None and record.get("split") != split:
                continue
            if subject_id is not None and record.get("subject_id") != subject_id:
                continue
            if roi is not None and record.get("roi") != roi:
                continue
            image_id = str(record["image_id"])
            manifest_image_path = str(record["image_path"])
            map_key = precomputed_map_key(manifest_image_path)
            routing_key = image_id if artifact_key == "image_id" else map_key
            if routing_key in seen:
                continue
            image_path = Path(record["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"External-model image not found: {image_path}")
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "manifest_image_path": manifest_image_path,
                    "map_key": map_key,
                }
            )
            seen.add(routing_key)
            if max_items is not None and len(rows) >= int(max_items):
                break
    if not rows:
        raise ValueError("No image rows matched the external-model filters")
    return rows


def _checkpoint_path(registry: Any, model: dict[str, Any]) -> Path | None:
    filename = model.get("checkpoint", {}).get("filename")
    if not filename:
        return None
    return registry.workspace_path("checkpoints") / str(model["id"]) / str(filename)


def _checkpoint_hash(model_id: str, checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is not None and checkpoint_path.exists():
        return sha256_tree(checkpoint_path)
    lock_path = resolve_path(
        f"configs/external_models/checkpoint_locks/{model_id}.json"
    )
    if not lock_path.is_file():
        return None
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    value = payload.get("sha256")
    return str(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--registry", default="configs/external_models/registry.yaml"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--image-root", default="data/raw/nsd_algonauts")
    parser.add_argument("--split")
    parser.add_argument("--subject-id")
    parser.add_argument("--roi")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--artifact-scope",
        choices=["full", "resource_only"],
        default="full",
    )
    parser.add_argument(
        "--artifact-key",
        choices=["image_id", "map_key"],
        default="image_id",
        help="Identity written to the artifact; behavior exports must use map_key.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-efficiency-profile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_only:
        result = smoke_external_model(
            args.model,
            registry_path=args.registry,
            device=args.device,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.manifest or not args.output:
        raise SystemExit("--manifest and --output are required unless --smoke-only is used")
    manifest = run_external_model(
        args.model,
        manifest_path=args.manifest,
        output_dir=args.output,
        registry_path=args.registry,
        image_root=args.image_root,
        split=args.split,
        subject_id=args.subject_id,
        roi=args.roi,
        max_items=args.max_items,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        artifact_scope=args.artifact_scope,
        artifact_key=args.artifact_key,
        profile_efficiency=not args.skip_efficiency_profile,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
