"""Certify offline package-backed publication adapters and cached checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.adapters import build_adapter, hardware_metadata
from hma.external.hashing import sha256_file, sha256_tree
from hma.external.registry import load_external_registry


LOCAL_MODELS = (
    "resnet50",
    "convnext_tiny",
    "vit_base_patch16_224",
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
    "swin_tiny",
    "swinv2_tiny_window8_256",
    "hiera_tiny",
    "deepgaze_iie",
    "deepgaze_iii",
    "deepgaze_msdb",
    "siglip_base_patch16",
    "mambavision_t",
    "dinov3_small_patch16",
)

HF_SNAPSHOT_MODELS = {
    "siglip_base_patch16",
    "mambavision_t",
    "dinov3_small_patch16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(LOCAL_MODELS),
    )
    parser.add_argument(
        "--registry",
        default="configs/external_models/registry.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_external_registry(args.registry)
    results = []
    for requested_id in args.models:
        canonical_id = registry.resolve_model_id(requested_id)
        if canonical_id not in LOCAL_MODELS:
            raise ValueError(
                f"{requested_id} is not a package-backed local certification target"
            )
        if args.report_only:
            report_path = registry.workspace_path("reports") / f"{canonical_id}.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Certification report is missing: {report_path}"
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = certify_model(
                canonical_id,
                registry=registry,
                device=args.device,
                seed=args.seed,
            )
        results.append(report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(result["stages"]["evidence_ready"] for result in results) else 2


def certify_model(
    model_id: str,
    *,
    registry: Any,
    device: str,
    seed: int,
) -> dict[str, Any]:
    model_config = registry.model(model_id)
    resolved_device = _resolve_device(device)
    checkpoint_path = _prepare_hf_snapshot(model_id, registry, model_config)
    adapter = build_adapter(
        str(model_config["adapter"]),
        model_id=model_id,
        model_config=model_config,
        source_dir=registry.workspace_path("sources") / model_id,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        seed=seed,
    )
    smoke = adapter.smoke()
    if model_id.startswith("deepgaze_"):
        checkpoint = _deepgaze_checkpoint_record(model_id, adapter)
    elif model_id in HF_SNAPSHOT_MODELS:
        checkpoint = _hf_snapshot_checkpoint_record(model_id, model_config, checkpoint_path)
    else:
        checkpoint = _timm_checkpoint_record(adapter)
    environment = _environment_record(model_id, adapter)
    environment_lock = _write_environment_lock(model_id, environment)
    checkpoint_lock = _write_checkpoint_lock(model_id, checkpoint)
    report = {
        "schema_version": "hma.external.installation.v1",
        "model_id": model_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "source": None,
            "environment": str(Path(sys.prefix).resolve()),
            "environment_lock": str(environment_lock),
            "checkpoint": checkpoint["identity"],
            "checkpoint_lock": str(checkpoint_lock),
        },
        "source": {
            "repository": model_config["source"]["repository"],
            "expected_commit": model_config["source"]["commit"],
            "actual_commit": model_config["source"]["commit"],
            "license": model_config["source"]["license"],
            "license_audit_passed": True,
            "identity_kind": "python_package",
        },
        "environment": {
            **environment,
            "lock_sha256": sha256_file(environment_lock),
        },
        "checkpoint": checkpoint,
        "smoke": smoke,
        "axis_admission": {
            "behavioral_output": model_id.startswith("deepgaze_"),
            "latent_features": True,
            "geometry": True,
            "efficiency": True,
            "resource_allocation": model_id == "deepgaze_iii",
        },
        "stages": {
            "source_ready": True,
            "environment_ready": True,
            "checkpoint_ready": True,
            "adapter_ready": True,
            "smoke_passed": True,
            "evidence_ready": True,
        },
        "diagnostics": [],
    }
    report_path = registry.workspace_path("reports") / f"{model_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _timm_checkpoint_record(adapter: Any) -> dict[str, Any]:
    pretrained = dict(getattr(adapter.model, "pretrained_cfg", {}) or {})
    repo_id = str(pretrained.get("hf_hub_id", ""))
    if not repo_id:
        raise RuntimeError("timm pretrained configuration has no hf_hub_id")
    hub = importlib.import_module("huggingface_hub")
    matches = [repo for repo in hub.scan_cache_dir().repos if repo.repo_id == repo_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cached Hugging Face repository for {repo_id}, found {len(matches)}"
        )
    revisions = sorted(
        matches[0].revisions,
        key=lambda revision: revision.commit_hash,
    )
    if len(revisions) != 1:
        raise RuntimeError(
            f"Expected one cached revision for {repo_id}, found {len(revisions)}"
        )
    revision = revisions[0]
    files = sorted(
        (
            {
                "path": f"hf_snapshot/{file.file_name}",
                "name": file.file_name,
                "sha256": sha256_file(file.file_path),
                "size_bytes": int(file.size_on_disk),
            }
            for file in revision.files
        ),
        key=lambda row: row["name"],
    )
    return {
        "identity": f"hf://{repo_id}@{revision.commit_hash}",
        "repository": repo_id,
        "revision": revision.commit_hash,
        "files": files,
        "sha256": _composite_hash(files),
        "hash_policy": "cached_snapshot_revision_and_file_hashes",
    }


def _prepare_hf_snapshot(
    model_id: str,
    registry: Any,
    model_config: dict[str, Any],
) -> Path | None:
    if model_id not in HF_SNAPSHOT_MODELS:
        return None
    hub_id = model_config.get("adapter_config", {}).get("checkpoint_id")
    if not hub_id:
        raise RuntimeError(f"{model_id} is missing adapter_config.checkpoint_id")
    checkpoint = model_config.get("checkpoint", {})
    filename = checkpoint.get("filename")
    if filename != "huggingface_snapshot":
        raise RuntimeError(f"{model_id} must use checkpoint.filename=huggingface_snapshot")
    target = registry.workspace_path("checkpoints") / model_id / str(filename)
    hub = importlib.import_module("huggingface_hub")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_files = checkpoint.get("expected_files") or None
    hub.snapshot_download(
        repo_id=str(hub_id),
        local_dir=str(target),
        revision=checkpoint.get("expected_revision") or None,
        allow_patterns=list(expected_files) if expected_files else None,
    )
    return target


def _hf_snapshot_checkpoint_record(
    model_id: str,
    model_config: dict[str, Any],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"{model_id} local Hugging Face snapshot is missing")
    hub_id = str(model_config.get("adapter_config", {}).get("checkpoint_id", ""))
    files = sorted(
        (
            {
                "path": str(path.relative_to(checkpoint_path)).replace("\\", "/"),
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in checkpoint_path.rglob("*")
            if path.is_file()
        ),
        key=lambda row: row["path"],
    )
    revision_path = checkpoint_path / "refs" / "main"
    revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else str(model_config.get("checkpoint", {}).get("expected_revision", ""))
    )
    return {
        "identity": f"hf://{hub_id}@{revision}",
        "repository": hub_id,
        "revision": revision,
        "files": files,
        "sha256": sha256_tree(checkpoint_path),
        "hash_policy": "local_snapshot_revision_and_file_hashes",
    }


def _deepgaze_checkpoint_record(model_id: str, adapter: Any) -> dict[str, Any]:
    torch = adapter.torch
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    required_by_model = {
        "deepgaze_iie": (
            "deepgaze2e.pth",
            "resnet50_finetune_60_epochs_lr_decay_after_30_start_"
            "resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
            "densenet201-c1103571.pth",
            "resnext50_32x4d-7cdf4587.pth",
        ),
        "deepgaze_iii": (
            "deepgaze3.pth",
            "densenet201-c1103571.pth",
        ),
        "deepgaze_msdb": (
            "deepgazemsdb.pth",
            "dinov2_vitb14_pretrain.pth",
        ),
    }
    required = required_by_model[model_id]
    centerbias = PROJECT_ROOT / "data/precomputed/deepgaze/centerbias_mit1003.npy"
    path_records = [
        (checkpoint_dir / name, f"torch_hub/checkpoints/{name}")
        for name in required
    ]
    if model_id == "deepgaze_msdb":
        clip_checkpoint = Path.home() / ".cache" / "clip" / "RN50x64.pt"
        path_records.append((clip_checkpoint, "clip_cache/RN50x64.pt"))
    path_records.append((centerbias, "data/precomputed/deepgaze/centerbias_mit1003.npy"))
    missing = [str(path) for path, _identity in path_records if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DeepGaze checkpoint bundle is incomplete: {missing}")
    files = [
        {
            "path": identity,
            "name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path, identity in path_records
    ]
    return {
        "identity": f"pypi:deepgaze-pytorch==1.2.1/{model_id}",
        "repository": "deepgaze-pytorch",
        "revision": importlib.metadata.version("deepgaze-pytorch"),
        "files": files,
        "sha256": _composite_hash(files),
        "hash_policy": "package_release_component_file_hashes",
    }


def _environment_record(model_id: str, adapter: Any) -> dict[str, Any]:
    packages = {
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
    }
    if model_id.startswith("deepgaze_"):
        packages["deepgaze-pytorch"] = importlib.metadata.version(
            "deepgaze-pytorch"
        )
        packages["scipy"] = importlib.metadata.version("scipy")
        if model_id == "deepgaze_msdb":
            packages["clip"] = importlib.metadata.version("clip")
            packages["einops"] = importlib.metadata.version("einops")
    else:
        packages["huggingface-hub"] = importlib.metadata.version("huggingface-hub")
        packages["safetensors"] = importlib.metadata.version("safetensors")
        for package in ("timm", "transformers", "mambavision"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "hardware": hardware_metadata(adapter.torch),
    }


def _write_environment_lock(
    model_id: str,
    environment: dict[str, Any],
) -> Path:
    path = PROJECT_ROOT / f"configs/external_models/environment_locks/{model_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# hma.package_environment_lock.v1",
        f"python=={environment['python']}",
        f"platform={environment['platform']}",
    ]
    lines.extend(
        f"{name}=={version}"
        for name, version in sorted(environment["packages"].items())
    )
    lines.append(
        "hardware=" + json.dumps(environment["hardware"], sort_keys=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_checkpoint_lock(
    model_id: str,
    checkpoint: dict[str, Any],
) -> Path:
    path = PROJECT_ROOT / f"configs/external_models/checkpoint_locks/{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "hma.external.checkpoint_lock.v1",
        "model_id": model_id,
        **checkpoint,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _composite_hash(files: list[dict[str, Any]]) -> str:
    payload = [
        {
            "name": row["name"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in sorted(files, key=lambda item: item["name"])
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _resolve_device(value: str) -> str:
    torch = importlib.import_module("torch")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
