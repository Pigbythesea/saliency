"""Install and audit a pinned Matrix V2 external model environment."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MICROMAMBA_VERSION = "2.8.1-0"
MICROMAMBA_ASSETS = {
    "x86_64": {
        "url": (
            "https://github.com/mamba-org/micromamba-releases/releases/download/"
            f"{MICROMAMBA_VERSION}/micromamba-linux-64"
        ),
        "sha256": "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82",
    },
    "aarch64": {
        "url": (
            "https://github.com/mamba-org/micromamba-releases/releases/download/"
            f"{MICROMAMBA_VERSION}/micromamba-linux-aarch64"
        ),
        "sha256": "e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a",
    },
}

from hma.external.hashing import sha256_file, sha256_tree


def _scratch_runtime_environment() -> dict[str, str]:
    """Keep package-manager caches and temporary files out of cluster home."""
    external_root = PROJECT_ROOT / "external"
    paths = {
        "MAMBA_ROOT_PREFIX": external_root / "cache" / "micromamba",
        "PIP_CACHE_DIR": external_root / "cache" / "pip",
        "XDG_CACHE_HOME": external_root / "cache" / "xdg",
        "TORCH_HOME": external_root / "cache" / "torch",
        "HF_HOME": external_root / "cache" / "huggingface",
        "TMPDIR": external_root / "tmp",
    }
    environment = dict(os.environ)
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    return environment


def _resolve_micromamba(value: str) -> str:
    explicit = Path(value).expanduser()
    if (explicit.is_absolute() or explicit.parent != Path(".")) and explicit.is_file():
        return str(explicit.resolve())
    discovered = shutil.which(value)
    if discovered:
        return discovered
    local = (
        PROJECT_ROOT
        / "external"
        / "tools"
        / "micromamba"
        / MICROMAMBA_VERSION
        / "micromamba"
    )
    if local.is_file():
        _verify_micromamba(local)
        return str(local)
    if platform.system() != "Linux":
        raise FileNotFoundError(
            f"micromamba executable '{value}' was not found. Automatic bootstrap "
            "is supported only inside Linux/WSL."
        )
    architecture = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    architecture = aliases.get(architecture, architecture)
    asset = MICROMAMBA_ASSETS.get(architecture)
    if asset is None:
        raise RuntimeError(
            "No pinned micromamba asset is configured for architecture "
            f"'{platform.machine()}'"
        )
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(".partial")
    print(
        f"Bootstrapping micromamba {MICROMAMBA_VERSION} to {local}",
        file=sys.stderr,
    )
    try:
        urllib.request.urlretrieve(str(asset["url"]), temporary)
        actual = sha256_file(temporary)
        if actual != asset["sha256"]:
            raise RuntimeError(
                "Downloaded micromamba SHA-256 mismatch: "
                f"expected {asset['sha256']}, found {actual}"
            )
        os.replace(temporary, local)
        local.chmod(
            local.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    _verify_micromamba(local)
    return str(local)


def _verify_micromamba(path: Path) -> None:
    architecture = platform.machine().lower()
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(
        architecture, architecture
    )
    asset = MICROMAMBA_ASSETS.get(architecture)
    if asset is None:
        raise RuntimeError(
            "No pinned micromamba checksum is configured for architecture "
            f"'{platform.machine()}'"
        )
    actual = sha256_file(path)
    if actual != asset["sha256"]:
        raise RuntimeError(
            f"Local micromamba SHA-256 mismatch at {path}: "
            f"expected {asset['sha256']}, found {actual}"
        )


def _ensure_setup_runtime() -> None:
    """Re-exec in a tiny micromamba runtime when bare Python lacks PyYAML."""
    if importlib.util.find_spec("yaml") is not None:
        return
    if os.environ.get("HMA_SETUP_BOOTSTRAPPED") == "1":
        raise RuntimeError(
            "The Matrix V2 setup bootstrap environment was created without PyYAML"
        )
    micromamba_name = "micromamba"
    if "--micromamba" in sys.argv:
        index = sys.argv.index("--micromamba")
        if index + 1 < len(sys.argv):
            micromamba_name = sys.argv[index + 1]
    executable = _resolve_micromamba(micromamba_name)
    bootstrap_dir = PROJECT_ROOT / "external" / "environments" / "hma_setup_bootstrap"
    history = bootstrap_dir / "conda-meta" / "history"
    if not history.is_file():
        bootstrap_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                executable,
                "create",
                "--yes",
                "--prefix",
                str(bootstrap_dir),
                "python=3.10",
                "pyyaml=6.0",
            ],
            env=_scratch_runtime_environment(),
            check=True,
        )
    environment = _scratch_runtime_environment()
    environment["HMA_SETUP_BOOTSTRAPPED"] = "1"
    completed = subprocess.run(
        [
            executable,
            "run",
            "--prefix",
            str(bootstrap_dir),
            "python",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


_ensure_setup_runtime()

from hma.external.registry import load_external_registry
from hma.utils.paths import resolve_path


def build_installation_report(
    model_id: str,
    *,
    registry_path: str | Path = "configs/external_models/registry.yaml",
) -> dict[str, Any]:
    registry = load_external_registry(registry_path)
    model = registry.model(model_id)
    canonical_id = str(model["id"])
    source_dir = registry.workspace_path("sources") / canonical_id
    environment_dir = registry.workspace_path("environments") / canonical_id
    environment_lock = _environment_lock_path(canonical_id)
    checkpoint_path = _checkpoint_path(registry, model)
    checkpoint_lock = _checkpoint_lock_path(canonical_id)
    actual_commit = _git_commit(source_dir)
    expected_commit = str(model["source"]["commit"])
    source_ready = (
        source_dir.is_dir()
        and expected_commit != "PIN_REQUIRED"
        and actual_commit == expected_commit
    )
    environment_ready = (
        (environment_dir / "conda-meta" / "history").is_file()
        and environment_lock.is_file()
    )
    manifest_sha256 = sha256_file(registry.environment_path(canonical_id))
    environment_lock_sha256 = (
        sha256_file(environment_lock) if environment_lock.is_file() else None
    )
    checkpoint_hash = (
        sha256_tree(checkpoint_path)
        if checkpoint_path is not None and checkpoint_path.exists()
        else None
    )
    lock = _read_json(checkpoint_lock)
    checkpoint_ready = bool(
        checkpoint_hash
        and lock
        and lock.get("sha256") == checkpoint_hash
        and lock.get("model_id") == canonical_id
    )
    adapter_ready = (
        _adapter_class_available(str(model["adapter"]))
        and model.get("adapter_status") == "implemented"
    )
    previous = _read_json(_report_path(registry, canonical_id))
    smoke_passed = bool(
        previous
        and previous.get("stages", {}).get("smoke_passed")
        and previous.get("environment", {}).get("manifest_sha256")
        == manifest_sha256
        and previous.get("environment", {}).get("lock_sha256")
        == environment_lock_sha256
    )
    license_name = str(model["source"].get("license", ""))
    license_audit_passed = not any(
        marker in license_name.lower()
        for marker in ("audit_required", "unknown", "unverified")
    )
    evidence_ready = all(
        (
            source_ready,
            environment_ready,
            checkpoint_ready,
            adapter_ready,
            smoke_passed,
            license_audit_passed,
        )
    )
    return {
        "schema_version": "hma.external.installation.v1",
        "model_id": canonical_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "source": str(source_dir),
            "environment": str(environment_dir),
            "environment_lock": str(environment_lock),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_lock": str(checkpoint_lock),
        },
        "source": {
            "repository": model["source"]["repository"],
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "license": license_name,
            "license_audit_passed": license_audit_passed,
        },
        "environment": {
            "manifest": str(registry.environment_path(canonical_id)),
            "manifest_sha256": manifest_sha256,
            "lock_sha256": environment_lock_sha256,
        },
        "checkpoint": {
            "sha256": checkpoint_hash,
            "lock_sha256": lock.get("sha256") if lock else None,
            "hash_policy": model["checkpoint"].get("hash_policy"),
        },
        "stages": {
            "source_ready": source_ready,
            "environment_ready": environment_ready,
            "checkpoint_ready": checkpoint_ready,
            "adapter_ready": adapter_ready,
            "smoke_passed": smoke_passed,
            "evidence_ready": evidence_ready,
        },
        "diagnostics": _diagnostics(
            model=model,
            source_ready=source_ready,
            environment_ready=environment_ready,
            checkpoint_ready=checkpoint_ready,
            adapter_ready=adapter_ready,
            smoke_passed=smoke_passed,
            license_audit_passed=license_audit_passed,
        ),
    }


def install_model(
    model_id: str,
    *,
    registry_path: str | Path,
    download_checkpoint: bool,
    run_smoke: bool,
    micromamba: str,
) -> dict[str, Any]:
    registry = load_external_registry(registry_path)
    model = registry.model(model_id)
    canonical_id = str(model["id"])
    source_dir = registry.workspace_path("sources") / canonical_id
    environment_dir = registry.workspace_path("environments") / canonical_id
    _prepare_source(model, source_dir)
    _prepare_environment(
        model=model,
        source_dir=source_dir,
        environment_dir=environment_dir,
        environment_manifest=registry.environment_path(canonical_id),
        micromamba=micromamba,
    )
    if download_checkpoint:
        _download_checkpoint(
            registry=registry,
            model=model,
            environment_dir=environment_dir,
            micromamba=micromamba,
        )
    report = build_installation_report(canonical_id, registry_path=registry_path)
    if run_smoke:
        if not report["stages"]["checkpoint_ready"]:
            raise RuntimeError(
                "Checkpoint must be downloaded and locked before installation smoke"
            )
        _run_adapter_smoke(
            canonical_id=canonical_id,
            environment_dir=environment_dir,
            micromamba=micromamba,
        )
        report["stages"]["smoke_passed"] = True
        report["stages"]["evidence_ready"] = all(
            report["stages"][stage]
            for stage in (
                "source_ready",
                "environment_ready",
                "checkpoint_ready",
                "adapter_ready",
                "smoke_passed",
            )
        ) and bool(report["source"]["license_audit_passed"])
    _write_report(registry, canonical_id, report)
    return report


def _prepare_source(model: dict[str, Any], source_dir: Path) -> None:
    repository = str(model["source"]["repository"])
    commit = str(model["source"]["commit"])
    if commit == "PIN_REQUIRED":
        raise RuntimeError(
            f"Model '{model['id']}' cannot be installed until its official source commit is pinned"
        )
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        _run(["git", "clone", repository, str(source_dir)])
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"External source path is not a git checkout: {source_dir}")
    _run(["git", "-C", str(source_dir), "fetch", "origin", commit])
    _run(["git", "-C", str(source_dir), "checkout", "--detach", commit])
    actual = _git_commit(source_dir)
    if actual != commit:
        raise RuntimeError(f"Source commit mismatch: expected {commit}, found {actual}")


def _prepare_environment(
    *,
    model: dict[str, Any],
    source_dir: Path,
    environment_dir: Path,
    environment_manifest: Path,
    micromamba: str,
) -> None:
    executable = _resolve_micromamba(micromamba)
    environment_dir.parent.mkdir(parents=True, exist_ok=True)
    environment_lock = _environment_lock_path(str(model["id"]))
    if environment_lock.exists():
        environment_lock.unlink()
    _run(
        [
            executable,
            "create",
            "--yes",
            "--channel-priority",
            "strict",
            "--prefix",
            str(environment_dir),
            "--file",
            str(environment_manifest),
        ]
    )
    _validate_torch_environment(
        environment_dir=environment_dir,
        micromamba=executable,
    )
    for command in _post_install_commands(str(model["id"])):
        _run(
            [
                executable,
                "run",
                "--prefix",
                str(environment_dir),
                "bash",
                "-lc",
                command.format(source=str(source_dir)),
            ]
        )
    _write_environment_lock(
        model_id=str(model["id"]),
        environment_dir=environment_dir,
        micromamba=executable,
    )


def _validate_torch_environment(
    *,
    environment_dir: Path,
    micromamba: str,
) -> None:
    code = (
        "import torch; "
        "assert torch.version.cuda, 'PyTorch was installed without CUDA support'; "
        "print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
    )
    _run(
        [
            micromamba,
            "run",
            "--prefix",
            str(environment_dir),
            "python",
            "-c",
            code,
        ]
    )


def _post_install_commands(model_id: str) -> list[str]:
    if model_id in {"deit_small_static", "tome_deit_small_r13"}:
        return [
            'python -m pip install -e "{source}"',
            'python scripts/apply_external_patches.py --model '
            f'{model_id} --source "{{source}}"',
        ]
    if model_id == "dynamicvit_deit_small_keep_0_7":
        return [
            'python scripts/apply_external_patches.py --model '
            f'{model_id} --source "{{source}}"'
        ]
    if model_id in {"mambavision_t", "dinov3_small_patch16"}:
        return ['python -m pip install -e "{source}"']
    if model_id == "siglip_base_patch16":
        return []
    if model_id == "hat":
        return [
            'python -m pip install -r "{source}/requirements.txt"',
            'python -m pip install "git+https://github.com/facebookresearch/detectron2.git"',
            'cd "{source}/hat/pixel_decoder/ops" && sh make.sh',
        ]
    if model_id == "scandiff":
        return ['python -m pip install -r "{source}/requirements.txt"']
    return []


def _download_checkpoint(
    *,
    registry: Any,
    model: dict[str, Any],
    environment_dir: Path,
    micromamba: str,
) -> None:
    canonical_id = str(model["id"])
    checkpoint = dict(model["checkpoint"])
    target = _checkpoint_path(registry, model)
    if target is None:
        raise RuntimeError(
            f"Model '{canonical_id}' has no released checkpoint URL recorded"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name == "huggingface_snapshot":
        hub_id = model.get("adapter_config", {}).get("checkpoint_id")
        if not hub_id:
            raise RuntimeError(f"Model '{canonical_id}' is missing adapter_config.checkpoint_id")
        executable = _resolve_micromamba(micromamba)
        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={hub_id!r}, local_dir={str(target)!r})"
        )
        _run(
            [
                str(executable),
                "run",
                "--prefix",
                str(environment_dir),
                "python",
                "-c",
                code,
            ]
        )
    else:
        url = checkpoint.get("url")
        if not url:
            raise RuntimeError(f"Model '{canonical_id}' has no checkpoint download URL")
        temporary = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(str(url), temporary)
        os.replace(temporary, target)
    digest = sha256_tree(target)
    lock = {
        "schema_version": "hma.external.checkpoint_lock.v1",
        "model_id": canonical_id,
        "source": checkpoint.get("url"),
        "sha256": digest,
        "path_kind": "directory" if target.is_dir() else "file",
    }
    lock_path = _checkpoint_lock_path(canonical_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")


def _run_adapter_smoke(
    *,
    canonical_id: str,
    environment_dir: Path,
    micromamba: str,
) -> None:
    executable = _resolve_micromamba(micromamba)
    _run(
        [
            str(executable),
            "run",
            "--prefix",
            str(environment_dir),
            "python",
            "scripts/run_external_model.py",
            "--model",
            canonical_id,
            "--smoke-only",
            "--device",
            "cpu",
        ]
    )


def _checkpoint_path(registry: Any, model: dict[str, Any]) -> Path | None:
    filename = model.get("checkpoint", {}).get("filename")
    if not filename:
        return None
    return registry.workspace_path("checkpoints") / str(model["id"]) / str(filename)


def _adapter_class_available(class_path: str) -> bool:
    module_name, separator, class_name = class_path.partition(":")
    if not separator or not module_name or not class_name:
        return False
    if module_name == "hma.external.adapters":
        adapter_path = SRC_ROOT / "hma" / "external" / "adapters.py"
        tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
        return any(
            isinstance(node, ast.ClassDef) and node.name == class_name
            for node in tree.body
        )
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    return spec is not None


def _checkpoint_lock_path(model_id: str) -> Path:
    return resolve_path(f"configs/external_models/checkpoint_locks/{model_id}.json")


def _environment_lock_path(model_id: str) -> Path:
    return resolve_path(f"configs/external_models/environment_locks/{model_id}.txt")


def _write_environment_lock(
    *,
    model_id: str,
    environment_dir: Path,
    micromamba: str,
) -> Path:
    result = subprocess.run(
        [
            micromamba,
            "list",
            "--prefix",
            str(environment_dir),
            "--explicit",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_scratch_runtime_environment(),
    )
    pip_result = subprocess.run(
        [
            micromamba,
            "run",
            "--prefix",
            str(environment_dir),
            "python",
            "-m",
            "pip",
            "freeze",
            "--all",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_scratch_runtime_environment(),
    )
    path = _environment_lock_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        result.stdout.rstrip()
        + "\n\n# pip-freeze\n"
        + pip_result.stdout,
        encoding="utf-8",
    )
    return path


def _report_path(registry: Any, model_id: str) -> Path:
    return registry.workspace_path("reports") / f"{model_id}.json"


def _write_report(registry: Any, model_id: str, report: dict[str, Any]) -> Path:
    path = _report_path(registry, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _git_commit(source_dir: Path) -> str | None:
    if not (source_dir / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _diagnostics(**stages: Any) -> list[str]:
    messages = []
    for name, ready in stages.items():
        if not ready:
            messages.append(f"{name}: integration work remains")
    return messages


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        env=_scratch_runtime_environment(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--registry", default="configs/external_models/registry.yaml"
    )
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--micromamba", default="micromamba")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.install:
            report = install_model(
                args.model,
                registry_path=args.registry,
                download_checkpoint=bool(args.download_checkpoint),
                run_smoke=bool(args.smoke),
                micromamba=args.micromamba,
            )
        else:
            report = build_installation_report(args.model, registry_path=args.registry)
    except Exception as exc:
        registry = load_external_registry(args.registry)
        canonical_id = registry.resolve_model_id(args.model)
        report = build_installation_report(
            canonical_id,
            registry_path=args.registry,
        )
        report["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["diagnostics"].append(
            f"last_error: {type(exc).__name__}: {exc}"
        )
        _write_report(registry, canonical_id, report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    else:
        registry = load_external_registry(args.registry)
        _write_report(registry, str(report["model_id"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stages"]["evidence_ready"] or args.report_only else 2


if __name__ == "__main__":
    sys.exit(main())
