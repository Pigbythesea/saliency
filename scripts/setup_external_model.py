"""Install and audit a pinned Matrix V2 external model environment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
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

SYMLINK_SENSITIVE_ENV_MODELS = {"hat", "scandiff"}

from hma.external.hashing import sha256_file, sha256_tree


def _project_is_on_wsl_windows_mount() -> bool:
    parts = PROJECT_ROOT.resolve().parts
    return platform.system() == "Linux" and len(parts) >= 3 and parts[1] == "mnt"


def _workspace_key() -> str:
    return hashlib.sha256(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]


def _scratch_runtime_environment() -> dict[str, str]:
    """Keep all setup subprocess state out of the cluster home directory."""
    external_root = _runtime_state_root()
    paths = {
        "HOME": external_root / "runtime_home",
        "MAMBA_ROOT_PREFIX": external_root / "cache" / "micromamba",
        "CONDA_PKGS_DIRS": external_root / "cache" / "conda" / "pkgs",
        "PIP_CACHE_DIR": external_root / "cache" / "pip",
        "XDG_CACHE_HOME": external_root / "cache" / "xdg",
        "TORCH_HOME": external_root / "cache" / "torch",
        "HF_HOME": external_root / "cache" / "huggingface",
        "HF_HUB_CACHE": external_root / "cache" / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": external_root / "cache" / "huggingface" / "transformers",
        "MPLCONFIGDIR": external_root / "cache" / "matplotlib",
        "CUDA_CACHE_PATH": external_root / "cache" / "cuda",
        "TRITON_CACHE_DIR": external_root / "cache" / "triton",
        "NUMBA_CACHE_DIR": external_root / "cache" / "numba",
        "IPYTHONDIR": external_root / "cache" / "ipython",
        "JUPYTER_CONFIG_DIR": external_root / "cache" / "jupyter",
        "UV_CACHE_DIR": external_root / "cache" / "uv",
        "PYTHONUSERBASE": external_root / "python_user",
        "PYTHONPYCACHEPREFIX": external_root / "cache" / "pycache",
        "TMPDIR": external_root / "tmp",
        "TMP": external_root / "tmp",
        "TEMP": external_root / "tmp",
    }
    environment = dict(os.environ)
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("CONDA_REMOTE_CONNECT_TIMEOUT_SECS", "60")
    environment.setdefault("CONDA_REMOTE_READ_TIMEOUT_SECS", "600")
    environment.setdefault("CONDA_REMOTE_MAX_RETRIES", "6")
    environment.setdefault("CONDA_REMOTE_BACKOFF_FACTOR", "2")
    environment.setdefault("MAMBA_REMOTE_CONNECT_TIMEOUT_SECS", "60")
    environment.setdefault("MAMBA_REMOTE_READ_TIMEOUT_SECS", "600")
    environment.setdefault("MAMBA_REMOTE_MAX_RETRIES", "6")
    environment.setdefault("MAMBA_REMOTE_BACKOFF_FACTOR", "2")
    environment.setdefault("MAMBA_NO_LOW_SPEED_LIMIT", "1")
    environment.setdefault("MAMBA_DOWNLOAD_TIMEOUT_SECONDS", "900")
    return environment


def _runtime_state_root() -> Path:
    if _project_is_on_wsl_windows_mount():
        return Path.home() / ".cache" / "hma_external" / "runtime" / _workspace_key()
    return PROJECT_ROOT / "external"


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
    bootstrap_dir = _runtime_state_root() / "environments" / "hma_setup_bootstrap"
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
    environment_dir = _model_environment_dir(registry, canonical_id)
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
    environment_dir = _model_environment_dir(registry, canonical_id)
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
        report["diagnostics"] = _diagnostics(
            model=model,
            source_ready=report["stages"]["source_ready"],
            environment_ready=report["stages"]["environment_ready"],
            checkpoint_ready=report["stages"]["checkpoint_ready"],
            adapter_ready=report["stages"]["adapter_ready"],
            smoke_passed=report["stages"]["smoke_passed"],
            license_audit_passed=report["source"]["license_audit_passed"],
        )
    _write_report(registry, canonical_id, report)
    return report


def _model_environment_dir(registry: Any, model_id: str) -> Path:
    default = registry.workspace_path("environments") / model_id
    if (
        model_id not in SYMLINK_SENSITIVE_ENV_MODELS
        or not _project_is_on_wsl_windows_mount()
    ):
        return default
    root_override = os.environ.get("HMA_EXTERNAL_ENV_ROOT")
    if root_override:
        root = Path(root_override).expanduser()
    else:
        root = Path.home() / ".cache" / "hma_external" / "environments" / _workspace_key()
    return root / model_id


def _prepare_source(model: dict[str, Any], source_dir: Path) -> None:
    model_id = str(model["id"])
    repository = str(model["source"]["repository"])
    commit = str(model["source"]["commit"])
    if commit == "PIN_REQUIRED":
        raise RuntimeError(
            f"Model '{model['id']}' cannot be installed until its official source commit is pinned"
        )
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        _run_source_command(model_id, ["git", "clone", repository, str(source_dir)])
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"External source path is not a git checkout: {source_dir}")
    actual = _git_commit(source_dir)
    if actual == commit:
        return
    _run_source_command(model_id, ["git", "-C", str(source_dir), "fetch", "origin", commit])
    _run_source_command(model_id, ["git", "-C", str(source_dir), "checkout", "--detach", commit])
    _run_source_command(model_id, ["git", "-C", str(source_dir), "reset", "--hard", commit])
    _run_source_command(model_id, ["git", "-C", str(source_dir), "clean", "-fdx"])
    actual = _git_commit(source_dir)
    if actual != commit:
        raise RuntimeError(f"Source commit mismatch: expected {commit}, found {actual}")


def _run_source_command(model_id: str, command: list[str]) -> None:
    log_dir = resolve_path("outputs/paper1_publication_v0/preflight/setup_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_id}_source_latest.log"
    with log_path.open("ab") as log_file:
        log_file.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
        completed = subprocess.run(
            command,
            check=False,
            cwd=PROJECT_ROOT,
            env=_scratch_runtime_environment(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.write(f"EXIT_CODE={completed.returncode}\n\n".encode("utf-8"))
    if completed.returncode != 0:
        raise RuntimeError(
            f"{model_id} source command failed; see {log_path.relative_to(PROJECT_ROOT)}"
        )


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
    if (environment_dir / "conda-meta" / "history").is_file():
        try:
            _repair_existing_environment(
                model_id=str(model["id"]),
                environment_dir=environment_dir,
                micromamba=executable,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "Existing environment repair failed; keeping "
                f"{environment_dir}: {exc}",
                file=sys.stderr,
            )
            raise
        try:
            _validate_model_environment(
                model_id=str(model["id"]),
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
            return
        except subprocess.CalledProcessError as exc:
            print(
                "Existing environment failed validation; rebuilding "
                f"{environment_dir}: {exc}",
                file=sys.stderr,
            )
            _remove_stale_environment(environment_dir)
            if environment_lock.exists():
                environment_lock.unlink()
    _remove_stale_environment(environment_dir)
    if environment_lock.exists():
        environment_lock.unlink()
    _run_environment_create(
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
        ],
        environment_dir=environment_dir,
    )
    _validate_model_environment(
        model_id=str(model["id"]),
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


def _remove_stale_environment(environment_dir: Path) -> None:
    target = environment_dir.resolve()
    allowed_roots = {
        (PROJECT_ROOT / "external" / "environments").resolve(),
        _native_environment_root().resolve(),
    }
    if target.parent not in allowed_roots:
        raise RuntimeError(
            "Refusing to remove external environment outside approved roots "
            f"{sorted(str(path) for path in allowed_roots)}: {target}"
        )
    if target.exists():
        shutil.rmtree(target)


def _native_environment_root() -> Path:
    root_override = os.environ.get("HMA_EXTERNAL_ENV_ROOT")
    if root_override:
        return Path(root_override).expanduser()
    return Path.home() / ".cache" / "hma_external" / "environments" / _workspace_key()


def _repair_existing_environment(
    *,
    model_id: str,
    environment_dir: Path,
    micromamba: str,
) -> None:
    if model_id == "mambavision_t":
        if not _mambavision_build_toolchain_available(environment_dir):
            _run(
                [
                    micromamba,
                    "install",
                    "--yes",
                    "--prefix",
                    str(environment_dir),
                    "conda-forge::gcc_linux-64=12",
                    "conda-forge::gxx_linux-64=12",
                    "conda-forge::ninja",
                    "conda-forge::packaging",
                    "nvidia::cuda-nvcc=12.6",
                    "nvidia::cuda-cudart-dev=12.6",
                    "nvidia::cuda-cccl",
                ]
            )
        return
    if model_id != "hat":
        return
    if not _hat_cuda_dev_headers_available(environment_dir):
        _run(
            [
                micromamba,
                "install",
                "--yes",
                "--prefix",
                str(environment_dir),
                "nvidia::cuda-cudart-dev=11.7.99",
                "nvidia::cuda-libraries-dev=11.7.*",
            ]
        )
    if not _hat_pillow_legacy_constants_available(
        environment_dir=environment_dir,
        micromamba=micromamba,
    ):
        _run(
            [
                micromamba,
                "install",
                "--yes",
                "--prefix",
                str(environment_dir),
                "pillow<10",
            ]
        )


def _hat_cuda_dev_headers_available(environment_dir: Path) -> bool:
    include_roots = [
        environment_dir / "include",
        environment_dir / "targets" / "x86_64-linux" / "include",
    ]
    required = (
        "cuda_runtime.h",
        "cublas_v2.h",
        "cublasLt.h",
        "cufft.h",
        "curand.h",
        "cusolverDn.h",
        "cusparse.h",
    )
    return all(
        any((root / header).is_file() for root in include_roots)
        for header in required
    )


def _hat_pillow_legacy_constants_available(
    *,
    environment_dir: Path,
    micromamba: str,
) -> bool:
    completed = subprocess.run(
        [
            micromamba,
            "run",
            "--prefix",
            str(environment_dir),
            "python",
            "-c",
            "from PIL import Image; raise SystemExit(0 if hasattr(Image, 'LINEAR') else 1)",
        ],
        check=False,
        env=_scratch_runtime_environment(),
    )
    return completed.returncode == 0


def _mambavision_build_toolchain_available(environment_dir: Path) -> bool:
    include_roots = [
        environment_dir / "include",
        environment_dir / "targets" / "x86_64-linux" / "include",
    ]
    required_headers = ("cuda_runtime.h", "cub/cub.cuh", "thrust/version.h")
    required_tools = (
        "bin/nvcc",
        "bin/x86_64-conda-linux-gnu-gcc",
        "bin/x86_64-conda-linux-gnu-c++",
    )
    return all((environment_dir / relative).exists() for relative in required_tools) and all(
        any((root / header).is_file() for root in include_roots)
        for header in required_headers
    )


def _validate_model_environment(
    *,
    model_id: str,
    environment_dir: Path,
    micromamba: str,
) -> None:
    _validate_torch_environment(
        environment_dir=environment_dir,
        micromamba=micromamba,
    )
    if model_id == "hat":
        for command in (
            'test -x "${CONDA_PREFIX}/bin/nvcc"',
            (
                'for header in cuda_runtime.h cublas_v2.h cublasLt.h cufft.h '
                'curand.h cusolverDn.h cusparse.h; do '
                'test -f "${CONDA_PREFIX}/include/${header}" || '
                'test -f "${CONDA_PREFIX}/targets/x86_64-linux/include/${header}" || '
                '(echo "${header} missing; install nvidia::cuda-libraries-dev=11.7.*" >&2; exit 1); '
                'done'
            ),
            (
                'compiler="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"; '
                'test -x "$compiler"; '
                'version="$("$compiler" -dumpfullversion -dumpversion)"; '
                'major="${version%%.*}"; '
                'test -n "$major"; '
                'test "$major" -le 11'
            ),
            (
                'python -c "import importlib.metadata as md; '
                "mm=lambda name: tuple(map(int, md.version(name).split('.')[:2])); "
                "assert mm('pip') <= (22, 3), md.version('pip'); "
                "assert mm('setuptools') <= (65, 5), md.version('setuptools'); "
                "assert mm('wheel') <= (0, 37), md.version('wheel')"
                '"'
            ),
        ):
            _run(
                [
                    micromamba,
                    "run",
                    "--prefix",
                    str(environment_dir),
                    "bash",
                    "-lc",
                    command,
                ]
            )
    if model_id == "mambavision_t":
        command = (
            "import torch; "
            "version = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])); "
            "assert version >= (2, 6), torch.__version__; "
            "assert torch.version.cuda == '12.6', torch.version.cuda"
        )
        _run(
            [
                micromamba,
                "run",
                "--prefix",
                str(environment_dir),
                "python",
                "-c",
                command,
            ]
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
    if model_id == "mambavision_t":
        return [
            (
                'python -c "from mamba_ssm.ops.selective_scan_interface '
                'import selective_scan_fn" || '
                '(set -o pipefail; '
                'mkdir -p outputs/paper1_publication_v0/preflight/setup_logs; '
                'log="outputs/paper1_publication_v0/preflight/setup_logs/'
                'mambavision_mamba_ssm_build_latest.log"; '
                'wheel="external/checkpoints/mambavision_t/dependencies/'
                'mamba_ssm-2.2.4+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"; '
                'cuda_includes="${{CONDA_PREFIX}}/include"; '
                'if [ -d "${{CONDA_PREFIX}}/targets/x86_64-linux/include" ]; then '
                'cuda_includes="${{CONDA_PREFIX}}/targets/x86_64-linux/include:${{cuda_includes}}"; '
                'fi; '
                'export CUDA_HOME="${{CONDA_PREFIX}}"; '
                'export CC="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-gcc"; '
                'export CXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++"; '
                'export CUDAHOSTCXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++"; '
                'export MAMBA_FORCE_BUILD=TRUE; '
                'export MAX_JOBS="${{MAX_JOBS:-1}}"; '
                'export CPATH="${{cuda_includes}}:${{CPATH:-}}"; '
                'export C_INCLUDE_PATH="${{cuda_includes}}:${{C_INCLUDE_PATH:-}}"; '
                'export CPLUS_INCLUDE_PATH="${{cuda_includes}}:${{CPLUS_INCLUDE_PATH:-}}"; '
                'export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-8.9}}"; '
                'if [ -f "$wheel" ]; then '
                'python -m pip install -vv --no-cache-dir "$wheel"; '
                'else '
                'python -m pip install -vv --no-build-isolation --no-cache-dir '
                '"https://github.com/state-spaces/mamba/archive/refs/tags/v2.2.4.tar.gz"; '
                'fi '
                '2>&1 | tee "$log")'
            ),
            'python -m pip install -e "{source}"',
        ]
    if model_id == "dinov3_small_patch16":
        return ['python -m pip install -e "{source}"']
    if model_id == "siglip_base_patch16":
        return []
    if model_id == "hat":
        return [
            "python -m pip install "
            '"numpy<2" '
            '"opencv-python==4.7.0.68" '
            '"scipy==1.10.0" '
            '"cython==0.29.32" '
            '"tqdm==4.64.1" '
            '"yacs==0.1.8" '
            '"fvcore==0.1.5.post20221221" '
            '"iopath==0.1.10" '
            '"pycocotools==2.0.6" '
            '"timm==0.6.12" '
            '"scikit-image==0.19.3" '
            '"tensorboard==2.11.0" '
            '"submitit==1.4.5" '
            '"tabulate==0.9.0" '
            '"protobuf==3.20.3" '
            '"fairscale==0.4.13"',
            (
                'python -c "import detectron2" || '
                '(set -o pipefail; mkdir -p outputs/paper1_publication_v0/preflight/setup_logs; '
                'log="outputs/paper1_publication_v0/preflight/setup_logs/hat_detectron2_build_latest.log"; '
                'cuda_includes="${{CONDA_PREFIX}}/include"; '
                'if [ -d "${{CONDA_PREFIX}}/targets/x86_64-linux/include" ]; then '
                'cuda_includes="${{CONDA_PREFIX}}/targets/x86_64-linux/include:${{cuda_includes}}"; '
                'fi; '
                'CC="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-gcc" '
                'CXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++" '
                'CUDAHOSTCXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++" '
                'CUDA_HOME="${{CONDA_PREFIX}}" FORCE_CUDA=1 '
                'SETUPTOOLS_USE_DISTUTILS=stdlib '
                'CPATH="${{cuda_includes}}:${{CPATH:-}}" '
                'C_INCLUDE_PATH="${{cuda_includes}}:${{C_INCLUDE_PATH:-}}" '
                'CPLUS_INCLUDE_PATH="${{cuda_includes}}:${{CPLUS_INCLUDE_PATH:-}}" '
                'TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}}" '
                'MAX_JOBS="${{MAX_JOBS:-2}}" '
                'python -m pip install -vv --no-build-isolation '
                '"git+https://github.com/facebookresearch/detectron2.git@v0.6" '
                '2>&1 | tee "$log")'
            ),
            (
                'python -c "import MultiScaleDeformableAttention" || '
                '(set -o pipefail; mkdir -p outputs/paper1_publication_v0/preflight/setup_logs; '
                'log="outputs/paper1_publication_v0/preflight/setup_logs/hat_msdeformattn_build_latest.log"; '
                'cd "{source}/hat/pixel_decoder/ops" && '
                'cuda_includes="${{CONDA_PREFIX}}/include"; '
                'if [ -d "${{CONDA_PREFIX}}/targets/x86_64-linux/include" ]; then '
                'cuda_includes="${{CONDA_PREFIX}}/targets/x86_64-linux/include:${{cuda_includes}}"; '
                'fi; '
                'CC="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-gcc" '
                'CXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++" '
                'CUDAHOSTCXX="${{CONDA_PREFIX}}/bin/x86_64-conda-linux-gnu-c++" '
                'CUDA_HOME="${{CONDA_PREFIX}}" FORCE_CUDA=1 '
                'SETUPTOOLS_USE_DISTUTILS=stdlib '
                'CPATH="${{cuda_includes}}:${{CPATH:-}}" '
                'C_INCLUDE_PATH="${{cuda_includes}}:${{C_INCLUDE_PATH:-}}" '
                'CPLUS_INCLUDE_PATH="${{cuda_includes}}:${{CPLUS_INCLUDE_PATH:-}}" '
                'TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}}" '
                'MAX_JOBS="${{MAX_JOBS:-2}}" sh make.sh 2>&1 | tee "$OLDPWD/$log")'
            ),
        ]
    if model_id == "scandiff":
        return [
            (
                'cd "{source}" && python -c "import hydra, omegaconf, timm; '
                'from src.model.components.dit_model import DiTModel"'
            )
        ]
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
    lock_path = _checkpoint_lock_path(canonical_id)
    lock = _read_json(lock_path)
    if target.exists() and lock:
        digest = sha256_tree(target)
        if lock.get("model_id") == canonical_id and lock.get("sha256") == digest:
            return
    if target.name == "huggingface_snapshot":
        hub_id = model.get("adapter_config", {}).get("checkpoint_id")
        if not hub_id:
            raise RuntimeError(f"Model '{canonical_id}' is missing adapter_config.checkpoint_id")
        expected_files = [str(value) for value in checkpoint.get("expected_files", [])]
        missing_files = [
            relative for relative in expected_files if not (target / relative).is_file()
        ]
        if target.is_dir() and expected_files and not missing_files:
            digest = sha256_tree(target)
            lock = {
                "schema_version": "hma.external.checkpoint_lock.v1",
                "model_id": canonical_id,
                "source": checkpoint.get("url"),
                "sha256": digest,
                "path_kind": "directory",
            }
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps(lock, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return
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
    elif canonical_id == "scandiff" and _scandiff_checkpoint_files_ready(target):
        pass
    else:
        url = checkpoint.get("url")
        if not url:
            raise RuntimeError(f"Model '{canonical_id}' has no checkpoint download URL")
        additional_urls = [str(value) for value in checkpoint.get("additional_urls", [])]
        if additional_urls:
            target.mkdir(parents=True, exist_ok=True)
            for asset_url in [str(url), *additional_urls]:
                _download_checkpoint_asset(asset_url, target)
        elif str(url).lower().endswith(".zip") and target.suffix.lower() != ".zip":
            target.mkdir(parents=True, exist_ok=True)
            _download_checkpoint_asset(str(url), target)
        else:
            _download_single_file(str(url), target)
    digest = sha256_tree(target)
    lock = {
        "schema_version": "hma.external.checkpoint_lock.v1",
        "model_id": canonical_id,
        "source": checkpoint.get("url"),
        "sha256": digest,
        "path_kind": "directory" if target.is_dir() else "file",
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")


def _scandiff_checkpoint_files_ready(target: Path) -> bool:
    required = (
        "scandiff_freeview.pth",
        "scandiff_visualsearch.pth",
        "task_embeddings.npy",
    )
    return target.is_dir() and all((target / name).is_file() for name in required)


def _download_checkpoint_asset(url: str, target_dir: Path) -> None:
    filename = _download_filename(url)
    downloaded = target_dir / filename
    _download_single_file(url, downloaded)
    if downloaded.suffix.lower() == ".zip":
        extract_dir = target_dir / downloaded.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded) as archive:
            archive.extractall(extract_dir)


def _download_single_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = _google_drive_download_url(url)
    temporary = target.with_name(f"{target.name}.partial")
    if target.is_file() and target.stat().st_size > 0:
        print(
            f"Checkpoint asset already present: {target} "
            f"({target.stat().st_size:,} bytes)",
            file=sys.stderr,
        )
        return
    attempts = int(os.environ.get("HMA_DOWNLOAD_ATTEMPTS", "6"))
    attempts = max(1, attempts)
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            _download_single_file_attempt(
                url=url,
                resolved=resolved,
                target=target,
                temporary=temporary,
                attempt=attempt,
                attempts=attempts,
            )
            return
        except KeyboardInterrupt:
            print(
                f"Interrupted checkpoint download; kept partial file at {temporary}",
                file=sys.stderr,
            )
            raise
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            print(
                f"Checkpoint download attempt {attempt}/{attempts} failed for "
                f"{target.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt == attempts:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(
        f"checkpoint download failed after {attempts} attempts; "
        f"partial file kept at {temporary}"
    ) from last_error


def _download_single_file_attempt(
    *,
    url: str,
    resolved: str,
    target: Path,
    temporary: Path,
    attempt: int,
    attempts: int,
) -> None:
    read_timeout = float(os.environ.get("HMA_DOWNLOAD_READ_TIMEOUT_SECS", "90"))
    progress_interval = float(os.environ.get("HMA_DOWNLOAD_PROGRESS_SECS", "15"))
    chunk_size = int(os.environ.get("HMA_DOWNLOAD_CHUNK_BYTES", str(1024 * 1024)))
    resume_from = temporary.stat().st_size if temporary.is_file() else 0
    request = urllib.request.Request(resolved)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    print(
        f"Downloading checkpoint asset {target.name} "
        f"(attempt {attempt}/{attempts}, resume={resume_from:,} bytes)",
        file=sys.stderr,
    )
    with urllib.request.urlopen(request, timeout=read_timeout) as response:
        status = getattr(response, "status", None)
        if resume_from and status != 206:
            print(
                f"Server did not honor resume for {target.name}; restarting file",
                file=sys.stderr,
            )
            resume_from = 0
        content_length = response.headers.get("Content-Length")
        total = int(content_length) + resume_from if content_length else None
        mode = "ab" if resume_from else "wb"
        downloaded = resume_from
        last_progress = time.monotonic()
        with temporary.open(mode) as handle:
            if not resume_from:
                first = response.read(512)
                content_type = response.headers.get("Content-Type", "")
                if (
                    "text/html" in content_type.lower()
                    or first.lstrip().lower().startswith(b"<!doctype html")
                    or first.lstrip().lower().startswith(b"<html")
                ):
                    raise RuntimeError(
                        "download returned an HTML page instead of checkpoint bytes; "
                        f"url={url}"
                    )
                handle.write(first)
                downloaded += len(first)
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_progress >= progress_interval:
                    total_text = f"{total:,}" if total else "unknown"
                    print(
                        f"Downloading {target.name}: {downloaded:,}/{total_text} bytes",
                        file=sys.stderr,
                    )
                    last_progress = now
    os.replace(temporary, target)
    print(
        f"Finished checkpoint asset {target.name}: {target.stat().st_size:,} bytes",
        file=sys.stderr,
    )


def _google_drive_download_url(url: str) -> str:
    marker = "drive.google.com/file/d/"
    if marker not in url:
        return url
    file_id = url.split(marker, 1)[1].split("/", 1)[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _download_filename(url: str) -> str:
    if "drive.google.com/file/d/" in url:
        return "google_drive_checkpoint"
    parsed = url.split("?", 1)[0].rstrip("/")
    name = Path(parsed).name
    if not name:
        raise RuntimeError(f"Cannot infer checkpoint filename from URL: {url}")
    return name


def _run_adapter_smoke(
    *,
    canonical_id: str,
    environment_dir: Path,
    micromamba: str,
) -> None:
    executable = _resolve_micromamba(micromamba)
    device = _adapter_smoke_device(
        canonical_id=canonical_id,
        environment_dir=environment_dir,
        micromamba=str(executable),
    )
    command = [
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
        device,
    ]
    log_path = (
        PROJECT_ROOT
        / "outputs"
        / "paper1_publication_v0"
        / "preflight"
        / "setup_logs"
        / f"{canonical_id}_smoke_latest.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=_scratch_runtime_environment(),
    )
    log_path.write_text(
        "$ "
        + " ".join(command)
        + f"\nEXIT_CODE={completed.returncode}\n\n"
        + "# stdout\n"
        + completed.stdout
        + "\n# stderr\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{canonical_id} smoke failed; see {log_path.relative_to(PROJECT_ROOT)}"
        )


def _adapter_smoke_device(
    *,
    canonical_id: str,
    environment_dir: Path,
    micromamba: str,
) -> str:
    if canonical_id not in {"mambavision_t", "scandiff"}:
        return "cpu"
    completed = subprocess.run(
        [
            micromamba,
            "run",
            "--prefix",
            str(environment_dir),
            "python",
            "-c",
            "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=_scratch_runtime_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not inspect CUDA availability for {canonical_id} smoke: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    device = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            f"{canonical_id} requires CUDA for smoke in this environment"
        )
    return device


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


def _run_environment_create(command: list[str], *, environment_dir: Path) -> None:
    attempts = int(os.environ.get("HMA_CONDA_CREATE_ATTEMPTS", "4"))
    attempts = max(1, attempts)
    last_error: subprocess.CalledProcessError | None = None
    log_dir = resolve_path("outputs/paper1_publication_v0/preflight/setup_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{environment_dir.name}_environment_create_latest.log"
    log_path.write_text("", encoding="utf-8")
    print(f"Writing environment create log to {log_path}", file=sys.stderr)
    for attempt in range(1, attempts + 1):
        with log_path.open("ab") as log_file:
            log_file.write(
                (
                    f"\n=== micromamba create attempt {attempt}/{attempts} ===\n"
                    f"{' '.join(command)}\n"
                ).encode("utf-8")
            )
            completed = subprocess.run(
                command,
                env=_scratch_runtime_environment(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            return
        last_error = subprocess.CalledProcessError(completed.returncode, command)
        if attempt >= attempts:
            break
        print(
            "External environment create failed; removing partial prefix "
            f"and retrying {attempt + 1}/{attempts}: {environment_dir}. "
            f"See {log_path}",
            file=sys.stderr,
        )
        _remove_stale_environment(environment_dir)
        time.sleep(min(30, 5 * attempt))
    if last_error is not None:
        raise last_error


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
