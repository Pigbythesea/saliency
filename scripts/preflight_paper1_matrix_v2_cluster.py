"""Validate cluster prerequisites before Matrix V2 Slurm submission."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.artifacts import validate_external_artifact
from hma.external.registry import load_external_registry
from hma.utils.paths import resolve_path
from scripts.setup_external_model import build_installation_report


MODELS = (
    "deit_small_static",
    "dynamicvit_deit_small_keep_0_7",
    "tome_deit_small_r13",
)
MANIFESTS = {
    "nsd": (
        "data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv",
        "data/raw/nsd_algonauts",
        9841,
    ),
    "salicon": (
        "data/manifests/v2/salicon_static2000_manifest.csv",
        "data/raw/SALICON",
        2000,
    ),
    "cat2000": (
        "data/manifests/v2/cat2000_static2000_manifest.csv",
        "data/raw/CAT2000",
        100,
    ),
    "coco_search18": (
        "data/manifests/v2/coco_search18_static2000_manifest.csv",
        "data/raw/COCO-Search18",
        635,
    ),
}
RUNTIME_PATH_VARIABLES = (
    "HOME",
    "MAMBA_ROOT_PREFIX",
    "CONDA_PKGS_DIRS",
    "PIP_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "MPLCONFIGDIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "IPYTHONDIR",
    "JUPYTER_CONFIG_DIR",
    "UV_CACHE_DIR",
    "PYTHONUSERBASE",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
    "TMP",
    "TEMP",
)


def run_preflight(*, mode: str, verify_all_images: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _record(checks, "linux", platform.system() == "Linux", platform.platform())
    for name, passed, detail in _runtime_environment_checks():
        _record(checks, f"scratch_runtime:{name}", passed, detail)
    _record(
        checks,
        "slurm_client",
        shutil.which("sinfo") is not None and shutil.which("sbatch") is not None,
        "sinfo and sbatch must be available",
    )
    micromamba = (
        PROJECT_ROOT
        / "external/tools/micromamba/2.8.1-0/micromamba"
    )
    _record(checks, "pinned_micromamba", micromamba.is_file(), str(micromamba))
    core_python = (
        PROJECT_ROOT
        / "external/environments/paper1_matrix_v2_core/bin/python"
    )
    _record(checks, "core_environment", core_python.is_file(), str(core_python))

    registry = load_external_registry()
    for model_id in MODELS:
        report = build_installation_report(model_id)
        _record(
            checks,
            f"external_model:{model_id}",
            all(
                report["stages"][stage]
                for stage in (
                    "source_ready",
                    "environment_ready",
                    "checkpoint_ready",
                    "adapter_ready",
                    "smoke_passed",
                )
            ),
            "; ".join(report["diagnostics"]) or "ready",
        )
        environment = registry.workspace_path("environments") / model_id
        _record(
            checks,
            f"external_python:{model_id}",
            (environment / "bin/python").is_file(),
            str(environment / "bin/python"),
        )
        cuda_runtime = _environment_cuda_runtime(environment)
        _record(
            checks,
            f"external_cuda:{model_id}",
            bool(cuda_runtime),
            (
                f"torch.version.cuda={cuda_runtime}"
                if cuda_runtime
                else "PyTorch is CPU-only or the environment cannot import torch"
            ),
        )

    for name, (manifest, root, expected_unique) in MANIFESTS.items():
        result = _validate_image_manifest(
            resolve_path(manifest),
            resolve_path(root),
            expected_unique=expected_unique,
            verify_all_images=verify_all_images,
        )
        _record(checks, f"manifest:{name}", result["passed"], result["detail"])

    required_gib = 20 if mode == "smoke" else 150
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    _record(
        checks,
        "scratch_free_space",
        free_gib >= required_gib,
        f"{free_gib:.1f} GiB free; requires at least {required_gib} GiB",
    )

    if mode == "analysis":
        for model_id in MODELS:
            artifact = resolve_path(
                f"outputs/paper1_matrix_v2/external_artifacts/full/{model_id}"
            )
            try:
                manifest = validate_external_artifact(artifact, verify_hashes=True)
                passed = int(manifest["num_images"]) == 9841
                detail = f"{manifest['num_images']} images"
            except Exception as exc:
                passed = False
                detail = f"{type(exc).__name__}: {exc}"
            _record(checks, f"full_artifact:{model_id}", passed, detail)

    return {
        "schema_version": "hma.matrix_v2.cluster_preflight.v1",
        "mode": mode,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _runtime_environment_checks() -> list[tuple[str, bool, str]]:
    external_root = (PROJECT_ROOT / "external").resolve()
    checks = []
    for name in RUNTIME_PATH_VARIABLES:
        value = os.environ.get(name, "")
        try:
            path = Path(value).expanduser().resolve() if value else None
            passed = path is not None and path.is_relative_to(external_root)
        except (OSError, RuntimeError):
            path = None
            passed = False
        detail = str(path) if path is not None else "unset"
        checks.append((name, passed, detail))
    return checks


def _validate_image_manifest(
    manifest_path: Path,
    image_root: Path,
    *,
    expected_unique: int,
    verify_all_images: bool,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"passed": False, "detail": f"missing {manifest_path}"}
    unique: dict[str, Path] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row["image_id"])
            path = Path(str(row["image_path"])).expanduser()
            if not path.is_absolute():
                path = image_root / path
            unique.setdefault(image_id, path)
    missing = []
    if verify_all_images:
        missing = [str(path) for path in unique.values() if not path.is_file()]
    passed = len(unique) == expected_unique and not missing
    detail = (
        f"{len(unique)} unique images; expected {expected_unique}; "
        f"missing files={len(missing)}"
    )
    return {"passed": passed, "detail": detail}


def _record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _environment_cuda_runtime(environment: Path) -> str:
    python = environment / "bin/python"
    if not python.is_file():
        return ""
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import torch; print(torch.version.cuda or '')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "analysis"], required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--no-verify-all-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(
        mode=args.mode,
        verify_all_images=not args.no_verify_all_images,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output:
        path = resolve_path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
