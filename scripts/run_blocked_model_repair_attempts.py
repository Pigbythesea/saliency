"""Run concrete blocked-model setup/certification attempts and record logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.registry import load_external_registry
from hma.utils.config import load_yaml

PREFLIGHT_ROOT = PROJECT_ROOT / "outputs" / "paper1_publication_v0" / "preflight"
LOG_ROOT = PREFLIGHT_ROOT / "setup_logs"
RECORDS_PATH = PREFLIGHT_ROOT / "setup_attempt_records.jsonl"

LOCAL_PYTHON = Path(".venv") / "Scripts" / "python.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "siglip_base_patch16",
            "mambavision_t",
            "dinov3_small_patch16",
            "hat",
            "scandiff",
            "adaptivenn_deit_small",
            "semba",
            "semba_fast",
        ],
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    runtime = load_external_registry("configs/external_models/registry.yaml")
    publication = load_yaml(PROJECT_ROOT / "configs/external_models/publication_registry.yaml")
    existing = _load_existing_records()
    results = []
    for model_id in args.models:
        record = run_attempt(
            model_id,
            runtime=runtime,
            publication=publication,
            timeout_seconds=args.timeout_seconds,
        )
        existing[str(record["runtime_model_id"])] = record
        results.append(record)
        print(json.dumps(record, indent=2, sort_keys=True))
    _write_records(existing)
    return 0 if all(int(row["exit_code"]) == 0 for row in results) else 2


def run_attempt(
    model_id: str,
    *,
    runtime: Any,
    publication: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    canonical_id = runtime.resolve_model_id(model_id)
    model = runtime.model(canonical_id)
    publication_ids = _publication_ids_for_runtime(publication, canonical_id)
    commands = _commands_for(canonical_id)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_ROOT / f"{_safe_name(canonical_id)}_{timestamp}.log"
    exit_code = 0
    timed_out = False
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for command in commands:
            log.write(f"$ {command['cmd']}\n")
            log.flush()
            try:
                completed = subprocess.run(
                    command["argv"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    env=_attempt_environment(),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = 124
                if exc.stdout:
                    log.write(str(exc.stdout))
                if exc.stderr:
                    log.write(str(exc.stderr))
                log.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
                break
            if completed.stdout:
                log.write(completed.stdout)
            if completed.stderr:
                log.write(completed.stderr)
            exit_code = int(completed.returncode)
            log.write(f"\nEXIT_CODE={exit_code}\n")
            log.flush()
            if exit_code != 0:
                break
    text = log_path.read_text(encoding="utf-8", errors="replace")
    missing, owner = _classify_failure(canonical_id, text, exit_code, timed_out)
    next_command = commands[-1]["cmd"] if commands else ""
    return {
        "schema_version": "hma.external.setup_attempt.v1",
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "model_id": canonical_id,
        "runtime_model_id": canonical_id,
        "publication_model_ids": publication_ids,
        "source": model["source"]["repository"],
        "source_revision": model["source"]["commit"],
        "source_license_or_access": model["source"].get("license", ""),
        "checkpoint_url": model["checkpoint"].get("url", ""),
        "checkpoint_expected_path": str(_checkpoint_path(runtime, model)),
        "command_cmd": " && ".join(command["cmd"] for command in commands),
        "exit_code": exit_code,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "error_summary": _tail_summary(text),
        "missing_external_requirement": "none" if exit_code == 0 else missing,
        "next_step_owner": "none" if exit_code == 0 else owner,
        "next_command_cmd": next_command,
    }


def _commands_for(model_id: str) -> list[dict[str, Any]]:
    python = str(LOCAL_PYTHON)
    if model_id == "siglip_base_patch16":
        return [
            _command(
                [python, "-m", "pip", "install", "transformers==4.56.0"],
                ".\\.venv\\Scripts\\python.exe -m pip install transformers==4.56.0",
            ),
            _command(
                [
                    python,
                    "scripts\\certify_local_publication_models.py",
                    "--models",
                    model_id,
                    "--device",
                    "cpu",
                ],
                ".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models siglip_base_patch16 --device cpu",
            ),
        ]
    if model_id == "mambavision_t":
        return [
            _command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "transformers==4.56.0",
                    "timm==1.0.15",
                    "mamba-ssm==2.2.4",
                ],
                ".\\.venv\\Scripts\\python.exe -m pip install transformers==4.56.0 timm==1.0.15 mamba-ssm==2.2.4",
            ),
            _command(
                [
                    python,
                    "scripts\\certify_local_publication_models.py",
                    "--models",
                    model_id,
                    "--device",
                    "auto",
                ],
                ".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models mambavision_t --device auto",
            ),
        ]
    if model_id == "dinov3_small_patch16":
        probe = (
            "from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id='facebook/dinov3-vits16-pretrain-lvd1689m', "
            "revision='114c1379950215c8b35dfcd4e90a5c251dde0d32', "
            "local_dir='external/checkpoints/dinov3_small_patch16/huggingface_snapshot')"
        )
        return [
            _command(
                [python, "-c", probe],
                ".\\.venv\\Scripts\\python.exe -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/dinov3-vits16-pretrain-lvd1689m', revision='114c1379950215c8b35dfcd4e90a5c251dde0d32', local_dir='external/checkpoints/dinov3_small_patch16/huggingface_snapshot')\"",
            ),
            _command(
                [python, "-m", "pip", "install", "transformers==4.56.0"],
                ".\\.venv\\Scripts\\python.exe -m pip install transformers==4.56.0",
            ),
            _command(
                [
                    python,
                    "scripts\\certify_local_publication_models.py",
                    "--models",
                    model_id,
                    "--device",
                    "cpu",
                ],
                ".\\.venv\\Scripts\\python.exe scripts\\certify_local_publication_models.py --models dinov3_small_patch16 --device cpu",
            ),
        ]
    if model_id in {"semba", "semba_fast"}:
        return [
            _command(
                [
                    python,
                    "scripts\\setup_external_model.py",
                    "--model",
                    model_id,
                    "--install",
                    "--download-checkpoint",
                    "--smoke",
                ],
                f".\\.venv\\Scripts\\python.exe scripts\\setup_external_model.py --model {model_id} --install --download-checkpoint --smoke",
            )
        ]
    return [
        _command(
            [
                "wsl",
                "-e",
                "bash",
                "-lc",
                "cd /mnt/d/Git/saliency && python3 scripts/setup_external_model.py "
                f"--model {model_id} --install --download-checkpoint --smoke",
            ],
            f"wsl -e bash -lc \"cd /mnt/d/Git/saliency && python3 scripts/setup_external_model.py --model {model_id} --install --download-checkpoint --smoke\"",
        )
    ]


def _command(argv: list[str], cmd: str) -> dict[str, Any]:
    return {"argv": argv, "cmd": cmd}


def _classify_failure(
    model_id: str,
    text: str,
    exit_code: int,
    timed_out: bool,
) -> tuple[str, str]:
    lower = text.lower()
    if exit_code == 0:
        return "none", "none"
    if timed_out:
        return "setup command exceeded timeout; rerun with longer timeout or on cluster", "user_required"
    if model_id.startswith("dinov3") and any(
        marker in lower for marker in ("gated", "403", "access", "manual")
    ):
        return (
            "accept Hugging Face gated access for facebook/dinov3-vits16-pretrain-lvd1689m and provide an authenticated HF token",
            "user_required",
        )
    if "wsl" in lower and any(
        marker in lower for marker in ("not recognized", "not installed", "no installed distributions")
    ):
        return "local WSL/Linux setup runtime is unavailable", "user_required"
    if model_id == "mambavision_t" and any(
        marker in lower
        for marker in (
            "mamba_ssm",
            "mamba-ssm",
            "nvcc was not found",
            "bare_metal_version",
        )
    ):
        return (
            "MambaVision requires mamba-ssm in a Linux/CUDA build environment; local Windows pip build failed",
            "user_required",
        )
    if "huggingface.co" in lower and any(
        marker in lower for marker in ("ssl", "read timed out", "connectionreseterror", "max retries exceeded")
    ):
        return "public Hugging Face snapshot download failed with network/SSL errors", "user_required"
    if "repo.anaconda.com" in lower and any(
        marker in lower for marker in ("timeout was reached", "operation too slow", "download error")
    ):
        return "conda package download timed out while building the pinned external environment", "user_required"
    if "github.com" in lower and any(
        marker in lower for marker in ("failed to connect", "gnutls recv error", "tls connection")
    ):
        return "git clone/fetch from the official GitHub source failed with network/TLS errors", "user_required"
    if "download returned an html page" in lower or "drive.google.com" in lower:
        return "manual Google Drive checkpoint download is required", "user_required"
    if "unresolved_official_source" in lower or "pin_required" in lower:
        return "official executable source/checkpoint API is unresolved", "user_required"
    if "404: not found" in lower and "license" in lower:
        return "official license file is absent at the probed source URL", "user_required"
    if "no module named" in lower or "modulenotfounderror" in lower:
        return "Python dependency installation/import failed", "codex_implementable"
    return "see setup log for the concrete failing command", "codex_implementable"


def _attempt_environment() -> dict[str, str]:
    environment = dict(os.environ)
    external = PROJECT_ROOT / "external"
    cache = external / "cache"
    environment.setdefault("HF_HOME", str(cache / "huggingface"))
    environment.setdefault("HF_HUB_CACHE", str(cache / "huggingface" / "hub"))
    environment.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
    environment.setdefault("TORCH_HOME", str(cache / "torch"))
    environment.setdefault("PYTHONNOUSERSITE", "1")
    return environment


def _publication_ids_for_runtime(publication: dict[str, Any], runtime_id: str) -> list[str]:
    models = publication.get("models", {})
    return [
        str(model_id)
        for model_id, entry in sorted(models.items())
        if str(entry.get("runtime_model_id", "")) == runtime_id
    ]


def _checkpoint_path(runtime: Any, model: dict[str, Any]) -> Path:
    filename = model.get("checkpoint", {}).get("filename") or "checkpoint"
    return runtime.workspace_path("checkpoints") / str(model["id"]) / str(filename)


def _load_existing_records() -> dict[str, dict[str, Any]]:
    if not RECORDS_PATH.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("runtime_model_id") or record.get("model_id") or "")
        if key:
            records[key] = record
    return records


def _write_records(records: dict[str, dict[str, Any]]) -> None:
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    RECORDS_PATH.write_text(
        "".join(json.dumps(records[key], sort_keys=True) + "\n" for key in sorted(records)),
        encoding="utf-8",
    )


def _tail_summary(text: str, *, max_lines: int = 8) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = lines[-max_lines:]
    summary = " | ".join(tail)
    return summary[:1000]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


if __name__ == "__main__":
    raise SystemExit(main())
