"""Run the Matrix V2 64-image local scientific smoke sequentially."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.artifacts import validate_external_artifact


MODELS = (
    "deit_small_static",
    "dynamicvit_deit_small_keep_0_7",
    "tome_deit_small_r13",
)
ROIS = ("v1", "ventral", "lateral", "parietal")
MICROMAMBA = "external/tools/micromamba/2.8.1-0/micromamba"
WSL_PROJECT = "/mnt/d/Git/saliency"


def build_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for model_id in MODELS:
        artifact = (
            PROJECT_ROOT
            / "outputs"
            / "paper1_matrix_v2"
            / "external_artifacts"
            / "scientific64"
            / model_id
        )
        shell_command = (
            f"cd {WSL_PROJECT} && {MICROMAMBA} run "
            f"-p external/environments/{model_id} "
            f"python scripts/run_external_model.py --model {model_id} "
            "--manifest data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv "
            "--split train --subject-id subj01 --roi V1 --max-items 64 "
            f"--output outputs/paper1_matrix_v2/external_artifacts/scientific64/{model_id}"
        )
        jobs.append(
            {
                "name": f"export_{model_id}",
                "kind": "export",
                "model_id": model_id,
                "artifact": artifact,
                "command": ["wsl", "-e", "bash", "-lc", shell_command],
            }
        )
    for model_id in MODELS:
        for roi in ROIS:
            config = (
                PROJECT_ROOT
                / "configs"
                / "experiments"
                / "paper1_matrix_v2"
                / "scientific64"
                / f"{model_id}_{roi}_scientific64.yaml"
            )
            jobs.append(
                {
                    "name": f"analysis_{model_id}_{roi}",
                    "kind": "analysis",
                    "model_id": model_id,
                    "config": config,
                    "command": [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "run_neural_alignment.py"),
                        "--config",
                        str(config),
                    ],
                }
            )
    jobs.append(
        {
            "name": "refresh_matrix_v2_audit",
            "kind": "audit",
            "command": [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "audit_paper1_matrix_v2.py"),
            ],
        }
    )
    return jobs


def run_jobs(*, dry_run: bool = False) -> int:
    log_root = PROJECT_ROOT / "logs" / "paper1_matrix_v2_scientific64"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    if dry_run:
        for job in jobs:
            print(job["name"], subprocess.list2cmdline(job["command"]))
        return 0

    keep_awake = _prevent_windows_sleep()
    results: list[dict[str, Any]] = []
    export_ready: dict[str, bool] = {model_id: False for model_id in MODELS}
    try:
        for index, job in enumerate(jobs, start=1):
            name = str(job["name"])
            model_id = job.get("model_id")
            if job["kind"] == "analysis" and not export_ready[str(model_id)]:
                result = {
                    "name": name,
                    "status": "skipped",
                    "reason": f"scientific64 artifact for {model_id} did not validate",
                }
                results.append(result)
                print(f"[{index}/{len(jobs)}] SKIP {name}: {result['reason']}")
                continue
            log_path = log_root / f"{name}.log"
            print(f"[{index}/{len(jobs)}] START {name}")
            started = datetime.now(timezone.utc)
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    job["command"],
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            result = {
                "name": name,
                "status": "passed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "log": str(log_path),
            }
            if completed.returncode == 0 and job["kind"] == "export":
                try:
                    manifest = validate_external_artifact(
                        job["artifact"], verify_hashes=True
                    )
                    if int(manifest["num_images"]) != 64:
                        raise ValueError(
                            f"Expected 64 images, found {manifest['num_images']}"
                        )
                except Exception as exc:
                    result["status"] = "failed"
                    result["validation_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    export_ready[str(model_id)] = True
                    result["artifact"] = str(job["artifact"])
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {str(result['status']).upper()} {name}; "
                f"log={log_path}"
            )
    finally:
        _restore_windows_sleep(keep_awake)

    summary_path = log_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "hma.matrix_v2.scientific64_run.v1",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failed = [row for row in results if row["status"] != "passed"]
    print(f"Summary: {summary_path}")
    print(f"Passed: {len(results) - len(failed)}; failed/skipped: {len(failed)}")
    return 1 if failed else 0


def _prevent_windows_sleep() -> bool:
    if os.name != "nt":
        return False
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(
        es_continuous | es_system_required
    )
    return bool(result)


def _restore_windows_sleep(enabled: bool) -> None:
    if enabled and os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_jobs(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
