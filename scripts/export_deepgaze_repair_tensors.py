"""Export bounded DeepGaze repair tensors and validate determinism."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.artifacts import validate_external_artifact
from hma.external.hashing import sha256_file

from scripts.run_external_model import run_external_model

PUBLICATION_EXTERNAL = PROJECT_ROOT / "outputs/paper1_publication_v0/external"

OUTPUT_DIRS = {
    "deepgaze_iie": PUBLICATION_EXTERNAL / "deepgaze_iie_latent",
    "deepgaze_iii": PUBLICATION_EXTERNAL / "deepgaze_iii_conditional",
    "deepgaze_msdb": PUBLICATION_EXTERNAL / "deepgaze_msdb_latent",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepgaze_iie", "deepgaze_iii", "deepgaze_msdb"],
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/salicon_manifest.csv",
    )
    parser.add_argument("--image-root", default="data/raw/SALICON")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--per-model-timeout-seconds", type=int, default=0)
    args = parser.parse_args()

    results = []
    for model_id in args.models:
        kwargs = {
            "manifest": args.manifest,
            "image_root": args.image_root,
            "split": args.split,
            "max_items": args.max_items,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
        }
        if args.per_model_timeout_seconds > 0:
            result = export_with_timeout(
                model_id,
                timeout_seconds=args.per_model_timeout_seconds,
                **kwargs,
            )
        else:
            result = export_and_validate(
                model_id,
                **kwargs,
            )
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(row["status"] == "deterministic" for row in results) else 2


def export_with_timeout(
    model_id: str,
    *,
    timeout_seconds: int,
    **kwargs: Any,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(
        target=_export_worker,
        args=(queue, model_id, kwargs),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        result = {
            "schema_version": "hma.external.deepgaze_determinism_validation.v1",
            "model_id": model_id,
            "status": "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error_type": "TimeoutExpired",
            "error_message": f"DeepGaze repair export exceeded {timeout_seconds} seconds",
        }
        validation_path = OUTPUT_DIRS[model_id] / "determinism_validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
    if not queue.empty():
        return queue.get()
    result = {
        "schema_version": "hma.external.deepgaze_determinism_validation.v1",
        "model_id": model_id,
        "status": "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "error_type": "ProcessExited",
        "error_message": f"DeepGaze repair export exited with code {process.exitcode}",
    }
    validation_path = OUTPUT_DIRS[model_id] / "determinism_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _export_worker(queue: Any, model_id: str, kwargs: dict[str, Any]) -> None:
    queue.put(export_and_validate(model_id, **kwargs))


def export_and_validate(
    model_id: str,
    *,
    manifest: str,
    image_root: str,
    split: str,
    max_items: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIRS[model_id]
    repeat_dir = output_dir / "_determinism_repeat"
    for path in (output_dir, repeat_dir):
        if path.exists():
            shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_dir.mkdir(parents=True, exist_ok=True)

    try:
        first_manifest = run_external_model(
            model_id,
            manifest_path=manifest,
            output_dir=output_dir,
            image_root=image_root,
            split=split,
            max_items=max_items,
            batch_size=batch_size,
            device=device,
            seed=seed,
            profile_efficiency=False,
        )
        second_manifest = run_external_model(
            model_id,
            manifest_path=manifest,
            output_dir=repeat_dir,
            image_root=image_root,
            split=split,
            max_items=max_items,
            batch_size=batch_size,
            device=device,
            seed=seed,
            profile_efficiency=False,
        )
        first = validate_external_artifact(output_dir, verify_hashes=True)
        second = validate_external_artifact(repeat_dir, verify_hashes=True)
        comparison = compare_manifests(output_dir, first, repeat_dir, second)
        status = "deterministic" if comparison["deterministic"] else "mismatch"
        result = {
            "schema_version": "hma.external.deepgaze_determinism_validation.v1",
            "model_id": model_id,
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "primary_manifest": str(first_manifest.relative_to(PROJECT_ROOT)),
            "repeat_manifest": str(second_manifest.relative_to(PROJECT_ROOT)),
            "manifest": {
                "num_images": first["num_images"],
                "features": sorted(first["features"]),
                "outputs": sorted(first["outputs"]),
                "resource_allocation": sorted(first["resource_allocation"]),
                "scanpaths_file": first.get("scanpaths_file"),
            },
            "comparison": comparison,
        }
    except Exception as exc:
        result = {
            "schema_version": "hma.external.deepgaze_determinism_validation.v1",
            "model_id": model_id,
            "status": "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    validation_path = output_dir / "determinism_validation.json"
    validation_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def compare_manifests(
    first_root: Path,
    first: dict[str, Any],
    second_root: Path,
    second: dict[str, Any],
) -> dict[str, Any]:
    keys = ("features", "outputs", "resource_allocation")
    mismatches: list[str] = []
    if first["num_images"] != second["num_images"]:
        mismatches.append("num_images")
    first_ids = json.loads((first_root / first["image_ids_file"]).read_text(encoding="utf-8"))
    second_ids = json.loads((second_root / second["image_ids_file"]).read_text(encoding="utf-8"))
    if first_ids != second_ids:
        mismatches.append("image_ids")
    for key in keys:
        if sorted(first[key]) != sorted(second[key]):
            mismatches.append(key)
            continue
        for output_name in sorted(first[key]):
            first_chunks = first[key][output_name]
            second_chunks = second[key][output_name]
            if len(first_chunks) != len(second_chunks):
                mismatches.append(f"{key}.{output_name}.chunk_count")
                continue
            for index, (left, right) in enumerate(zip(first_chunks, second_chunks)):
                for field in ("start", "stop", "shape", "dtype", "sha256"):
                    if left.get(field) != right.get(field):
                        mismatches.append(f"{key}.{output_name}.chunk_{index}.{field}")
    if first.get("scanpaths_file") or second.get("scanpaths_file"):
        if first.get("scanpaths_file") != second.get("scanpaths_file"):
            mismatches.append("scanpaths_file_name")
        else:
            first_hash = sha256_file(first_root / str(first["scanpaths_file"]))
            second_hash = sha256_file(second_root / str(second["scanpaths_file"]))
            if first_hash != second_hash:
                mismatches.append("scanpaths_sha256")
    return {
        "deterministic": not mismatches,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    raise SystemExit(main())
