"""Run the comparable Matrix V2 efficiency protocol for one external model."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.adapters import build_adapter  # noqa: E402
from hma.external.registry import load_external_registry  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402
from scripts.run_external_model import _checkpoint_path, _load_image_rows  # noqa: E402


def profile_model(
    model_id: str,
    *,
    manifest_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    registry_path: str | Path = "configs/external_models/registry.yaml",
    split: str = "train",
    subject_id: str | None = None,
    roi: str | None = None,
    device: str = "cuda",
    seed: int = 123,
) -> Path:
    registry = load_external_registry(registry_path)
    model = registry.model(model_id)
    canonical_id = str(model["id"])
    rows = _load_image_rows(
        manifest_path,
        image_root=image_root,
        split=split,
        subject_id=subject_id,
        roi=roi,
        max_items=16,
    )
    if len(rows) != 16:
        raise ValueError(f"Efficiency profiling requires 16 images, found {len(rows)}")
    adapter = build_adapter(
        str(model["adapter"]),
        model_id=canonical_id,
        model_config=model,
        source_dir=registry.workspace_path("sources") / canonical_id,
        checkpoint_path=_checkpoint_path(registry, model),
        device=device,
        seed=seed,
    )
    images = [
        Image.open(row["image_path"]).convert("RGB")
        for row in rows
    ]
    try:
        payload = adapter.profile_efficiency(
            images,
            warmup_runs=20,
            measured_runs=100,
            repeats=5,
        )
    finally:
        for image in images:
            image.close()
    payload.update(
        {
            "schema_version": "hma.external.efficiency_profile.v2",
            "model_id": canonical_id,
            "manifest_path": str(resolve_path(manifest_path)),
            "map_keys": [row["map_key"] for row in rows],
            "preprocessing": model.get("preprocessing", {}),
            "seed": int(seed),
        }
    )
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_failure_profile(
    *,
    model_id: str,
    manifest_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    split: str,
    subject_id: str | None,
    roi: str | None,
    device: str,
    seed: int,
    exc: BaseException,
) -> Path:
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    publication_model_id = output.parent.name or model_id
    payload: dict[str, Any] = {
        "schema_version": "hma.external.efficiency_profile.v2",
        "model_id": model_id,
        "publication_model_id": publication_model_id,
        "profile_status": "profile_failed",
        "status": "profile_failed",
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "manifest_path": str(resolve_path(manifest_path)),
        "image_root": str(resolve_path(image_root)),
        "split": split,
        "subject_id": subject_id or "",
        "roi": roi or "",
        "device": device,
        "map_keys": [],
        "preprocessing": {},
        "seed": int(seed),
        "latency_ms_per_image": "",
        "peak_memory_bytes": "",
        "parameter_count": "",
        "theoretical_flops_or_macs": "",
        "profiler_realized_flops_or_macs": "",
        "total_cost_per_image": "",
        "total_cost_per_task": "",
        "sequential_step_count": "",
        "stopping_behavior": "",
        "resource_summary": {
            "profile_failure": {
                "status": "profile_failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--registry",
        default="configs/external_models/registry.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--subject-id")
    parser.add_argument("--roi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--allow-failure-profile",
        action="store_true",
        help=(
            "Write an explicit failed-profile JSON and exit successfully when "
            "the comparable profiling protocol cannot run."
        ),
    )
    args = parser.parse_args()
    try:
        print(
            profile_model(
                args.model,
                manifest_path=args.manifest,
                image_root=args.image_root,
                output_path=args.output,
                registry_path=args.registry,
                split=args.split,
                subject_id=args.subject_id,
                roi=args.roi,
                device=args.device,
                seed=args.seed,
            )
        )
    except Exception as exc:
        if not args.allow_failure_profile:
            raise
        traceback.print_exc()
        print(
            write_failure_profile(
                model_id=args.model,
                manifest_path=args.manifest,
                image_root=args.image_root,
                output_path=args.output,
                split=args.split,
                subject_id=args.subject_id,
                roi=args.roi,
                device=args.device,
                seed=args.seed,
                exc=exc,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
