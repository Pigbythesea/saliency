"""Generate the adaptive-computation Matrix V2 neural experiment configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.registry import load_external_registry
from hma.utils.config import load_yaml, save_yaml
from hma.utils.paths import resolve_path


STAGE_SETTINGS = {
    "adapter16": {"max_items": 16, "pca_components": 8, "subset_sizes": [8]},
    "scientific64": {"max_items": 64, "pca_components": 32, "subset_sizes": [32]},
    "full": {
        "max_items": 9841,
        "pca_components": 512,
        "subset_sizes": [512, 1024, 2048],
    },
}


def create_configs(
    *,
    matrix_path: str | Path = "configs/paper1_matrix_v2.yaml",
    output_root: str | Path = "configs/experiments/paper1_matrix_v2",
) -> list[Path]:
    matrix = load_yaml(resolve_path(matrix_path))
    registry = load_external_registry(matrix["external_registry"])
    evidence = dict(matrix["first_evidence_run"])
    generated: list[Path] = []
    for stage, settings in STAGE_SETTINGS.items():
        for model_id in evidence["models"]:
            model = registry.model(str(model_id))
            for roi_cell in evidence["roi_cells"]:
                roi = str(roi_cell["roi"])
                name = f"{model_id}_{roi.lower()}_{stage}"
                config = _experiment_config(
                    name=name,
                    stage=stage,
                    model=model,
                    subject_id=str(evidence["subject_id"]),
                    roi=roi,
                    roi_cell=dict(roi_cell),
                    settings=settings,
                )
                path = resolve_path(output_root) / stage / f"{name}.yaml"
                save_yaml(config, path)
                generated.append(path)
    return generated


def _experiment_config(
    *,
    name: str,
    stage: str,
    model: dict[str, Any],
    subject_id: str,
    roi: str,
    roi_cell: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(model["id"])
    return {
        "seed": 123,
        "device": "cpu",
        "experiment": {
            "name": name,
            "matrix_version": "paper1_matrix_v2",
            "stage": stage,
        },
        "dataset": {
            "name": "nsd_algonauts",
            "label": name,
            "root": "data/raw/nsd_algonauts",
            "manifest_path": roi_cell["manifest_path"],
            "split": "train",
            "subject_id": subject_id,
            "roi": roi,
            "max_items": int(settings["max_items"]),
            "validate_files": True,
        },
        "model": {
            "name": model_id,
            "backend": "external_artifact",
            "pretrained": True,
            "eval_mode": True,
        },
        "external_artifact": {
            "path": f"outputs/paper1_matrix_v2/external_artifacts/{stage}/{model_id}",
            "verify_hashes": True,
            **(
                {
                    "feature_cache_dir": (
                        f"outputs/paper1_matrix_v2/cache/{stage}/"
                        f"{model_id}/raw_features"
                    )
                }
                if stage in {"scientific64", "full"}
                else {}
            ),
        },
        "preprocessing": dict(model.get("preprocessing", {})),
        "neural": {
            "layers": list(model.get("feature_layers", [])),
            "response_key": "roi_responses",
            "noise_ceiling_key": "noise_ceiling",
            "feature_reduction": "flatten_pca",
            "pca_components": int(settings["pca_components"]),
            "pca_solver": "randomized",
            "pca_whiten": False,
            "feature_reduction_seed": 123,
            **(
                {
                    "pca_cache_dir": (
                        f"outputs/paper1_matrix_v2/cache/{stage}/"
                        f"{model_id}/pca"
                    )
                }
                if stage in {"scientific64", "full"}
                else {}
            ),
            "train_fraction": 0.8,
            "ridge_alpha": 1.0,
            "ridge_alphas": [
                0.001,
                0.01,
                0.1,
                1.0,
                10.0,
                100.0,
                1000.0,
                10000.0,
                100000.0,
                1000000.0,
                10000000.0,
            ],
            "validation_fraction": 0.2,
            "metric": "correlation",
            "selection": {
                "enabled": True,
                "validation_fraction": 0.2,
                "primary_score": "mean_noise_normalized_score",
            },
            "rsa": {"enabled": False},
            "geometry": {
                "enabled": True,
                "methods": ["linear_cka", "subset_rsa"],
                "subset_sizes": list(settings["subset_sizes"]),
                "subset_seed": 123,
            },
        },
        "output": {
            "dir": f"outputs/paper1_matrix_v2/neural/{stage}/{model_id}/{roi.lower()}"
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/paper1_matrix_v2.yaml")
    parser.add_argument(
        "--output-root", default="configs/experiments/paper1_matrix_v2"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = create_configs(matrix_path=args.matrix, output_root=args.output_root)
    print(f"Generated {len(paths)} Matrix V2 configs under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
