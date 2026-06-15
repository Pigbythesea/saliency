"""Recompute Matrix V2 geometry from retained selected activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.experiments.neural_alignment import (  # noqa: E402
    _compute_geometry_rows,
    _write_geometry_rows,
)
from hma.utils.config import load_experiment_config  # noqa: E402
from scripts.compute_matched_geometry import (  # noqa: E402
    _dataset_image_ids_and_responses,
)


def recompute_geometry(
    root: str | Path = "outputs/paper1_matrix_v2/neural/full",
) -> list[Path]:
    output_root = Path(root)
    written: list[Path] = []
    response_cache = {}
    for metadata_path in sorted(output_root.glob("*/*/metadata.json")):
        output_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = load_experiment_config(metadata["config_path"])
        geometry = dict(config["neural"]["geometry"])
        activations = np.load(output_dir / "activations.npz", allow_pickle=True)
        selected_layer = str(metadata["selected_layer"])
        image_ids = [str(value) for value in activations["image_ids"].tolist()]
        dataset_ids, responses = _dataset_image_ids_and_responses(
            config,
            response_cache,
        )
        if image_ids != dataset_ids:
            raise ValueError(f"Activation image order mismatch: {output_dir}")
        rows = _compute_geometry_rows(
            {selected_layer: np.asarray(activations[selected_layer])},
            responses,
            layers=[selected_layer],
            methods=[str(value) for value in geometry["methods"]],
            subset_sizes=[int(value) for value in geometry["subset_sizes"]],
            subset_seeds=[int(value) for value in geometry["subset_seeds"]],
            null_control_seeds=[
                int(value) for value in geometry["null_control_seeds"]
            ],
            bootstrap_resamples=int(geometry.get("bootstrap_resamples", 0)),
            bootstrap_seed=int(geometry.get("bootstrap_seed", 123)),
            bootstrap_confidence=float(
                geometry.get("bootstrap_confidence", 0.95)
            ),
            row_context={
                "dataset": metadata.get("dataset", ""),
                "model": metadata.get("model_name") or metadata.get("model", ""),
                "subject_id": (metadata.get("subjects") or [""])[0],
                "roi": (metadata.get("rois") or [""])[0],
            },
            model_feature_reduction=str(metadata["feature_reduction"]),
        )
        path = output_dir / "geometry_scores.csv"
        _write_geometry_rows(path, rows)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/paper1_matrix_v2/neural/full",
    )
    args = parser.parse_args()
    for path in recompute_geometry(args.root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
