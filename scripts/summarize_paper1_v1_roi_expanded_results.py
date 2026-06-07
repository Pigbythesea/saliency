"""Summarize Paper 1 V1 ROI-expanded neural and geometry outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from hma.experiments import summarize_neural_roi_results
from hma.utils.config import load_yaml


DEFAULT_CONFIG = Path("configs/paper1_experiment_v1.yaml")


def summarize_paper1_v1_roi_expanded_results(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Path]:
    config = load_yaml(config_path)
    discovery = config["discovery_matrix"]
    config_root = Path(discovery["config_root"])
    output_root = Path(discovery["output_root"])
    input_dirs = [output_root / path.stem for path in sorted(config_root.glob("*.yaml"))]
    behavioral_csv = config.get("baseline_inputs", {}).get("behavioral_csv")
    return summarize_neural_roi_results(
        input_dirs,
        discovery["summary_output_dir"],
        behavioral_csv=behavioral_csv,
        scope_config=config_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = summarize_paper1_v1_roi_expanded_results(args.config)
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
