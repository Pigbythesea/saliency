"""Write compact summaries for neural ROI alignment outputs."""

from __future__ import annotations

import argparse

from hma.experiments import summarize_neural_roi_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize neural ROI alignment outputs.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Neural output directories containing encoding_scores.csv and optional rsa_scores.csv.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--behavioral-csv", default=None)
    parser.add_argument("--efficiency-csv", default=None)
    parser.add_argument(
        "--scope-config",
        default=None,
        help="Optional Paper 1 scope config, e.g. configs/paper1_experiment_v1.yaml.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = summarize_neural_roi_results(
        args.input_dirs,
        args.output_dir,
        behavioral_csv=args.behavioral_csv,
        efficiency_csv=args.efficiency_csv,
        scope_config=args.scope_config,
    )
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
