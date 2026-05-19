"""Run the static saliency benchmark."""

from __future__ import annotations

import argparse

from hma.experiments import run_saliency_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an HMA static saliency benchmark.")
    parser.add_argument(
        "--config",
        default="configs/experiments/saliency_static_debug.yaml",
        help="Path to an experiment YAML config.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-image progress output.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Print progress every N images. Defaults to about 20 updates per run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate = run_saliency_benchmark(
        args.config,
        progress=not args.no_progress,
        progress_interval=args.progress_interval,
    )
    print(f"Per-image CSV: {aggregate['per_image_csv']}")
    print(f"Aggregate JSON: {aggregate['aggregate_json']}")
    for metric, value in aggregate["metrics"].items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
