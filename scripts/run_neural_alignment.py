"""Run a neural alignment smoke experiment."""

from __future__ import annotations

import argparse

from hma.experiments import run_neural_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HMA neural encoding alignment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/neural_smoke_dummy.yaml",
        help="Path to a neural experiment YAML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_neural_alignment(args.config)
    print(f"Activations: {result['activations']}")
    print(f"Encoding scores: {result['encoding_scores']}")
    if result.get("rsa_scores"):
        print(f"RSA scores: {result['rsa_scores']}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
