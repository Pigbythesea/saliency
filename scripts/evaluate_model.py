"""Placeholder model evaluation CLI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HMA model placeholder.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    print(f"Model evaluation is not implemented yet: {args.config}")


if __name__ == "__main__":
    main()
