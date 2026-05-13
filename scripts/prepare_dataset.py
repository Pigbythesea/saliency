"""Placeholder dataset preparation CLI.

Real dataset download and manifest creation will be added in later tasks.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an HMA dataset placeholder.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    print(f"Dataset preparation is not implemented yet: {args.config}")


if __name__ == "__main__":
    main()
