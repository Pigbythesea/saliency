"""Audit and remove package/model caches accidentally created in cluster home."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


CACHE_PATHS = (
    ".cache/huggingface",
    ".cache/mamba",
    ".cache/pip",
    ".cache/torch",
    ".cache/torch_extensions",
    ".conda/pkgs",
    ".local/share/mamba",
    ".nv/ComputeCache",
    ".triton",
)


def audit_home(
    *,
    clean: bool = False,
    home_dir: str | Path | None = None,
) -> dict[str, Any]:
    home = (
        Path(home_dir).expanduser().resolve()
        if home_dir is not None
        else Path.home().resolve()
    )
    entries = []
    for relative in CACHE_PATHS:
        path = (home / relative).resolve()
        if not path.is_relative_to(home) or path == home:
            raise RuntimeError(f"Refusing unsafe cluster-home path: {path}")
        existed = path.exists()
        size_bytes = _path_size(path) if existed else 0
        if clean and existed:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        entries.append(
            {
                "path": str(path),
                "existed": existed,
                "size_bytes": size_bytes,
                "removed": bool(clean and existed and not path.exists()),
            }
        )
    return {
        "schema_version": "hma.cluster_home_cleanup.v1",
        "home": str(home),
        "clean": bool(clean),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "entries": entries,
    }


def _path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return int(path.stat().st_size)
    total = 0
    for root, _directories, files in os.walk(path):
        for name in files:
            candidate = Path(root) / name
            try:
                total += int(candidate.stat().st_size)
            except FileNotFoundError:
                continue
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--json-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_home(clean=bool(args.clean))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output:
        path = Path(args.json_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
