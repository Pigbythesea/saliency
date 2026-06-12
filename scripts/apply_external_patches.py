"""Apply audited compatibility patches inside isolated external environments."""

from __future__ import annotations

import argparse
import site
from pathlib import Path


REPLACEMENTS = {
    "from torch._six import inf": "from math import inf",
    "from torch._six import container_abcs": "import collections.abc as container_abcs",
    "from torch._six import string_classes": "string_classes = (str,)",
}


def apply_patches(model_id: str, source_dir: str | Path) -> list[Path]:
    if model_id not in {
        "deit_small_static",
        "dynamicvit_deit_small_keep_0_7",
        "tome_deit_small_r13",
    }:
        return []
    candidates = list(Path(source_dir).rglob("*.py"))
    for package_root in site.getsitepackages():
        timm_root = Path(package_root) / "timm"
        if timm_root.is_dir():
            candidates.extend(timm_root.rglob("*.py"))
    changed: list[Path] = []
    for path in sorted(set(candidates)):
        original = path.read_text(encoding="utf-8")
        patched = original
        for old, new in REPLACEMENTS.items():
            patched = patched.replace(old, new)
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            changed.append(path)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = apply_patches(args.model, args.source)
    for path in changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
