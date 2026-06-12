"""Standard-library hashing helpers for external sources and checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a file or directory deterministically for checkpoint snapshots."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    for file_path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()
