"""Audit local NSD / Algonauts files for neural reliability metadata."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable


KEYWORDS = (
    "reliab",
    "noise",
    "ceiling",
    "repeat",
    "trial",
    "split",
    "variance",
    "ncsnr",
)
USABLE_SUFFIXES = {".csv", ".json", ".mat", ".npy", ".npz", ".tsv", ".txt"}
SKIP_DIRS = {"responses", "training_images"}


def audit_neural_reliability_metadata(
    root: str | Path = "data/raw/nsd_algonauts",
    output_dir: str | Path = "outputs/neural_reliability_audit_v1",
) -> dict[str, Path | int | bool]:
    """Scan local neural data for candidate reliability / noise-ceiling files."""
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _candidate_rows(root_path)
    candidate_csv = output_path / "candidate_reliability_files.csv"
    _write_candidate_csv(candidate_csv, rows)

    usable_rows = [
        row
        for row in rows
        if row["suffix"].lower() in USABLE_SUFFIXES and row["kind"] == "file"
    ]
    note = output_path / "README.md"
    _write_note(note, root_path=root_path, rows=rows, usable_rows=usable_rows)
    return {
        "candidate_csv": candidate_csv,
        "note": note,
        "candidate_count": len(rows),
        "usable_candidate_count": len(usable_rows),
        "usable_metadata_found": bool(usable_rows),
    }


def _candidate_rows(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []

    rows: list[dict[str, str]] = []
    for directory_name, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        directory = Path(directory_name)
        for name in filenames:
            lowered = name.lower()
            matched = [keyword for keyword in KEYWORDS if keyword in lowered]
            if not matched:
                continue
            path = directory / name
            rows.append(
                {
                    "path": path.resolve().as_posix(),
                    "relative_path": path.resolve().relative_to(root).as_posix(),
                    "kind": "file",
                    "suffix": path.suffix.lower(),
                    "matched_keywords": " ".join(matched),
                    "size_bytes": str(path.stat().st_size),
                }
            )
    rows.sort(key=lambda row: row["relative_path"])
    return rows


def _write_candidate_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = [
        "path",
        "relative_path",
        "kind",
        "suffix",
        "matched_keywords",
        "size_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_note(
    path: Path,
    *,
    root_path: Path,
    rows: list[dict[str, str]],
    usable_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Neural Reliability Metadata Audit V1",
        "",
        f"- Root scanned: `{root_path}`.",
        f"- Candidate files: {len(rows)}.",
        f"- Usable candidate files: {len(usable_rows)}.",
        "",
        "## Conclusion",
        "",
    ]
    if usable_rows:
        lines.append(
            "Potential reliability/noise-ceiling metadata files were found. Inspect "
            "`candidate_reliability_files.csv` before treating current neural scores "
            "as noise-normalized."
        )
        lines.extend(["", "## Candidate Files", ""])
        for row in usable_rows[:20]:
            lines.append(
                f"- `{row['relative_path']}` "
                f"({row['matched_keywords']}, {row['size_bytes']} bytes)."
            )
        if len(usable_rows) > 20:
            lines.append(f"- ... plus {len(usable_rows) - 20} more.")
    else:
        lines.append(
            "No usable local reliability, noise-ceiling, repeat-trial, or split-half "
            "metadata file was found by filename audit. Current ROI500 neural outputs "
            "should remain labeled `benchmark_style_non_noise_normalized` unless "
            "external target-level ceiling metadata is added."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local NSD / Algonauts files for reliability metadata."
    )
    parser.add_argument("--root", default="data/raw/nsd_algonauts")
    parser.add_argument("--output-dir", default="outputs/neural_reliability_audit_v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_neural_reliability_metadata(
        root=args.root,
        output_dir=args.output_dir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
