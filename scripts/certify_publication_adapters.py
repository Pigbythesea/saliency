"""Generate publication adapter certification and setup-blocker artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.certification import (
    build_certification_records,
    update_scope_reset_adapter_tables,
    write_certification_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-registry",
        default="configs/external_models/publication_registry.yaml",
    )
    parser.add_argument(
        "--runtime-registry",
        default="configs/external_models/registry.yaml",
    )
    parser.add_argument(
        "--jsonl",
        default="outputs/paper1_scope_reset/adapter_certification_records.jsonl",
    )
    parser.add_argument(
        "--csv",
        default="outputs/paper1_scope_reset/adapter_certification_summary.csv",
    )
    parser.add_argument("--update-scope-tables", action="store_true")
    args = parser.parse_args()
    records = build_certification_records(
        publication_registry_path=args.publication_registry,
        runtime_registry_path=args.runtime_registry,
    )
    jsonl, summary = write_certification_records(
        records,
        jsonl_path=args.jsonl,
        csv_path=args.csv,
    )
    if args.update_scope_tables:
        update_scope_reset_adapter_tables(
            records,
            comparability_path=(
                "outputs/paper1_scope_reset/model_adapter_comparability_table.csv"
            ),
            role_matrix_path="outputs/paper1_scope_reset/model_role_matrix.csv",
        )
    print(jsonl)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
