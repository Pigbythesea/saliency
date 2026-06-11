"""Audit feasibility for adding a modern free-viewing fixation reference."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

from hma.utils.paths import ensure_dir


DEFAULT_OUTPUT = (
    "outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv"
)
DEFAULT_PYPROJECT = "pyproject.toml"
DEFAULT_EXPORTER = "scripts/export_deepgaze_maps.py"
DEFAULT_CONFIG_GENERATOR = "scripts/create_deepgaze_reference_configs.py"
DEFAULT_DEEPGAZE_ROOT = "data/precomputed/deepgaze"
DEFAULT_SALICON_MANIFEST = "data/manifests/v2/salicon_static2000_manifest.csv"
DEFAULT_CAT2000_MANIFEST = "data/manifests/v2/cat2000_static2000_manifest.csv"

FIELDNAMES = [
    "candidate_reference",
    "viewing_regime",
    "dataset_scope",
    "local_support",
    "requires_download",
    "requires_new_dependency",
    "estimated_run_scope",
    "decision",
    "next_action",
    "detail",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a free-viewing reference feasibility decision table."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--pyproject", default=DEFAULT_PYPROJECT)
    parser.add_argument("--exporter", default=DEFAULT_EXPORTER)
    parser.add_argument("--config-generator", default=DEFAULT_CONFIG_GENERATOR)
    parser.add_argument("--deepgaze-root", default=DEFAULT_DEEPGAZE_ROOT)
    parser.add_argument("--salicon-manifest", default=DEFAULT_SALICON_MANIFEST)
    parser.add_argument("--cat2000-manifest", default=DEFAULT_CAT2000_MANIFEST)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = audit_free_viewing_reference_feasibility(
        pyproject=args.pyproject,
        exporter=args.exporter,
        config_generator=args.config_generator,
        deepgaze_root=args.deepgaze_root,
        salicon_manifest=args.salicon_manifest,
        cat2000_manifest=args.cat2000_manifest,
        output=args.output,
    )
    print(f"Free-viewing reference feasibility: {Path(args.output).resolve()} ({len(rows)} rows)")


def audit_free_viewing_reference_feasibility(
    *,
    pyproject: str | Path = DEFAULT_PYPROJECT,
    exporter: str | Path = DEFAULT_EXPORTER,
    config_generator: str | Path = DEFAULT_CONFIG_GENERATOR,
    deepgaze_root: str | Path = DEFAULT_DEEPGAZE_ROOT,
    salicon_manifest: str | Path = DEFAULT_SALICON_MANIFEST,
    cat2000_manifest: str | Path = DEFAULT_CAT2000_MANIFEST,
    output: str | Path | None = DEFAULT_OUTPUT,
) -> list[dict[str, str]]:
    """Return and optionally write the free-viewing reference feasibility rows."""
    paths = {
        "pyproject": Path(pyproject),
        "exporter": Path(exporter),
        "config_generator": Path(config_generator),
        "deepgaze_root": Path(deepgaze_root),
        "salicon_manifest": Path(salicon_manifest),
        "cat2000_manifest": Path(cat2000_manifest),
    }
    rows = build_feasibility_rows(paths)
    if output is not None:
        _write_rows(Path(output), rows)
    return rows


def build_feasibility_rows(paths: dict[str, Path]) -> list[dict[str, str]]:
    salicon_rows = _csv_row_count(paths["salicon_manifest"])
    cat2000_rows = _csv_row_count(paths["cat2000_manifest"])
    iie_salicon_maps = _npy_count(paths["deepgaze_root"] / "salicon_static2000")
    iie_cat2000_maps = _npy_count(paths["deepgaze_root"] / "cat2000_static2000")
    has_deepgaze_dependency = _module_available("deepgaze_pytorch")
    has_iie_class = _deepgaze_class_available("DeepGazeIIE")
    has_msdb_class = _deepgaze_class_available("DeepGazeMSDB")
    exporter_text = _read_optional_text(paths["exporter"])
    config_text = _read_optional_text(paths["config_generator"])
    pyproject_text = _read_optional_text(paths["pyproject"])
    exporter_supports_msdb = "--model" in exporter_text and "deepgaze_msdb" in exporter_text
    config_supports_scoped_datasets = "--datasets" in config_text and "--reference-name" in config_text
    manifests_ready = salicon_rows > 0 and cat2000_rows > 0
    msdb_feasible = (
        has_deepgaze_dependency
        and has_msdb_class
        and exporter_supports_msdb
        and config_supports_scoped_datasets
        and manifests_ready
    )

    return [
        {
            "candidate_reference": "DeepGaze MSDB",
            "viewing_regime": "free_viewing",
            "dataset_scope": "SALICON/CAT2000",
            "local_support": _join_support(
                [
                    _flag("deepgaze_pytorch importable", has_deepgaze_dependency),
                    _flag("DeepGazeMSDB class available", has_msdb_class),
                    _flag("exporter supports --model deepgaze_msdb", exporter_supports_msdb),
                    _flag("config generator supports scoped reference configs", config_supports_scoped_datasets),
                    f"SALICON manifest rows={salicon_rows}",
                    f"CAT2000 manifest rows={cat2000_rows}",
                ]
            ),
            "requires_download": "yes",
            "requires_new_dependency": "no" if has_deepgaze_dependency and has_msdb_class else "yes",
            "estimated_run_scope": (
                f"{max(salicon_rows, 0) + max(cat2000_rows, 0)} free-viewing images "
                f"({max(salicon_rows, 0)} SALICON + {max(cat2000_rows, 0)} CAT2000); "
                "export maps, run two benchmarks, then aggregate separately"
            ),
            "decision": "feasible_now" if msdb_feasible else "requires_download_or_dependency",
            "next_action": (
                "run the documented cmd smoke export, then full SALICON/CAT2000 export and scoring"
                if msdb_feasible
                else "defer until deepgaze_pytorch MSDB support, manifests, and exporter/config hooks are present"
            ),
            "detail": (
                "Class-level MSDB support is present locally; this audit does not instantiate pretrained "
                "weights, so first export may download weights if they are not already cached."
                if msdb_feasible
                else "MSDB is not fully supported by the local dependency/code/config surface yet."
            ),
        },
        {
            "candidate_reference": "DeepGaze IIE current reference",
            "viewing_regime": "free_viewing",
            "dataset_scope": "SALICON/CAT2000",
            "local_support": _join_support(
                [
                    _flag("deepgaze_pytorch importable", has_deepgaze_dependency),
                    _flag("DeepGazeIIE class available", has_iie_class),
                    f"SALICON precomputed maps={iie_salicon_maps}",
                    f"CAT2000 precomputed maps={iie_cat2000_maps}",
                    _flag("pyproject keeps DeepGaze as external optional support", "deepgaze" not in pyproject_text.lower()),
                ]
            ),
            "requires_download": "no",
            "requires_new_dependency": "no" if has_deepgaze_dependency and has_iie_class else "yes",
            "estimated_run_scope": "already exported and scored as the current accepted DeepGaze-class reference",
            "decision": "feasible_now",
            "next_action": "keep as accepted free-viewing reference unless MSDB rows are exported and scored",
            "detail": (
                "Current DeepGaze IIE reference maps are present for the free-viewing static subsets; "
                "CAT2000 can be over-complete because repeated image IDs use map-key filenames."
            ),
        },
        {
            "candidate_reference": "comparable modern free-viewing reference",
            "viewing_regime": "free_viewing",
            "dataset_scope": "SALICON/CAT2000",
            "local_support": "no named alternative implementation, dependency, or precomputed-map source is present",
            "requires_download": "yes",
            "requires_new_dependency": "yes",
            "estimated_run_scope": "unknown until a concrete model/source and map format are selected",
            "decision": "defer_or_document_limitation",
            "next_action": (
                "do not add a generic saliency leaderboard branch; document the limitation unless a "
                "specific comparable reference with local support is selected"
            ),
            "detail": (
                "The only concrete modern candidate with local class support is DeepGaze MSDB; "
                "other references would require a separate feasibility decision."
            ),
        },
    ]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _deepgaze_class_available(class_name: str) -> bool:
    try:
        import deepgaze_pytorch
    except ImportError:
        return False
    return hasattr(deepgaze_pytorch, class_name)


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _npy_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.npy"))


def _read_optional_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _flag(label: str, value: bool) -> str:
    return f"{label}={'yes' if value else 'no'}"


def _join_support(parts: list[str]) -> str:
    return "; ".join(parts)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
