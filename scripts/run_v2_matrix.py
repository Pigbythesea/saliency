"""Run the controlled V2 saliency matrix with reliability gating and reporting."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hma.experiments.aggregate_results import aggregate_result_files, save_aggregate_table
from hma.experiments.summarize_results import summarize_aggregate_results
from hma.utils.config import load_yaml
from hma.utils.paths import ensure_dir
from hma.viz.plot_metrics import (
    load_csv_rows,
    metric_higher_is_better,
    plot_alignment_vs_efficiency,
    plot_model_ranking,
)


DEFAULT_CONFIG_DIR = Path("configs/experiments/real_matrix_v2")
DEFAULT_OUTPUT_ROOT = Path("outputs/real_matrix_v2")
DEFAULT_RELIABILITY_CHECKS = {
    ("salicon_pilot500", "vit_base_patch16_224", "attention_rollout"),
    ("salicon_pilot500", "deit_small_patch16_224", "attention_rollout"),
    ("salicon_pilot500", "convnext_tiny", "gradcam"),
    ("salicon_pilot500", "vit_small_patch14_dinov2", "attention_rollout"),
    ("salicon_pilot500", "vit_base_patch16_clip_224", "attention_rollout"),
    ("salicon_pilot500", "resnet50_clip", "gradcam"),
}
REPORT_METRICS = ["nss", "shuffled_auc", "auc_borji", "auc_judd", "cc", "similarity", "kl"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HMA real_matrix_v2 benchmarks.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--phase",
        choices=["pilot", "static2000", "all"],
        default="pilot",
        help="Matrix phase to run after reliability checks.",
    )
    parser.add_argument("--ledger", default=None, help="Run ledger CSV path.")
    parser.add_argument("--aggregate-csv", default=None, help="Aggregate CSV path.")
    parser.add_argument("--summary-dir", default=None, help="Summary output directory.")
    parser.add_argument("--plots-dir", default=None, help="Plot output directory.")
    parser.add_argument(
        "--efficiency-csv",
        default="outputs/real_matrix_v2/efficiency/model_efficiency.csv",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip configs with aggregate JSON.")
    parser.add_argument(
        "--reliability-checks",
        nargs="*",
        default=None,
        metavar="DATASET:MODEL:METHOD",
        help=(
            "Reliability gate entries. If omitted, built-in V2 and SSL gates are used. "
            "Pass the flag with no entries to disable reliability gates."
        ),
    )
    parser.add_argument(
        "--reliability-checks-csv",
        default=None,
        help=(
            "CSV with dataset, model, and saliency_method/method columns. "
            "When provided, these checks replace the built-in defaults."
        ),
    )
    parser.add_argument("--no-report", action="store_true", help="Skip aggregation/summaries/plots.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Forward per-image progress interval to benchmark runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    phase_label = "pilot" if args.phase == "pilot" else "static2000" if args.phase == "static2000" else "all"
    ledger = Path(args.ledger) if args.ledger else output_root / "run_ledgers" / f"{phase_label}_run_ledger.csv"
    aggregate_csv = (
        Path(args.aggregate_csv)
        if args.aggregate_csv
        else output_root / "aggregated" / ("pilot_results.csv" if args.phase == "pilot" else "results.csv")
    )
    summary_dir = Path(args.summary_dir) if args.summary_dir else aggregate_csv.parent / f"{aggregate_csv.stem}_summary"
    plots_dir = Path(args.plots_dir) if args.plots_dir else aggregate_csv.parent / f"{aggregate_csv.stem}_plots"

    configs = [_config_info(path) for path in sorted(config_dir.glob("*.yaml"))]
    selected = _selected_configs(configs, args.phase)
    reliability_checks = _reliability_checks_from_args(args)
    reliability = [info for info in configs if _reliability_key(info) in reliability_checks]
    run_order = _dedupe_infos([*reliability, *selected])
    planned_runs = _planned_run_count(run_order, args.max_runs)
    if not args.no_progress:
        print(
            f"[progress] V2 matrix: phase={args.phase}, planned configs={planned_runs}, "
            f"resume={args.resume}",
            flush=True,
        )

    failed_capabilities: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for index, info in enumerate(run_order, start=1):
        capability = (info["model"], info["saliency_method"])
        if info["scale"] == "static2000" and capability in failed_capabilities:
            rows.append(_skipped_row(info, "unsupported_after_pilot_reliability_failure"))
            _print_config_progress(
                enabled=not args.no_progress,
                index=len(rows),
                total=planned_runs,
                info=info,
                status="skipped",
                reason="unsupported_after_pilot_reliability_failure",
            )
            continue
        if args.max_runs is not None and len(rows) >= args.max_runs:
            break
        row = _run_one(
            info,
            dry_run=args.dry_run,
            resume=args.resume,
            progress=not args.no_progress,
            progress_index=len(rows) + 1,
            progress_total=planned_runs,
            progress_interval=args.progress_interval,
        )
        rows.append(row)
        if _reliability_key(info) in reliability_checks and row["status"] == "failed":
            failed_capabilities.add(capability)
        _write_ledger(ledger, rows)

    if rows:
        _write_ledger(ledger, rows)
        print(f"Run ledger: {ledger.resolve()}")
    else:
        print("Run ledger unchanged: no configs were executed or skipped.")
    if not args.no_report and not args.dry_run:
        _write_reports(output_root, aggregate_csv, summary_dir, plots_dir, args.efficiency_csv)


def _reliability_checks_from_args(args: argparse.Namespace) -> set[tuple[str, str, str]]:
    supplied_inline = args.reliability_checks is not None
    supplied_csv = bool(args.reliability_checks_csv)
    if not supplied_inline and not supplied_csv:
        return set(DEFAULT_RELIABILITY_CHECKS)

    checks: set[tuple[str, str, str]] = set()
    if supplied_csv:
        checks.update(_read_reliability_checks_csv(Path(args.reliability_checks_csv)))
    for entry in args.reliability_checks or []:
        checks.add(_parse_reliability_check(entry))
    return checks


def _read_reliability_checks_csv(path: Path) -> set[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        checks = set()
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            model = str(row.get("model", "")).strip()
            method = str(row.get("saliency_method") or row.get("method") or "").strip()
            if not dataset or not model or not method:
                raise ValueError(
                    "Reliability CSV rows must include dataset, model, and "
                    "saliency_method or method"
                )
            checks.add((dataset, model, method))
    return checks


def _parse_reliability_check(entry: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in entry.split(":")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "Reliability checks must use DATASET:MODEL:METHOD, "
            f"got {entry!r}"
        )
    return parts[0], parts[1], parts[2]


def _selected_configs(configs: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase == "all":
        return configs
    return [info for info in configs if info["scale"] == phase]


def _config_info(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    saliency = config.get("saliency", {})
    output_dir = Path(config.get("output", {}).get("dir", ""))
    label = str(dataset.get("label") or dataset.get("name") or "")
    return {
        "config_path": path,
        "dataset": label,
        "scale": "static2000" if "static2000" in label else "pilot",
        "model": str(model.get("name", "")),
        "saliency_method": str(saliency.get("method", "")),
        "saliency_family": _saliency_family(str(saliency.get("method", ""))),
        "output_dir": output_dir,
    }


def _run_one(
    info: dict[str, Any],
    *,
    dry_run: bool,
    resume: bool,
    progress: bool,
    progress_index: int,
    progress_total: int,
    progress_interval: int | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    aggregate_json = info["output_dir"] / "aggregate_metrics.json"
    if resume and aggregate_json.is_file():
        _print_config_progress(
            enabled=progress,
            index=progress_index,
            total=progress_total,
            info=info,
            status="skipped",
            reason="already_completed",
        )
        return _base_row(info, "skipped", start, "already_completed")
    if dry_run:
        _print_config_progress(
            enabled=progress,
            index=progress_index,
            total=progress_total,
            info=info,
            status="dry_run",
        )
        return _base_row(info, "dry_run", start, "")

    _print_config_progress(
        enabled=progress,
        index=progress_index,
        total=progress_total,
        info=info,
        status="starting",
    )
    command = [
        sys.executable,
        "scripts/run_saliency_benchmark.py",
        "--config",
        str(info["config_path"]),
    ]
    if not progress:
        command.append("--no-progress")
    if progress_interval is not None:
        command.extend(["--progress-interval", str(progress_interval)])

    returncode, output = _run_streaming(command)
    status = "succeeded" if returncode == 0 else "failed"
    _print_config_progress(
        enabled=progress,
        index=progress_index,
        total=progress_total,
        info=info,
        status=status,
        reason=f"{time.perf_counter() - start:.1f}s",
    )
    return _base_row(info, status, start, _compact_error(output), returncode=returncode)


def _write_reports(
    output_root: Path,
    aggregate_csv: Path,
    summary_dir: Path,
    plots_dir: Path,
    efficiency_csv: str | None,
) -> None:
    rows = aggregate_result_files([output_root])
    output_path = save_aggregate_table(rows, aggregate_csv)
    print(f"Aggregate CSV: {output_path}")
    efficiency_path = Path(efficiency_csv) if efficiency_csv else None
    usable_efficiency = efficiency_path if efficiency_path and efficiency_path.is_file() else None
    outputs = summarize_aggregate_results(
        output_path,
        summary_dir,
        efficiency_csv=usable_efficiency,
    )
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")
    _write_plots(rows, plots_dir, usable_efficiency)


def _write_plots(
    rows: list[dict[str, Any]],
    plots_dir: Path,
    efficiency_csv: Path | None,
) -> None:
    if not rows:
        return
    ensure_dir(plots_dir)
    efficiency_rows = load_csv_rows(efficiency_csv) if efficiency_csv else None
    for metric in REPORT_METRICS:
        metric_rows = [row for row in rows if str(row.get("metric")) == metric]
        if not metric_rows:
            continue
        plot_model_ranking(
            metric_rows,
            metric,
            plots_dir / f"ranking_{metric}.png",
            higher_is_better=metric_higher_is_better(metric),
        )
        if efficiency_rows is not None:
            try:
                plot_alignment_vs_efficiency(
                    metric_rows,
                    efficiency_rows,
                    metric,
                    "latency_mean_ms",
                    plots_dir / f"{metric}_vs_latency_mean_ms.png",
                )
            except ValueError:
                pass


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "config_path",
        "dataset",
        "scale",
        "model",
        "saliency_method",
        "saliency_family",
        "status",
        "returncode",
        "runtime_seconds",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(
    info: dict[str, Any],
    status: str,
    start: float,
    error: str,
    *,
    returncode: int | str = "",
) -> dict[str, Any]:
    return {
        "config_path": str(info["config_path"]),
        "dataset": info["dataset"],
        "scale": info["scale"],
        "model": info["model"],
        "saliency_method": info["saliency_method"],
        "saliency_family": info["saliency_family"],
        "status": status,
        "returncode": returncode,
        "runtime_seconds": f"{time.perf_counter() - start:.3f}",
        "error": error,
    }


def _skipped_row(info: dict[str, Any], reason: str) -> dict[str, Any]:
    return _base_row(info, "skipped", time.perf_counter(), reason)


def _dedupe_infos(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for info in infos:
        key = str(info["config_path"])
        if key in seen:
            continue
        seen.add(key)
        output.append(info)
    return output


def _reliability_key(info: dict[str, Any]) -> tuple[str, str, str]:
    return (info["dataset"], info["model"], info["saliency_method"])


def _saliency_family(method_name: str) -> str:
    if method_name in {"vanilla_gradient", "integrated_gradients"}:
        return "evidence_sensitivity"
    if method_name == "occlusion":
        return "perturbation"
    if method_name == "gradcam":
        return "class_localization"
    if method_name in {"attention_rollout", "rollout"}:
        return "internal_routing"
    if method_name in {"center_bias", "random_saliency"}:
        return "baseline"
    if method_name in {"precomputed_map", "deepgaze_precomputed"}:
        return "reference"
    return "unknown"


def _compact_error(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-5:])[:1500]


def _run_streaming(command: list[str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    return process.wait(), "".join(output_lines)


def _planned_run_count(run_order: list[dict[str, Any]], max_runs: int | None) -> int:
    if max_runs is None:
        return len(run_order)
    return min(max_runs, len(run_order))


def _print_config_progress(
    *,
    enabled: bool,
    index: int,
    total: int,
    info: dict[str, Any],
    status: str,
    reason: str = "",
) -> None:
    if not enabled:
        return
    label = (
        f"{info['dataset']} :: {info['model']} + {info['saliency_method']}"
    )
    suffix = f" ({reason})" if reason else ""
    print(f"[progress] V2 matrix {index}/{total}: {status} {label}{suffix}", flush=True)


if __name__ == "__main__":
    main()
