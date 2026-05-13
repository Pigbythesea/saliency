"""Aggregate HMA benchmark result directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from hma.experiments.aggregate_results import (
    aggregate_result_files,
    save_aggregate_table,
)
from hma.utils.paths import ensure_dir
from hma.viz.plot_metrics import (
    load_csv_rows,
    plot_alignment_vs_efficiency,
    plot_model_ranking,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate HMA benchmark results.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result CSV files or directories containing per_image_metrics.csv files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/aggregated/results.csv",
        help="Aggregate CSV output path.",
    )
    parser.add_argument(
        "--plots-dir",
        default="outputs/aggregated/plots",
        help="Directory for optional plot outputs.",
    )
    parser.add_argument(
        "--plot-metric",
        default=None,
        help="Metric to plot. Defaults to the first metric in the aggregate table.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Sort ranking plots with lower metric values first.",
    )
    parser.add_argument(
        "--efficiency-csv",
        default=None,
        help="Optional efficiency CSV for alignment-vs-efficiency scatter plot.",
    )
    parser.add_argument(
        "--efficiency-field",
        default="latency_mean_ms",
        help="Efficiency CSV field for scatter plot x-axis.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only write aggregate CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = aggregate_result_files(args.paths)
    output_path = save_aggregate_table(rows, args.output)
    print(f"Aggregate CSV: {output_path}")

    if args.no_plots or not rows:
        return

    plots_dir = ensure_dir(args.plots_dir)
    metric = args.plot_metric or str(rows[0]["metric"])
    ranking_paths = plot_model_ranking(
        rows,
        metric,
        plots_dir / f"ranking_{metric}.png",
        higher_is_better=not args.lower_is_better,
    )
    print(f"Ranking plot: {ranking_paths[0]} and {ranking_paths[1]}")

    if args.efficiency_csv:
        efficiency_rows = load_csv_rows(args.efficiency_csv)
        scatter_paths = plot_alignment_vs_efficiency(
            rows,
            efficiency_rows,
            metric,
            args.efficiency_field,
            plots_dir / f"{metric}_vs_{args.efficiency_field}.png",
        )
        print(f"Efficiency scatter: {scatter_paths[0]} and {scatter_paths[1]}")


if __name__ == "__main__":
    main()
