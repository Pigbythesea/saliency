"""Matplotlib plots for aggregated HMA metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from hma.utils.paths import ensure_dir


def plot_model_ranking(
    aggregate_rows: Iterable[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    higher_is_better: bool = True,
) -> tuple[Path, Path]:
    """Save PNG and PDF bar plots ranking models by an aggregate metric."""
    plt = _import_pyplot()
    rows = _filter_metric_rows(aggregate_rows, metric)
    if not rows:
        raise ValueError(f"No aggregate rows found for metric '{metric}'")

    rows.sort(key=lambda row: float(row["mean"]), reverse=higher_is_better)
    labels = [_ranking_label(row) for row in rows]
    values = [float(row["mean"]) for row in rows]

    width = max(6.0, min(14.0, 1.0 + len(rows) * 1.4))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(labels, values, color="#3b82f6")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.set_title(f"Model ranking by {metric}")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    return _save_png_and_pdf(fig, output_path, plt)


def plot_alignment_vs_efficiency(
    aggregate_rows: Iterable[dict[str, Any]],
    efficiency_rows: Iterable[dict[str, Any]] | str | Path,
    metric: str,
    efficiency_field: str,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and PDF scatter plots for alignment metric vs efficiency field."""
    plt = _import_pyplot()
    metric_rows = _filter_metric_rows(aggregate_rows, metric)
    efficiency_by_model = _index_efficiency_rows(efficiency_rows)

    points: list[tuple[str, float, float]] = []
    for row in metric_rows:
        model = str(row.get("model", ""))
        efficiency = efficiency_by_model.get(model)
        if not efficiency:
            continue
        x_value = _parse_float(efficiency.get(efficiency_field))
        y_value = _parse_float(row.get("mean"))
        if x_value is not None and y_value is not None:
            points.append((model, x_value, y_value))

    if not points:
        raise ValueError(
            "No matching aggregate and efficiency rows found for "
            f"metric '{metric}' and field '{efficiency_field}'"
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_values = [point[1] for point in points]
    y_values = [point[2] for point in points]
    ax.scatter(x_values, y_values, color="#ef4444")
    for model, x_value, y_value in points:
        ax.annotate(model, (x_value, y_value), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(efficiency_field)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {efficiency_field}")
    fig.tight_layout()
    return _save_png_and_pdf(fig, output_path, plt)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load rows from a CSV file for plotting helpers."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _filter_metric_rows(
    rows: Iterable[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("metric")) == metric]


def _index_efficiency_rows(
    rows_or_path: Iterable[dict[str, Any]] | str | Path,
) -> dict[str, dict[str, Any]]:
    rows = load_csv_rows(rows_or_path) if isinstance(rows_or_path, (str, Path)) else rows_or_path
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row.get("model") or row.get("model_name")
        if model:
            indexed[str(model)] = row
    return indexed


def _ranking_label(row: dict[str, Any]) -> str:
    label = str(row.get("model", "unknown"))
    saliency_method = str(row.get("saliency_method", "unknown"))
    dataset = str(row.get("dataset", "unknown"))
    if saliency_method != "unknown":
        label = f"{label}\n{saliency_method}"
    if dataset != "unknown":
        label = f"{label}\n{dataset}"
    return label


def _save_png_and_pdf(fig: Any, output_path: str | Path, plt: Any) -> tuple[Path, Path]:
    path = Path(output_path).expanduser()
    ensure_dir(path.parent)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path.resolve(), pdf_path.resolve()


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with the analysis extra: "
            'uv pip install -e ".[analysis]"'
        ) from exc
    return plt
