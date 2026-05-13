"""Matplotlib plots for aggregated HMA metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

from hma.utils.paths import ensure_dir


LOWER_IS_BETTER_METRICS = {"kl", "kl_divergence", "emd", "emd_2d", "mae"}


MODEL_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#f59e0b",
    "#0891b2",
    "#be123c",
    "#4b5563",
]
METHOD_COLORS = {
    "center_bias": "#0f766e",
    "random_saliency": "#9ca3af",
    "vanilla_gradient": "#2563eb",
    "gradcam": "#dc2626",
    "integrated_gradients": "#7c3aed",
    "attention_rollout": "#f59e0b",
    "rollout": "#f59e0b",
}
METHOD_MARKERS = {
    "vanilla_gradient": "o",
    "gradcam": "s",
    "integrated_gradients": "D",
    "attention_rollout": "P",
    "rollout": "P",
}


def plot_model_ranking(
    aggregate_rows: Iterable[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    higher_is_better: bool = True,
) -> tuple[Path, Path]:
    """Save faceted horizontal ranking plots grouped by dataset."""
    plt = _import_pyplot()
    rows = _filter_metric_rows(aggregate_rows, metric)
    if not rows:
        raise ValueError(f"No aggregate rows found for metric '{metric}'")

    grouped = _group_by_facet(rows)
    datasets = sorted(grouped)
    max_group_size = max(len(values) for values in grouped.values())
    fig_height = max(4.0, min(18.0, 1.8 + len(datasets) * (1.0 + max_group_size * 0.22)))
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11.5, fig_height),
        squeeze=False,
    )

    for ax, dataset in zip(axes[:, 0], datasets):
        dataset_rows = sorted(
            grouped[dataset],
            key=lambda row: float(row["mean"]),
            reverse=higher_is_better,
        )
        labels = _ranking_labels_for_dataset(dataset_rows)
        values = [float(row["mean"]) for row in dataset_rows]
        methods = [str(row.get("saliency_method", "unknown")) for row in dataset_rows]
        colors = [_method_color(method) for method in methods]

        positions = list(range(len(dataset_rows)))
        bars = ax.barh(positions, values, color=colors, alpha=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(_facet_title(dataset), loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(metric)
        ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
        ax.margins(x=0.08)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            _label_bar(ax, bar, value)

    _add_method_color_legend(fig, rows, plt)
    fig.suptitle(f"{metric} Ranking By Dataset", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.82, 0.98))
    return _save_png_and_pdf(fig, output_path, plt)


def plot_alignment_vs_efficiency(
    aggregate_rows: Iterable[dict[str, Any]],
    efficiency_rows: Iterable[dict[str, Any]] | str | Path,
    metric: str,
    efficiency_field: str,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Save faceted efficiency scatter plots grouped by dataset."""
    plt = _import_pyplot()
    metric_rows = _filter_metric_rows(aggregate_rows, metric)
    efficiency_by_model = _index_efficiency_rows(efficiency_rows)
    model_color = _model_color_map(metric_rows)

    points = []
    for row in metric_rows:
        model = str(row.get("model", ""))
        efficiency = efficiency_by_model.get(model)
        if not efficiency:
            continue
        x_value = _parse_float(efficiency.get(efficiency_field))
        y_value = _parse_float(row.get("mean"))
        if x_value is None or y_value is None:
            continue
        points.append(
            {
                "dataset": str(row.get("dataset", "unknown")),
                "model": model,
                "saliency_method": str(row.get("saliency_method", "unknown")),
                "saliency_family": str(row.get("saliency_family", "unknown")),
                "x": x_value,
                "y": y_value,
            }
        )

    if not points:
        raise ValueError(
            "No matching aggregate and efficiency rows found for "
            f"metric '{metric}' and field '{efficiency_field}'"
        )

    grouped = _group_by_facet(points)
    datasets = sorted(grouped)
    fig_height = max(4.0, 2.8 * len(datasets))
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(10.8, fig_height),
        squeeze=False,
        sharex=True,
    )

    for ax, dataset in zip(axes[:, 0], datasets):
        for point in grouped[dataset]:
            method = point["saliency_method"]
            ax.scatter(
                point["x"],
                point["y"],
                color=model_color.get(point["model"], "#6b7280"),
                marker=_method_marker(method),
                s=72,
                alpha=0.88,
                edgecolors="#111827",
                linewidths=0.45,
            )
        ax.set_title(_facet_title(dataset), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel(efficiency_field)
    _add_model_color_legend(fig, model_color, plt)
    _add_method_marker_legend(fig, points, plt)
    fig.suptitle(f"{metric} vs {efficiency_field}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.78, 0.98))
    return _save_png_and_pdf(fig, output_path, plt)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load rows from a CSV file for plotting helpers."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _filter_metric_rows(
    rows: Iterable[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if str(row.get("metric")) != metric:
            continue
        mean = _parse_float(row.get("mean"))
        if mean is None:
            continue
        filtered.append(row)
    return filtered


def _group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return grouped


def _group_by_facet(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _facet_key(row)
        grouped.setdefault(key, []).append(row)
    return grouped


def metric_higher_is_better(metric: str) -> bool:
    return str(metric) not in LOWER_IS_BETTER_METRICS


def _facet_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "unknown"))
    family = str(row.get("saliency_family", "unknown"))
    return f"{dataset}::{family}"


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


def _model_color_map(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    models = sorted(
        {
            str(row.get("model", "unknown"))
            for row in rows
            if str(row.get("saliency_family", "")) != "baseline"
        }
    )
    return {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#6b7280")


def _method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def _label_bar(ax: Any, bar: Any, value: float) -> None:
    width = bar.get_width()
    x_limits = ax.get_xlim()
    span = abs(x_limits[1] - x_limits[0]) or 1.0
    offset = span * 0.01
    if width >= 0:
        x = width + offset
        ha = "left"
    else:
        x = offset
        ha = "left"
    ax.text(
        x,
        bar.get_y() + bar.get_height() / 2.0,
        _format_value(value),
        va="center",
        ha=ha,
        fontsize=7,
        color="#374151",
    )


def _ranking_labels_for_dataset(rows: list[dict[str, Any]]) -> list[str]:
    model_counts: dict[str, int] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        model_counts[model] = model_counts.get(model, 0) + 1

    labels = []
    for row in rows:
        model = str(row.get("model", "unknown"))
        label = _short_model(model)
        if model_counts[model] > 1:
            label = f"{label} ({_short_method(str(row.get('saliency_method', 'unknown')))})"
        labels.append(label)
    return labels


def _add_method_color_legend(fig: Any, rows: list[dict[str, Any]], plt: Any) -> None:
    handles = []
    for method in sorted({str(row.get("saliency_method", "unknown")) for row in rows}):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=_method_color(method),
                markeredgecolor=_method_color(method),
                markersize=8,
                label=_short_method(method),
            )
        )
    fig.legend(
        handles=handles,
        title="Saliency method",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )


def _add_model_color_legend(fig: Any, model_color: dict[str, str], plt: Any) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=_short_model(model),
        )
        for model, color in model_color.items()
    ]
    fig.legend(
        handles=handles,
        title="Model",
        loc="center right",
        bbox_to_anchor=(0.995, 0.62),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )


def _add_method_marker_legend(fig: Any, points: list[dict[str, Any]], plt: Any) -> None:
    methods = sorted({str(point["saliency_method"]) for point in points})
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=_method_marker(method),
            linestyle="",
            markerfacecolor="#d1d5db",
            markeredgecolor="#111827",
            markersize=7,
            label=_short_method(method),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        title="Saliency method",
        loc="center right",
        bbox_to_anchor=(0.995, 0.25),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )


def _short_dataset(dataset: str) -> str:
    replacements = {
        "salicon_pilot500": "SALICON",
        "cat2000_pilot500": "CAT2000",
        "coco_search18_pilot500": "COCO-Search18",
        "salicon": "SALICON",
        "cat2000": "CAT2000",
        "coco_search18": "COCO-Search18",
    }
    return replacements.get(dataset, dataset)


def _facet_title(facet_key: str) -> str:
    dataset, _, family = facet_key.partition("::")
    if family and family != "unknown":
        return f"{_short_dataset(dataset)} - {_short_family(family)}"
    return _short_dataset(dataset)


def _short_family(family: str) -> str:
    replacements = {
        "baseline": "Baseline",
        "evidence_sensitivity": "Evidence sensitivity",
        "class_localization": "Class localization",
        "internal_routing": "Internal routing",
    }
    return replacements.get(family, family)


def _short_method(method: str) -> str:
    replacements = {
        "vanilla_gradient": "Gradient",
        "gradcam": "Grad-CAM",
        "center_bias": "Center bias",
        "random_saliency": "Random",
        "integrated_gradients": "Integrated gradients",
        "attention_rollout": "Attention rollout",
        "rollout": "Attention rollout",
    }
    return replacements.get(method, method)


def _short_model(model: str) -> str:
    replacements = {
        "center_bias_baseline": "Center bias",
        "random_baseline": "Random",
        "vit_base_patch16_224": "ViT-B/16",
        "deit_small_patch16_224": "DeiT-S/16",
        "swin_tiny_patch4_window7_224": "Swin-T",
        "convnext_tiny": "ConvNeXt-T",
    }
    return replacements.get(model, model)


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.3g}"


def _save_png_and_pdf(fig: Any, output_path: str | Path, plt: Any) -> tuple[Path, Path]:
    path = Path(output_path).expanduser()
    ensure_dir(path.parent)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path.resolve(), pdf_path.resolve()


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


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
