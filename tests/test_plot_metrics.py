import csv

from hma.viz.plot_metrics import (
    metric_higher_is_better,
    plot_alignment_vs_efficiency,
    plot_model_ranking,
)


def test_plot_model_ranking_saves_png_and_pdf(tmp_path):
    rows = [
        {
            "dataset": "dummy",
            "model": "model_a",
            "saliency_method": "gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "nss",
            "mean": 0.7,
        },
        {
            "dataset": "dummy",
            "model": "model_b",
            "saliency_method": "gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "nss",
            "mean": 0.4,
        },
    ]

    png_path, pdf_path = plot_model_ranking(rows, "nss", tmp_path / "ranking.png")

    assert png_path.is_file()
    assert pdf_path.is_file()


def test_plot_alignment_vs_efficiency_saves_png_and_pdf(tmp_path):
    aggregate_rows = [
        {
            "dataset": "dummy",
            "model": "model_a",
            "saliency_method": "gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "nss",
            "mean": 0.7,
        },
        {
            "dataset": "dummy",
            "model": "model_b",
            "saliency_method": "gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "nss",
            "mean": 0.4,
        },
    ]
    efficiency_csv = tmp_path / "efficiency.csv"
    with efficiency_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "latency_mean_ms"])
        writer.writeheader()
        writer.writerow({"model_name": "model_a", "latency_mean_ms": 2.0})
        writer.writerow({"model_name": "model_b", "latency_mean_ms": 5.0})

    png_path, pdf_path = plot_alignment_vs_efficiency(
        aggregate_rows,
        efficiency_csv,
        "nss",
        "latency_mean_ms",
        tmp_path / "scatter.png",
    )

    assert png_path.is_file()
    assert pdf_path.is_file()


def test_matrix_style_plots_handle_multiple_datasets_and_methods(tmp_path):
    rows = [
        {
            "dataset": "salicon_pilot500",
            "model": "resnet50",
            "saliency_method": "vanilla_gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "auc_judd",
            "mean": 0.62,
        },
        {
            "dataset": "cat2000_pilot500",
            "model": "resnet50",
            "saliency_method": "gradcam",
            "saliency_family": "class_localization",
            "metric": "auc_judd",
            "mean": 0.75,
        },
        {
            "dataset": "coco_search18_pilot500",
            "model": "convnext_tiny",
            "saliency_method": "vanilla_gradient",
            "saliency_family": "evidence_sensitivity",
            "metric": "auc_judd",
            "mean": 0.73,
        },
    ]
    efficiency_csv = tmp_path / "efficiency.csv"
    with efficiency_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "latency_mean_ms"])
        writer.writeheader()
        writer.writerow({"model_name": "resnet50", "latency_mean_ms": 55.0})
        writer.writerow({"model_name": "convnext_tiny", "latency_mean_ms": 58.0})

    ranking_png, ranking_pdf = plot_model_ranking(
        rows,
        "auc_judd",
        tmp_path / "matrix_ranking.png",
    )
    scatter_png, scatter_pdf = plot_alignment_vs_efficiency(
        rows,
        efficiency_csv,
        "auc_judd",
        "latency_mean_ms",
        tmp_path / "matrix_scatter.png",
    )

    assert ranking_png.is_file()
    assert ranking_pdf.is_file()
    assert scatter_png.is_file()
    assert scatter_pdf.is_file()


def test_lower_is_better_metric_direction_is_known():
    assert metric_higher_is_better("nss")
    assert not metric_higher_is_better("kl")
    assert not metric_higher_is_better("emd")
