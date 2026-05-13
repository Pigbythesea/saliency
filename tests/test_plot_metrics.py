import csv

from hma.viz.plot_metrics import plot_alignment_vs_efficiency, plot_model_ranking


def test_plot_model_ranking_saves_png_and_pdf(tmp_path):
    rows = [
        {
            "dataset": "dummy",
            "model": "model_a",
            "saliency_method": "gradient",
            "metric": "nss",
            "mean": 0.7,
        },
        {
            "dataset": "dummy",
            "model": "model_b",
            "saliency_method": "gradient",
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
            "metric": "nss",
            "mean": 0.7,
        },
        {
            "dataset": "dummy",
            "model": "model_b",
            "saliency_method": "gradient",
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
