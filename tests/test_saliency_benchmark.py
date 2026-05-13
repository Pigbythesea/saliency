import csv
import json

from hma.experiments import run_saliency_benchmark
from hma.utils.config import save_yaml


def test_saliency_benchmark_writes_csv_json_and_visualizations(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "outputs"
    save_yaml(
        {
            "seed": 123,
            "device": "cpu",
            "dataset": {
                "name": "dummy_static_saliency",
                "max_items": 2,
                "image_shape": [3, 8, 8],
                "map_shape": [8, 8],
                "seed": 5,
            },
            "model": {
                "name": "dummy_vision_encoder",
                "noise_scale": 0.0,
                "seed": 7,
            },
            "saliency": {"method": "dummy_gradient_free"},
            "metrics": ["nss", "cc", "similarity", "kl"],
            "output": {
                "dir": str(output_dir),
                "save_visualizations": True,
                "num_visualizations": 1,
            },
        },
        config_path,
    )

    aggregate = run_saliency_benchmark(config_path)

    per_image_csv = output_dir / "per_image_metrics.csv"
    aggregate_json = output_dir / "aggregate_metrics.json"
    visualization = output_dir / "visualizations" / "val_0000.png"

    assert per_image_csv.is_file()
    assert aggregate_json.is_file()
    assert visualization.is_file()
    assert aggregate["num_items"] == 2
    assert set(aggregate["metrics"]) == {"nss", "cc", "similarity", "kl"}

    with per_image_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["image_id"] == "val_0000"
    assert set(rows[0]) == {"image_id", "image_path", "nss", "cc", "similarity", "kl"}

    saved = json.loads(aggregate_json.read_text(encoding="utf-8"))
    assert saved["num_items"] == 2
    assert saved["dataset"] == "dummy_static_saliency"
    assert saved["model"] == "dummy_vision_encoder"
    assert saved["saliency_method"] == "dummy_gradient_free"
