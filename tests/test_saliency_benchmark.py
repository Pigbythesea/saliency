import csv
import json

import numpy as np
import pytest
from PIL import Image

import hma.experiments.saliency_benchmark as benchmark_module
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


@pytest.mark.parametrize(
    ("method", "saliency_extra", "expected_family"),
    [
        ("vanilla_gradient", {}, "evidence_sensitivity"),
        ("integrated_gradients", {"steps": 2}, "evidence_sensitivity"),
        ("gradcam", {"target_layer": "features.0"}, "class_localization"),
    ],
)
def test_saliency_benchmark_runs_real_torch_path(
    tmp_path,
    monkeypatch,
    method,
    saliency_extra,
    expected_family,
):
    torch = pytest.importorskip("torch")

    class TinyDataset:
        def __iter__(self):
            for index in range(2):
                image = Image.fromarray(
                    np.full((8, 8, 3), 32 + index * 64, dtype=np.uint8),
                    mode="RGB",
                )
                fixation_map = np.zeros((8, 8), dtype=np.float32)
                fixation_map[3:5, 3:5] = 1.0
                yield {
                    "image": image,
                    "image_id": f"tiny_{index}",
                    "image_path": f"tiny_{index}.png",
                    "fixation_map": fixation_map,
                    "metadata": {"dataset": "tiny"},
                }

    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, images):
            features = self.features(images)
            pooled = self.pool(features).flatten(1)
            return self.classifier(pooled)

    class TinyWrapper:
        def __init__(self):
            torch.manual_seed(0)
            self.model = TinyCNN().eval()

        def get_last_logits(self, images):
            return self.model(images)

    monkeypatch.setattr(benchmark_module, "build_dataset", lambda _config: TinyDataset())
    monkeypatch.setattr(benchmark_module, "build_model", lambda _config: TinyWrapper())

    config_path = tmp_path / f"{method}.yaml"
    output_dir = tmp_path / method
    saliency_config = {"method": method, **saliency_extra}
    save_yaml(
        {
            "device": "cpu",
            "dataset": {"name": "tiny_static", "split": "val"},
            "model": {"name": "tiny_torch"},
            "preprocessing": {
                "input_size": [8, 8],
                "mean": "none",
                "std": "none",
            },
            "saliency": saliency_config,
            "metrics": ["nss", "auc_judd", "cc", "similarity", "kl"],
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )

    aggregate = run_saliency_benchmark(config_path)

    assert (output_dir / "per_image_metrics.csv").is_file()
    assert (output_dir / "aggregate_metrics.json").is_file()
    assert aggregate["num_items"] == 2
    assert aggregate["saliency_family"] == expected_family
    assert set(aggregate["metrics"]) == {"nss", "auc_judd", "cc", "similarity", "kl"}


def test_saliency_benchmark_cache_writes_reuses_and_invalidates(tmp_path, monkeypatch):
    calls = {"count": 0}

    def counting_saliency(_model, _image, target_map=None, **_kwargs):
        calls["count"] += 1
        return np.asarray(target_map, dtype=np.float32)

    monkeypatch.setattr(
        benchmark_module,
        "build_saliency_method",
        lambda _config: counting_saliency,
    )

    def write_config(path, method):
        save_yaml(
            {
                "device": "cpu",
                "dataset": {
                    "name": "dummy_static_saliency",
                    "max_items": 2,
                    "image_shape": [3, 8, 8],
                    "map_shape": [8, 8],
                    "seed": 5,
                },
                "model": {"name": "dummy_vision_encoder"},
                "saliency": {"method": method},
                "cache": {"enabled": True, "dir": "saliency_maps", "reuse": True},
                "metrics": ["cc"],
                "output": {"dir": str(tmp_path / "cached_outputs")},
            },
            path,
        )

    first_config = tmp_path / "first.yaml"
    write_config(first_config, "center_bias")

    first = run_saliency_benchmark(first_config)
    assert calls["count"] == 2
    assert first["cache_hits"] == 0
    assert first["cache_writes"] == 2
    assert len(list((tmp_path / "cached_outputs" / "saliency_maps").glob("*.npy"))) == 2

    second = run_saliency_benchmark(first_config)
    assert calls["count"] == 2
    assert second["cache_hits"] == 2
    assert second["cache_writes"] == 0

    changed_config = tmp_path / "changed.yaml"
    write_config(changed_config, "random_saliency")
    changed = run_saliency_benchmark(changed_config)
    assert calls["count"] == 4
    assert changed["cache_hits"] == 0
    assert changed["cache_writes"] == 2
