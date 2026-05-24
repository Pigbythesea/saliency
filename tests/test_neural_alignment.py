import csv
import json

import numpy as np
import pytest

from hma.experiments.neural_alignment import run_neural_alignment
from hma.neural import (
    benchmark_encoding_target_scores,
    compare_rdms,
    compute_rdm,
    evaluate_encoding,
    fit_ridge_encoding,
    predict_ridge_encoding,
    save_activations,
)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_ridge_encoding_recovers_signal_better_than_shuffled_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    weights = rng.normal(size=(6, 4))
    Y = X @ weights + 0.05 * rng.normal(size=(80, 4))

    model = fit_ridge_encoding(X[:60], Y[:60], alpha=0.1)
    predictions = predict_ridge_encoding(model, X[60:])
    scores = evaluate_encoding(predictions, Y[60:], metric="correlation")
    shuffled_scores = evaluate_encoding(predictions, Y[60:][::-1], metric="correlation")

    assert scores.shape == (4,)
    assert scores.mean() > shuffled_scores.mean()
    assert scores.mean() > 0.8


def test_ridge_encoding_r2_shape_and_finite_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 2))
    model = fit_ridge_encoding(X[:10], Y[:10], alpha=1.0)

    scores = evaluate_encoding(model, X[10:], Y[10:], metric="r2")

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_benchmark_encoding_target_scores_reports_squared_r_and_r2():
    target = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    predictions = target.copy()

    rows = benchmark_encoding_target_scores(predictions, target)

    assert len(rows) == 2
    assert rows[0]["pearson_r"] == pytest.approx(1.0)
    assert rows[0]["r2_score_from_r"] == pytest.approx(1.0)
    assert rows[0]["prediction_r2"] == pytest.approx(1.0)
    assert rows[0]["noise_normalized_score"] == ""
    assert rows[0]["metric_scope"] == "benchmark_style_non_noise_normalized"


def test_benchmark_encoding_target_scores_marks_zero_variance():
    target = np.array([[1.0], [1.0], [1.0]])
    predictions = np.array([[0.1], [0.2], [0.3]])

    row = benchmark_encoding_target_scores(predictions, target)[0]

    assert row["pearson_r"] == 0.0
    assert row["r2_score_from_r"] == 0.0
    assert row["prediction_r2"] == 0.0
    assert row["valid_prediction_variance"] == "true"
    assert row["valid_target_variance"] == "false"


def test_benchmark_encoding_target_scores_noise_normalizes_when_ceiling_exists():
    target = np.array([[0.0], [1.0], [2.0], [3.0]])
    predictions = target.copy()

    row = benchmark_encoding_target_scores(
        predictions,
        target,
        noise_ceiling=np.array([0.5]),
    )[0]

    assert row["noise_ceiling"] == pytest.approx(0.5)
    assert row["noise_normalized_score"] == pytest.approx(2.0)
    assert row["valid_noise_ceiling"] == "true"
    assert row["metric_scope"] == "benchmark_style_noise_normalized"


def test_benchmark_encoding_target_scores_excludes_zero_and_invalid_ceilings():
    target = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    predictions = target.copy()

    rows = benchmark_encoding_target_scores(
        predictions,
        target,
        noise_ceiling=np.array([0.0, -1.0, np.nan]),
    )

    assert [row["valid_noise_ceiling"] for row in rows] == ["false", "false", "false"]
    assert all(row["noise_normalized_score"] == "" for row in rows)
    assert all(row["metric_scope"] == "benchmark_style_non_noise_normalized" for row in rows)


def test_rdm_shape_symmetry_and_comparison():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(6, 5))

    rdm = compute_rdm(features, metric="correlation")
    euclidean = compute_rdm(features, metric="euclidean")
    similarity = compare_rdms(rdm, rdm, method="spearman")

    assert rdm.shape == (6, 6)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)
    assert euclidean.shape == (6, 6)
    assert np.isfinite(similarity)
    assert similarity > 0.99


def test_save_activations_writes_npz(tmp_path):
    output = save_activations(
        {
            "image_ids": np.array(["a", "b"], dtype=object),
            "layer1": np.ones((2, 3), dtype=np.float32),
        },
        tmp_path / "activations",
    )

    assert output.suffix == ".npz"
    loaded = np.load(output, allow_pickle=True)
    assert set(loaded.files) == {"image_ids", "layer1"}


def test_neural_runner_moves_model_to_resolved_device(monkeypatch, tmp_path):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    moved = {}

    class DeviceAwareDummy:
        def to(self, device):
            moved["device"] = device

        def get_features(self, images, layers=None):
            return {"embedding": np.ones((1, 3), dtype=np.float32)}

    config_path = tmp_path / "neural.yaml"
    save_yaml(
        {
            "seed": 123,
            "device": "cpu",
            "dataset": {
                "name": "dummy_static_saliency",
                "label": "dummy_neural",
                "num_items": 3,
                "image_shape": [3, 8, 8],
                "map_shape": [8, 8],
                "roi_response_dim": 2,
            },
            "model": {"name": "dummy_vision_encoder"},
            "preprocessing": {"input_size": [8, 8], "mean": "none", "std": "none"},
            "neural": {"layers": ["embedding"], "response_key": "roi_responses"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: DeviceAwareDummy())

    neural_alignment.run_neural_alignment(config_path)

    assert moved["device"] == "cpu"


def test_run_neural_alignment_smoke_writes_outputs(tmp_path):
    from hma.utils.config import save_yaml

    config_path = tmp_path / "neural.yaml"
    output_dir = tmp_path / "outputs"
    save_yaml(
        {
            "seed": 123,
            "device": "cpu",
            "dataset": {
                "name": "dummy_static_saliency",
                "label": "dummy_neural",
                "num_items": 8,
                "image_shape": [3, 8, 8],
                "map_shape": [8, 8],
                "roi_response_dim": 3,
                "seed": 2,
            },
            "model": {"name": "dummy_vision_encoder", "noise_scale": 0.0},
            "preprocessing": {"input_size": [8, 8], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "feature_reduction": "spatial_mean",
                "train_fraction": 0.75,
                "ridge_alpha": 1.0,
                "metric": "correlation",
                "rsa": {
                    "enabled": True,
                    "rdm_metric": "correlation",
                    "response_rdm_metric": "correlation",
                    "compare_method": "spearman",
                },
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )

    result = run_neural_alignment(config_path)

    assert (output_dir / "activations.npz").is_file()
    assert (output_dir / "encoding_scores.csv").is_file()
    assert (output_dir / "encoding_target_scores.csv").is_file()
    assert (output_dir / "rsa_scores.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert result["num_items"] == 8
    assert result["model_name"] == "dummy_vision_encoder"
    assert result["model_backend"] == "timm"
    assert result["model_pretrained"] is False
    assert result["score_rows"][0]["layer"] == "embedding"
    assert result["score_rows"][0]["alpha_selection_mode"] == "fixed"
    assert result["score_rows"][0]["metric_scope"] == "benchmark_style_non_noise_normalized"
    assert result["target_score_rows"]
    assert result["score_rows"][0]["dataset"] == "dummy_neural"
    assert result["score_rows"][0]["roi"] == "dummy_roi"
    assert result["rsa_rows"][0]["layer"] == "embedding"
    assert result["rsa_scores"].endswith("rsa_scores.csv")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "dummy_vision_encoder"
    assert metadata["model_backend"] == "timm"
    assert metadata["model_pretrained"] is False
    assert metadata["metric_scope"] == "benchmark_style_non_noise_normalized"
    assert metadata["alpha_selection_modes"] == ["fixed"]
    target_rows = _read_csv(output_dir / "encoding_target_scores.csv")
    assert {row["metric_scope"] for row in target_rows} == {
        "benchmark_style_non_noise_normalized"
    }


def test_run_neural_alignment_noise_normalizes_with_ceiling_metadata(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 12

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array([feature, 2.0 * feature], dtype=np.float32),
                    "noise_ceiling": np.array([0.5, 0.25], dtype=np.float32),
                    "noise_ceiling_source": "synthetic_test_ceiling",
                },
            }

    class SyntheticModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and isinstance(images, torch.Tensor):
                feature = float(images.mean().detach().cpu().numpy())
            else:
                feature = float(np.asarray(images).mean())
            return {layer: np.array([[feature]], dtype=np.float32) for layer in layers}

    config_path = tmp_path / "neural_ceiling.yaml"
    output_dir = tmp_path / "outputs"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "train_fraction": 0.75,
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SyntheticModel())

    result = run_neural_alignment(config_path)

    assert result["noise_ceiling_available"] is True
    assert result["noise_ceiling_source"] == "synthetic_test_ceiling"
    assert result["score_rows"][0]["metric_scope"] == "benchmark_style_noise_normalized"
    assert result["score_rows"][0]["valid_noise_ceiling_targets"] == 2
    assert result["score_rows"][0]["zero_noise_ceiling_targets"] == 0
    assert result["score_rows"][0]["invalid_noise_ceiling_targets"] == 0
    assert result["score_rows"][0]["mean_noise_normalized_score"]
    target_rows = _read_csv(output_dir / "encoding_target_scores.csv")
    assert {row["metric_scope"] for row in target_rows} == {
        "benchmark_style_noise_normalized"
    }
    assert all(row["noise_ceiling"] for row in target_rows)
    assert all(row["noise_normalized_score"] for row in target_rows)
    assert all(row["valid_noise_ceiling"] == "true" for row in target_rows)


def test_run_neural_alignment_counts_zero_noise_ceiling_targets(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 12

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array([feature, 2.0 * feature], dtype=np.float32),
                    "noise_ceiling": np.array([0.5, 0.0], dtype=np.float32),
                },
            }

    class SyntheticModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            feature = float(images.mean().detach().cpu().numpy())
            return {layer: np.array([[feature]], dtype=np.float32) for layer in layers}

    config_path = tmp_path / "neural_zero_ceiling.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {"layers": ["embedding"], "response_key": "roi_responses"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SyntheticModel())

    result = run_neural_alignment(config_path)

    row = result["score_rows"][0]
    assert row["valid_noise_ceiling_targets"] == 1
    assert row["zero_noise_ceiling_targets"] == 1
    assert row["invalid_noise_ceiling_targets"] == 0
    target_rows = result["target_score_rows"]
    assert [target["valid_noise_ceiling"] for target in target_rows] == ["true", "false"]
    assert target_rows[1]["noise_normalized_score"] == ""


def test_run_neural_alignment_rejects_noise_ceiling_length_mismatch(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 4

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index)
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "metadata": {
                    "roi_responses": np.array([feature, feature + 1.0], dtype=np.float32),
                    "noise_ceiling": np.array([0.5], dtype=np.float32),
                },
            }

    class SyntheticModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            return {layer: np.ones((1, 1), dtype=np.float32) for layer in layers}

    config_path = tmp_path / "neural_bad_ceiling.yaml"
    save_yaml(
        {
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {"layers": ["embedding"], "response_key": "roi_responses"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SyntheticModel())

    with pytest.raises(ValueError, match="one value per target"):
        run_neural_alignment(config_path)


def test_run_neural_alignment_rejects_partially_missing_noise_ceiling(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 4

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index)
            metadata = {
                "roi_responses": np.array([feature, feature + 1.0], dtype=np.float32),
            }
            if index != 0:
                metadata["noise_ceiling"] = np.array([0.5, 0.7], dtype=np.float32)
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "metadata": metadata,
            }

    class SyntheticModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            return {layer: np.ones((1, 1), dtype=np.float32) for layer in layers}

    config_path = tmp_path / "neural_partial_ceiling.yaml"
    save_yaml(
        {
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {"layers": ["embedding"], "response_key": "roi_responses"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SyntheticModel())

    with pytest.raises(ValueError, match="partially missing"):
        run_neural_alignment(config_path)


def test_run_neural_alignment_selects_cv_ridge_alpha(monkeypatch, tmp_path):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 30

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 100.0
            image = np.full((3, 4, 4), feature, dtype=np.float32)
            return {
                "image": image,
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array(
                        [2.0 * feature + 0.1, -feature + 0.2],
                        dtype=np.float32,
                    ),
                },
            }

    class SyntheticModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and isinstance(images, torch.Tensor):
                feature = images.mean().detach().cpu().numpy()
            else:
                feature = np.asarray(images).mean()
            array = np.array([[feature, feature * feature]], dtype=np.float32)
            return {layer: array for layer in (layers or ["embedding"])}

    config_path = tmp_path / "neural_cv.yaml"
    output_dir = tmp_path / "outputs"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "train_fraction": 0.8,
                "ridge_alpha": 1.0,
                "ridge_alphas": [1000000.0, 0.001],
                "validation_fraction": 0.25,
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SyntheticModel())

    result = run_neural_alignment(config_path)

    row = result["score_rows"][0]
    assert row["alpha_selection_mode"] == "cv_inner_validation"
    assert row["selected_ridge_alpha"] == pytest.approx(0.001)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_ridge_alphas"]["embedding"] == pytest.approx(0.001)


def test_flatten_pca_fits_on_train_rows_only():
    from hma.experiments import neural_alignment

    train_idx = np.array([0, 1, 2, 3])
    features = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    altered_test_features = features.copy()
    altered_test_features[4:] = np.array(
        [[100.0, -50.0, 25.0, 1.0], [-100.0, 75.0, -25.0, 1.0]],
        dtype=np.float32,
    )

    reduced, metadata = neural_alignment._fit_transform_flatten_pca(
        features,
        layer="embedding",
        input_shape=[2, 2],
        train_idx=train_idx,
        pca_components=2,
        pca_solver="full",
        pca_whiten=False,
        random_seed=0,
    )
    altered_reduced, altered_metadata = neural_alignment._fit_transform_flatten_pca(
        altered_test_features,
        layer="embedding",
        input_shape=[2, 2],
        train_idx=train_idx,
        pca_components=2,
        pca_solver="full",
        pca_whiten=False,
        random_seed=0,
    )

    assert np.allclose(reduced[train_idx], altered_reduced[train_idx])
    assert metadata["train_only_fit"] is True
    assert metadata["effective_components"] == 2
    assert altered_metadata["n_train_fit"] == len(train_idx)


def test_run_neural_alignment_flatten_pca_smoke_writes_reduced_outputs(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 10

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index)
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array(
                        [feature, feature * 0.5 + 1.0],
                        dtype=np.float32,
                    ),
                },
            }

    class SpatialFeatureModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            base = np.array(
                [[[value, value + 1.0, value + 2.0], [value * 2.0, 1.0, -value]]],
                dtype=np.float32,
            )
            return {layer: base for layer in layers}

    config_path = tmp_path / "neural_flatten_pca.yaml"
    output_dir = tmp_path / "outputs"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "feature_reduction": "flatten_pca",
                "pca_components": 3,
                "pca_solver": "full",
                "train_fraction": 0.6,
                "ridge_alpha": 1.0,
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SpatialFeatureModel())

    result = run_neural_alignment(config_path)

    assert (output_dir / "activations.npz").is_file()
    assert (output_dir / "encoding_scores.csv").is_file()
    assert (output_dir / "encoding_target_scores.csv").is_file()
    metadata_path = output_dir / "feature_reduction_metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layer_metadata = metadata["layers"][0]
    assert layer_metadata["method"] == "flatten_pca"
    assert layer_metadata["input_feature_shape"] == [2, 3]
    assert layer_metadata["effective_components"] == 3
    assert layer_metadata["train_only_fit"] is True
    loaded = np.load(output_dir / "activations.npz", allow_pickle=True)
    assert loaded["embedding"].shape == (10, 3)
    assert result["feature_reduction_metadata"] == str(metadata_path)


def test_run_neural_alignment_flatten_pca_requires_components(tmp_path):
    from hma.utils.config import save_yaml

    config_path = tmp_path / "missing_pca_components.yaml"
    save_yaml(
        {
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "neural": {"feature_reduction": "flatten_pca"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )

    with pytest.raises(ValueError, match="pca_components is required"):
        run_neural_alignment(config_path)


def test_run_neural_alignment_flatten_pca_rejects_too_many_components(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    class SyntheticDataset:
        def __len__(self):
            return 6

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index)
            return {
                "image": np.full((3, 4, 4), feature, dtype=np.float32),
                "metadata": {
                    "roi_responses": np.array([feature, feature + 1.0], dtype=np.float32),
                },
            }

    class FourFeatureModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            return {
                layer: np.array([[value, value + 1.0, value + 2.0, value + 3.0]], dtype=np.float32)
                for layer in layers
            }

    config_path = tmp_path / "too_many_pca_components.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "feature_reduction": "flatten_pca",
                "pca_components": 4,
                "pca_solver": "full",
                "train_fraction": 0.5,
            },
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: FourFeatureModel())

    with pytest.raises(ValueError, match="pca_components must be <= min"):
        run_neural_alignment(config_path)


def test_run_neural_alignment_missing_roi_response_fails(tmp_path):
    from hma.utils.config import save_yaml

    config_path = tmp_path / "missing.yaml"
    save_yaml(
        {
            "device": "cpu",
            "dataset": {
                "name": "dummy_static_saliency",
                "num_items": 3,
                "image_shape": [3, 8, 8],
                "map_shape": [8, 8],
            },
            "model": {"name": "dummy_vision_encoder"},
            "preprocessing": {"input_size": [8, 8], "mean": "none", "std": "none"},
            "neural": {"layers": ["embedding"], "response_key": "roi_responses"},
            "output": {"dir": str(tmp_path / "outputs")},
        },
        config_path,
    )

    with pytest.raises(ValueError, match="missing neural response"):
        run_neural_alignment(config_path)
