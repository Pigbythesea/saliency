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
    fuse_spatial_feature_layers,
    normalize_spatial_features,
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


def test_normalize_spatial_features_accepts_token_and_spatial_layouts():
    token_features = np.zeros((2, 5, 3), dtype=np.float32)
    channels_last = np.zeros((2, 4, 5, 3), dtype=np.float32)
    channels_first = np.zeros((2, 3, 4, 5), dtype=np.float32)

    assert normalize_spatial_features(token_features).shape == (2, 5, 3)
    assert normalize_spatial_features(channels_last, layout="channels_last").shape == (2, 20, 3)
    assert normalize_spatial_features(channels_first, layout="channels_first").shape == (2, 20, 3)


def test_normalize_spatial_features_rejects_non_spatial_matrix():
    with pytest.raises(ValueError, match="learned_spatial_readout expects"):
        normalize_spatial_features(np.zeros((4, 8), dtype=np.float32))


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


def _run_synthetic_selection(
    monkeypatch,
    tmp_path,
    *,
    output_name="outputs",
    include_noise_ceiling=True,
    corrupt_indices=None,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    corrupt_indices = set(corrupt_indices or [])

    class SyntheticDataset:
        def __len__(self):
            return 24

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            image_feature = 100.0 + feature if index in corrupt_indices else feature
            response_feature = -100.0 - feature if index in corrupt_indices else feature
            metadata = {
                "roi": "synthetic_roi",
                "subject_id": "subj_test",
                "roi_responses": np.array(
                    [2.0 * response_feature + 0.1, -response_feature + 0.2],
                    dtype=np.float32,
                ),
            }
            if include_noise_ceiling:
                metadata["noise_ceiling"] = np.array([0.5, 0.25], dtype=np.float32)
            return {
                "image": np.full((3, 4, 4), image_feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": metadata,
            }

    class LayerSelectionModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            item_index = int(round(value * 10.0))
            bad_feature = float((item_index % 3) - 1)
            outputs = {
                "bad": np.array([[bad_feature, -bad_feature, 0.25]], dtype=np.float32),
                "good": np.array([[value, value * value, 1.0]], dtype=np.float32),
            }
            return {layer: outputs[layer] for layer in layers}

    output_dir = tmp_path / output_name
    config_path = tmp_path / f"{output_name}.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "layers": ["bad", "good"],
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "feature_reduction": "flatten_pca",
                "pca_components": 2,
                "pca_solver": "full",
                "train_fraction": 0.75,
                "ridge_alpha": 1.0,
                "ridge_alphas": [0.01, 0.1, 1.0],
                "metric": "correlation",
                "selection": {
                    "enabled": True,
                    "validation_fraction": 0.25,
                    "primary_score": "mean_noise_normalized_score",
                },
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: LayerSelectionModel())

    return neural_alignment.run_neural_alignment(config_path), output_dir


def test_run_neural_alignment_selection_writes_artifacts_and_selected_scores(
    monkeypatch,
    tmp_path,
):
    result, output_dir = _run_synthetic_selection(monkeypatch, tmp_path)

    assert (output_dir / "selection_candidates.csv").is_file()
    assert (output_dir / "selection_artifact.json").is_file()
    assert result["selection_enabled"] is True
    assert result["selected_layer"] == "good"
    assert result["selected_feature_reduction"] == "flatten_pca"
    assert result["selection_candidates"].endswith("selection_candidates.csv")
    assert result["selection_artifact"].endswith("selection_artifact.json")

    candidate_rows = _read_csv(output_dir / "selection_candidates.csv")
    assert len(candidate_rows) == 2
    assert [row["selected"] for row in candidate_rows].count("true") == 1
    assert [row["layer"] for row in candidate_rows] == ["bad", "good"]
    selected_row = [row for row in candidate_rows if row["selected"] == "true"][0]
    assert selected_row["validation_score_type"] == "noise_normalized"
    assert selected_row["layer"] == "good"

    encoding_rows = _read_csv(output_dir / "encoding_scores.csv")
    assert len(encoding_rows) == 1
    assert encoding_rows[0]["layer"] == "good"
    assert encoding_rows[0]["alpha_selection_mode"] == "selection_validation"

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    artifact = json.loads((output_dir / "selection_artifact.json").read_text(encoding="utf-8"))
    feature_metadata = json.loads(
        (output_dir / "feature_reduction_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["selected_layer"] == "good"
    assert len(artifact["selection_train_image_ids"]) == int(
        selected_row["selection_n_train"]
    )
    assert len(artifact["selection_validation_image_ids"]) == int(
        selected_row["selection_n_validation"]
    )
    assert feature_metadata["layers"][0]["n_train_fit"] == len(
        artifact["outer_train_image_ids"]
    )
    assert int(selected_row["pca_n_train_fit"]) == len(artifact["selection_train_image_ids"])


def test_run_neural_alignment_selection_ignores_heldout_test_rows(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment

    clean_result, clean_output = _run_synthetic_selection(
        monkeypatch,
        tmp_path,
        output_name="clean",
    )
    train_idx, test_idx = neural_alignment._outer_train_test_indices(
        24,
        0.75,
        np.random.default_rng(0),
    )
    assert train_idx.size and test_idx.size

    corrupt_result, corrupt_output = _run_synthetic_selection(
        monkeypatch,
        tmp_path,
        output_name="corrupt",
        corrupt_indices=set(int(index) for index in test_idx),
    )

    assert clean_result["selected_layer"] == corrupt_result["selected_layer"] == "good"
    clean_candidates = _read_csv(clean_output / "selection_candidates.csv")
    corrupt_candidates = _read_csv(corrupt_output / "selection_candidates.csv")
    assert [
        (row["layer"], row["validation_score"], row["selected_ridge_alpha"])
        for row in clean_candidates
    ] == [
        (row["layer"], row["validation_score"], row["selected_ridge_alpha"])
        for row in corrupt_candidates
    ]


def test_run_neural_alignment_selection_falls_back_to_raw_score_without_ceiling(
    monkeypatch,
    tmp_path,
):
    result, output_dir = _run_synthetic_selection(
        monkeypatch,
        tmp_path,
        include_noise_ceiling=False,
    )

    assert result["selected_layer"] == "good"
    selected_row = [
        row for row in _read_csv(output_dir / "selection_candidates.csv") if row["selected"] == "true"
    ][0]
    assert selected_row["validation_score_type"] == "raw"


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


def _run_synthetic_learned_readout(
    monkeypatch,
    tmp_path,
    *,
    output_name="learned",
    corrupt_indices=None,
):
    pytest.importorskip("torch")
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    corrupt_indices = set(corrupt_indices or [])

    class SyntheticDataset:
        def __len__(self):
            return 18

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            image_feature = 100.0 + feature if index in corrupt_indices else feature
            response_feature = -100.0 - feature if index in corrupt_indices else feature
            return {
                "image": np.full((3, 4, 4), image_feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array(
                        [
                            2.0 * response_feature + 0.1,
                            -response_feature + 0.2,
                            response_feature * 0.5,
                        ],
                        dtype=np.float32,
                    ),
                    "noise_ceiling": np.array([0.5, 0.0, np.nan], dtype=np.float32),
                },
            }

    class SpatialFeatureModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            tokens = np.array(
                [
                    [value, value * value, 1.0],
                    [value + 0.5, -value, 0.25],
                    [0.1, value * 0.5, -0.5],
                ],
                dtype=np.float32,
            )
            return {layer: tokens for layer in layers}

    output_dir = tmp_path / output_name
    config_path = tmp_path / f"{output_name}.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "encoding_method": "learned_spatial_readout",
                "layers": ["blocks.3"],
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "train_fraction": 0.75,
                "metric": "correlation",
                "learned_readout": {
                    "validation_fraction": 0.25,
                    "max_epochs": 6,
                    "batch_size": 4,
                    "target_batch_size": 2,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "patience": 3,
                    "objective": "pearson",
                },
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: SpatialFeatureModel())
    return neural_alignment.run_neural_alignment(config_path), output_dir


def test_run_neural_alignment_learned_readout_writes_outputs(monkeypatch, tmp_path):
    result, output_dir = _run_synthetic_learned_readout(monkeypatch, tmp_path)

    assert (output_dir / "encoding_scores.csv").is_file()
    assert (output_dir / "encoding_target_scores.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "learned_readout_metadata.json").is_file()
    assert (output_dir / "feature_reduction_metadata.json").is_file()
    row = result["score_rows"][0]
    assert row["layer"] == "blocks.3"
    assert row["feature_reduction"] == "learned_spatial_readout"
    assert row["selected_ridge_alpha"] == ""
    assert row["alpha_selection_mode"] == "early_stopping_validation"
    assert row["valid_noise_ceiling_targets"] == 1
    assert row["zero_noise_ceiling_targets"] == 1
    assert row["invalid_noise_ceiling_targets"] == 1
    target_rows = _read_csv(output_dir / "encoding_target_scores.csv")
    assert target_rows[0]["feature_reduction"] == "learned_spatial_readout"
    assert {"pearson_r", "r2_score_from_r", "prediction_r2", "noise_ceiling"} <= set(
        target_rows[0]
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["encoding_method"] == "learned_spatial_readout"
    assert metadata["feature_reduction"] == "learned_spatial_readout"
    readout_metadata = json.loads(
        (output_dir / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    assert readout_metadata["method"] == "learned_spatial_readout"
    assert readout_metadata["layer"] == "blocks.3"
    assert readout_metadata["best_epoch"] >= 1


def test_run_neural_alignment_learned_readout_ignores_heldout_for_best_epoch(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment

    clean_result, _clean_output = _run_synthetic_learned_readout(
        monkeypatch,
        tmp_path,
        output_name="clean_learned",
    )
    _train_idx, test_idx = neural_alignment._outer_train_test_indices(
        18,
        0.75,
        np.random.default_rng(0),
    )
    corrupt_result, _corrupt_output = _run_synthetic_learned_readout(
        monkeypatch,
        tmp_path,
        output_name="corrupt_learned",
        corrupt_indices=set(int(index) for index in test_idx),
    )

    assert clean_result["score_rows"][0]["layer"] == corrupt_result["score_rows"][0]["layer"]
    assert clean_result["learned_readout_metadata"]
    clean_metadata = json.loads(
        (_clean_output / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    corrupt_metadata = json.loads(
        (_corrupt_output / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    assert clean_metadata["best_epoch"] == corrupt_metadata["best_epoch"]
    assert clean_metadata["validation_score"] == pytest.approx(
        corrupt_metadata["validation_score"]
    )


def test_run_neural_alignment_learned_readout_summary_compatible(
    monkeypatch,
    tmp_path,
):
    from hma.experiments.summarize_neural_roi_results import summarize_neural_roi_results

    _result, output_dir = _run_synthetic_learned_readout(monkeypatch, tmp_path)

    outputs = summarize_neural_roi_results([output_dir], tmp_path / "summary")

    combined_rows = _read_csv(outputs["combined_encoding_scores"])
    ranking_rows = _read_csv(outputs["neural_model_rankings"])
    assert combined_rows[0]["feature_reduction"] == "learned_spatial_readout"
    assert combined_rows[0]["source_dir"] == str(output_dir.resolve())
    assert ranking_rows


def test_fuse_spatial_feature_layers_channel_concatenates_by_image():
    layer_a = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    layer_b = np.array(
        [
            [[10.0], [20.0]],
            [[30.0], [40.0]],
        ],
        dtype=np.float32,
    )

    fused, metadata = fuse_spatial_feature_layers(
        {"a": layer_a, "b": layer_b},
        layers=["a", "b"],
        fusion_method="channel_concat",
    )

    assert fused.shape == (2, 2, 3)
    np.testing.assert_array_equal(fused[0], np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]))
    np.testing.assert_array_equal(fused[1], np.array([[5.0, 6.0, 30.0], [7.0, 8.0, 40.0]]))
    assert metadata["layers"] == ["a", "b"]
    assert metadata["fusion_method"] == "channel_concat"
    assert metadata["fused_feature_shape"] == [2, 3]


def test_fuse_spatial_feature_layers_rejects_mismatched_positions():
    with pytest.raises(ValueError, match="same number of spatial positions"):
        fuse_spatial_feature_layers(
            {
                "a": np.zeros((2, 2, 3), dtype=np.float32),
                "b": np.zeros((2, 3, 1), dtype=np.float32),
            },
            layers=["a", "b"],
            fusion_method="channel_concat",
        )


def _run_synthetic_multilayer_learned_readout(
    monkeypatch,
    tmp_path,
    *,
    output_name="multilayer_learned",
    corrupt_indices=None,
    mismatched_positions=False,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    corrupt_indices = set(corrupt_indices or [])
    captured = {}

    class SyntheticDataset:
        def __len__(self):
            return 24

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            image_feature = 100.0 + feature if index in corrupt_indices else feature
            response_feature = -100.0 - feature if index in corrupt_indices else feature
            return {
                "image": np.full((3, 4, 4), image_feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": {
                    "roi": "synthetic_roi",
                    "subject_id": "subj_test",
                    "roi_responses": np.array(
                        [response_feature, -response_feature, 0.5 * response_feature],
                        dtype=np.float32,
                    ),
                    "noise_ceiling": np.array([0.5, 0.25, 0.1], dtype=np.float32),
                },
            }

    class MultiLayerModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            outputs = {
                "early": np.array([[value, value + 1.0], [value + 2.0, value + 3.0]], dtype=np.float32),
                "middle": np.array([[10.0 + value], [20.0 + value]], dtype=np.float32),
                "late": np.array([[100.0 + value, 200.0 + value], [300.0 + value, 400.0 + value]], dtype=np.float32),
            }
            if mismatched_positions:
                outputs["late"] = np.array(
                    [[100.0 + value], [200.0 + value], [300.0 + value]],
                    dtype=np.float32,
                )
            return {layer: outputs[layer] for layer in layers}

    def fake_fit_spatial_readout(features, responses, *, train_idx, validation_idx, config):
        captured.setdefault("fit_features", np.asarray(features).copy())
        captured["validation_features"] = np.asarray(features)[validation_idx].copy()
        captured["train_idx"] = np.asarray(train_idx).copy()
        captured["validation_idx"] = np.asarray(validation_idx).copy()
        return {
            "responses": np.asarray(responses),
            "metadata": {
                "method": "learned_spatial_readout",
                "input_feature_shape": [int(dim) for dim in np.asarray(features).shape[1:]],
                "normalized_feature_shape": [int(dim) for dim in np.asarray(features).shape[1:]],
                "best_epoch": 1,
                "validation_score": float(np.asarray(features)[validation_idx].mean()),
                "validation_score_type": "mean_pearson",
                "epochs_ran": 1,
            },
        }

    def fake_predict_spatial_readout(bundle, _features, *, indices=None):
        selected = np.arange(bundle["responses"].shape[0]) if indices is None else np.asarray(indices)
        return bundle["responses"][selected]

    output_dir = tmp_path / output_name
    config_path = tmp_path / f"{output_name}.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "encoding_method": "learned_spatial_readout",
                "layers": ["early", "middle", "late"],
                "layer_fusion": "channel_concat",
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "train_fraction": 0.75,
                "metric": "correlation",
                "learned_readout": {
                    "validation_fraction": 0.25,
                    "max_epochs": 2,
                    "batch_size": 4,
                    "target_batch_size": 2,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "patience": 1,
                    "objective": "pearson",
                },
                "rsa": {"enabled": False},
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: MultiLayerModel())
    monkeypatch.setattr(neural_alignment, "fit_spatial_readout", fake_fit_spatial_readout)
    monkeypatch.setattr(
        neural_alignment,
        "predict_spatial_readout",
        fake_predict_spatial_readout,
    )
    return neural_alignment.run_neural_alignment(config_path), output_dir, captured


def test_run_neural_alignment_multilayer_learned_readout_writes_outputs(
    monkeypatch,
    tmp_path,
):
    from hma.experiments.summarize_neural_roi_results import summarize_neural_roi_results

    result, output_dir, captured = _run_synthetic_multilayer_learned_readout(
        monkeypatch,
        tmp_path,
    )

    assert (output_dir / "encoding_scores.csv").is_file()
    assert (output_dir / "encoding_target_scores.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "feature_reduction_metadata.json").is_file()
    assert (output_dir / "learned_readout_metadata.json").is_file()
    assert not (output_dir / "feature_cache").exists()
    assert result["score_rows"][0]["layer"] == "early+middle+late"
    assert result["score_rows"][0]["feature_reduction"] == "learned_spatial_readout"
    assert result["score_rows"][0]["alpha_selection_mode"] == "early_stopping_validation"
    assert captured["fit_features"].shape == (24, 2, 5)

    readout_metadata = json.loads(
        (output_dir / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    assert readout_metadata["layers"] == ["early", "middle", "late"]
    assert readout_metadata["fusion_method"] == "channel_concat"
    assert readout_metadata["fused_feature_shape"] == [2, 5]
    feature_metadata = json.loads(
        (output_dir / "feature_reduction_metadata.json").read_text(encoding="utf-8")
    )
    assert feature_metadata["layers"][0]["layer"] == "early+middle+late"
    assert feature_metadata["layers"][0]["fusion_method"] == "channel_concat"

    outputs = summarize_neural_roi_results([output_dir], tmp_path / "summary")
    combined_rows = _read_csv(outputs["combined_encoding_scores"])
    assert combined_rows[0]["layer"] == "early+middle+late"
    assert combined_rows[0]["feature_reduction"] == "learned_spatial_readout"


def test_run_neural_alignment_multilayer_fusion_preserves_image_alignment(
    monkeypatch,
    tmp_path,
):
    _result, _output_dir, captured = _run_synthetic_multilayer_learned_readout(
        monkeypatch,
        tmp_path,
    )

    features = captured["fit_features"]
    np.testing.assert_array_equal(
        features[3],
        np.array(
            [
                [0.3, 1.3, 10.3, 100.3, 200.3],
                [2.3, 3.3, 20.3, 300.3, 400.3],
            ],
            dtype=np.float32,
        ),
    )


def test_run_neural_alignment_multilayer_ignores_heldout_for_validation(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment

    clean_result, clean_output, _clean_captured = _run_synthetic_multilayer_learned_readout(
        monkeypatch,
        tmp_path,
        output_name="clean_multilayer",
    )
    _train_idx, test_idx = neural_alignment._outer_train_test_indices(
        24,
        0.75,
        np.random.default_rng(0),
    )
    corrupt_result, corrupt_output, _corrupt_captured = _run_synthetic_multilayer_learned_readout(
        monkeypatch,
        tmp_path,
        output_name="corrupt_multilayer",
        corrupt_indices=set(int(index) for index in test_idx),
    )

    assert clean_result["score_rows"][0]["layer"] == corrupt_result["score_rows"][0]["layer"]
    clean_metadata = json.loads(
        (clean_output / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    corrupt_metadata = json.loads(
        (corrupt_output / "learned_readout_metadata.json").read_text(encoding="utf-8")
    )
    assert clean_metadata["validation_score"] == pytest.approx(
        corrupt_metadata["validation_score"]
    )


def test_run_neural_alignment_multilayer_requires_matching_positions(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(ValueError, match="same number of spatial positions"):
        _run_synthetic_multilayer_learned_readout(
            monkeypatch,
            tmp_path,
            mismatched_positions=True,
        )


def _run_synthetic_learned_readout_selection(
    monkeypatch,
    tmp_path,
    *,
    output_name="learned_selection",
    include_noise_ceiling=True,
    corrupt_indices=None,
):
    from hma.experiments import neural_alignment
    from hma.utils.config import save_yaml

    corrupt_indices = set(corrupt_indices or [])

    class SyntheticDataset:
        def __len__(self):
            return 24

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

        def __getitem__(self, index):
            feature = float(index) / 10.0
            image_feature = 100.0 + feature if index in corrupt_indices else feature
            response_feature = -100.0 - feature if index in corrupt_indices else feature
            metadata = {
                "roi": "synthetic_roi",
                "subject_id": "subj_test",
                "roi_responses": np.array(
                    [
                        response_feature,
                        -response_feature,
                        0.5 * response_feature,
                    ],
                    dtype=np.float32,
                ),
            }
            if include_noise_ceiling:
                metadata["noise_ceiling"] = np.array([0.5, 0.25, 0.1], dtype=np.float32)
            return {
                "image": np.full((3, 4, 4), image_feature, dtype=np.float32),
                "image_id": f"item_{index:04d}",
                "metadata": metadata,
            }

    class LayerSelectionModel:
        def to(self, device):
            return None

        def get_features(self, images, layers=None):
            value = float(images.mean().detach().cpu().numpy())
            outputs = {
                "bad": np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
                "good": np.array([[[value, -value, 0.5 * value]]], dtype=np.float32),
            }
            return {layer: outputs[layer] for layer in layers}

    def fake_fit_spatial_readout(features, responses, *, train_idx, validation_idx, config):
        return {
            "is_good": bool(np.max(np.asarray(features)) > 0.0),
            "responses": np.asarray(responses),
            "metadata": {
                "method": "learned_spatial_readout",
                "best_epoch": 1,
                "validation_score": 1.0,
                "validation_score_type": "mean_pearson",
                "epochs_ran": 1,
            }
        }

    def fake_predict_spatial_readout(_bundle, features, *, indices=None):
        selected = np.arange(features.shape[0]) if indices is None else np.asarray(indices)
        if _bundle["is_good"]:
            return _bundle["responses"][selected]
        return np.asarray(features[selected, 0, :], dtype=np.float32)

    output_dir = tmp_path / output_name
    config_path = tmp_path / f"{output_name}.yaml"
    save_yaml(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {"name": "synthetic"},
            "model": {"name": "synthetic_model"},
            "preprocessing": {"input_size": [4, 4], "mean": "none", "std": "none"},
            "neural": {
                "encoding_method": "learned_spatial_readout",
                "layers": ["bad", "good"],
                "response_key": "roi_responses",
                "noise_ceiling_key": "noise_ceiling",
                "train_fraction": 0.75,
                "metric": "correlation",
                "selection": {
                    "enabled": True,
                    "validation_fraction": 0.25,
                    "primary_score": "mean_noise_normalized_score",
                },
                "learned_readout": {
                    "validation_fraction": 0.25,
                    "max_epochs": 2,
                    "batch_size": 4,
                    "target_batch_size": 2,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "patience": 1,
                    "objective": "pearson",
                },
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )
    monkeypatch.setattr(neural_alignment, "build_dataset", lambda _config: SyntheticDataset())
    monkeypatch.setattr(neural_alignment, "build_model", lambda _config: LayerSelectionModel())
    monkeypatch.setattr(neural_alignment, "fit_spatial_readout", fake_fit_spatial_readout)
    monkeypatch.setattr(
        neural_alignment,
        "predict_spatial_readout",
        fake_predict_spatial_readout,
    )
    return neural_alignment.run_neural_alignment(config_path), output_dir


def test_run_neural_alignment_learned_readout_selection_writes_selected_outputs(
    monkeypatch,
    tmp_path,
):
    result, output_dir = _run_synthetic_learned_readout_selection(monkeypatch, tmp_path)

    assert (output_dir / "selection_candidates.csv").is_file()
    assert (output_dir / "selection_artifact.json").is_file()
    assert (output_dir / "learned_readout_metadata.json").is_file()
    assert result["selection_enabled"] is True
    assert result["selected_layer"] == "good"
    assert result["selected_feature_reduction"] == "learned_spatial_readout"
    assert result["selected_ridge_alpha"] == ""

    candidate_rows = _read_csv(output_dir / "selection_candidates.csv")
    assert len(candidate_rows) == 2
    assert [row["selected"] for row in candidate_rows].count("true") == 1
    selected_row = [row for row in candidate_rows if row["selected"] == "true"][0]
    assert selected_row["layer"] == "good"
    assert selected_row["validation_score_type"] == "noise_normalized"
    assert selected_row["selected_ridge_alpha"] == ""

    encoding_rows = _read_csv(output_dir / "encoding_scores.csv")
    assert len(encoding_rows) == 1
    assert encoding_rows[0]["layer"] == "good"
    assert encoding_rows[0]["alpha_selection_mode"] == (
        "learned_readout_selection_validation"
    )


def test_run_neural_alignment_learned_readout_selection_ignores_heldout_rows(
    monkeypatch,
    tmp_path,
):
    from hma.experiments import neural_alignment

    clean_result, clean_output = _run_synthetic_learned_readout_selection(
        monkeypatch,
        tmp_path,
        output_name="clean_learned_selection",
    )
    _train_idx, test_idx = neural_alignment._outer_train_test_indices(
        24,
        0.75,
        np.random.default_rng(0),
    )
    corrupt_result, corrupt_output = _run_synthetic_learned_readout_selection(
        monkeypatch,
        tmp_path,
        output_name="corrupt_learned_selection",
        corrupt_indices=set(int(index) for index in test_idx),
    )

    assert clean_result["selected_layer"] == corrupt_result["selected_layer"] == "good"
    clean_candidates = _read_csv(clean_output / "selection_candidates.csv")
    corrupt_candidates = _read_csv(corrupt_output / "selection_candidates.csv")
    assert [
        (row["layer"], row["validation_score"], row["selected"])
        for row in clean_candidates
    ] == [
        (row["layer"], row["validation_score"], row["selected"])
        for row in corrupt_candidates
    ]


def test_run_neural_alignment_learned_readout_selection_falls_back_to_raw_score(
    monkeypatch,
    tmp_path,
):
    result, output_dir = _run_synthetic_learned_readout_selection(
        monkeypatch,
        tmp_path,
        include_noise_ceiling=False,
    )

    assert result["selected_layer"] == "good"
    selected_row = [
        row for row in _read_csv(output_dir / "selection_candidates.csv") if row["selected"] == "true"
    ][0]
    assert selected_row["validation_score_type"] == "raw"


def test_run_neural_alignment_learned_readout_selection_summary_compatible(
    monkeypatch,
    tmp_path,
):
    from hma.experiments.summarize_neural_roi_results import summarize_neural_roi_results

    _result, output_dir = _run_synthetic_learned_readout_selection(monkeypatch, tmp_path)

    outputs = summarize_neural_roi_results([output_dir], tmp_path / "summary")

    combined_rows = _read_csv(outputs["combined_encoding_scores"])
    assert combined_rows[0]["feature_reduction"] == "learned_spatial_readout"
    assert combined_rows[0]["layer"] == "good"


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
