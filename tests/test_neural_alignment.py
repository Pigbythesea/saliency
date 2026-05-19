import numpy as np
import pytest

from hma.experiments.neural_alignment import run_neural_alignment
from hma.neural import (
    compare_rdms,
    compute_rdm,
    evaluate_encoding,
    fit_ridge_encoding,
    predict_ridge_encoding,
    save_activations,
)


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
                "train_fraction": 0.75,
                "ridge_alpha": 1.0,
                "metric": "correlation",
            },
            "output": {"dir": str(output_dir)},
        },
        config_path,
    )

    result = run_neural_alignment(config_path)

    assert (output_dir / "activations.npz").is_file()
    assert (output_dir / "encoding_scores.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert result["num_items"] == 8
    assert result["score_rows"][0]["layer"] == "embedding"


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
