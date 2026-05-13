import numpy as np

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
