from hma.pipelines.dummy import run_dummy_pipeline


def test_dummy_pipeline_runs_end_to_end():
    metrics = run_dummy_pipeline("configs/experiments/dummy_pipeline.yaml")

    assert metrics["experiment"] == "dummy_pipeline"
    assert metrics["num_items"] == 3
    assert set(metrics) == {"experiment", "num_items", "mae", "pearson"}
    assert 0.0 <= metrics["mae"] <= 1.0
    assert -1.0 <= metrics["pearson"] <= 1.0
