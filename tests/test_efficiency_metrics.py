import sys

import pytest

from hma.metrics.efficiency_metrics import (
    count_parameters,
    estimate_flops,
    estimate_model_size_mb,
)


class FakeParameter:
    def __init__(self, numel, element_size=4):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeModel:
    def parameters(self):
        return [FakeParameter(10, 4), FakeParameter(5, 2)]


class FakeWrapper:
    def __init__(self):
        self.model = FakeModel()


def test_count_parameters_supports_plain_model_and_wrapper():
    assert count_parameters(FakeModel()) == 15
    assert count_parameters(FakeWrapper()) == 15


def test_estimate_model_size_mb_uses_element_size():
    expected_bytes = 10 * 4 + 5 * 2

    assert estimate_model_size_mb(FakeModel()) == expected_bytes / (1024.0 * 1024.0)


def test_estimate_flops_returns_none_when_optional_dependencies_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "fvcore.nn", None)
    monkeypatch.setitem(sys.modules, "ptflops", None)

    with pytest.warns(UserWarning, match="requires torch"):
        assert estimate_flops(FakeModel(), input_shape=(1, 3, 8, 8)) is None


def test_measure_latency_smoke_test_with_tiny_torch_model():
    torch = pytest.importorskip("torch")
    from hma.metrics.efficiency_metrics import measure_latency

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 4 * 4, 2),
    )

    result = measure_latency(
        model,
        input_shape=(1, 3, 4, 4),
        device="cpu",
        warmup=0,
        repeats=1,
    )

    assert result["latency_mean_ms"] >= 0.0
    assert result["latency_min_ms"] >= 0.0
