import sys
from types import SimpleNamespace

from hma.utils.device import resolve_device


def test_resolve_device_defaults_to_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    assert resolve_device("auto") == "cpu"
    assert resolve_device(None) == "cpu"


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto") == "cuda"
    assert resolve_device("gpu") == "cuda"


def test_resolve_device_preserves_explicit_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"
