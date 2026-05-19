import sys
import types

import pytest

from hma.models import DummySaliencyModel, TimmModelWrapper, build_model


class FakeParameter:
    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class FakeHookHandle:
    def __init__(self, layer, hook):
        self.layer = layer
        self.hook = hook

    def remove(self):
        self.layer.hooks.remove(self.hook)


class FakeHookLayer:
    def __init__(self, name):
        self.name = name
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHookHandle(self, hook)

    def __call__(self, value):
        output = f"{self.name}:{value}"
        for hook in list(self.hooks):
            hook(self, (value,), output)
        return output


class FakeTimmModel:
    def __init__(self):
        self.eval_called = False
        self.layer1 = FakeHookLayer("layer1")
        self.block = types.SimpleNamespace(conv=FakeHookLayer("block.conv"))

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, images):
        hidden = self.layer1(images)
        self.block.conv(hidden)
        return {"logits": images}

    def forward_features(self, images):
        return {"features": images}

    def parameters(self):
        return [FakeParameter(2), FakeParameter(3)]

    def named_modules(self):
        return [
            ("", self),
            ("layer1", self.layer1),
            ("block.conv", self.block.conv),
        ]


def test_build_model_preserves_dummy_saliency_model():
    model = build_model({"model": {"name": "dummy_saliency", "noise_scale": 0.1}})

    assert isinstance(model, DummySaliencyModel)
    assert model.noise_scale == 0.1


def test_timm_wrapper_uses_create_model_and_eval(monkeypatch):
    calls = {}
    fake_model = FakeTimmModel()

    def create_model(model_name, pretrained=False, **kwargs):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["kwargs"] = kwargs
        return fake_model

    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(create_model=create_model),
    )

    wrapper = TimmModelWrapper(
        "resnet50",
        pretrained=False,
        eval_mode=True,
        features_only=True,
        num_classes=10,
    )

    assert calls == {
        "model_name": "resnet50",
        "pretrained": False,
        "kwargs": {"features_only": True, "num_classes": 10},
    }
    assert fake_model.eval_called is True
    assert wrapper.forward("images") == {"logits": "images"}
    assert wrapper.get_last_logits("images") == {"logits": "images"}
    assert wrapper.get_features("images") == {"features": "images"}
    assert wrapper.metadata == {"model_name": "resnet50", "parameter_count": 5}


def test_timm_wrapper_named_layers_use_forward_hooks(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeTimmModel()),
    )
    wrapper = TimmModelWrapper("resnet50", pretrained=False)

    features = wrapper.get_features("images", layers=["layer1", "block.conv"])

    assert features == {
        "layer1": "layer1:images",
        "block.conv": "block.conv:layer1:images",
    }


def test_timm_wrapper_unknown_named_layer_has_clear_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeTimmModel()),
    )
    wrapper = TimmModelWrapper("resnet50", pretrained=False)

    with pytest.raises(ValueError, match="Unknown timm layer"):
        wrapper.get_features("images", layers=["layer4"])


def test_build_model_supports_timm_backend(monkeypatch):
    calls = {}

    def create_model(model_name, pretrained=False, **kwargs):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["kwargs"] = kwargs
        return FakeTimmModel()

    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(create_model=create_model),
    )

    model = build_model(
        {
            "model": {
                "name": "convnext_tiny",
                "backend": "timm",
                "pretrained": False,
                "eval_mode": True,
                "kwargs": {"num_classes": 7},
            }
        }
    )

    assert isinstance(model, TimmModelWrapper)
    assert calls == {
        "model_name": "convnext_tiny",
        "pretrained": False,
        "kwargs": {"num_classes": 7},
    }


def test_timm_wrapper_reports_clear_import_error(monkeypatch):
    import hma.models.timm_wrappers as timm_wrappers

    def missing_timm(module_name):
        if module_name == "timm":
            raise ImportError("missing timm")
        return __import__(module_name)

    monkeypatch.setattr(timm_wrappers.importlib, "import_module", missing_timm)

    with pytest.raises(ImportError, match="timm is required"):
        TimmModelWrapper("resnet50", pretrained=False)
