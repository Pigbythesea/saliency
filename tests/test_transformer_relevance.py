import pytest

from hma.saliency import (
    UnsupportedModelError,
    build_saliency_method,
    transformer_relevance_saliency,
    transformer_relevance_to_saliency_map,
)


def _tiny_wrapper(torch):
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.head_dim = 2
            self.attn_dim = 4
            self.qkv = torch.nn.Linear(4, 12)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attn_drop = torch.nn.Identity()
            self.norm = torch.nn.Identity()
            self.proj = torch.nn.Linear(4, 4)
            self.proj_drop = torch.nn.Identity()
            self.scale = self.head_dim**-0.5

        def forward(self, tokens):
            return tokens

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv2d(3, 4, kernel_size=4, stride=4)
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, 4))
            self.blocks = torch.nn.ModuleList(
                [torch.nn.ModuleDict({"attn": TinyAttention()})]
            )
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, images):
            patches = self.patch(images).flatten(2).transpose(1, 2)
            cls = self.cls.expand(images.shape[0], -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            for block in self.blocks:
                tokens = block["attn"](tokens)
            return self.classifier(tokens[:, 0])

    class TinyWrapper:
        def __init__(self):
            torch.manual_seed(0)
            self.model = TinyTransformer().eval()

        def get_last_logits(self, images):
            return self.model(images)

    return TinyWrapper()


def test_transformer_relevance_saliency_shape_range_and_finite_values():
    torch = pytest.importorskip("torch")
    wrapper = _tiny_wrapper(torch)
    images = torch.randn(2, 3, 8, 8)

    maps = transformer_relevance_saliency(wrapper, images, target_class=1, grid_size=[2, 2])

    assert tuple(maps.shape) == (2, 1, 8, 8)
    assert torch.isfinite(maps).all()
    assert float(maps.min()) >= 0.0
    assert float(maps.max()) <= 1.0


def test_transformer_relevance_saliency_supports_target_class_list():
    torch = pytest.importorskip("torch")
    wrapper = _tiny_wrapper(torch)
    images = torch.randn(2, 3, 8, 8)

    maps = transformer_relevance_saliency(wrapper, images, target_class=[0, 2], grid_size=[2, 2])

    assert tuple(maps.shape) == (2, 1, 8, 8)


def test_transformer_relevance_target_class_list_length_must_match_batch():
    torch = pytest.importorskip("torch")
    wrapper = _tiny_wrapper(torch)
    images = torch.randn(2, 3, 8, 8)

    with pytest.raises(ValueError, match="target_class list length"):
        transformer_relevance_saliency(wrapper, images, target_class=[1], grid_size=[2, 2])


def test_transformer_relevance_unsupported_wrapper_raises_helpful_error():
    torch = pytest.importorskip("torch")
    images = torch.randn(1, 3, 8, 8)

    with pytest.raises(UnsupportedModelError, match="model_wrapper.model"):
        transformer_relevance_saliency(object(), images)


def test_transformer_relevance_requires_supported_attention_modules():
    torch = pytest.importorskip("torch")

    class CNNWrapper:
        def __init__(self):
            self.model = torch.nn.Conv2d(3, 4, kernel_size=3)

    images = torch.randn(1, 3, 8, 8)

    with pytest.raises(UnsupportedModelError, match="No compatible timm ViT attention"):
        transformer_relevance_saliency(CNNWrapper(), images)


def test_transformer_relevance_to_saliency_map_accepts_explicit_non_square_grid():
    torch = pytest.importorskip("torch")
    attention = torch.eye(7).reshape(1, 1, 7, 7).requires_grad_(True)
    score = attention[:, :, 0, 1:].sum()
    score.backward()

    maps = transformer_relevance_to_saliency_map(
        [attention],
        image_shape=(4, 6),
        grid_size=[2, 3],
    )

    assert tuple(maps.shape) == (1, 1, 4, 6)
    assert torch.isfinite(maps).all()


def test_transformer_relevance_to_saliency_map_requires_grid_for_non_square_patches():
    torch = pytest.importorskip("torch")
    attention = torch.eye(7).reshape(1, 1, 7, 7).requires_grad_(True)
    attention.sum().backward()

    with pytest.raises(UnsupportedModelError, match="Patch count is not square"):
        transformer_relevance_to_saliency_map([attention], image_shape=(4, 4))


def test_saliency_registry_builds_transformer_relevance_callable():
    torch = pytest.importorskip("torch")
    wrapper = _tiny_wrapper(torch)
    images = torch.randn(1, 3, 8, 8)
    method = build_saliency_method(
        {
            "saliency": {
                "method": "transformer_relevance",
                "grid_size": [2, 2],
                "head_fusion": "mean",
            }
        }
    )

    maps = method(wrapper, images)

    assert tuple(maps.shape) == (1, 1, 8, 8)
