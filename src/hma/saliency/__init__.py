"""Saliency map helpers."""

from hma.saliency.attention_rollout import (
    UnsupportedModelError,
    attention_rollout_saliency,
    attention_rollout_to_saliency_map,
    cls_to_patch_relevance,
    compute_attention_rollout,
    patch_relevance_to_grid,
)
from hma.saliency.baselines import center_bias_saliency, random_saliency
from hma.saliency.baselines import (
    COCOSearch18TaskPrior,
    coco_search18_task_prior_saliency,
)
from hma.saliency.gradcam import gradcam_saliency
from hma.saliency.gradients import vanilla_gradient_saliency
from hma.saliency.integrated_gradients import integrated_gradients_saliency
from hma.saliency.occlusion import occlusion_saliency
from hma.saliency.postprocess import (
    normalize_saliency_map,
    postprocess_saliency_map,
    resize_saliency_map,
)
from hma.saliency.registry import build_saliency_method
from hma.saliency.transformer_relevance import (
    compute_transformer_relevance_rollout,
    transformer_relevance_saliency,
    transformer_relevance_to_saliency_map,
)

__all__ = [
    "UnsupportedModelError",
    "COCOSearch18TaskPrior",
    "attention_rollout_saliency",
    "attention_rollout_to_saliency_map",
    "build_saliency_method",
    "center_bias_saliency",
    "cls_to_patch_relevance",
    "coco_search18_task_prior_saliency",
    "compute_attention_rollout",
    "gradcam_saliency",
    "integrated_gradients_saliency",
    "normalize_saliency_map",
    "occlusion_saliency",
    "patch_relevance_to_grid",
    "postprocess_saliency_map",
    "random_saliency",
    "resize_saliency_map",
    "compute_transformer_relevance_rollout",
    "transformer_relevance_saliency",
    "transformer_relevance_to_saliency_map",
    "vanilla_gradient_saliency",
]
