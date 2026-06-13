"""Adapters executed inside pinned external-model environments."""

from __future__ import annotations

import importlib
import json
import sys
import time
from abc import ABC, abstractmethod
from collections import abc as collections_abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ExternalIntegrationError(RuntimeError):
    """An actionable external integration failure."""


@dataclass
class ExternalBatchOutput:
    features: dict[str, Any]
    logits: Any | None = None
    task_outputs: dict[str, Any] = field(default_factory=dict)
    resource_allocation: dict[str, Any] = field(default_factory=dict)
    scanpaths: list[dict[str, Any]] = field(default_factory=list)


class ExternalModelAdapter(ABC):
    """Common model boundary required by Matrix V2."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: dict[str, Any],
        source_dir: str | Path,
        checkpoint_path: str | Path | None,
        device: str,
        seed: int,
    ) -> None:
        self.model_id = model_id
        self.model_config = dict(model_config)
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        )
        self.device = device
        self.seed = int(seed)
        self.layers = [str(value) for value in model_config.get("feature_layers", [])]
        self.adapter_config = dict(model_config.get("adapter_config", {}))
        if self.source_dir.is_dir() and str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        self.torch = _require_module("torch")
        _install_legacy_torch_six_compatibility(self.torch)
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """Load the pinned model and checkpoint."""

    @abstractmethod
    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        """Run one forward pass and expose all required outputs."""

    def extract_features(self, images: list[Image.Image]) -> dict[str, Any]:
        return self.run_batch(images, [str(index) for index in range(len(images))]).features

    def extract_resource_allocation(self, images: list[Image.Image]) -> dict[str, Any]:
        return self.run_batch(
            images, [str(index) for index in range(len(images))]
        ).resource_allocation

    def predict_scanpaths(
        self, images: list[Image.Image], image_ids: list[str]
    ) -> list[dict[str, Any]]:
        return self.run_batch(images, image_ids).scanpaths

    def profile_efficiency(
        self,
        images: list[Image.Image],
        *,
        warmup_runs: int = 2,
        measured_runs: int = 5,
    ) -> dict[str, Any]:
        if not images:
            return {}
        torch = self.torch
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(max(0, warmup_runs)):
            self.run_batch(images, [str(index) for index in range(len(images))])
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, measured_runs)):
            self.run_batch(images, [str(index) for index in range(len(images))])
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        parameter_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        peak_memory = (
            int(torch.cuda.max_memory_allocated())
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        theoretical_flops = self.adapter_config.get("full_token_flops")
        realized_flops = _profile_flops(
            self.model,
            self.preprocess(images[:1]),
        )
        if theoretical_flops is None and realized_flops is not None:
            theoretical_flops = realized_flops
        return {
            "parameters": parameter_count,
            "latency_ms_per_image": 1000.0 * elapsed / (max(1, measured_runs) * len(images)),
            "peak_memory_bytes": peak_memory,
            "theoretical_flops": (
                int(theoretical_flops) if theoretical_flops is not None else None
            ),
            "realized_flops": (
                int(realized_flops) if realized_flops is not None else None
            ),
        }

    def smoke(self) -> dict[str, Any]:
        image = Image.new("RGB", (224, 224), color=(127, 127, 127))
        output = self.run_batch([image], ["smoke"])
        if self.layers and set(output.features) != set(self.layers):
            raise ExternalIntegrationError(
                f"Smoke feature layers differ from registry: {sorted(output.features)}"
            )
        return {
            "feature_shapes": {
                layer: list(_to_numpy(values).shape)
                for layer, values in output.features.items()
            },
            "resource_outputs": sorted(output.resource_allocation),
            "has_logits": output.logits is not None,
        }

    def preprocess(self, images: list[Image.Image]) -> Any:
        torch = self.torch
        torchvision = _require_module("torchvision")
        preprocessing = dict(self.model_config.get("preprocessing", {}))
        size = preprocessing.get("input_size", [224, 224])
        height, width = int(size[0]), int(size[1])
        mean = preprocessing.get("mean", [0.485, 0.456, 0.406])
        std = preprocessing.get("std", [0.229, 0.224, 0.225])
        crop_pct = float(preprocessing.get("crop_pct", 1.0))
        if crop_pct <= 0.0 or crop_pct > 1.0:
            raise ExternalIntegrationError("preprocessing.crop_pct must be in (0, 1]")
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        resize_size: int | tuple[int, int]
        if height == width:
            resize_size = int(round(height / crop_pct))
        else:
            resize_size = (
                int(round(height / crop_pct)),
                int(round(width / crop_pct)),
            )
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(resize_size, interpolation=interpolation),
                torchvision.transforms.CenterCrop((height, width)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        return torch.stack([transform(image.convert("RGB")) for image in images]).to(self.device)


class ToMeAdapter(ExternalModelAdapter):
    """Paired static DeiT-S and ToMe-DeiT-S adapter with source tracing."""

    def load_model(self) -> Any:
        timm = _require_module("timm")
        model_name = str(self.adapter_config.get("model_name", "deit_small_patch16_224"))
        model = timm.create_model(model_name, pretrained=False)
        merge_r = int(self.adapter_config.get("merge_r", 0))
        if merge_r > 0:
            tome = _require_module("tome")
            tome.patch.timm(
                model,
                trace_source=bool(self.adapter_config.get("trace_source", True)),
                prop_attn=bool(self.adapter_config.get("proportional_attention", True)),
            )
            model.r = merge_r
        _load_checkpoint(model, self.checkpoint_path, self.torch)
        return model.eval().to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        torch = self.torch
        tensor = self.preprocess(images)
        features: dict[str, Any] = {}
        source_by_layer: dict[str, Any] = {}
        modules = dict(self.model.named_modules())
        handles = []

        def capture(layer: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                features[layer] = output.detach()
                info = getattr(self.model, "_tome_info", {})
                source = info.get("source")
                if source is not None:
                    source_by_layer[layer] = source.detach().clone()

            return hook

        for layer in self.layers:
            if layer not in modules:
                raise ExternalIntegrationError(f"ToMe layer not found: {layer}")
            handles.append(modules[layer].register_forward_hook(capture(layer)))
        try:
            with torch.inference_mode():
                logits = self.model(tensor)
        finally:
            for handle in handles:
                handle.remove()
        resource: dict[str, Any] = {
            f"realized_token_counts.{layer}": np.full(
                len(images), int(features[layer].shape[1]), dtype=np.int32
            )
            for layer in self.layers
        }
        if int(self.adapter_config.get("merge_r", 0)) == 0:
            token_count = int(next(iter(features.values())).shape[1] - 1)
            resource["full_token_mask"] = np.ones(
                (len(images), token_count), dtype=np.uint8
            )
        else:
            for layer, source in source_by_layer.items():
                resource[f"token_source_assignments.{layer}"] = source
        return ExternalBatchOutput(
            features=features,
            logits=logits,
            resource_allocation=resource,
        )


class DynamicViTAdapter(ExternalModelAdapter):
    """DynamicViT adapter that preserves token decisions in original patch coordinates."""

    def load_model(self) -> Any:
        dyvit = importlib.import_module("models.dyvit")
        architecture = str(self.adapter_config.get("architecture", "deit-s"))
        if architecture != "deit-s":
            raise ExternalIntegrationError(
                f"Unsupported DynamicViT architecture for Matrix V2: {architecture}"
            )
        keep_ratio = float(self.adapter_config.get("keep_ratio", 0.7))
        pruning_locations = [
            int(value)
            for value in self.adapter_config.get("pruning_locations", [3, 6, 9])
        ]
        model = dyvit.VisionTransformerDiffPruning(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            pruning_loc=pruning_locations,
            token_ratio=[keep_ratio, keep_ratio**2, keep_ratio**3],
        )
        _load_checkpoint(model, self.checkpoint_path, self.torch)
        return model.eval().to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        torch = self.torch
        tensor = self.preprocess(images)
        batch_index_select = getattr(_require_module("utils"), "batch_index_select")
        model = self.model
        with torch.inference_mode():
            batch_size = tensor.shape[0]
            x = model.patch_embed(tensor)
            original_patch_count = int(x.shape[1])
            cls_tokens = model.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            original_indices = torch.arange(
                original_patch_count, device=x.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
            prev_decision = torch.ones(
                batch_size, original_patch_count, 1, dtype=x.dtype, device=x.device
            )
            features: dict[str, Any] = {}
            resource: dict[str, Any] = {}
            pruning_index = 0
            for block_index, block in enumerate(model.blocks):
                if block_index in model.pruning_loc:
                    scores = model.score_predictor[pruning_index](
                        x[:, 1:], prev_decision
                    ).reshape(batch_size, -1, 2)[:, :, 0]
                    keep_count = int(original_patch_count * model.token_ratio[pruning_index])
                    local_keep = torch.argsort(scores, dim=1, descending=True)[:, :keep_count]
                    original_indices = batch_index_select(original_indices, local_keep)
                    cls_policy = torch.zeros(
                        batch_size, 1, dtype=local_keep.dtype, device=local_keep.device
                    )
                    x = batch_index_select(
                        x, torch.cat([cls_policy, local_keep + 1], dim=1)
                    )
                    prev_decision = batch_index_select(prev_decision, local_keep)
                    mask = torch.zeros(
                        batch_size,
                        original_patch_count,
                        dtype=torch.uint8,
                        device=x.device,
                    )
                    mask.scatter_(1, original_indices, 1)
                    resource[f"prediction_masks.stage_{pruning_index}"] = mask
                    resource[
                        f"retained_original_token_indices.stage_{pruning_index}"
                    ] = original_indices.clone()
                    resource[f"realized_token_counts.stage_{pruning_index}"] = torch.full(
                        (batch_size,), keep_count, dtype=torch.int32, device=x.device
                    )
                    x = block(x)
                    pruning_index += 1
                else:
                    x = block(x)
                layer = f"blocks.{block_index}"
                if layer in self.layers:
                    features[layer] = x.detach().clone()
            normalized = model.norm(x)
            logits = model.head(model.pre_logits(normalized[:, 0]))
        missing = [layer for layer in self.layers if layer not in features]
        if missing:
            raise ExternalIntegrationError(f"DynamicViT layers not captured: {missing}")
        return ExternalBatchOutput(
            features=features,
            logits=logits,
            resource_allocation=resource,
        )


class SigLIPAdapter(ExternalModelAdapter):
    """Official Transformers SigLIP vision encoder adapter."""

    def load_model(self) -> Any:
        transformers = _require_module("transformers")
        checkpoint = str(self.checkpoint_path or self.adapter_config["checkpoint_id"])
        self.processor = transformers.AutoProcessor.from_pretrained(checkpoint)
        parent = transformers.AutoModel.from_pretrained(checkpoint)
        model = getattr(parent, "vision_model", parent)
        return model.eval().to(self.device)

    def preprocess(self, images: list[Image.Image]) -> Any:
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        tensor = self.preprocess(images)
        modules = dict(self.model.named_modules())
        features, handles = {}, []
        for layer in self.layers:
            if layer not in modules:
                raise ExternalIntegrationError(f"SigLIP layer not found: {layer}")
            handles.append(
                modules[layer].register_forward_hook(
                    lambda _module, _inputs, output, name=layer: features.__setitem__(
                        name, _first_tensor(output).detach()
                    )
                )
            )
        try:
            with self.torch.inference_mode():
                output = self.model(pixel_values=tensor)
        finally:
            for handle in handles:
                handle.remove()
        logits = getattr(output, "pooler_output", None)
        return ExternalBatchOutput(features=features, logits=logits)


class MambaVisionAdapter(ExternalModelAdapter):
    """Official MambaVision adapter with explicit stage hooks."""

    def load_model(self) -> Any:
        timm = _require_module("timm")
        importlib.import_module("mambavision.models")
        model = timm.create_model("mamba_vision_T", pretrained=False)
        _load_checkpoint(model, self.checkpoint_path, self.torch)
        return model.eval().to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        tensor = self.preprocess(images)
        modules = dict(self.model.named_modules())
        features, handles = {}, []
        for layer in self.layers:
            if layer not in modules:
                raise ExternalIntegrationError(f"MambaVision stage not found: {layer}")
            handles.append(
                modules[layer].register_forward_hook(
                    lambda _module, _inputs, output, name=layer: features.__setitem__(
                        name, _first_tensor(output).detach()
                    )
                )
            )
        try:
            with self.torch.inference_mode():
                logits = self.model(tensor)
        finally:
            for handle in handles:
                handle.remove()
        return ExternalBatchOutput(features=features, logits=logits)


class _AuditPendingAdapter(ExternalModelAdapter):
    audit_message = "Adapter requires an official checkpoint/API audit before execution."

    def load_model(self) -> Any:
        raise ExternalIntegrationError(self.audit_message)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        raise ExternalIntegrationError(self.audit_message)


class DINOv3Adapter(_AuditPendingAdapter):
    audit_message = (
        "DINOv3 source is pinned, but the gated official checkpoint and constructor "
        "must be recorded before adapter_ready can pass."
    )


class HieraAdapter(_AuditPendingAdapter):
    audit_message = "Hiera requires a resolved official source pin and checkpoint lock."


class SwinAdapter(_AuditPendingAdapter):
    audit_message = "Swin requires a resolved official source pin and checkpoint lock."


class HATAdapter(_AuditPendingAdapter):
    audit_message = (
        "HAT execution is cluster-only until Detectron2, MSDeformableAttn, data, "
        "and checkpoint installation gates pass."
    )


class ScanDiffAdapter(_AuditPendingAdapter):
    audit_message = (
        "ScanDiff requires the official Hydra checkpoint/data snapshot before "
        "scanpath export can execute."
    )


def build_adapter(
    class_path: str,
    **kwargs: Any,
) -> ExternalModelAdapter:
    module_name, separator, class_name = class_path.partition(":")
    if not separator:
        raise ValueError("Adapter path must use 'module:ClassName' syntax")
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    return adapter_class(**kwargs)


def adapter_class_available(class_path: str) -> bool:
    try:
        module_name, separator, class_name = class_path.partition(":")
        return bool(separator) and hasattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return False


def hardware_metadata(torch: Any) -> dict[str, Any]:
    metadata = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
    }
    if torch.cuda.is_available():
        metadata["device_name"] = str(torch.cuda.get_device_name(0))
        metadata["device_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        metadata["device_name"] = "cpu"
    return metadata


def _install_legacy_torch_six_compatibility(torch: Any) -> None:
    """Provide symbols required by pinned pre-1.0 timm releases."""
    legacy = getattr(torch, "_six", None)
    if legacy is None:
        try:
            legacy = importlib.import_module("torch._six")
        except ImportError:
            return
        setattr(torch, "_six", legacy)
    if not hasattr(legacy, "container_abcs"):
        legacy.container_abcs = collections_abc
    if not hasattr(legacy, "string_classes"):
        legacy.string_classes = (str,)
    if not hasattr(legacy, "inf"):
        legacy.inf = float("inf")


def _load_checkpoint(model: Any, path: Path | None, torch: Any) -> None:
    if path is None or not path.exists():
        raise ExternalIntegrationError(f"Checkpoint is not ready: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ExternalIntegrationError("Checkpoint does not contain a state dictionary")
    normalized = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }
    result = model.load_state_dict(normalized, strict=False)
    missing = [
        key
        for key in getattr(result, "missing_keys", [])
        if not key.endswith("num_batches_tracked")
    ]
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise ExternalIntegrationError(
            "Checkpoint does not match the registered architecture; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )


def _require_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ExternalIntegrationError(
            f"Required external dependency '{name}' is not installed in this model environment"
        ) from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise ExternalIntegrationError(f"Could not identify tensor output in {type(value)!r}")


def _profile_flops(model: Any, tensor: Any) -> int | None:
    try:
        analysis_class = importlib.import_module("fvcore.nn").FlopCountAnalysis
        analysis = analysis_class(model, tensor)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return int(analysis.total())
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None
