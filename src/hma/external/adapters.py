"""Adapters executed inside pinned external-model environments."""

from __future__ import annotations

import importlib
import json
import sys
import time
import statistics
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections import abc as collections_abc
from contextlib import contextmanager
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
        warmup_runs: int = 20,
        measured_runs: int = 100,
        repeats: int = 5,
    ) -> dict[str, Any]:
        if not images:
            return {}
        torch = self.torch
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        image_ids = [str(index) for index in range(len(images))]
        for _ in range(max(0, warmup_runs)):
            self.run_batch(images, image_ids)
        cuda_synchronized = bool(
            self.device.startswith("cuda") and torch.cuda.is_available()
        )
        repeat_latencies: list[float] = []
        for _repeat in range(max(1, repeats)):
            if cuda_synchronized:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(max(1, measured_runs)):
                self.run_batch(images, image_ids)
            if cuda_synchronized:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            repeat_latencies.append(
                1000.0 * elapsed / (max(1, measured_runs) * len(images))
            )
        parameter_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        trainable_parameter_count = sum(
            int(parameter.numel())
            for parameter in self.model.parameters()
            if bool(getattr(parameter, "requires_grad", False))
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated())
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        peak_reserved_memory = (
            int(torch.cuda.max_memory_reserved())
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        theoretical_flops = self.adapter_config.get("full_token_flops")
        flop_profile = _profile_flops(
            self.model,
            self.preprocess(images[:1]),
        )
        realized_flops = flop_profile["total_flops"]
        if theoretical_flops is None and realized_flops is not None:
            theoretical_flops = realized_flops
        return {
            "parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "latency_ms_per_image": statistics.mean(repeat_latencies),
            "latency_ms_per_image_std": (
                statistics.stdev(repeat_latencies)
                if len(repeat_latencies) > 1
                else 0.0
            ),
            "latency_ms_per_image_cv": (
                statistics.stdev(repeat_latencies)
                / statistics.mean(repeat_latencies)
                if len(repeat_latencies) > 1
                and statistics.mean(repeat_latencies) != 0.0
                else 0.0
            ),
            "latency_ms_per_image_min": min(repeat_latencies),
            "latency_ms_per_image_max": max(repeat_latencies),
            "latency_repeats_ms_per_image": repeat_latencies,
            "batch_size": len(images),
            "warmup_batches": int(warmup_runs),
            "measured_batches_per_repeat": int(measured_runs),
            "timing_repeats": int(repeats),
            "cuda_synchronized": cuda_synchronized,
            "hardware": hardware_metadata(torch),
            "peak_memory_bytes": peak_memory,
            "peak_reserved_memory_bytes": peak_reserved_memory,
            "theoretical_flops": (
                int(theoretical_flops) if theoretical_flops is not None else None
            ),
            "realized_flops": (
                int(realized_flops) if realized_flops is not None else None
            ),
            "fvcore_unsupported_ops": flop_profile["unsupported_ops"],
            "fvcore_uncalled_modules": flop_profile["uncalled_modules"],
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
            "task_outputs": sorted(output.task_outputs),
            "scanpath_records": len(output.scanpaths),
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
        _require_module("timm")
        transforms_factory = importlib.import_module("timm.data.transforms_factory")
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
        self.processor = transformers.AutoImageProcessor.from_pretrained(checkpoint)
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
    """Official Hugging Face MambaVision adapter exposing four stage features."""

    def load_model(self) -> Any:
        transformers = _require_module("transformers")
        timm = _require_module("timm")
        checkpoint = str(self.checkpoint_path or self.adapter_config["checkpoint_id"])
        model = transformers.AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        config = getattr(model, "config", None)
        input_resolution = tuple(self.preprocessing.get("input_size", [224, 224]))
        self.transform = transforms_factory.create_transform(
            input_size=(3, int(input_resolution[0]), int(input_resolution[1])),
            is_training=False,
            mean=getattr(config, "mean", self.mean),
            std=getattr(config, "std", self.std),
            crop_mode=getattr(config, "crop_mode", "center"),
            crop_pct=getattr(config, "crop_pct", self.crop_pct),
        )
        return model.eval().to(self.device)

    def preprocess(self, images: list[Image.Image]) -> Any:
        tensors = [self.transform(image.convert("RGB")) for image in images]
        return self.torch.stack(tensors, dim=0).to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        tensor = self.preprocess(images)
        with self.torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, tuple):
            pooled, stage_features = output
        else:
            pooled = getattr(output, "pooler_output", None)
            stage_features = getattr(output, "hidden_states", None)
        if stage_features is None:
            raise ExternalIntegrationError("MambaVision did not return stage features")
        stage_features = list(stage_features)
        features: dict[str, Any] = {}
        for layer in self.layers:
            if not layer.startswith("levels."):
                continue
            index = int(layer.split(".", 1)[1])
            if index >= len(stage_features):
                raise ExternalIntegrationError(
                    f"MambaVision stage for {layer} is unavailable"
                )
            features[layer] = _first_tensor(stage_features[index]).detach()
        missing = [layer for layer in self.layers if layer not in features]
        if missing:
            raise ExternalIntegrationError(f"MambaVision stages not captured: {missing}")
        logits = pooled
        return ExternalBatchOutput(features=features, logits=logits)


class TimmFeatureAdapter(ExternalModelAdapter):
    """Generic publication adapter for timm image-only feature backbones."""

    def load_model(self) -> Any:
        timm = _require_module("timm")
        model_name = str(self.adapter_config["model_name"])
        pretrained = bool(
            self.adapter_config.get("pretrained", self.checkpoint_path is None)
        )
        model = timm.create_model(model_name, pretrained=pretrained)
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            _load_checkpoint(model, self.checkpoint_path, self.torch)
        return model.eval().to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        tensor = self.preprocess(images)
        modules = dict(self.model.named_modules())
        features: dict[str, Any] = {}
        handles = []
        for layer in self.layers:
            if layer not in modules:
                raise ExternalIntegrationError(f"timm feature layer not found: {layer}")
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
        missing = [layer for layer in self.layers if layer not in features]
        if missing:
            raise ExternalIntegrationError(f"timm feature layers not captured: {missing}")
        return ExternalBatchOutput(features=features, logits=logits)


class _AuditPendingAdapter(ExternalModelAdapter):
    audit_message = "Adapter requires an official checkpoint/API audit before execution."

    def load_model(self) -> Any:
        raise ExternalIntegrationError(self.audit_message)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        raise ExternalIntegrationError(self.audit_message)


class DINOv3Adapter(ExternalModelAdapter):
    """Official Hugging Face DINOv3 adapter prepared for gated local snapshots."""

    def load_model(self) -> Any:
        transformers = _require_module("transformers")
        checkpoint = str(self.checkpoint_path or self.adapter_config["checkpoint_id"])
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        model = transformers.AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        return model.eval().to(self.device)

    def preprocess(self, images: list[Image.Image]) -> Any:
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        tensor = self.preprocess(images)
        with self.torch.inference_mode():
            output = self.model(pixel_values=tensor, output_hidden_states=True)
        hidden_states = list(getattr(output, "hidden_states", []) or [])
        features: dict[str, Any] = {}
        for layer in self.layers:
            if layer.startswith("blocks."):
                index = int(layer.split(".", 1)[1])
                hidden_index = index + 1
                if hidden_index >= len(hidden_states):
                    raise ExternalIntegrationError(
                        f"DINOv3 hidden state for {layer} is unavailable"
                    )
                features[layer] = hidden_states[hidden_index].detach()
        missing = [layer for layer in self.layers if layer not in features]
        if missing:
            raise ExternalIntegrationError(f"DINOv3 layers not captured: {missing}")
        logits = getattr(output, "pooler_output", None)
        if logits is None:
            last_hidden = getattr(output, "last_hidden_state", None)
            if last_hidden is not None:
                logits = last_hidden[:, 0]
        return ExternalBatchOutput(features=features, logits=logits)


class HieraAdapter(_AuditPendingAdapter):
    audit_message = "Hiera requires a resolved official source pin and checkpoint lock."


class SwinAdapter(_AuditPendingAdapter):
    audit_message = "Swin requires a resolved official source pin and checkpoint lock."


class HATAdapter(ExternalModelAdapter):
    """Official HAT free-viewing adapter prepared around the demo inference API."""

    def load_model(self) -> Any:
        self._ensure_hat_assets()
        _install_pillow_legacy_resampling_compatibility()
        JsonConfig = importlib.import_module("common.config").JsonConfig
        hat_models = importlib.import_module("hat.models")
        config_path = self.source_dir / str(
            self.adapter_config.get("config_path", "configs/coco_freeview_dense_SSL.json")
        )
        if not config_path.is_file():
            raise ExternalIntegrationError(f"HAT config is missing: {config_path}")
        hparams = JsonConfig(str(config_path))
        hparams.Data.backbone_config = str(self._absolute_backbone_config())
        hparams.Train.parallel = False
        self.hparams = hparams
        with _hat_detectron2_resnet_stage_compatibility(self.torch):
            model = hat_models.HumanAttnTransformer(
                hparams.Data,
                num_decoder_layers=hparams.Model.n_dec_layers,
                hidden_dim=hparams.Model.embedding_dim,
                nhead=hparams.Model.n_heads,
                ntask=1 if hparams.Data.TAP == "FV" else 18,
                tgt_vocab_size=hparams.Data.patch_count + len(hparams.Data.special_symbols),
                num_output_layers=hparams.Model.num_output_layers,
                separate_fix_arch=hparams.Model.separate_fix_arch,
                train_encoder=hparams.Train.train_backbone,
                train_pixel_decoder=hparams.Train.train_pixel_decoder,
                use_dino=hparams.Train.use_dino_pretrained_model,
                dropout=hparams.Train.dropout,
                dim_feedforward=hparams.Model.hidden_dim,
                parallel_arch=hparams.Model.parallel_arch,
                dorsal_source=hparams.Model.dorsal_source,
                num_encoder_layers=hparams.Model.n_enc_layers,
                output_centermap="centermap_pred" in hparams.Train.losses,
                output_saliency="saliency_pred" in hparams.Train.losses,
                output_target_map="target_map_pred" in hparams.Train.losses,
                transfer_learning_setting=hparams.Train.transfer_learn,
                project_queries=hparams.Train.project_queries,
                is_pretraining=False,
                output_feature_map_name=hparams.Model.output_feature_map_name,
            )
        checkpoint = self._hat_checkpoint_file()
        payload = self.torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise ExternalIntegrationError(f"HAT checkpoint has no model state: {checkpoint}")
        normalized = _normalize_hat_checkpoint_state_dict(state)
        result = model.load_state_dict(normalized, strict=False)
        missing = [
            key
            for key in getattr(result, "missing_keys", [])
            if not key.endswith("num_batches_tracked")
        ]
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            raise ExternalIntegrationError(
                "HAT checkpoint did not match the official architecture; "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model.eval().to(self.device)

    def _ensure_hat_assets(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.is_dir():
            raise ExternalIntegrationError(f"HAT checkpoint directory is missing: {self.checkpoint_path}")
        required = [
            "M2F_R50.pkl",
            "M2F_R50_MSDeformAttnPixelDecoder.pkl",
            str(self.adapter_config.get("hat_checkpoint", "HAT_FV.pt")),
        ]
        missing = [name for name in required if not (self.checkpoint_path / name).is_file()]
        if missing:
            raise ExternalIntegrationError(
                f"HAT checkpoint directory is missing files: {missing}"
            )

    def _absolute_backbone_config(self) -> Path:
        source_config = self.source_dir / "configs" / "resnet50.yaml"
        if self.checkpoint_path is None:
            raise ExternalIntegrationError("HAT checkpoint path is unresolved")
        target = (
            self.checkpoint_path.parent
            / "generated_configs"
            / "hma_resnet50_absolute.yaml"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        weight_path = (self.checkpoint_path / "M2F_R50.pkl").resolve().as_posix()
        lines = source_config.read_text(encoding="utf-8").splitlines()
        rewritten = []
        replaced = False
        for line in lines:
            if line.strip().startswith("WEIGHTS:"):
                indent = line[: len(line) - len(line.lstrip())]
                rewritten.append(f'{indent}WEIGHTS: "{weight_path}"')
                replaced = True
            else:
                rewritten.append(line)
        if not replaced:
            raise ExternalIntegrationError(f"HAT backbone config has no WEIGHTS field: {source_config}")
        target.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        return target

    def _hat_checkpoint_file(self) -> Path:
        if self.checkpoint_path is None:
            raise ExternalIntegrationError("HAT checkpoint path is unresolved")
        return self.checkpoint_path / str(
            self.adapter_config.get("hat_checkpoint", "HAT_FV.pt")
        )

    def preprocess(self, images: list[Image.Image]) -> Any:
        torchvision = _require_module("torchvision")
        size = (int(self.hparams.Data.im_h), int(self.hparams.Data.im_w))
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return self.torch.stack([transform(image.convert("RGB")) for image in images]).to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        if not images:
            raise ExternalIntegrationError("HAT requires at least one image")
        tensor = self.preprocess(images)
        task_ids = self.torch.full(
            (len(images),),
            int(self.adapter_config.get("task_id", 0)),
            dtype=self.torch.long,
            device=self.device,
        )
        with self.torch.inference_mode():
            dorsal_embs, dorsal_pos, dorsal_mask, high_res_featmaps = self.model.encode(tensor)
            decode = self._decode_greedy(
                dorsal_embs=dorsal_embs,
                dorsal_pos=dorsal_pos,
                dorsal_mask=dorsal_mask,
                high_res_featmaps=high_res_featmaps,
                task_ids=task_ids,
            )
        query_state = dorsal_pos[0] if isinstance(dorsal_pos, tuple) else dorsal_pos
        if getattr(query_state, "ndim", 0) == 3:
            query_state = query_state.permute(1, 0, 2).contiguous()
        feature_candidates = {
            "dorsal_embeddings": dorsal_embs.permute(1, 0, 2).contiguous().detach().clone(),
            "high_res_feature_map": high_res_featmaps[0].detach().clone(),
            "working_memory": decode["normalized_fixations"].detach().clone(),
            "query_state": query_state.detach().clone(),
        }
        missing = [layer for layer in self.layers if layer not in feature_candidates]
        if missing:
            raise ExternalIntegrationError(f"HAT latent layers not captured: {missing}")
        scanpaths = self._scanpaths(
            image_ids=image_ids,
            images=images,
            normalized_fixations=decode["normalized_fixations"],
            stop_trace=decode["stop_trace"],
        )
        resource = self._resource_allocation(
            normalized_fixations=decode["normalized_fixations"],
            stop_trace=decode["stop_trace"],
        )
        return ExternalBatchOutput(
            features={layer: feature_candidates[layer] for layer in self.layers},
            task_outputs={
                "conditional_next_fixation": decode["first_fixation_map"],
                "termination": decode["stop_trace"],
            },
            resource_allocation=resource,
            scanpaths=scanpaths,
        )

    def _decode_greedy(
        self,
        *,
        dorsal_embs: Any,
        dorsal_pos: Any,
        dorsal_mask: Any,
        high_res_featmaps: Any,
        task_ids: Any,
    ) -> dict[str, Any]:
        utils = importlib.import_module("common.utils")
        evaluation = importlib.import_module("hat.evaluation")
        pa = self.hparams.Data
        batch_size = int(task_ids.shape[0])
        max_fixations = int(
            self.adapter_config.get("max_fixations", min(6, int(pa.max_traj_length)))
        )
        normalized_fixations = self.torch.zeros(batch_size, 1, 2).fill_(0.5)
        action_mask = evaluation.get_IOR_mask(
            np.ones(batch_size) * 0.5,
            np.ones(batch_size) * 0.5,
            int(pa.im_h),
            int(pa.im_w),
            int(pa.IOR_radius),
        ).to(self.device).bool()
        stop_values = []
        first_fixation_map = None
        for _step in range(max_fixations):
            padding_mask = self.torch.zeros(
                batch_size,
                normalized_fixations.shape[1],
                dtype=self.torch.bool,
            )
            ys, ys_high = utils.transform_fixations(
                normalized_fixations,
                padding_mask,
                pa,
                False,
                return_highres=True,
            )
            out = self.model.decode_and_predict(
                dorsal_embs.clone(),
                dorsal_pos,
                dorsal_mask,
                high_res_featmaps,
                ys.to(self.device),
                None,
                ys_high.to(self.device),
                task_ids,
            )
            fixation_map = out["pred_fixation_map"]
            if first_fixation_map is None:
                first_fixation_map = fixation_map.detach().clone()
            stop_values.append(out["pred_termination"].detach().reshape(batch_size))
            probability = fixation_map.reshape(batch_size, -1).clone()
            if bool(getattr(pa, "enforce_IOR", True)):
                probability = probability.masked_fill(action_mask, 0)
            next_word = self.torch.argmax(probability, dim=1).detach().cpu()
            norm_y = (next_word // int(pa.im_w)).to(dtype=self.torch.float32) / float(pa.im_h)
            norm_x = (next_word % int(pa.im_w)).to(dtype=self.torch.float32) / float(pa.im_w)
            normalized_fixations = self.torch.cat(
                [
                    normalized_fixations,
                    self.torch.stack([norm_x, norm_y], dim=1).unsqueeze(1),
                ],
                dim=1,
            )
            new_mask = evaluation.get_IOR_mask(
                norm_x.numpy(),
                norm_y.numpy(),
                int(pa.im_h),
                int(pa.im_w),
                int(pa.IOR_radius),
            ).to(self.device).bool()
            action_mask = self.torch.logical_or(action_mask, new_mask)
        if first_fixation_map is None:
            raise ExternalIntegrationError("HAT decode produced no fixation map")
        return {
            "first_fixation_map": first_fixation_map,
            "normalized_fixations": normalized_fixations.to(self.device),
            "stop_trace": self.torch.stack(stop_values, dim=1),
        }

    def _scanpaths(
        self,
        *,
        image_ids: list[str],
        images: list[Image.Image],
        normalized_fixations: Any,
        stop_trace: Any,
    ) -> list[dict[str, Any]]:
        values = normalized_fixations.detach().cpu().numpy()
        stops = stop_trace.detach().cpu().numpy()
        threshold = float(
            self.adapter_config.get(
                "terminate_threshold",
                getattr(self.hparams.Data, "terminate_threshold", 0.75),
            )
        )
        records: list[dict[str, Any]] = []
        for batch_index, image_id in enumerate(image_ids):
            width, height = images[batch_index].size
            stop_indices = np.where(stops[batch_index] > threshold)[0]
            predicted_steps = (
                int(stop_indices[0]) + 1 if len(stop_indices) else values.shape[1] - 1
            )
            fixations = []
            for step_index, (x_norm, y_norm) in enumerate(
                values[batch_index, : predicted_steps + 1]
            ):
                fixations.append(
                    {
                        "step": step_index,
                        "x": float(x_norm) * width,
                        "y": float(y_norm) * height,
                        "x_norm": float(x_norm),
                        "y_norm": float(y_norm),
                        "stop": bool(
                            step_index > 0
                            and stops[batch_index, step_index - 1] > threshold
                        ),
                    }
                )
            records.append(
                {
                    "image_id": str(image_id),
                    "model_id": self.model_id,
                    "condition": "freeview_initial_center_greedy",
                    "fixations": fixations,
                    "scanpath_length": len(fixations),
                }
            )
        return records

    def _resource_allocation(
        self,
        *,
        normalized_fixations: Any,
        stop_trace: Any,
    ) -> dict[str, Any]:
        values = normalized_fixations.detach().cpu().numpy()
        stops = stop_trace.detach().cpu().numpy()
        threshold = float(
            self.adapter_config.get(
                "terminate_threshold",
                getattr(self.hparams.Data, "terminate_threshold", 0.75),
            )
        )
        counts = []
        lengths = []
        for batch_index in range(values.shape[0]):
            stop_indices = np.where(stops[batch_index] > threshold)[0]
            predicted_steps = (
                int(stop_indices[0]) + 1 if len(stop_indices) else values.shape[1] - 1
            )
            counts.append(predicted_steps + 1)
            points = values[batch_index, : predicted_steps + 1]
            scaled = points * np.asarray(
                [float(self.hparams.Data.im_w), float(self.hparams.Data.im_h)],
                dtype=np.float32,
            )
            lengths.append(
                float(np.linalg.norm(np.diff(scaled, axis=0), axis=1).sum())
                if len(scaled) > 1
                else 0.0
            )
        fovea_area = np.pi * float(self.hparams.Data.IOR_radius) ** 2
        return {
            "fixation_count": np.asarray(counts, dtype=np.int32),
            "selected_glimpses": normalized_fixations,
            "stopping_behavior": (stop_trace > threshold).to(dtype=self.torch.uint8),
            "high_resolution_sampled_area": np.asarray(
                [count * fovea_area for count in counts],
                dtype=np.float32,
            ),
            "scanpath_length": np.asarray(lengths, dtype=np.float32),
        }

    def profile_efficiency(
        self,
        images: list[Image.Image],
        *,
        warmup_runs: int = 2,
        measured_runs: int = 5,
        repeats: int = 2,
    ) -> dict[str, Any]:
        if not images:
            return {}
        repeat_latencies: list[float] = []
        image_ids = [str(index) for index in range(len(images))]
        for _ in range(max(0, warmup_runs)):
            self.run_batch(images, image_ids)
        cuda_synchronized = bool(
            self.device.startswith("cuda") and self.torch.cuda.is_available()
        )
        if cuda_synchronized:
            self.torch.cuda.reset_peak_memory_stats()
        for _repeat in range(max(1, repeats)):
            if cuda_synchronized:
                self.torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(max(1, measured_runs)):
                self.run_batch(images, image_ids)
            if cuda_synchronized:
                self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            repeat_latencies.append(
                1000.0 * elapsed / (max(1, measured_runs) * len(images))
            )
        return {
            "parameters": sum(int(parameter.numel()) for parameter in self.model.parameters()),
            "latency_ms_per_image": statistics.mean(repeat_latencies),
            "latency_repeats_ms_per_image": repeat_latencies,
            "batch_size": len(images),
            "warmup_batches": int(warmup_runs),
            "measured_batches_per_repeat": int(measured_runs),
            "timing_repeats": int(repeats),
            "cuda_synchronized": cuda_synchronized,
            "hardware": hardware_metadata(self.torch),
            "peak_memory_bytes": (
                int(self.torch.cuda.max_memory_allocated()) if cuda_synchronized else 0
            ),
            "theoretical_flops": None,
            "realized_flops": None,
        }


class ScanDiffAdapter(_AuditPendingAdapter):
    audit_message = (
        "ScanDiff requires the official Hydra checkpoint/data snapshot before "
        "scanpath export can execute."
    )


class DeepGazeIIEAdapter(ExternalModelAdapter):
    """DeepGaze IIE adapter with frozen center-bias and latent hooks."""

    def load_model(self) -> Any:
        deepgaze = _require_module("deepgaze_pytorch")
        centerbias_path = Path(
            self.adapter_config.get(
                "centerbias_path",
                "data/precomputed/deepgaze/centerbias_mit1003.npy",
            )
        )
        if not centerbias_path.is_absolute():
            centerbias_path = Path.cwd() / centerbias_path
        if not centerbias_path.is_file():
            raise ExternalIntegrationError(
                f"DeepGaze center-bias asset is missing: {centerbias_path}"
            )
        self.centerbias_template = np.asarray(
            np.load(centerbias_path),
            dtype=np.float32,
        )
        model = deepgaze.DeepGazeIIE(pretrained=True)
        return model.eval().to(self.device)

    def run_batch(
        self,
        images: list[Image.Image],
        image_ids: list[str],
    ) -> ExternalBatchOutput:
        if not images:
            raise ExternalIntegrationError("DeepGaze requires at least one image")
        shapes = {(image.height, image.width) for image in images}
        if len(shapes) != 1:
            raise ExternalIntegrationError(
                "DeepGaze batches require images with a common spatial shape"
            )
        arrays = [
            np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
            for image in images
        ]
        tensor = self.torch.as_tensor(
            np.stack(arrays, axis=0),
            dtype=self.torch.float32,
            device=self.device,
        )
        height, width = next(iter(shapes))
        centerbias = _resize_log_density(
            self.centerbias_template,
            target_shape=(height, width),
        )
        centerbias_tensor = self.torch.as_tensor(
            np.repeat(centerbias[None], len(images), axis=0),
            dtype=self.torch.float32,
            device=self.device,
        )
        features, handles = _register_feature_hooks(
            self.model,
            self.layers,
            label="DeepGaze IIE",
        )
        try:
            with self.torch.inference_mode():
                log_density = self.model(tensor, centerbias_tensor)
        finally:
            for handle in handles:
                handle.remove()
        probability = self.torch.exp(log_density)
        if probability.ndim == 4 and probability.shape[1] == 1:
            probability = probability[:, 0]
        return ExternalBatchOutput(
            features=features,
            task_outputs={"human_gaze_density": probability},
        )


class DeepGazeIIIAdapter(ExternalModelAdapter):
    """DeepGaze III conditional next-fixation and greedy scanpath adapter."""

    def load_model(self) -> Any:
        deepgaze = _require_module("deepgaze_pytorch")
        self.centerbias_template = _load_deepgaze_centerbias(self.adapter_config)
        self.history_length = int(self.adapter_config.get("history_length", 4))
        self.max_fixations = int(self.adapter_config.get("max_fixations", 6))
        if self.history_length <= 0:
            raise ExternalIntegrationError("DeepGaze III history_length must be positive")
        if self.max_fixations <= 0:
            raise ExternalIntegrationError("DeepGaze III max_fixations must be positive")
        model = deepgaze.DeepGazeIII(pretrained=True)
        return model.eval().to(self.device)

    def run_batch(
        self,
        images: list[Image.Image],
        image_ids: list[str],
    ) -> ExternalBatchOutput:
        tensor, centerbias_tensor, height, width = _deepgaze_inputs(
            images,
            torch_module=self.torch,
            device=self.device,
            centerbias_template=self.centerbias_template,
        )
        x_hist, y_hist = _initial_center_history(
            len(images),
            width=width,
            height=height,
            history_length=self.history_length,
            torch_module=self.torch,
            device=self.device,
        )
        features, handles = _register_feature_hooks(
            self.model,
            self.layers,
            label="DeepGaze III",
        )
        try:
            with self.torch.inference_mode():
                log_density = self.model(
                    tensor,
                    centerbias_tensor,
                    x_hist=x_hist,
                    y_hist=y_hist,
                )
        finally:
            for handle in handles:
                handle.remove()
        probability = self.torch.exp(log_density)
        if probability.ndim == 4 and probability.shape[1] == 1:
            probability = probability[:, 0]
        scanpaths, resource = _deepgaze_greedy_scanpaths(
            self.model,
            tensor,
            centerbias_tensor,
            image_ids=image_ids,
            width=width,
            height=height,
            history_length=self.history_length,
            max_fixations=self.max_fixations,
            torch_module=self.torch,
        )
        resource["history_length"] = np.full(
            len(images), self.history_length, dtype=np.int32
        )
        return ExternalBatchOutput(
            features=features,
            task_outputs={"conditional_next_fixation": probability},
            resource_allocation=resource,
            scanpaths=scanpaths,
        )


class DeepGazeMSDBAdapter(ExternalModelAdapter):
    """DeepGaze MSDB saliency-density adapter with backbone/readout hooks."""

    def load_model(self) -> Any:
        deepgaze = _require_module("deepgaze_pytorch")
        self.centerbias_template = _load_deepgaze_centerbias(self.adapter_config)
        self.pixel_per_dva = float(self.adapter_config.get("pixel_per_dva", 21.7))
        dataset = self.adapter_config.get("dataset")
        self.dataset = None if dataset in (None, "", "none") else int(dataset)
        if self.pixel_per_dva <= 0:
            raise ExternalIntegrationError("DeepGaze MSDB pixel_per_dva must be positive")
        model = deepgaze.DeepGazeMSDB(pretrained=True)
        model = model.to(self.device)
        model.train(False)
        return model

    def run_batch(
        self,
        images: list[Image.Image],
        image_ids: list[str],
    ) -> ExternalBatchOutput:
        tensor, centerbias_tensor, _height, _width = _deepgaze_inputs(
            images,
            torch_module=self.torch,
            device=self.device,
            centerbias_template=self.centerbias_template,
        )
        features, handles = _register_feature_hooks(
            self.model,
            self.layers,
            label="DeepGaze MSDB",
        )
        try:
            with self.torch.inference_mode():
                log_density = self.model(
                    tensor,
                    centerbias_tensor,
                    pixel_per_dva=self.pixel_per_dva,
                    dataset=self.dataset,
                )
        finally:
            for handle in handles:
                handle.remove()
        probability = self.torch.exp(log_density)
        if probability.ndim == 4 and probability.shape[1] == 1:
            probability = probability[:, 0]
        return ExternalBatchOutput(
            features=features,
            task_outputs={"human_gaze_density": probability},
        )


class AdaptiveNNAdapter(ExternalModelAdapter):
    """Official AdaptiveNN-DeiT-S adapter exposing policy states and glimpses."""

    def load_model(self) -> Any:
        self._use_vendored_timm()
        dynamic_deit = importlib.import_module("models.dynamic_deitS")
        seq_len = int(self.adapter_config.get("seq_l", 4))
        model = dynamic_deit.dynamic_deitS(
            seq_len=seq_len,
            feature_in_chans=int(self.adapter_config.get("feature_in_chans", 384)),
            recover_n=int(self.adapter_config.get("recover_n", 5)),
            remaining_blocks=int(self.adapter_config.get("remaining_blocks", 4)),
            glance_input_size=int(self.adapter_config.get("glance_input_size", 112)),
            glance_net_depth=int(self.adapter_config.get("glance_net_depth", 12)),
            glance_net_mlp_ratio=float(self.adapter_config.get("glance_net_mlp_ratio", 4)),
            glance_net_drop_path=float(self.adapter_config.get("glance_net_drop_path", 0.1)),
            focus_patch_size=int(self.adapter_config.get("focus_patch_size", 112)),
            focus_net_reg_size=int(self.adapter_config.get("focus_net_reg_size", 160)),
            focus_net_depth=int(self.adapter_config.get("focus_net_depth", 12)),
            focus_net_mlp_ratio=float(self.adapter_config.get("focus_net_mlp_ratio", 4)),
            focus_net_drop_path=float(self.adapter_config.get("focus_net_drop_path", 0.1)),
            multi_cls_drop_path=float(self.adapter_config.get("multi_cls_drop_path", 0.1)),
            policy_net_hidden_chans=int(self.adapter_config.get("policy_net_hidden_chans", 128)),
            policy_net_kernel_size=int(self.adapter_config.get("policy_net_kernel_size", 3)),
            pretrained=False,
            num_classes=int(self.adapter_config.get("num_classes", 1000)),
        )
        _load_checkpoint(model, self.checkpoint_path, self.torch)
        return model.eval().to(self.device)

    def _use_vendored_timm(self) -> None:
        vendor_root = self.source_dir / "models"
        if not (vendor_root / "timm").is_dir():
            raise ExternalIntegrationError(
                f"AdaptiveNN vendored timm package is missing: {vendor_root / 'timm'}"
            )
        vendor_root_text = str(vendor_root)
        if vendor_root_text in sys.path:
            sys.path.remove(vendor_root_text)
        sys.path.insert(0, vendor_root_text)
        existing = sys.modules.get("timm")
        existing_path = Path(str(getattr(existing, "__file__", ""))) if existing else None
        if existing_path and vendor_root not in existing_path.parents:
            for name in list(sys.modules):
                if name == "timm" or name.startswith("timm."):
                    sys.modules.pop(name, None)
            for name in list(sys.modules):
                if name == "models.dynamic_deitS" or name.startswith("models.dynamic_deitS."):
                    sys.modules.pop(name, None)

    def preprocess(self, images: list[Image.Image]) -> Any:
        torch = self.torch
        torchvision = _require_module("torchvision")
        input_size = int(self.adapter_config.get("input_size", 288))
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (input_size, input_size),
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=self.adapter_config.get("mean", [0.485, 0.456, 0.406]),
                    std=self.adapter_config.get("std", [0.229, 0.224, 0.225]),
                ),
            ]
        )
        return torch.stack([transform(image.convert("RGB")) for image in images]).to(self.device)

    def run_batch(self, images: list[Image.Image], image_ids: list[str]) -> ExternalBatchOutput:
        torch = self.torch
        tensor = self.preprocess(images)
        seq_len = int(self.adapter_config.get("seq_l", 4))
        with torch.inference_mode():
            output = self.model(tensor, seq_l=seq_len, flag="forward_backbone")
        states = [state.detach().clone() for state in output.get("states", [])]
        focus_logits = [value.detach().clone() for value in output.get("x_focus", [])]
        glance_logits = [value.detach().clone() for value in output.get("x_glance", [])]
        actions = [value.detach().clone() for value in output.get("actions", [])]
        state_values = [
            value.detach().clone() for value in output.get("_state_values", [])
        ]
        features: dict[str, Any] = {}
        if states:
            features["glance_state"] = states[0]
            features["integrated_states"] = torch.stack(states, dim=1)
            features["final_state"] = states[-1]
        focus_regression = None
        if "outputs_reg_focus_net" in output and output["outputs_reg_focus_net"] is not None:
            focus_regression = output["outputs_reg_focus_net"].detach().clone()
        missing = [layer for layer in self.layers if layer not in features]
        if missing:
            raise ExternalIntegrationError(f"AdaptiveNN layers not captured: {missing}")
        selected_glimpses = torch.stack(actions, dim=1) if actions else None
        stopping_behavior = self._stopping_behavior(focus_logits, glance_logits)
        resource: dict[str, Any] = {
            "fixation_count": np.full(len(images), len(actions), dtype=np.int32),
            "stopping_behavior": stopping_behavior,
            "high_resolution_sampled_area": np.full(
                len(images),
                len(actions)
                * int(self.adapter_config.get("focus_patch_size", 112)) ** 2,
                dtype=np.int32,
            ),
        }
        if selected_glimpses is not None:
            resource["selected_glimpses"] = selected_glimpses
        if state_values:
            resource["policy_state_values"] = torch.stack(state_values, dim=1)
        scanpaths = self._scanpaths(
            image_ids=image_ids,
            images=images,
            actions=actions,
            stopping_behavior=stopping_behavior,
        )
        logits = focus_logits[-1] if focus_logits else (glance_logits[-1] if glance_logits else None)
        return ExternalBatchOutput(
            features=features,
            logits=logits,
            task_outputs={
                "selected_glimpse_map": selected_glimpses,
                "stopping_trace": stopping_behavior,
                "focus_regression": focus_regression,
            },
            resource_allocation=resource,
            scanpaths=scanpaths,
        )

    def _stopping_behavior(self, focus_logits: list[Any], glance_logits: list[Any]) -> np.ndarray:
        if not focus_logits:
            batch_size = int(glance_logits[0].shape[0]) if glance_logits else 0
            return np.zeros((batch_size, 0), dtype=np.uint8)
        torch = self.torch
        scores = torch.stack(
            [
                torch.softmax(logit.float(), dim=1).max(dim=1).values
                for logit in focus_logits
            ],
            dim=1,
        )
        threshold = float(self.adapter_config.get("stop_confidence_threshold", 0.0))
        if threshold <= 0.0:
            decisions = torch.zeros_like(scores, dtype=torch.uint8)
            decisions[:, -1] = 1
            return decisions.cpu().numpy()
        return (scores >= threshold).to(torch.uint8).cpu().numpy()

    def _scanpaths(
        self,
        *,
        image_ids: list[str],
        images: list[Image.Image],
        actions: list[Any],
        stopping_behavior: np.ndarray,
    ) -> list[dict[str, Any]]:
        if not actions:
            return []
        focus_patch_size = int(self.adapter_config.get("focus_patch_size", 112))
        stacked = self.torch.stack(actions, dim=1).detach().cpu().numpy()
        records: list[dict[str, Any]] = []
        for batch_index, image_id in enumerate(image_ids):
            width, height = images[batch_index].size
            fixations = []
            for step_index, action in enumerate(stacked[batch_index]):
                x_norm, y_norm = float(action[0]), float(action[1])
                fixations.append(
                    {
                        "step": step_index,
                        "x": x_norm * max(1, width - focus_patch_size),
                        "y": y_norm * max(1, height - focus_patch_size),
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                        "patch_size": focus_patch_size,
                        "stop": bool(stopping_behavior[batch_index, step_index]),
                    }
                )
            records.append(
                {
                    "image_id": image_id,
                    "model_id": self.model_id,
                    "fixations": fixations,
                    "scanpath_length": len(fixations),
                }
            )
        return records


class SemBAAdapter(_AuditPendingAdapter):
    audit_message = (
        "SemBA execution is blocked until an official source, checkpoint, "
        "license, and callable inference API are published."
    )


class SemBAFastAdapter(_AuditPendingAdapter):
    audit_message = (
        "SemBA-FAST execution is blocked until an official source, checkpoint, "
        "license, and target-present/absent inference API are published."
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


def _install_pillow_legacy_resampling_compatibility() -> None:
    """Provide Pillow aliases required by Detectron2 0.6 on Pillow >= 10."""
    resampling = getattr(Image, "Resampling", None)
    aliases = {
        "LINEAR": "BILINEAR",
        "CUBIC": "BICUBIC",
        "ANTIALIAS": "LANCZOS",
    }
    for legacy_name, modern_name in aliases.items():
        if legacy_name in vars(Image):
            continue
        value = getattr(resampling, modern_name, None) if resampling is not None else None
        if value is None:
            value = getattr(Image, modern_name, None)
        if value is not None:
            setattr(Image, legacy_name, value)


@contextmanager
def _hat_detectron2_resnet_stage_compatibility(torch: Any) -> Any:
    """Adapt official HAT ResNet weights to the installed Detectron2 key names."""
    module_class = torch.nn.modules.module.Module
    original_load_state_dict = module_class.load_state_dict

    def compatible_load_state_dict(module: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        if _hat_resnet_needs_stage_prefix_stripped(module, state_dict):
            state_dict = _strip_state_dict_prefix(state_dict, "stages.")
        return original_load_state_dict(module, state_dict, *args, **kwargs)

    module_class.load_state_dict = compatible_load_state_dict
    try:
        yield
    finally:
        module_class.load_state_dict = original_load_state_dict


def _hat_resnet_needs_stage_prefix_stripped(module: Any, state_dict: Any) -> bool:
    keys = [str(key) for key in state_dict.keys()]
    if not any(key.startswith("stages.res") for key in keys):
        return False
    child_names = set(getattr(module, "_modules", {}).keys())
    return {"stem", "res2", "res3", "res4", "res5"}.issubset(child_names)


def _strip_state_dict_prefix(state_dict: Any, prefix: str) -> Any:
    rewritten = OrderedDict()
    for key, value in state_dict.items():
        text = str(key)
        if text.startswith(prefix):
            text = text[len(prefix):]
        rewritten[text] = value
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        rewritten._metadata = metadata  # type: ignore[attr-defined]
    return rewritten


def _normalize_hat_checkpoint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = OrderedDict()
    for key, value in state_dict.items():
        text = str(key)
        if text.startswith("module."):
            text = text[7:]
        if text.startswith("encoder.backbone.stages."):
            text = "encoder.backbone." + text[len("encoder.backbone.stages."):]
        normalized[text] = value
    return normalized


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


def _register_feature_hooks(
    model: Any,
    layers: list[str],
    *,
    label: str,
) -> tuple[dict[str, Any], list[Any]]:
    modules = dict(model.named_modules())
    features: dict[str, Any] = {}
    handles = []
    for layer in layers:
        if layer not in modules:
            raise ExternalIntegrationError(f"{label} layer not found: {layer}")
        handles.append(
            modules[layer].register_forward_hook(
                lambda _module, _inputs, output, name=layer: features.__setitem__(
                    name, _first_tensor(output).detach().clone()
                )
            )
        )
    return features, handles


def _load_deepgaze_centerbias(adapter_config: dict[str, Any]) -> np.ndarray:
    centerbias_path = Path(
        adapter_config.get(
            "centerbias_path",
            "data/precomputed/deepgaze/centerbias_mit1003.npy",
        )
    )
    if not centerbias_path.is_absolute():
        centerbias_path = Path.cwd() / centerbias_path
    if not centerbias_path.is_file():
        raise ExternalIntegrationError(
            f"DeepGaze center-bias asset is missing: {centerbias_path}"
        )
    return np.asarray(np.load(centerbias_path), dtype=np.float32)


def _deepgaze_inputs(
    images: list[Image.Image],
    *,
    torch_module: Any,
    device: str,
    centerbias_template: np.ndarray,
) -> tuple[Any, Any, int, int]:
    if not images:
        raise ExternalIntegrationError("DeepGaze requires at least one image")
    shapes = {(image.height, image.width) for image in images}
    if len(shapes) != 1:
        raise ExternalIntegrationError(
            "DeepGaze batches require images with a common spatial shape"
        )
    arrays = [
        np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
        for image in images
    ]
    tensor = torch_module.as_tensor(
        np.stack(arrays, axis=0),
        dtype=torch_module.float32,
        device=device,
    )
    height, width = next(iter(shapes))
    centerbias = _resize_log_density(
        centerbias_template,
        target_shape=(height, width),
    )
    centerbias_tensor = torch_module.as_tensor(
        np.repeat(centerbias[None], len(images), axis=0),
        dtype=torch_module.float32,
        device=device,
    )
    return tensor, centerbias_tensor, height, width


def _initial_center_history(
    batch_size: int,
    *,
    width: int,
    height: int,
    history_length: int,
    torch_module: Any,
    device: str,
) -> tuple[Any, Any]:
    x = torch_module.full(
        (batch_size, history_length),
        float((width - 1) / 2.0),
        dtype=torch_module.float32,
        device=device,
    )
    y = torch_module.full(
        (batch_size, history_length),
        float((height - 1) / 2.0),
        dtype=torch_module.float32,
        device=device,
    )
    return x, y


def _deepgaze_greedy_scanpaths(
    model: Any,
    tensor: Any,
    centerbias_tensor: Any,
    *,
    image_ids: list[str],
    width: int,
    height: int,
    history_length: int,
    max_fixations: int,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_hist, y_hist = _initial_center_history(
        len(image_ids),
        width=width,
        height=height,
        history_length=history_length,
        torch_module=torch_module,
        device=str(tensor.device),
    )
    scanpaths = [
        {
            "image_id": str(image_id),
            "seed": 0,
            "condition": "initial_center_greedy",
            "fixations": [],
        }
        for image_id in image_ids
    ]
    for _step in range(max_fixations):
        with torch_module.inference_mode():
            log_density = model(
                tensor,
                centerbias_tensor,
                x_hist=x_hist,
                y_hist=y_hist,
            )
        if log_density.ndim == 4 and log_density.shape[1] == 1:
            log_density = log_density[:, 0]
        flat = log_density.reshape(log_density.shape[0], -1)
        indices = torch_module.argmax(flat, dim=1)
        next_y = (indices // width).to(dtype=torch_module.float32)
        next_x = (indices % width).to(dtype=torch_module.float32)
        for row_index, record in enumerate(scanpaths):
            record["fixations"].append(
                {
                    "x": float(next_x[row_index].detach().cpu().item()),
                    "y": float(next_y[row_index].detach().cpu().item()),
                    "duration": None,
                }
            )
        x_hist = torch_module.cat([next_x[:, None], x_hist[:, :-1]], dim=1)
        y_hist = torch_module.cat([next_y[:, None], y_hist[:, :-1]], dim=1)
    resource = {
        "fixation_count": np.full(len(image_ids), max_fixations, dtype=np.int32),
        "stopping_behavior": np.zeros(len(image_ids), dtype=np.uint8),
        "scanpath_length": np.asarray(
            [
                _scanpath_length_pixels(record["fixations"])
                for record in scanpaths
            ],
            dtype=np.float32,
        ),
        "total_cost_per_image_task": np.full(
            len(image_ids), max_fixations, dtype=np.float32
        ),
    }
    return scanpaths, resource


def _scanpath_length_pixels(fixations: list[dict[str, Any]]) -> float:
    if len(fixations) < 2:
        return 0.0
    points = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in fixations],
        dtype=np.float32,
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _profile_flops(model: Any, tensor: Any) -> dict[str, Any]:
    try:
        analysis_class = importlib.import_module("fvcore.nn").FlopCountAnalysis
        analysis = analysis_class(model, tensor)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = int(analysis.total())
        unsupported = {
            str(key): int(value)
            for key, value in analysis.unsupported_ops().items()
        }
        uncalled = sorted(str(value) for value in analysis.uncalled_modules())
        return {
            "total_flops": total,
            "unsupported_ops": unsupported,
            "uncalled_modules": uncalled,
        }
    except (ImportError, RuntimeError, TypeError, ValueError):
        return {
            "total_flops": None,
            "unsupported_ops": {},
            "uncalled_modules": [],
        }


def _resize_log_density(
    values: np.ndarray,
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    try:
        scipy_ndimage = importlib.import_module("scipy.ndimage")
        resized = scipy_ndimage.zoom(
            values,
            (
                target_shape[0] / values.shape[0],
                target_shape[1] / values.shape[1],
            ),
            order=0,
            mode="nearest",
        )
    except ImportError:
        image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
        resized = np.asarray(
            image.resize(
                (target_shape[1], target_shape[0]),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
    maximum = float(np.max(resized))
    log_normalizer = maximum + float(np.log(np.sum(np.exp(resized - maximum))))
    return np.asarray(resized - log_normalizer, dtype=np.float32)
