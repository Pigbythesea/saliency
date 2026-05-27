"""Learned spatial readout heads for frozen visual features."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialReadoutConfig:
    """Training options for a target-wise spatial readout."""

    variant: str = "separable"
    spatial_rank: int = 1
    max_epochs: int = 100
    batch_size: int = 32
    target_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-6
    validation_fraction: float = 0.2
    objective: str = "pearson"
    seed: int = 0
    device: str = "cpu"
    progress: bool = False
    progress_every: int = 1


def normalize_spatial_features(
    features: np.ndarray,
    *,
    layout: str = "auto",
) -> np.ndarray:
    """Normalize spatial features to ``[n_images, n_positions, n_channels]``."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 3:
        return np.ascontiguousarray(array)
    if array.ndim != 4:
        raise ValueError(
            "learned_spatial_readout expects features with shape [N, T, C], "
            "[N, H, W, C], or [N, C, H, W]"
        )

    if layout == "channels_last":
        return np.ascontiguousarray(array.reshape(array.shape[0], -1, array.shape[-1]))
    if layout == "channels_first":
        transposed = np.moveaxis(array, 1, -1)
        return np.ascontiguousarray(transposed.reshape(array.shape[0], -1, array.shape[1]))
    if layout != "auto":
        raise ValueError("layout must be 'auto', 'channels_last', or 'channels_first'")

    if array.shape[-1] >= array.shape[1]:
        return np.ascontiguousarray(array.reshape(array.shape[0], -1, array.shape[-1]))
    transposed = np.moveaxis(array, 1, -1)
    return np.ascontiguousarray(transposed.reshape(array.shape[0], -1, array.shape[1]))


def fuse_spatial_feature_layers(
    features_by_layer: dict[str, np.ndarray],
    *,
    layers: list[str],
    fusion_method: str = "channel_concat",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse spatial feature layers while preserving image and token alignment."""
    if fusion_method != "channel_concat":
        raise ValueError("learned_spatial_readout layer_fusion must be 'channel_concat'")
    if not layers:
        raise ValueError("At least one layer is required for learned_spatial_readout fusion")

    normalized_layers: list[np.ndarray] = []
    per_layer_shapes: dict[str, dict[str, list[int]]] = {}
    expected_images: int | None = None
    expected_positions: int | None = None
    for layer in layers:
        if layer not in features_by_layer:
            raise KeyError(f"Missing features for layer '{layer}'")
        raw = np.asarray(features_by_layer[layer], dtype=np.float32)
        normalized = normalize_spatial_features(raw)
        n_images = int(normalized.shape[0])
        n_positions = int(normalized.shape[1])
        if expected_images is None:
            expected_images = n_images
            expected_positions = n_positions
        elif n_images != expected_images:
            raise ValueError(
                "All fused layers must have the same number of images: "
                f"{layer} has {n_images}, expected {expected_images}"
            )
        elif n_positions != expected_positions:
            raise ValueError(
                "All fused layers must have the same number of spatial positions: "
                f"{layer} has {n_positions}, expected {expected_positions}"
            )
        normalized_layers.append(normalized)
        per_layer_shapes[layer] = {
            "input_feature_shape": [int(dim) for dim in raw.shape[1:]],
            "normalized_feature_shape": [int(dim) for dim in normalized.shape[1:]],
        }

    fused = np.ascontiguousarray(np.concatenate(normalized_layers, axis=2))
    metadata = {
        "fusion_method": fusion_method,
        "layers": list(layers),
        "per_layer_input_feature_shapes": {
            layer: shapes["input_feature_shape"]
            for layer, shapes in per_layer_shapes.items()
        },
        "per_layer_normalized_feature_shapes": {
            layer: shapes["normalized_feature_shape"]
            for layer, shapes in per_layer_shapes.items()
        },
        "fused_feature_shape": [int(dim) for dim in fused.shape[1:]],
    }
    return fused, metadata


def fit_spatial_readout(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    config: SpatialReadoutConfig,
) -> dict[str, Any]:
    """Fit a target-wise spatial readout and return model state plus metadata."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("learned_spatial_readout requires PyTorch") from exc

    if config.objective != "pearson":
        raise ValueError("learned_spatial_readout objective must be 'pearson'")
    if train_idx.size < 1 or validation_idx.size < 1:
        raise ValueError("learned_spatial_readout requires non-empty train and validation splits")

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    y = np.asarray(responses, dtype=np.float32)
    feature_matrix = np.asarray(features)
    if feature_matrix.shape[0] != y.shape[0]:
        raise ValueError("features and responses must have the same number of rows")
    if y.ndim != 2:
        raise ValueError("learned_spatial_readout expects a 2D response matrix")
    sample_features = normalize_spatial_features(feature_matrix[0][None, ...])

    y_mean = y[train_idx].mean(axis=0, keepdims=True)
    y_std = y[train_idx].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    y_mean = y_mean.astype(np.float32)
    y_standardized = (y - y_mean) / y_std

    requested_device = config.device
    device = torch.device(
        "cuda"
        if str(requested_device).startswith("cuda") and torch.cuda.is_available()
        else requested_device
    )
    model = _build_spatial_readout_model(
        variant=config.variant,
        n_positions=int(sample_features.shape[1]),
        n_channels=int(sample_features.shape[2]),
        n_targets=int(y.shape[1]),
        target_batch_size=int(config.target_batch_size),
        spatial_rank=int(config.spatial_rank),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = nn.MSELoss()

    train_dataset = _SpatialFeatureDataset(
        feature_matrix,
        y_standardized,
        train_idx,
    )
    generator = torch.Generator()
    generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
    )

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    stopped_early = False
    started_at = perf_counter()

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_predictions = _predict_standardized_in_batches(
                model,
                feature_matrix,
                validation_idx,
                batch_size=int(config.batch_size),
                device=device,
            )
            y_val = torch.from_numpy(y_standardized[validation_idx]).to(device)
            val_loss = float(loss_fn(val_predictions, y_val).detach().cpu().item())
            val_score = float(_mean_columnwise_pearson_torch(val_predictions, y_val))
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_mean_pearson": val_score,
            }
        )

        improved = val_score > best_score + float(config.min_delta)
        if val_score > best_score + float(config.min_delta):
            best_score = val_score
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                stopped_early = True
                _print_progress(
                    config=config,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_score=val_score,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    started_at=started_at,
                    forced=True,
                    stopped_early=True,
                )
                break
        _print_progress(
            config=config,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_score=val_score,
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            started_at=started_at,
            forced=improved,
            stopped_early=False,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    metadata = {
        "method": "learned_spatial_readout",
        "readout_variant": str(config.variant),
        "spatial_rank": int(config.spatial_rank),
        "input_feature_shape": [int(dim) for dim in np.asarray(features).shape[1:]],
        "normalized_feature_shape": [int(dim) for dim in sample_features.shape[1:]],
        "n_train": int(train_idx.size),
        "n_validation": int(validation_idx.size),
        "max_epochs": int(config.max_epochs),
        "batch_size": int(config.batch_size),
        "target_batch_size": int(config.target_batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "objective": config.objective,
        "best_epoch": int(best_epoch),
        "validation_score": float(best_score),
        "validation_score_type": "mean_pearson",
        "stopped_early": bool(stopped_early),
        "epochs_ran": int(len(history)),
        "progress": bool(config.progress),
        "progress_every": int(config.progress_every),
        "parameter_count": int(sum(value.numel() for value in state.values())),
        "trainable_parameter_count": int(sum(value.numel() for value in state.values())),
        "history": history,
    }
    return {
        "state_dict": state,
        "y_mean": y_mean,
        "y_std": y_std,
        "metadata": metadata,
        "config": config,
    }


def predict_spatial_readout(
    model_bundle: dict[str, Any],
    features: np.ndarray,
    *,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Predict unstandardized responses from a fitted spatial readout bundle."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("learned_spatial_readout requires PyTorch") from exc

    feature_matrix = np.asarray(features)
    state = model_bundle["state_dict"]
    spatial_logits = state["spatial_logits"]
    channel_weights = state["channel_weights"]
    if spatial_logits.ndim == 2:
        n_targets, n_positions = spatial_logits.shape
        n_channels = int(channel_weights.shape[1])
    else:
        n_targets = int(spatial_logits.shape[0])
        n_positions = int(spatial_logits.shape[2])
        n_channels = int(channel_weights.shape[2])
    sample_index = 0 if indices is None else int(np.asarray(indices).ravel()[0])
    sample_features = normalize_spatial_features(feature_matrix[sample_index][None, ...])
    if sample_features.shape[1] != n_positions:
        raise ValueError("features do not match fitted readout spatial shape")
    variant = str(getattr(model_bundle["config"], "variant", "separable"))
    spatial_rank = int(getattr(model_bundle["config"], "spatial_rank", 1))
    model = _build_spatial_readout_model(
        variant=variant,
        n_positions=int(n_positions),
        n_channels=n_channels,
        n_targets=int(n_targets),
        target_batch_size=int(model_bundle["config"].target_batch_size),
        spatial_rank=spatial_rank,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        if indices is None:
            prediction_indices = np.arange(feature_matrix.shape[0])
        else:
            prediction_indices = np.asarray(indices, dtype=np.int64).ravel()
        standardized_tensor = _predict_standardized_in_batches(
            model,
            feature_matrix,
            prediction_indices,
            batch_size=int(model_bundle["config"].batch_size),
            device=torch.device("cpu"),
        )
    standardized = standardized_tensor.cpu().numpy().astype(np.float32, copy=False)
    return standardized * model_bundle["y_std"] + model_bundle["y_mean"]


class TargetWiseSpatialReadout:  # pragma: no cover - exercised through training tests
    """Target-wise spatial pooling plus target-wise channel projection."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("learned_spatial_readout requires PyTorch") from exc

        class _TargetWiseSpatialReadout(nn.Module):
            def __init__(
                self,
                *,
                n_positions: int,
                n_channels: int,
                n_targets: int,
                target_batch_size: int,
            ) -> None:
                super().__init__()
                self.n_targets = int(n_targets)
                self.target_batch_size = int(target_batch_size)
                self.spatial_logits = nn.Parameter(
                    torch.zeros(int(n_targets), int(n_positions))
                )
                self.channel_weights = nn.Parameter(
                    torch.empty(int(n_targets), int(n_channels))
                )
                nn.init.normal_(self.channel_weights, mean=0.0, std=0.01)
                self.bias = nn.Parameter(torch.zeros(int(n_targets)))

            def forward(self, features: Any) -> Any:
                weights = torch.softmax(self.spatial_logits, dim=1)
                chunks = []
                for start in range(0, self.n_targets, self.target_batch_size):
                    end = min(start + self.target_batch_size, self.n_targets)
                    spatial = weights[start:end]
                    pooled = torch.einsum("bpc,tp->btc", features, spatial)
                    values = (pooled * self.channel_weights[start:end]).sum(dim=2)
                    values = values + self.bias[start:end]
                    chunks.append(values)
                return torch.cat(chunks, dim=1)

        return _TargetWiseSpatialReadout(*args, **kwargs)


def _build_spatial_readout_model(
    *,
    variant: str,
    n_positions: int,
    n_channels: int,
    n_targets: int,
    target_batch_size: int,
    spatial_rank: int,
) -> Any:
    if variant == "separable":
        return TargetWiseSpatialReadout(
            n_positions=n_positions,
            n_channels=n_channels,
            n_targets=n_targets,
            target_batch_size=target_batch_size,
        )
    if variant == "voxel_specific_lowrank":
        return LowRankVoxelSpatialReadout(
            n_positions=n_positions,
            n_channels=n_channels,
            n_targets=n_targets,
            target_batch_size=target_batch_size,
            spatial_rank=spatial_rank,
        )
    raise ValueError(
        "learned_spatial_readout variant must be 'separable' or "
        "'voxel_specific_lowrank'"
    )


class LowRankVoxelSpatialReadout:  # pragma: no cover - exercised through training tests
    """Low-rank per-target spatial-channel readout for voxel-specific routing."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("learned_spatial_readout requires PyTorch") from exc

        class _LowRankVoxelSpatialReadout(nn.Module):
            def __init__(
                self,
                *,
                n_positions: int,
                n_channels: int,
                n_targets: int,
                target_batch_size: int,
                spatial_rank: int,
            ) -> None:
                super().__init__()
                if int(spatial_rank) < 1:
                    raise ValueError("voxel_specific_lowrank spatial_rank must be >= 1")
                self.n_targets = int(n_targets)
                self.target_batch_size = int(target_batch_size)
                self.spatial_rank = int(spatial_rank)
                self.spatial_logits = nn.Parameter(
                    torch.zeros(int(n_targets), self.spatial_rank, int(n_positions))
                )
                self.channel_weights = nn.Parameter(
                    torch.empty(int(n_targets), self.spatial_rank, int(n_channels))
                )
                nn.init.normal_(self.channel_weights, mean=0.0, std=0.01)
                self.bias = nn.Parameter(torch.zeros(int(n_targets)))

            def forward(self, features: Any) -> Any:
                weights = torch.softmax(self.spatial_logits, dim=2)
                chunks = []
                for start in range(0, self.n_targets, self.target_batch_size):
                    end = min(start + self.target_batch_size, self.n_targets)
                    spatial = weights[start:end]
                    pooled = torch.einsum("bpc,trp->btrc", features, spatial)
                    values = (pooled * self.channel_weights[start:end]).sum(dim=(2, 3))
                    values = values + self.bias[start:end]
                    chunks.append(values)
                return torch.cat(chunks, dim=1)

        return _LowRankVoxelSpatialReadout(*args, **kwargs)


def _mean_columnwise_pearson_torch(predictions: Any, target: Any) -> float:
    pred = predictions - predictions.mean(dim=0, keepdim=True)
    true = target - target.mean(dim=0, keepdim=True)
    pred_norm = torch.linalg.norm(pred, dim=0)
    true_norm = torch.linalg.norm(true, dim=0)
    denominator = pred_norm * true_norm
    values = torch.where(
        denominator > 1e-12,
        (pred * true).sum(dim=0) / denominator.clamp_min(1e-12),
        torch.zeros_like(denominator),
    )
    return float(values.mean().detach().cpu().item())


class _SpatialFeatureDataset:  # pragma: no cover - exercised through runner tests
    def __init__(
        self,
        features: np.ndarray,
        responses: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.responses = responses
        self.indices = np.asarray(indices, dtype=np.int64).ravel()

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        item_index = int(self.indices[index])
        feature = normalize_spatial_features(self.features[item_index][None, ...])[0]
        return torch.from_numpy(feature), torch.from_numpy(self.responses[item_index])


def _predict_standardized_in_batches(
    model: Any,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> Any:
    predictions = []
    for start in range(0, int(indices.size), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = normalize_spatial_features(features[batch_indices])
        batch_tensor = torch.from_numpy(batch).to(device)
        predictions.append(model(batch_tensor))
    return torch.cat(predictions, dim=0)


def _print_progress(
    *,
    config: SpatialReadoutConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_score: float,
    best_score: float,
    best_epoch: int,
    epochs_without_improvement: int,
    started_at: float,
    forced: bool,
    stopped_early: bool,
) -> None:
    if not config.progress:
        return
    every = max(1, int(config.progress_every))
    if not forced and epoch % every != 0 and epoch != int(config.max_epochs):
        return
    elapsed = perf_counter() - started_at
    seconds_per_epoch = elapsed / max(1, epoch)
    remaining_epochs = max(0, int(config.max_epochs) - int(epoch))
    eta_seconds = seconds_per_epoch * remaining_epochs
    status = " stopped_early" if stopped_early else ""
    print(
        "[learned_readout] "
        f"epoch={epoch}/{int(config.max_epochs)} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_pearson={val_score:.4f} "
        f"best={best_score:.4f}@{best_epoch} "
        f"no_improve={epochs_without_improvement}/{int(config.patience)} "
        f"elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta_seconds)}"
        f"{status}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


try:  # Keep torch optional at import time for lightweight environments.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
