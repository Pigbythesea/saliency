"""Neural alignment utilities."""

from hma.neural.activations import extract_activations, save_activations
from hma.neural.encoding import (
    benchmark_encoding_target_scores,
    evaluate_encoding,
    fit_ridge_encoding,
    predict_ridge_encoding,
)
from hma.neural.learned_readout import (
    SpatialReadoutConfig,
    fit_spatial_readout,
    fuse_spatial_feature_layers,
    normalize_spatial_features,
    predict_spatial_readout,
)
from hma.neural.rsa import compare_rdms, compute_rdm

__all__ = [
    "compare_rdms",
    "compute_rdm",
    "benchmark_encoding_target_scores",
    "evaluate_encoding",
    "extract_activations",
    "fit_ridge_encoding",
    "fit_spatial_readout",
    "fuse_spatial_feature_layers",
    "normalize_spatial_features",
    "predict_ridge_encoding",
    "predict_spatial_readout",
    "save_activations",
    "SpatialReadoutConfig",
]
