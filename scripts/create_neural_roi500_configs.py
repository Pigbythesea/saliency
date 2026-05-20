"""Generate ROI500 neural alignment configs for Algonauts PRF visual ROIs."""

from __future__ import annotations

import argparse
import csv
import importlib
import re
from pathlib import Path
from typing import Any, Iterable

from hma.utils.config import save_yaml


CONFIG_ROOT = Path("configs/experiments/neural_roi500")
DEBUG_CONFIG_ROOT = Path("configs/experiments/neural_roi500_debug")
SSL_CANDIDATE_CONFIG_ROOT = Path("configs/experiments/neural_roi500_ssl_candidates_debug")
SSL_PRETRAINED_DEBUG_CONFIG_ROOT = Path(
    "configs/experiments/neural_roi500_ssl_pretrained_debug"
)
SSL_CONFIG_ROOT = Path("configs/experiments/neural_roi500_ssl")
MANIFEST_PATH = "data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv"
OUTPUT_ROOT = "outputs/neural_roi500"
DEBUG_OUTPUT_ROOT = "outputs/neural_roi500_debug"
SSL_CANDIDATE_OUTPUT_ROOT = "outputs/neural_roi500_ssl_candidates_debug"
SSL_PRETRAINED_DEBUG_OUTPUT_ROOT = "outputs/neural_roi500_ssl_pretrained_debug"
SSL_OUTPUT_ROOT = "outputs/neural_roi500_ssl"
ROI_CONFIGS = [
    ("V1", "v1"),
    ("V2", "v2"),
    ("V3", "v3"),
    ("hV4", "hv4"),
]
MODEL_SPECS = {
    "resnet50": {
        "layers": ["layer1", "layer2", "layer3", "layer4"],
    },
    "convnext_tiny": {
        "layers": ["stages.0", "stages.1", "stages.2", "stages.3"],
    },
    "deit_small_patch16_224": {
        "layers": ["blocks.0", "blocks.3", "blocks.6", "blocks.9", "blocks.11"],
    },
    "vit_base_patch16_224": {
        "layers": ["blocks.0", "blocks.3", "blocks.6", "blocks.9", "blocks.11"],
    },
}
SSL_MULTIMODAL_CANDIDATES = [
    {"model_name": "vit_small_patch14_dinov2", "family": "DINOv2"},
    {"model_name": "vit_base_patch14_dinov2", "family": "DINOv2"},
    {"model_name": "vit_small_patch16_dinov3", "family": "DINOv3"},
    {"model_name": "vit_base_patch16_dinov3", "family": "DINOv3"},
    {"model_name": "vit_base_patch16_clip_224", "family": "CLIP"},
    {"model_name": "resnet50_clip", "family": "CLIP"},
    {"model_name": "vit_base_patch16_siglip_224", "family": "SigLIP"},
    {"model_name": "eva02_base_patch16_clip_224", "family": "EVA-CLIP"},
]
DEFAULT_SSL_PRETRAINED_MODELS = [
    "vit_small_patch14_dinov2",
    "vit_base_patch16_clip_224",
    "resnet50_clip",
]
REQUIRED_NEURAL_OUTPUT_FILES = [
    "activations.npz",
    "encoding_scores.csv",
    "rsa_scores.csv",
    "metadata.json",
]
SSL_CANDIDATE_FIELDNAMES = [
    "model_name",
    "family",
    "available_in_timm",
    "architecture_hint",
    "proposed_layers",
    "verified_layers",
    "missing_layers",
    "wrapper_compatible",
    "pretrained_weights_run",
    "debug_config_path",
    "pretrained_debug_config_path",
    "pretrained_output_dir",
    "pretrained_run_status",
    "pretrained_weight_status",
    "pretrained_run_error",
    "inspection_error",
]


def create_neural_roi500_configs(
    *,
    output_dir: str | Path = CONFIG_ROOT,
    manifest_path: str = MANIFEST_PATH,
    output_root: str | Path = OUTPUT_ROOT,
    models: Iterable[str] | None = None,
    rois: Iterable[str] | None = None,
    max_items: int = 500,
    name_suffix: str = "500",
) -> list[Path]:
    """Write ROI neural configs and return their paths."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_models = list(models) if models is not None else list(MODEL_SPECS)
    selected_rois = list(rois) if rois is not None else [roi for roi, _slug in ROI_CONFIGS]
    roi_slug_by_name = dict(ROI_CONFIGS)
    written = []
    for model_name in selected_models:
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"Unknown model spec '{model_name}'. Available specs: {sorted(MODEL_SPECS)}"
            )
        for roi in selected_rois:
            if roi not in roi_slug_by_name:
                raise ValueError(
                    f"Unknown ROI '{roi}'. Available ROIs: {sorted(roi_slug_by_name)}"
                )
            slug = roi_slug_by_name[roi]
            config = _config_for_roi(
                model_name=model_name,
                roi=roi,
                slug=slug,
                manifest_path=manifest_path,
                output_root=str(output_root),
                max_items=max_items,
                name_suffix=name_suffix,
            )
            run_name = _run_name(model_name, slug, name_suffix)
            path = root / f"{run_name}.yaml"
            save_yaml(config, path)
            written.append(path)
    return written


def inspect_ssl_multimodal_candidates(
    *,
    output_csv: str | Path | None = None,
    candidates: Iterable[dict[str, str]] | None = None,
    timm_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Dry-inspect SSL/multimodal timm candidates without loading pretrained weights."""
    candidate_specs = list(candidates) if candidates is not None else SSL_MULTIMODAL_CANDIDATES
    timm = timm_module or importlib.import_module("timm")
    available_models = set(timm.list_models()) if hasattr(timm, "list_models") else set()
    rows = []
    for candidate in candidate_specs:
        model_name = candidate["model_name"]
        family = candidate["family"]
        row: dict[str, Any] = {
            "model_name": model_name,
            "family": family,
            "available_in_timm": str(model_name in available_models).lower(),
            "architecture_hint": _architecture_hint(model_name),
            "proposed_layers": "",
            "verified_layers": "",
            "missing_layers": "",
            "wrapper_compatible": "false",
            "pretrained_weights_run": "false",
            "debug_config_path": "",
            "pretrained_debug_config_path": "",
            "pretrained_output_dir": "",
            "pretrained_run_status": "not_run",
            "pretrained_weight_status": "",
            "pretrained_run_error": "",
            "inspection_error": "",
        }
        if model_name not in available_models:
            row["inspection_error"] = "model_not_listed_by_timm"
            rows.append(row)
            continue
        try:
            model = timm.create_model(model_name, pretrained=False)
            module_names = _module_names(model)
            proposed = _candidate_layers(model_name, module_names)
            missing = [layer for layer in proposed if layer not in module_names]
            row["proposed_layers"] = " ".join(proposed)
            row["verified_layers"] = " ".join(
                layer for layer in proposed if layer in module_names
            )
            row["missing_layers"] = " ".join(missing)
            row["wrapper_compatible"] = str(bool(proposed) and not missing).lower()
        except Exception as exc:  # pragma: no cover - exact timm failures vary by install
            row["inspection_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    if output_csv is not None:
        _write_candidate_inventory(Path(output_csv), rows)
    return rows


def create_ssl_multimodal_debug_configs(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path = SSL_CANDIDATE_CONFIG_ROOT,
    output_root: str | Path = SSL_CANDIDATE_OUTPUT_ROOT,
    manifest_path: str = MANIFEST_PATH,
    roi: str = "V1",
    max_items: int = 16,
) -> list[Path]:
    """Write pretrained=False debug configs for compatible candidate rows."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    roi_slug_by_name = dict(ROI_CONFIGS)
    if roi not in roi_slug_by_name:
        raise ValueError(f"Unknown ROI '{roi}'. Available ROIs: {sorted(roi_slug_by_name)}")
    written = []
    for row in candidate_rows:
        if str(row.get("wrapper_compatible", "")).lower() != "true":
            continue
        model_name = str(row["model_name"])
        layers = str(row.get("verified_layers", "")).split()
        if not layers:
            continue
        name = _run_name(model_name, roi_slug_by_name[roi], "debug")
        config = _config_for_candidate_roi(
            model_name=model_name,
            layers=layers,
            roi=roi,
            name=name,
            manifest_path=manifest_path,
            output_root=str(output_root),
            max_items=max_items,
        )
        path = root / f"{name}.yaml"
        save_yaml(config, path)
        row["debug_config_path"] = str(path)
        written.append(path)
    return written


def create_ssl_multimodal_candidate_configs(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    output_root: str | Path,
    manifest_path: str = MANIFEST_PATH,
    rois: Iterable[str] | None = None,
    max_items: int = 500,
    name_suffix: str = "500",
    pretrained: bool = True,
    models: Iterable[str] | None = None,
    config_path_field: str | None = None,
) -> list[Path]:
    """Write SSL/multimodal configs for compatible candidate rows."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    roi_slug_by_name = dict(ROI_CONFIGS)
    selected_rois = list(rois) if rois is not None else [roi for roi, _slug in ROI_CONFIGS]
    for roi in selected_rois:
        if roi not in roi_slug_by_name:
            raise ValueError(f"Unknown ROI '{roi}'. Available ROIs: {sorted(roi_slug_by_name)}")
    selected_models = set(models) if models is not None else None

    written = []
    for row in candidate_rows:
        if str(row.get("wrapper_compatible", "")).lower() != "true":
            continue
        model_name = str(row["model_name"])
        if selected_models is not None and model_name not in selected_models:
            continue
        layers = str(row.get("verified_layers", "")).split()
        if not layers:
            continue
        for roi in selected_rois:
            name = _run_name(model_name, roi_slug_by_name[roi], name_suffix)
            config = _config_for_candidate_roi(
                model_name=model_name,
                layers=layers,
                roi=roi,
                name=name,
                manifest_path=manifest_path,
                output_root=str(output_root),
                max_items=max_items,
                pretrained=pretrained,
            )
            path = root / f"{name}.yaml"
            save_yaml(config, path)
            if config_path_field is not None:
                row[config_path_field] = str(path)
            if pretrained and name_suffix == "pretrained_debug":
                row["pretrained_output_dir"] = config["output"]["dir"]
            written.append(path)
    return written


def create_ssl_multimodal_pretrained_debug_configs(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path = SSL_PRETRAINED_DEBUG_CONFIG_ROOT,
    output_root: str | Path = SSL_PRETRAINED_DEBUG_OUTPUT_ROOT,
    manifest_path: str = MANIFEST_PATH,
    models: Iterable[str] | None = None,
) -> list[Path]:
    """Write pretrained=True V1 debug configs for selected SSL/multimodal candidates."""
    return create_ssl_multimodal_candidate_configs(
        candidate_rows,
        output_dir=output_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        rois=["V1"],
        max_items=16,
        name_suffix="pretrained_debug",
        pretrained=True,
        models=models or DEFAULT_SSL_PRETRAINED_MODELS,
        config_path_field="pretrained_debug_config_path",
    )


def create_ssl_multimodal_roi500_configs(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path = SSL_CONFIG_ROOT,
    output_root: str | Path = SSL_OUTPUT_ROOT,
    manifest_path: str = MANIFEST_PATH,
    models: Iterable[str] | None = None,
) -> list[Path]:
    """Write pretrained=True full ROI500 configs for selected SSL/multimodal candidates."""
    return create_ssl_multimodal_candidate_configs(
        candidate_rows,
        output_dir=output_dir,
        output_root=output_root,
        manifest_path=manifest_path,
        rois=[roi for roi, _slug in ROI_CONFIGS],
        max_items=500,
        name_suffix="500",
        pretrained=True,
        models=models,
    )


def refresh_ssl_pretrained_status(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    output_root: str | Path = SSL_PRETRAINED_DEBUG_OUTPUT_ROOT,
) -> list[dict[str, Any]]:
    """Refresh pretrained debug status by scanning expected neural output files."""
    rows = list(candidate_rows)
    root = Path(output_root)
    for row in rows:
        model_name = str(row.get("model_name", ""))
        output_dir = Path(
            row.get("pretrained_output_dir")
            or root / _run_name(model_name, "v1", "pretrained_debug")
        )
        row["pretrained_output_dir"] = str(output_dir)
        missing = [
            name
            for name in REQUIRED_NEURAL_OUTPUT_FILES
            if not (output_dir / name).is_file()
        ]
        existing = [
            name for name in REQUIRED_NEURAL_OUTPUT_FILES if (output_dir / name).is_file()
        ]
        if not existing:
            row["pretrained_weights_run"] = "false"
            row["pretrained_run_status"] = row.get("pretrained_run_status") or "not_run"
            row["pretrained_weight_status"] = row.get("pretrained_weight_status", "")
            row["pretrained_run_error"] = row.get("pretrained_run_error", "")
            continue
        if missing:
            row["pretrained_weights_run"] = "false"
            row["pretrained_run_status"] = "incomplete"
            row["pretrained_weight_status"] = _metadata_pretrained_status(output_dir)
            row["pretrained_run_error"] = "missing " + " ".join(missing)
            continue
        row["pretrained_weights_run"] = "true"
        row["pretrained_run_status"] = "complete"
        row["pretrained_weight_status"] = _metadata_pretrained_status(output_dir)
        row["pretrained_run_error"] = ""
    return rows


def _config_for_roi(
    *,
    model_name: str,
    roi: str,
    slug: str,
    manifest_path: str,
    output_root: str,
    max_items: int,
    name_suffix: str,
) -> dict:
    name = _run_name(model_name, slug, name_suffix)
    layers = MODEL_SPECS[model_name]["layers"]
    return {
        "seed": 123,
        "device": "auto",
        "experiment": {"name": name},
        "dataset": {
            "name": "nsd_algonauts",
            "label": name,
            "root": "data/raw/nsd_algonauts",
            "manifest_path": manifest_path,
            "split": "train",
            "subject_id": "subj01",
            "roi": roi,
            "max_items": int(max_items),
            "validate_files": True,
        },
        "model": {
            "name": model_name,
            "backend": "timm",
            "pretrained": True,
            "eval_mode": True,
        },
        "preprocessing": {
            "input_size": [224, 224],
            "mean": "imagenet",
            "std": "imagenet",
        },
        "neural": {
            "layers": list(layers),
            "response_key": "roi_responses",
            "feature_reduction": "spatial_mean",
            "train_fraction": 0.8,
            "ridge_alpha": 1.0,
            "metric": "correlation",
            "rsa": {
                "enabled": True,
                "rdm_metric": "correlation",
                "response_rdm_metric": "correlation",
                "compare_method": "spearman",
            },
        },
        "output": {"dir": f"{output_root}/{name}"},
    }


def _config_for_candidate_roi(
    *,
    model_name: str,
    layers: list[str],
    roi: str,
    name: str,
    manifest_path: str,
    output_root: str,
    max_items: int,
    pretrained: bool = False,
) -> dict:
    return {
        "seed": 123,
        "device": "auto",
        "experiment": {"name": name},
        "dataset": {
            "name": "nsd_algonauts",
            "label": name,
            "root": "data/raw/nsd_algonauts",
            "manifest_path": manifest_path,
            "split": "train",
            "subject_id": "subj01",
            "roi": roi,
            "max_items": int(max_items),
            "validate_files": True,
        },
        "model": {
            "name": model_name,
            "backend": "timm",
            "pretrained": bool(pretrained),
            "eval_mode": True,
        },
        "preprocessing": {
            "input_size": _input_size_for_candidate(model_name),
            "mean": "imagenet",
            "std": "imagenet",
        },
        "neural": {
            "layers": list(layers),
            "response_key": "roi_responses",
            "feature_reduction": "spatial_mean",
            "train_fraction": 0.8,
            "ridge_alpha": 1.0,
            "metric": "correlation",
            "rsa": {
                "enabled": True,
                "rdm_metric": "correlation",
                "response_rdm_metric": "correlation",
                "compare_method": "spearman",
            },
        },
        "output": {"dir": f"{output_root}/{name}"},
    }


def _run_name(model_name: str, roi_slug: str, suffix: str) -> str:
    return f"{model_name}_{roi_slug}_{suffix}"


def _architecture_hint(model_name: str) -> str:
    if model_name.startswith("resnet"):
        return "resnet"
    if "clip" in model_name and model_name.startswith("eva"):
        return "eva_vit"
    if model_name.startswith("vit") or model_name.startswith("eva"):
        return "vit"
    return "unknown"


def _module_names(model: Any) -> set[str]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return set()
    return {name for name, _module in named_modules() if name}


def _candidate_layers(model_name: str, module_names: set[str]) -> list[str]:
    if _architecture_hint(model_name) == "resnet":
        plain_layers = [
            layer for layer in ["layer1", "layer2", "layer3", "layer4"] if layer in module_names
        ]
        if plain_layers:
            return plain_layers
        return [
            layer for layer in ["stages.0", "stages.1", "stages.2", "stages.3"] if layer in module_names
        ]

    block_indices = sorted(
        {
            int(match.group(1))
            for name in module_names
            for match in [re.fullmatch(r"blocks\.(\d+)", name)]
            if match is not None
        }
    )
    if not block_indices:
        return []
    selected_indices = [index for index in [0, 3, 6, 9] if index in block_indices]
    last_index = block_indices[-1]
    if last_index not in selected_indices:
        selected_indices.append(last_index)
    return [f"blocks.{index}" for index in selected_indices]


def _input_size_for_candidate(model_name: str) -> list[int]:
    if "dinov2" in model_name:
        return [518, 518]
    return [224, 224]


def _write_candidate_inventory(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SSL_CANDIDATE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_pretrained_status(output_dir: Path) -> str:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return ""
    try:
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return "metadata_unreadable"
    pretrained = metadata.get("model_pretrained")
    if pretrained is True:
        return "pretrained_true"
    if pretrained is False:
        return "pretrained_false"
    return "metadata_missing_model_pretrained"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ROI500 neural alignment configs for one or more timm models."
    )
    parser.add_argument("--output-dir", default=str(CONFIG_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--manifest-path", default=MANIFEST_PATH)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS))
    parser.add_argument("--rois", nargs="+", choices=[roi for roi, _slug in ROI_CONFIGS])
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--name-suffix", default="500")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate V1-only debug configs with max_items=16 under debug paths.",
    )
    parser.add_argument(
        "--inspect-ssl-candidates",
        action="store_true",
        help="Dry-inspect SSL/multimodal timm candidates with pretrained=False.",
    )
    parser.add_argument(
        "--candidate-output",
        default="outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv",
    )
    parser.add_argument(
        "--write-ssl-debug-configs",
        action="store_true",
        help="Write pretrained=False V1 debug configs for compatible candidates.",
    )
    parser.add_argument(
        "--write-ssl-pretrained-debug-configs",
        action="store_true",
        help="Write pretrained=True V1 debug configs for selected compatible candidates.",
    )
    parser.add_argument(
        "--write-ssl-roi500-configs",
        action="store_true",
        help="Write pretrained=True full ROI500 configs for selected compatible candidates.",
    )
    parser.add_argument(
        "--refresh-ssl-pretrained-status",
        action="store_true",
        help="Refresh candidate pretrained run status by scanning pretrained debug outputs.",
    )
    parser.add_argument(
        "--ssl-models",
        nargs="+",
        default=None,
        help="Optional SSL/multimodal model names to generate configs for.",
    )
    parser.add_argument("--ssl-debug-output-dir", default=str(SSL_CANDIDATE_CONFIG_ROOT))
    parser.add_argument("--ssl-debug-output-root", default=str(SSL_CANDIDATE_OUTPUT_ROOT))
    parser.add_argument(
        "--ssl-pretrained-debug-output-dir",
        default=str(SSL_PRETRAINED_DEBUG_CONFIG_ROOT),
    )
    parser.add_argument(
        "--ssl-pretrained-debug-output-root",
        default=str(SSL_PRETRAINED_DEBUG_OUTPUT_ROOT),
    )
    parser.add_argument("--ssl-roi500-output-dir", default=str(SSL_CONFIG_ROOT))
    parser.add_argument("--ssl-roi500-output-root", default=str(SSL_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inspect_ssl_candidates:
        rows = inspect_ssl_multimodal_candidates(output_csv=args.candidate_output)
        if args.write_ssl_debug_configs:
            create_ssl_multimodal_debug_configs(
                rows,
                output_dir=args.ssl_debug_output_dir,
                output_root=args.ssl_debug_output_root,
                manifest_path=args.manifest_path,
            )
        if args.write_ssl_pretrained_debug_configs:
            create_ssl_multimodal_pretrained_debug_configs(
                rows,
                output_dir=args.ssl_pretrained_debug_output_dir,
                output_root=args.ssl_pretrained_debug_output_root,
                manifest_path=args.manifest_path,
                models=args.ssl_models,
            )
        if args.write_ssl_roi500_configs:
            create_ssl_multimodal_roi500_configs(
                rows,
                output_dir=args.ssl_roi500_output_dir,
                output_root=args.ssl_roi500_output_root,
                manifest_path=args.manifest_path,
                models=args.ssl_models,
            )
        if args.refresh_ssl_pretrained_status:
            refresh_ssl_pretrained_status(
                rows,
                output_root=args.ssl_pretrained_debug_output_root,
            )
        if (
            args.write_ssl_debug_configs
            or args.write_ssl_pretrained_debug_configs
            or args.write_ssl_roi500_configs
            or args.refresh_ssl_pretrained_status
        ):
            _write_candidate_inventory(Path(args.candidate_output), rows)
        print(args.candidate_output)
        return

    output_dir = args.output_dir
    output_root = args.output_root
    max_items = args.max_items
    name_suffix = args.name_suffix
    rois = args.rois
    if args.debug:
        output_dir = str(DEBUG_CONFIG_ROOT)
        output_root = str(DEBUG_OUTPUT_ROOT)
        max_items = 16
        name_suffix = "debug"
        rois = ["V1"]
    written = create_neural_roi500_configs(
        output_dir=output_dir,
        manifest_path=args.manifest_path,
        output_root=output_root,
        models=args.models,
        rois=rois,
        max_items=max_items,
        name_suffix=name_suffix,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
