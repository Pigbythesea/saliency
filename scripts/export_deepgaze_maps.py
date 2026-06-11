"""Export precomputed DeepGaze IIE saliency maps for manifest rows."""

from __future__ import annotations

import argparse
import csv
import time
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hma.saliency.precomputed import precomputed_map_key


DEFAULT_CENTERBIAS = "data/precomputed/deepgaze/centerbias_mit1003.npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DeepGaze IIE maps as .npy files.")
    parser.add_argument(
        "--model",
        choices=["deepgaze_iie", "deepgaze_msdb"],
        default="deepgaze_iie",
        help="DeepGaze model to export. Defaults to the current IIE reference.",
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV with image_id and image_path.")
    parser.add_argument("--image-root", required=True, help="Root used to resolve relative image_path values.")
    parser.add_argument("--output-dir", required=True, help="Directory for <image_id>.npy outputs.")
    parser.add_argument("--centerbias", default=DEFAULT_CENTERBIAS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--pixel-per-dva",
        type=float,
        default=21.7,
        help=(
            "Pixels per degree of visual angle for DeepGaze MSDB. Ignored by IIE. "
            "Default follows the MSDB generalized-parameter example."
        ),
    )
    parser.add_argument(
        "--msdb-dataset",
        choices=["averaged", "mit1003", "cat2000", "coco_freeview", "daemons", "figrim"],
        default="averaged",
        help=(
            "Optional DeepGaze MSDB dataset-specific parameter set. Defaults to averaged "
            "parameters for cross-dataset generalization."
        ),
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for smoke exports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing maps.")
    parser.add_argument("--dry-run", action="store_true", help="List planned work without loading DeepGaze.")
    parser.add_argument("--save-log-density", action="store_true", help="Save log-density instead of probability map.")
    parser.add_argument(
        "--filename-template",
        default="{image_id}.npy",
        help="Output filename template. Supports {image_id} and collision-safe {map_key}.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_manifest_rows(args.manifest)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    centerbias_template = np.load(args.centerbias)

    planned = [
        (
            row,
            output_path_for_row(
                output_dir,
                row,
                image_root=image_root,
                filename_template=args.filename_template,
            ),
        )
        for row in rows
    ]
    if args.dry_run:
        print(f"Manifest rows: {len(rows)}")
        print(f"Image root: {image_root}")
        print(f"Output dir: {output_dir}")
        for row, output_path in planned[:10]:
            print(f"{row['image_id']}: {resolve_image_path(image_root, row['image_path'])} -> {output_path}")
        if len(planned) > 10:
            print(f"... {len(planned) - 10} more")
        return

    ensure_dir(output_dir)
    torch, model, device = build_deepgaze_model(args.device, model_name=args.model)
    model.eval()
    msdb_dataset = resolve_msdb_dataset_index(args.msdb_dataset)

    start = time.perf_counter()
    exported = 0
    skipped = 0
    for index, (row, output_path) in enumerate(planned, start=1):
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            maybe_print_progress(index, len(planned), exported, skipped, start, args.progress_interval)
            continue

        image = load_rgb_image(resolve_image_path(image_root, row["image_path"]))
        prediction = predict_deepgaze_map(
            model,
            torch,
            image,
            centerbias_template,
            device=device,
            model_name=args.model,
            pixel_per_dva=args.pixel_per_dva,
            msdb_dataset=msdb_dataset,
            save_log_density=args.save_log_density,
        )
        np.save(output_path, prediction.astype(np.float32, copy=False))
        exported += 1
        maybe_print_progress(index, len(planned), exported, skipped, start, args.progress_interval)

    elapsed = time.perf_counter() - start
    print(
        f"Finished DeepGaze export: exported={exported}, skipped={skipped}, "
        f"total={len(planned)}, elapsed={elapsed:.1f}s"
    )


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"image_id", "image_path"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def build_deepgaze_model(
    device_config: str,
    *,
    model_name: str = "deepgaze_iie",
) -> tuple[Any, Any, str]:
    try:
        import torch
        import deepgaze_pytorch
    except ImportError as exc:
        raise ImportError(
            "DeepGaze export requires torch and deepgaze_pytorch. Install DeepGaze "
            "and its optional dependencies before exporting maps."
        ) from exc

    device = resolve_device(torch, device_config)
    model_class = resolve_deepgaze_model_class(deepgaze_pytorch, model_name)
    try:
        model = model_class(pretrained=True).to(device)
    except (urllib.error.URLError, OSError) as exc:
        if model_name == "deepgaze_msdb":
            raise RuntimeError(
                "DeepGaze MSDB model setup failed while downloading or reading its "
                "external pretrained assets. MSDB uses torch.hub to load DINOv2 from "
                "facebookresearch/dinov2:6a62615, so the first run must successfully "
                "cache that GitHub source package and its weights. Retry after network "
                "stabilizes or pre-cache the DINOv2 torch hub dependency, then rerun "
                "the export command."
            ) from exc
        raise
    return torch, model, device


def resolve_deepgaze_model_class(deepgaze_module: Any, model_name: str) -> Any:
    """Return the DeepGaze class for a supported export model name."""
    classes = {
        "deepgaze_iie": "DeepGazeIIE",
        "deepgaze_msdb": "DeepGazeMSDB",
    }
    try:
        class_name = classes[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeepGaze model: {model_name}") from exc
    try:
        return getattr(deepgaze_module, class_name)
    except AttributeError as exc:
        raise ImportError(f"deepgaze_pytorch does not provide {class_name}") from exc


def predict_deepgaze_map(
    model: Any,
    torch: Any,
    image: np.ndarray,
    centerbias_template: np.ndarray,
    *,
    device: str,
    model_name: str = "deepgaze_iie",
    pixel_per_dva: float = 21.7,
    msdb_dataset: int | None = None,
    save_log_density: bool = False,
) -> np.ndarray:
    centerbias = resize_and_normalize_centerbias(centerbias_template, image.shape[:2])
    image_tensor = torch.as_tensor(
        image.transpose(2, 0, 1)[None].copy(),
        dtype=torch.float32,
        device=device,
    )
    centerbias_tensor = torch.as_tensor(centerbias[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        if model_name == "deepgaze_msdb":
            log_density = model(
                image_tensor,
                centerbias_tensor,
                pixel_per_dva=float(pixel_per_dva),
                dataset=msdb_dataset,
            )
        else:
            log_density = model(image_tensor, centerbias_tensor)
    array = log_density.detach().cpu().numpy()[0]
    if array.ndim == 3:
        array = array[0]
    if save_log_density:
        return array.astype(np.float32, copy=False)
    return np.exp(array).astype(np.float32, copy=False)


def resize_and_normalize_centerbias(
    centerbias_template: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    height, width = int(target_shape[0]), int(target_shape[1])
    resized = resize_centerbias_nearest(centerbias_template, (height, width))
    return resized - logsumexp_2d(resized)


def resize_centerbias_nearest(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    try:
        from scipy.ndimage import zoom

        return zoom(
            values,
            (target_shape[0] / values.shape[0], target_shape[1] / values.shape[1]),
            order=0,
            mode="nearest",
        )
    except ImportError:
        image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
        return np.asarray(image.resize((target_shape[1], target_shape[0]), Image.NEAREST))


def logsumexp_2d(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.nanmax(array))
    if not np.isfinite(maximum):
        return 0.0
    return maximum + float(np.log(np.sum(np.exp(array - maximum))))


def resolve_image_path(root: Path, image_path: str) -> Path:
    candidate = Path(image_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def output_path_for_image_id(output_dir: Path, image_id: str) -> Path:
    if "/" in image_id or "\\" in image_id:
        raise ValueError(f"image_id cannot contain path separators for map export: {image_id}")
    return output_dir / f"{image_id}.npy"


def output_path_for_row(
    output_dir: Path,
    row: dict[str, str],
    *,
    image_root: Path,
    filename_template: str,
) -> Path:
    resolved_image_path = resolve_image_path(image_root, row["image_path"])
    map_key = precomputed_map_key(resolved_image_path)
    filename = filename_template.format(image_id=row["image_id"], map_key=map_key)
    if "/" in filename or "\\" in filename:
        raise ValueError(f"filename_template cannot produce nested paths: {filename}")
    if not filename.endswith((".npy", ".npz", ".png", ".jpg", ".jpeg")):
        filename = f"{filename}.npy"
    return output_dir / filename


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resolve_device(torch: Any, value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def resolve_msdb_dataset_index(value: str) -> int | None:
    """Return the DeepGaze MSDB dataset index, or None for averaged parameters."""
    indices = {
        "averaged": None,
        "mit1003": 0,
        "cat2000": 1,
        "coco_freeview": 2,
        "daemons": 3,
        "figrim": 4,
    }
    try:
        return indices[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeepGaze MSDB dataset: {value}") from exc


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_print_progress(
    index: int,
    total: int,
    exported: int,
    skipped: int,
    start: float,
    interval: int,
) -> None:
    if index != total and index % max(1, interval) != 0:
        return
    elapsed = time.perf_counter() - start
    rate = index / elapsed if elapsed > 0 else 0.0
    print(
        f"[progress] {index}/{total} exported={exported} skipped={skipped} "
        f"rate={rate:.2f} image/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
