"""CAT2000 manifest-based dataset loader."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from hma.datasets.base import BaseVisionDataset, VisionDatasetItem
from hma.datasets.registry import register_dataset
from hma.saliency.postprocess import postprocess_saliency_map


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "image_path",
    "fixation_map_path",
    "category",
    "split",
    "width",
    "height",
}


@dataclass(frozen=True)
class _CAT2000Row:
    image_id: str
    image_path: Path
    fixation_map_path: Path
    fixation_points_path: Path | None
    category: str
    split: str
    width: int | None
    height: int | None


class CAT2000Dataset(BaseVisionDataset):
    """CAT2000 loader backed by a portable manifest CSV."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        split: str = "val",
        categories: list[str] | tuple[str, ...] | set[str] | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        max_items: int | None = None,
        image_size: int | tuple[int, int] | list[int] | None = None,
        validate_files: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self._resolve_path(manifest_path, Path.cwd())
        self.split = split
        self.categories = set(categories) if categories is not None else None
        self.transform = transform
        self.max_items = max_items
        self.image_size = _parse_image_size(image_size)
        self.validate_files = validate_files
        self.rows = self._load_rows()

        if self.validate_files:
            self._validate_files_exist()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CAT2000Dataset":
        return cls(
            root=config.get("root", "data/cat2000"),
            manifest_path=config.get(
                "manifest_path", "data/manifests/cat2000_manifest.csv"
            ),
            split=config.get("split", "val"),
            categories=config.get("categories"),
            max_items=config.get("max_items"),
            image_size=config.get("image_size"),
            validate_files=bool(config.get("validate_files", False)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[VisionDatasetItem]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> VisionDatasetItem:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        target_shape = None
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.BILINEAR)
            target_shape = (height, width)

        output_image = self.transform(image) if self.transform is not None else image
        fixation_map = _load_fixation_map(row.fixation_map_path, target_shape)

        return {
            "image": output_image,
            "image_id": row.image_id,
            "image_path": str(row.image_path),
            "fixation_map": fixation_map,
            "fixation_points": None,
            "metadata": {
                "dataset": "cat2000",
                "category": row.category,
                "split": row.split,
                "width": row.width,
                "height": row.height,
                "fixation_map_path": str(row.fixation_map_path),
                "fixation_points_path": str(row.fixation_points_path or ""),
            },
        }

    def _load_rows(self) -> list[_CAT2000Row]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"CAT2000 manifest not found: {self.manifest_path}")

        rows: list[_CAT2000Row] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_MANIFEST_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"CAT2000 manifest missing required columns: {sorted(missing)}"
                )

            for record in reader:
                if record["split"] != self.split:
                    continue
                category = record["category"]
                if self.categories is not None and category not in self.categories:
                    continue
                rows.append(
                    _CAT2000Row(
                        image_id=record["image_id"],
                        image_path=self._resolve_path(record["image_path"], self.root),
                        fixation_map_path=self._resolve_path(
                            record["fixation_map_path"], self.root
                        ),
                        fixation_points_path=self._optional_resolve_path(
                            record.get("fixation_points_path"), self.root
                        ),
                        category=category,
                        split=record["split"],
                        width=_optional_int(record.get("width")),
                        height=_optional_int(record.get("height")),
                    )
                )

        if self.max_items is not None:
            rows = rows[: int(self.max_items)]
        return rows

    def _validate_files_exist(self) -> None:
        for row in self.rows:
            if not row.image_path.is_file():
                raise FileNotFoundError(f"CAT2000 image not found: {row.image_path}")
            if not row.fixation_map_path.is_file():
                raise FileNotFoundError(
                    f"CAT2000 fixation map not found: {row.fixation_map_path}"
                )
            if row.fixation_points_path is not None and not row.fixation_points_path.is_file():
                raise FileNotFoundError(
                    f"CAT2000 fixation points not found: {row.fixation_points_path}"
                )

    @staticmethod
    def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(base_dir).expanduser().resolve() / candidate).resolve()

    @staticmethod
    def _optional_resolve_path(path: str | Path | None, base_dir: str | Path) -> Path | None:
        if path is None or str(path) == "":
            return None
        return CAT2000Dataset._resolve_path(path, base_dir)


def _load_fixation_map(
    path: Path,
    target_shape: tuple[int, int] | None,
) -> np.ndarray:
    with Image.open(path) as image:
        map_array = np.asarray(image.convert("L"), dtype=np.float32)
    return postprocess_saliency_map(map_array, target_shape=target_shape)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_image_size(
    image_size: int | tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or length-2 sequence")
    height, width = image_size
    return (int(height), int(width))


register_dataset("cat2000", CAT2000Dataset)
register_dataset("CAT2000", CAT2000Dataset)
