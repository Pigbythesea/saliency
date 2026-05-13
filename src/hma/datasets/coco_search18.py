"""COCO-Search18 manifest-based dataset loader."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from hma.datasets.base import BaseVisionDataset, VisionDatasetItem
from hma.datasets.fixation_utils import points_to_fixation_map
from hma.datasets.registry import register_dataset


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "image_path",
    "split",
    "width",
    "height",
    "target_category",
    "task",
    "fixation_points",
    "subject_id",
    "trial_id",
}


@dataclass(frozen=True)
class _COCOSearch18Row:
    image_id: str
    image_path: Path
    split: str
    width: int | None
    height: int | None
    target_category: str
    task: str
    fixation_points: np.ndarray
    subject_id: str | None
    trial_id: str | None


class COCOSearch18Dataset(BaseVisionDataset):
    """COCO-Search18 loader backed by a portable manifest CSV."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image], Any] | None = None,
        max_items: int | None = None,
        image_size: int | tuple[int, int] | list[int] | None = None,
        generate_fixation_map: bool = True,
        fixation_sigma: float = 10.0,
        validate_files: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self._resolve_path(manifest_path, Path.cwd())
        self.split = split
        self.transform = transform
        self.max_items = max_items
        self.image_size = _parse_image_size(image_size)
        self.generate_fixation_map = generate_fixation_map
        self.fixation_sigma = float(fixation_sigma)
        self.validate_files = validate_files
        self.rows = self._load_rows()

        if self.validate_files:
            self._validate_files_exist()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "COCOSearch18Dataset":
        return cls(
            root=config.get("root", "data/coco_search18"),
            manifest_path=config.get(
                "manifest_path", "data/manifests/coco_search18_manifest.csv"
            ),
            split=config.get("split", "train"),
            max_items=config.get("max_items"),
            image_size=config.get("image_size"),
            generate_fixation_map=bool(config.get("generate_fixation_map", True)),
            fixation_sigma=float(config.get("fixation_sigma", 10.0)),
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
        original_width, original_height = image.size
        fixation_points = row.fixation_points.copy()
        map_height, map_width = original_height, original_width

        if self.image_size is not None:
            map_height, map_width = self.image_size
            if original_width > 0 and original_height > 0 and fixation_points.size:
                fixation_points[:, 0] *= map_width / float(original_width)
                fixation_points[:, 1] *= map_height / float(original_height)
            image = image.resize((map_width, map_height), Image.BILINEAR)

        output_image = self.transform(image) if self.transform is not None else image
        fixation_map = None
        if self.generate_fixation_map:
            fixation_map = points_to_fixation_map(
                fixation_points,
                height=map_height,
                width=map_width,
                sigma=self.fixation_sigma,
            )

        return {
            "image": output_image,
            "image_id": row.image_id,
            "image_path": str(row.image_path),
            "fixation_map": fixation_map,
            "fixation_points": fixation_points,
            "metadata": {
                "dataset": "coco_search18",
                "split": row.split,
                "target_category": row.target_category,
                "task": row.task,
                "subject_id": row.subject_id,
                "trial_id": row.trial_id,
                "width": row.width,
                "height": row.height,
            },
        }

    def _load_rows(self) -> list[_COCOSearch18Row]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"COCO-Search18 manifest not found: {self.manifest_path}"
            )

        rows: list[_COCOSearch18Row] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_MANIFEST_COLUMNS - columns
            if missing:
                raise ValueError(
                    "COCO-Search18 manifest missing required columns: "
                    f"{sorted(missing)}"
                )

            for record in reader:
                if record["split"] != self.split:
                    continue
                rows.append(
                    _COCOSearch18Row(
                        image_id=record["image_id"],
                        image_path=self._resolve_path(record["image_path"], self.root),
                        split=record["split"],
                        width=_optional_int(record.get("width")),
                        height=_optional_int(record.get("height")),
                        target_category=record["target_category"],
                        task=record["task"],
                        fixation_points=_parse_points(record["fixation_points"]),
                        subject_id=_optional_str(record.get("subject_id")),
                        trial_id=_optional_str(record.get("trial_id")),
                    )
                )

        if self.max_items is not None:
            rows = rows[: int(self.max_items)]
        return rows

    def _validate_files_exist(self) -> None:
        for row in self.rows:
            if not row.image_path.is_file():
                raise FileNotFoundError(
                    f"COCO-Search18 image not found: {row.image_path}"
                )

    @staticmethod
    def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(base_dir).expanduser().resolve() / candidate).resolve()


def _parse_points(raw_points: str) -> np.ndarray:
    if raw_points == "":
        return np.zeros((0, 2), dtype=np.float32)
    points = json.loads(raw_points)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return array.reshape(-1, 2)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


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


register_dataset("coco_search18", COCOSearch18Dataset)
register_dataset("coco-search18", COCOSearch18Dataset)
register_dataset("COCO-Search18", COCOSearch18Dataset)
