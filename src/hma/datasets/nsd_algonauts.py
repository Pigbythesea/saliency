"""NSD / Algonauts manifest-based dataset loader."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from hma.datasets.base import BaseVisionDataset, VisionDatasetItem
from hma.datasets.registry import register_dataset


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "image_path",
    "split",
    "subject_id",
}


@dataclass(frozen=True)
class _NSDAlgonautsRow:
    image_id: str
    image_path: Path
    split: str
    subject_id: str
    roi: str | None
    roi_response_path: Path | None
    roi_responses: np.ndarray | None


class NSDAlgonautsDataset(BaseVisionDataset):
    """Manifest-based loader for NSD / Algonauts-style image-response rows."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image], Any] | None = None,
        max_items: int | None = None,
        validate_files: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self._resolve_path(manifest_path, Path.cwd())
        self.split = split
        self.transform = transform
        self.max_items = max_items
        self.validate_files = validate_files
        self.rows = self._load_rows()

        if self.validate_files:
            self._validate_files_exist()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NSDAlgonautsDataset":
        return cls(
            root=config.get("root", "data/nsd_algonauts"),
            manifest_path=config.get(
                "manifest_path", "data/manifests/nsd_algonauts_manifest.csv"
            ),
            split=config.get("split", "train"),
            max_items=config.get("max_items"),
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
        output_image = self.transform(image) if self.transform is not None else image
        roi_responses = self._load_roi_responses(row)

        return {
            "image": output_image,
            "image_id": row.image_id,
            "image_path": str(row.image_path),
            "fixation_map": None,
            "fixation_points": None,
            "metadata": {
                "dataset": "nsd_algonauts",
                "split": row.split,
                "subject_id": row.subject_id,
                "roi": row.roi,
                "roi_responses": roi_responses,
                "roi_response_path": (
                    str(row.roi_response_path) if row.roi_response_path else None
                ),
            },
        }

    def _load_rows(self) -> list[_NSDAlgonautsRow]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"NSD / Algonauts manifest not found: {self.manifest_path}"
            )

        rows: list[_NSDAlgonautsRow] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_MANIFEST_COLUMNS - columns
            if missing:
                raise ValueError(
                    "NSD / Algonauts manifest missing required columns: "
                    f"{sorted(missing)}"
                )

            for record in reader:
                if record["split"] != self.split:
                    continue
                response_path = _optional_str(record.get("roi_response_path"))
                rows.append(
                    _NSDAlgonautsRow(
                        image_id=record["image_id"],
                        image_path=self._resolve_path(record["image_path"], self.root),
                        split=record["split"],
                        subject_id=record["subject_id"],
                        roi=_optional_str(record.get("roi")),
                        roi_response_path=(
                            self._resolve_path(response_path, self.root)
                            if response_path
                            else None
                        ),
                        roi_responses=_parse_roi_responses(record.get("roi_responses")),
                    )
                )

        if self.max_items is not None:
            rows = rows[: int(self.max_items)]
        return rows

    def _load_roi_responses(self, row: _NSDAlgonautsRow) -> np.ndarray | None:
        if row.roi_response_path is not None:
            return _load_response_file(row.roi_response_path)
        return row.roi_responses

    def _validate_files_exist(self) -> None:
        for row in self.rows:
            if not row.image_path.is_file():
                raise FileNotFoundError(
                    f"NSD / Algonauts image not found: {row.image_path}"
                )
            if row.roi_response_path is not None and not row.roi_response_path.is_file():
                raise FileNotFoundError(
                    f"NSD / Algonauts response file not found: {row.roi_response_path}"
                )

    @staticmethod
    def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(base_dir).expanduser().resolve() / candidate).resolve()


def _load_response_file(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "responses" in data:
            return np.asarray(data["responses"], dtype=np.float32)
        first_key = data.files[0]
        return np.asarray(data[first_key], dtype=np.float32)
    return np.asarray(np.load(path), dtype=np.float32)


def _parse_roi_responses(raw: str | None) -> np.ndarray | None:
    value = _optional_str(raw)
    if value is None:
        return None
    return np.asarray(json.loads(value), dtype=np.float32)


def _optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


register_dataset("nsd_algonauts", NSDAlgonautsDataset)
register_dataset("nsd", NSDAlgonautsDataset)
register_dataset("algonauts", NSDAlgonautsDataset)
