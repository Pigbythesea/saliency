"""Clustered and hierarchical uncertainty for behavioral metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    """A machine-readable bootstrap summary."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int
    uncertainty_unit: str
    valid_resamples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "seed": self.seed,
            "uncertainty_unit": self.uncertainty_unit,
            "valid_resamples": self.valid_resamples,
        }


def image_cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    value_key: str,
    image_key: str = "image_path",
    resamples: int = 2000,
    seed: int = 123,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap image clusters while retaining all rows within sampled images."""
    records = _validated_records(rows, value_key=value_key, keys=(image_key,))
    grouped = _group_records(records, keys=(image_key,))
    return _hierarchical_bootstrap(
        grouped,
        value_key=value_key,
        nested_key=None,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        uncertainty_unit=image_key,
    )


def salicon_hierarchical_interval(
    rows: Iterable[Mapping[str, Any]],
    *,
    value_key: str,
    image_key: str = "image_path",
    worker_key: str = "worker_id",
    resamples: int = 2000,
    seed: int = 123,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap SALICON images, then workers within each sampled image."""
    records = _validated_records(
        rows,
        value_key=value_key,
        keys=(image_key, worker_key),
    )
    grouped = _group_records(records, keys=(image_key,))
    return _hierarchical_bootstrap(
        grouped,
        value_key=value_key,
        nested_key=worker_key,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        uncertainty_unit=f"{worker_key}_within_{image_key}",
    )


def coco_search18_hierarchical_interval(
    rows: Iterable[Mapping[str, Any]],
    *,
    value_key: str,
    image_key: str = "image_path",
    task_key: str = "target_category",
    subject_key: str = "subject_id",
    resamples: int = 2000,
    seed: int = 123,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap image/task clusters, then subjects within each cluster."""
    records = _validated_records(
        rows,
        value_key=value_key,
        keys=(image_key, task_key, subject_key),
    )
    grouped = _group_records(records, keys=(image_key, task_key))
    return _hierarchical_bootstrap(
        grouped,
        value_key=value_key,
        nested_key=subject_key,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        uncertainty_unit=f"{subject_key}_within_{image_key}_{task_key}",
    )


def _hierarchical_bootstrap(
    grouped: dict[tuple[str, ...], list[dict[str, Any]]],
    *,
    value_key: str,
    nested_key: str | None,
    resamples: int,
    seed: int,
    confidence: float,
    uncertainty_unit: str,
) -> BootstrapInterval:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    cluster_keys = sorted(grouped)
    if not cluster_keys:
        raise ValueError("at least one cluster is required")
    estimate = float(
        np.mean(
            [
                float(record[value_key])
                for records in grouped.values()
                for record in records
            ]
        )
    )
    rng = np.random.default_rng(seed)
    statistics: list[float] = []
    for _ in range(resamples):
        sampled_clusters = rng.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
        sampled_values: list[float] = []
        for cluster_index in sampled_clusters:
            records = grouped[cluster_keys[int(cluster_index)]]
            if nested_key is None:
                sampled_values.extend(float(record[value_key]) for record in records)
                continue
            by_nested = _group_records(records, keys=(nested_key,))
            nested_ids = sorted(by_nested)
            sampled_nested = rng.choice(
                len(nested_ids),
                size=len(nested_ids),
                replace=True,
            )
            for nested_index in sampled_nested:
                nested_records = by_nested[nested_ids[int(nested_index)]]
                record_index = int(rng.integers(0, len(nested_records)))
                sampled_values.append(float(nested_records[record_index][value_key]))
        if sampled_values:
            value = float(np.mean(sampled_values))
            if np.isfinite(value):
                statistics.append(value)
    if not statistics:
        raise ValueError("bootstrap produced no finite resamples")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(statistics), [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=float(confidence),
        resamples=int(resamples),
        seed=int(seed),
        uncertainty_unit=uncertainty_unit,
        valid_resamples=len(statistics),
    )


def _validated_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    value_key: str,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = (value_key, *keys)
    for row in rows:
        record = dict(row)
        missing = [key for key in required if record.get(key) in (None, "")]
        if missing:
            raise ValueError(f"behavioral uncertainty row is missing {missing}")
        value = float(record[value_key])
        if not np.isfinite(value):
            continue
        record[value_key] = value
        records.append(record)
    if not records:
        raise ValueError("no finite behavioral metric rows were supplied")
    return records


def _group_records(
    records: Iterable[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = tuple(str(row[name]) for name in keys)
        grouped.setdefault(key, []).append(dict(row))
    return grouped
