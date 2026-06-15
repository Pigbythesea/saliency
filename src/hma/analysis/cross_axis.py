"""Family-aware cross-axis sensitivity without paper interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class CrossAxisInterval:
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    valid_resamples: int
    seed: int
    uncertainty_unit: str = "model_family"

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "valid_resamples": self.valid_resamples,
            "seed": self.seed,
            "uncertainty_unit": self.uncertainty_unit,
        }


def cross_axis_panel_preflight(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_key: str = "model_id",
    family_key: str = "family",
    role_key: str = "role",
    regime_key: str = "behavioral_regime",
    object_key: str = "behavioral_object",
    min_models: int = 6,
    min_families: int = 3,
    min_roles: int = 2,
) -> dict[str, Any]:
    """Check panel size and reject pooled behavioral regimes or objects."""
    records = [dict(row) for row in rows]
    if not records:
        return {"status": "empty_panel", "passed": False}
    regimes = _nonempty_values(records, regime_key)
    objects = _nonempty_values(records, object_key)
    if len(regimes) > 1:
        raise ValueError("cross-axis panels may not pool behavioral regimes")
    if len(objects) > 1:
        raise ValueError("cross-axis panels may not pool behavioral objects")
    models = _nonempty_values(records, model_key)
    families = _nonempty_values(records, family_key)
    roles = _nonempty_values(records, role_key)
    passed = (
        len(models) >= min_models
        and len(families) >= min_families
        and len(roles) >= min_roles
    )
    return {
        "status": "ready" if passed else "insufficient_panel",
        "passed": passed,
        "n_models": len(models),
        "n_families": len(families),
        "n_roles": len(roles),
        "behavioral_regime": next(iter(regimes), ""),
        "behavioral_object": next(iter(objects), ""),
        "required_models": int(min_models),
        "required_families": int(min_families),
        "required_roles": int(min_roles),
    }


def leave_one_family_sensitivity(
    rows: Iterable[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    family_key: str = "family",
) -> list[dict[str, Any]]:
    """Recompute Spearman correlation after omitting each model family."""
    records = _numeric_records(rows, x_key=x_key, y_key=y_key, family_key=family_key)
    baseline = _spearman(records, x_key=x_key, y_key=y_key)
    output: list[dict[str, Any]] = []
    for family in sorted({str(row[family_key]) for row in records}):
        retained = [row for row in records if str(row[family_key]) != family]
        score = _spearman(retained, x_key=x_key, y_key=y_key)
        output.append(
            {
                "sensitivity_type": "leave_one_family",
                "omitted_family": family,
                "baseline_spearman": baseline if baseline is not None else "",
                "sensitivity_spearman": score if score is not None else "",
                "n_models": len(retained),
                "n_families": len({str(row[family_key]) for row in retained}),
                "status": "complete" if score is not None else "insufficient_panel",
            }
        )
    return output


def family_block_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    family_key: str = "family",
    resamples: int = 2000,
    seed: int = 123,
    confidence: float = 0.95,
) -> CrossAxisInterval:
    """Bootstrap model families as blocks and retain models within each block."""
    records = _numeric_records(rows, x_key=x_key, y_key=y_key, family_key=family_key)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row[family_key]), []).append(row)
    families = sorted(grouped)
    if len(families) < 2:
        raise ValueError("family bootstrap requires at least two families")
    estimate = _spearman(records, x_key=x_key, y_key=y_key)
    if estimate is None:
        raise ValueError("cross-axis panel requires at least three model rows")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(len(families), size=len(families), replace=True)
        bootstrap_rows = [
            row
            for index in sampled
            for row in grouped[families[int(index)]]
        ]
        score = _spearman(bootstrap_rows, x_key=x_key, y_key=y_key)
        if score is not None and np.isfinite(score):
            scores.append(float(score))
    if not scores:
        raise ValueError("family bootstrap produced no valid correlations")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(scores), [alpha, 1.0 - alpha])
    return CrossAxisInterval(
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        confidence=float(confidence),
        resamples=int(resamples),
        valid_resamples=len(scores),
        seed=int(seed),
    )


def _numeric_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    family_key: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get(family_key) in (None, ""):
            raise ValueError(f"cross-axis row is missing {family_key}")
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            records.append({**dict(row), x_key: x, y_key: y})
    if len(records) < 3:
        raise ValueError("cross-axis analysis requires at least three finite rows")
    return records


def _spearman(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> float | None:
    if len(rows) < 3:
        return None
    x = _rankdata(np.asarray([row[x_key] for row in rows], dtype=np.float64))
    y = _rankdata(np.asarray([row[y_key] for row in rows], dtype=np.float64))
    x -= float(x.mean())
    y -= float(y.mean())
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denominator <= 1e-12 else float(np.dot(x, y) / denominator)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _nonempty_values(rows: list[dict[str, Any]], key: str) -> set[str]:
    return {
        str(row[key])
        for row in rows
        if row.get(key) not in (None, "")
    }
