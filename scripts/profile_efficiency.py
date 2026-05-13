"""Profile model efficiency metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hma.metrics.efficiency_metrics import (
    count_parameters,
    estimate_flops,
    estimate_model_size_mb,
    measure_latency,
)
from hma.models import build_model
from hma.utils.config import load_yaml
from hma.utils.paths import ensure_dir


LATENCY_FIELDS = [
    "latency_mean_ms",
    "latency_median_ms",
    "latency_min_ms",
    "latency_max_ms",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile HMA model efficiency.")
    parser.add_argument("--config", required=True, help="Path to model or experiment YAML.")
    parser.add_argument(
        "--output",
        default="outputs/efficiency/profile.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for profiling. 'auto' uses GPU when available, otherwise CPU.",
    )
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--flops", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = profile_config(
        config_path=args.config,
        device=args.device,
        input_shape=_parse_input_shape(args.input_shape),
        warmup=args.warmup,
        repeats=args.repeats,
        include_flops=args.flops,
    )
    output = Path(args.output)
    ensure_dir(output.parent)
    _write_rows(output, rows, include_flops=args.flops)
    print(f"Efficiency profile written to {output.resolve()}")


def profile_config(
    config_path: str | Path,
    device: str,
    input_shape: tuple[int, ...],
    warmup: int,
    repeats: int,
    include_flops: bool,
) -> list[dict[str, object]]:
    config = load_yaml(config_path)
    model_configs = _extract_model_configs(config)
    rows = []
    for model_config in model_configs:
        model = build_model({"model": model_config})
        row: dict[str, object] = {
            "model_name": model_config.get("name") or model_config.get("model_name"),
            "parameter_count": count_parameters(model),
            "model_size_mb": estimate_model_size_mb(model),
        }
        try:
            row.update(
                measure_latency(
                    model,
                    input_shape=input_shape,
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                )
            )
        except ImportError as exc:
            print(f"Latency skipped for {row['model_name']}: {exc}")
            for field in LATENCY_FIELDS:
                row[field] = ""
        if include_flops:
            row["flops"] = estimate_flops(model, input_shape=input_shape, device=device)
        rows.append(row)
    return rows


def _extract_model_configs(config: dict) -> list[dict]:
    if isinstance(config.get("models"), list):
        return list(config["models"])
    if isinstance(config.get("model"), dict):
        return [config["model"]]
    if "name" in config:
        return [config]
    raise KeyError("Config must contain model, models, or a model name")


def _write_rows(path: Path, rows: list[dict[str, object]], include_flops: bool) -> None:
    fieldnames = [
        "model_name",
        "parameter_count",
        "model_size_mb",
        *LATENCY_FIELDS,
    ]
    if include_flops:
        fieldnames.append("flops")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_input_shape(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
