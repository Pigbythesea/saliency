# Human-Machine Visual Alignment

This repository is the starting point for a benchmark comparing computer vision models with human and neural visual alignment signals.

Planned benchmark layers include:

- Static human saliency and fixation maps from datasets such as SALICON and CAT2000.
- Task-driven gaze and scanpaths from datasets such as COCO-Search18.
- fMRI and neural response alignment using NSD / Algonauts-style encoding and RSA.
- Representational geometry analyses such as RSA and CKA.
- Brain-Score Vision style external comparisons.
- Computational efficiency metrics such as parameters, FLOPs, latency, and retained tokens.

The initial implementation is intentionally small. It includes a config-driven dummy end-to-end pipeline that creates synthetic data, generates fake saliency maps, and computes simple metrics. It does not download datasets or load real pretrained models.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On systems where the Python launcher is preferred:

```powershell
py -3.10 -m pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest
```

## Run The Dummy Pipeline

```powershell
python scripts/run_dummy_pipeline.py --config configs/experiments/dummy_pipeline.yaml
```

The dummy pipeline validates the first repo wiring:

1. YAML config loading.
2. Synthetic dataset creation.
3. Fake saliency prediction.
4. Shared saliency normalization.
5. Metric computation.

## Repository Layout

```text
configs/      YAML configs for datasets, models, and experiments.
scripts/      CLI entrypoints and placeholders.
src/hma/      Python package code.
tests/        Pytest coverage for the dummy pipeline and metrics.
```

## Current Scope

This skeleton does not implement dataset downloads, real model loading, fMRI encoding, Brain-Score integration, or video evaluation. Those pieces should be added incrementally behind small modules and config-driven scripts.
