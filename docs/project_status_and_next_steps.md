# HMA Project Status And Next Steps

Date: 2026-05-13

## Current Status

The repository is now a working Python benchmark skeleton for the Human-Machine Visual Alignment project. It can load configs, build datasets/models/saliency methods, run a dummy static saliency benchmark end to end, compute metrics, write results, aggregate result CSVs, plot summaries, and run neural-alignment utilities on synthetic data.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `76 passed, 3 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

## What Has Been Built

Core infrastructure:

- Python package `hma` with config/path utilities.
- Dataset and model registries.
- Script entrypoints for dummy pipeline, dataset preparation, saliency benchmarking, efficiency profiling, and result aggregation.
- Synced virtual environment with dependencies including `torch`, `timm`, `scipy`, `sklearn`, `pandas`, `matplotlib`, and optional profiling packages.
- Default model-running device policy: `device: auto`, which resolves to GPU when available and CPU otherwise.

Datasets:

- Dummy saliency dataset.
- Manifest-based SALICON loader.
- Manifest-based CAT2000 loader with category filtering.
- COCO-Search18 loader with task-driven fixation points and generated fixation maps.
- NSD / Algonauts-style manifest loader for image, subject, and ROI-response data.
- Dataset preparation support for SALICON, CAT2000, and COCO-Search18.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- Center-bias map utility.
- Efficiency metrics: parameter count, model size, latency, and optional FLOPs.
- Neural utilities: ridge encoding, per-voxel/ROI evaluation, RSA, and RDM comparison.

Models and saliency:

- Base model wrapper.
- `timm` image model wrapper.
- Dummy model path for offline tests.
- Gradient saliency.
- Integrated Gradients.
- Minimal Grad-CAM.
- Attention rollout for ViT-like attention tensors.
- Saliency-method registry.

Experiments and outputs:

- Static saliency benchmark runner.
- Per-image CSV output.
- Aggregate JSON output.
- Optional visualization PNGs.
- Result aggregation across runs.
- Model-ranking plots.
- Alignment-vs-efficiency scatter plots.

## Alignment With Proposal

The proposal frames human-machine visual alignment as a multi-level benchmark rather than a single saliency score. The current codebase now supports the foundation for most of Phase 1:

- Behavioral fixation/saliency alignment.
- Multiple saliency definitions: gradients, Integrated Gradients, Grad-CAM, attention rollout, and dummy routing.
- Static saliency datasets: SALICON and CAT2000.
- Task-driven gaze: COCO-Search18.
- Neural-alignment skeleton for NSD / Algonauts-style encoding and RSA.
- Efficiency profiling.
- Result aggregation and plotting.

Still missing:

- Real benchmark execution over actual SALICON/CAT2000 files.
- Robust image preprocessing for real `timm` models inside benchmark runs.
- Saliency caching for expensive methods.
- Center-bias, random, and DeepGaze-style baselines as first-class benchmark entries.
- Shuffled AUC, AUC-Borji, EMD, and inter-observer ceiling.
- Real fMRI activation extraction over torch dataloaders.
- Brain-Score integration.
- Selective-computation models, token pruning, and foveation.
- Video extension.

## Proposal Review

The proposal's strongest idea is that human-like attention should not be reduced to one heatmap score. It separates:

- Behavioral saliency: agreement with human fixation maps.
- Evidence sensitivity: gradients, Integrated Gradients, and perturbation.
- Class localization: Grad-CAM-style maps.
- Internal routing: attention, rollout, and token masks.
- Computation allocation: retained tokens, glimpses, and sparse frames.
- Neural alignment: fMRI encoding and RSA.
- Efficiency alignment: alignment per parameter, FLOP, or latency.

The codebase should continue preserving this separation. It should not collapse all saliency methods into one undifferentiated attention score.

The main engineering risk now matches the main scientific risk: the benchmark will only be meaningful if preprocessing, saliency generation, caching, and metric aggregation are standardized across models and datasets.

## Recommended Next Step

The next milestone should be **Real Static Saliency Benchmark V1**.

This step should bridge the current skeleton to actual SALICON/CAT2000 experiments with real `timm` models on CPU or GPU.

Recommended additions:

1. Model preprocessing layer
   - Convert PIL/NumPy dataset images into torch tensors.
   - Support resize, normalization, batch dimension, and device placement.
   - Keep `device: auto` as the default so runs use GPU when available and CPU otherwise.
   - Use ImageNet mean/std by default.
   - Keep preprocessing config-driven.

2. Real torch saliency benchmark path
   - Let `run_saliency_benchmark.py` process real datasets with `TimmModelWrapper`.
   - Support `vanilla_gradient`, `integrated_gradients`, `gradcam`, and `attention_rollout`.
   - Use argmax target class by default, with optional fixed-class config.

3. Saliency cache
   - Save predicted saliency maps to `outputs/.../saliency_maps/*.npy`.
   - Reuse cached maps when config/model/dataset/image/method match.
   - This is necessary before running expensive saliency methods.

4. Baseline methods
   - Add `center_bias` and `random_saliency` saliency methods.
   - These directly address the proposal's center-bias and sanity-control requirements.

5. Expanded experiment config
   - Add a real debug config such as `configs/experiments/salicon_resnet50_debug.yaml`.
   - Use a SALICON manifest, `resnet50`, Grad-CAM or gradient saliency, and a small `max_items` default.
   - Include metrics such as `nss`, `auc_judd`, `cc`, `similarity`, and `kl`.

6. Tests
   - Use fake PIL images/maps and a tiny torch model.
   - Verify the benchmark can run through the real torch path.
   - Verify saliency cache files are written and reused.
   - Verify center-bias baseline beats random on synthetic center-fixation data.

This milestone would turn the project from a well-tested architecture skeleton into a usable first experimental benchmark.
