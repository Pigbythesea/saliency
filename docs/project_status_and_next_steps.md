# HMA Project Status And Next Steps

Date: 2026-05-13

## Current Status

The repository is now a working Python benchmark for the first phase of the Human-Machine Visual Alignment project. It can load configs, build datasets/models/saliency methods, run dummy and torch-backed static saliency benchmarks, compute metrics, cache saliency maps, write result artifacts, aggregate result CSVs, plot summaries, and run neural-alignment utilities on synthetic data.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `87 passed, 4 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

The real SALICON debug command was also attempted:

```powershell
.\.venv\Scripts\python.exe scripts/run_saliency_benchmark.py --config configs/experiments/salicon_resnet50_debug.yaml
```

It reaches dataset construction and currently stops because the local manifest is missing:

```text
SALICON manifest not found: D:\Git\saliency\data\manifests\salicon_manifest.csv
```

This is a data-availability blocker, not a benchmark-code failure. The torch benchmark path is covered by synthetic integration tests.

## What Has Been Built

Core infrastructure:

- Python package `hma` with config/path utilities.
- Dataset and model registries.
- Script entrypoints for dummy pipeline, dataset preparation, saliency benchmarking, efficiency profiling, and result aggregation.
- Synced virtual environment with dependencies including `torch`, `timm`, `scipy`, `sklearn`, `pandas`, `matplotlib`, and optional profiling packages.
- Default model-running device policy: `device: auto`, which resolves to GPU when available and CPU otherwise.
- Config defaults for preprocessing and saliency-map caching.

Datasets:

- Dummy saliency dataset.
- Manifest-based SALICON loader.
- Manifest-based CAT2000 loader with category filtering.
- COCO-Search18 loader with task-driven fixation points and generated fixation maps.
- NSD / Algonauts-style manifest loader for image, subject, and ROI-response data.
- Dataset preparation support for SALICON, CAT2000, and COCO-Search18.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- `auc_judd` is now wired into the saliency benchmark config path.
- Center-bias map utility.
- Efficiency metrics: parameter count, model size, latency, and optional FLOPs.
- Neural utilities: ridge encoding, per-voxel/ROI evaluation, RSA, and RDM comparison.

Models and saliency:

- Base model wrapper.
- `timm` image model wrapper.
- Dummy model path for offline tests.
- Config-driven torch image preprocessing for PIL, NumPy, and tensor inputs.
- Gradient saliency.
- Integrated Gradients.
- Minimal Grad-CAM.
- Attention rollout for ViT-like attention tensors.
- First-class `center_bias` and `random_saliency` baselines.
- Saliency-method registry with saliency-family metadata support.

Experiments and outputs:

- Static saliency benchmark runner.
- Real torch saliency benchmark path for `vanilla_gradient`, `integrated_gradients`, `gradcam`, and `attention_rollout`.
- Optional fixed `target_class`; otherwise saliency methods use argmax logits.
- Saliency-map caching under `outputs/.../saliency_maps/*.npy` with metadata-based reuse and invalidation.
- Per-image CSV output.
- Aggregate JSON output with `saliency_family`, cache hit count, and cache write count.
- Optional visualization PNGs.
- Result aggregation across runs.
- Model-ranking plots.
- Alignment-vs-efficiency scatter plots.
- Real debug config at `configs/experiments/salicon_resnet50_debug.yaml`.

## Alignment With Proposal

The proposal frames human-machine visual alignment as a multi-level benchmark rather than a single saliency score. The current codebase now supports the foundation for Phase 1:

- Behavioral fixation/saliency alignment.
- Multiple saliency definitions: gradients, Integrated Gradients, Grad-CAM, attention rollout, center-bias baseline, random baseline, and dummy routing.
- Explicit saliency families: evidence sensitivity, class localization, internal routing, and baseline controls.
- Static saliency datasets: SALICON and CAT2000.
- Task-driven gaze: COCO-Search18.
- Neural-alignment skeleton for NSD / Algonauts-style encoding and RSA.
- Efficiency profiling.
- Result aggregation and plotting.

Still missing:

- Local SALICON/CAT2000 manifest files and actual benchmark runs over real images.
- DeepGaze-style upper/reference baseline.
- Shuffled AUC, AUC-Borji, EMD, and inter-observer ceiling.
- Larger architecture comparison matrix across CNNs, ViTs, hierarchical transformers, and efficient/adaptive models.
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

The main engineering risk now matches the main scientific risk: the benchmark will only be meaningful if preprocessing, saliency generation, caching, and metric aggregation remain standardized across models and datasets.

## Completed This Session

The **Real Static Saliency Benchmark V1** milestone has been implemented.

Completed additions:

1. Model preprocessing layer
   - Converts PIL, NumPy, and torch tensor images into `BxCxHxW` tensors.
   - Supports resize, RGB/channel handling, float scaling, ImageNet mean/std normalization, batch dimension, and device placement.
   - Uses config-driven defaults under `preprocessing`.

2. Real torch saliency benchmark path
   - Benchmark runner now preprocesses images for torch saliency methods.
   - Model is moved to the resolved device before real torch saliency execution.
   - Supports `vanilla_gradient`, `integrated_gradients`, `gradcam`, and `attention_rollout`.
   - Supports optional fixed `target_class`; otherwise existing argmax-target behavior remains.
   - Dummy benchmark behavior remains covered and unchanged.

3. Saliency cache
   - Saves raw saliency predictions to `outputs/.../saliency_maps/*.npy`.
   - Saves matching cache metadata as JSON.
   - Reuses cache entries when dataset, split, image id/path, model config, saliency config, preprocessing config, and target shape match.
   - Invalidates cache automatically when method/model/preprocessing config changes.

4. Baseline methods
   - Added `center_bias`.
   - Added seeded `random_saliency`.
   - Both are registered as first-class saliency methods.

5. Expanded experiment config
   - Added `configs/experiments/salicon_resnet50_debug.yaml`.
   - Uses SALICON, `resnet50`, Grad-CAM, `max_items: 5`, cache enabled, and metrics `nss`, `auc_judd`, `cc`, `similarity`, and `kl`.
   - Keeps `pretrained: false` for offline reproducibility, with an inline note to use `pretrained: true` for meaningful model results.

6. Tests
   - Added preprocessing tests.
   - Added torch benchmark integration tests with fake PIL images and a tiny torch model.
   - Added cache write/reuse/invalidation tests.
   - Added baseline sanity tests showing center bias beats random on synthetic center fixation.

## Recommended Next Steps

The next milestone should be **Real Data Static Benchmark V1 Results**.

This should turn the implemented V1 benchmark into actual SALICON/CAT2000 experimental outputs.

Recommended additions:

1. Data onboarding and manifest validation
   - Create or place `data/manifests/salicon_manifest.csv`.
   - Create or place `data/manifests/cat2000_manifest.csv`.
   - Run manifest validation with `validate_files: true`.
   - Add a small documented manifest example for each dataset.
   - Confirm image paths, fixation-map paths, split labels, and image dimensions.

2. First real benchmark runs
   - Run SALICON debug with `resnet50` and Grad-CAM.
   - Run SALICON debug with `center_bias` and `random_saliency`.
   - Run CAT2000 debug with the same first model/baseline set.
   - Save outputs under separate experiment directories so aggregation can compare model methods and baselines.

3. Baseline and sanity-control expansion
   - Add Gaussian center-prior variants if needed.
   - Add a DeepGaze-style reference baseline when a dependency or exported prediction format is chosen.
   - Add model-weight or label-randomization sanity checks after the first real runs are stable.

4. Metric expansion
   - Add shuffled AUC to control for dataset center bias.
   - Add AUC-Borji and EMD.
   - Add inter-observer ceiling only when the dataset representation exposes individual observer fixation data or equivalent splits.

5. Model comparison matrix
   - Start with a small, balanced set: `resnet50`, `convnext_tiny`, `vit_base_patch16_224`, `deit_small_patch16_224`, and `swin_tiny_patch4_window7_224`.
   - Compare saliency families separately instead of averaging them into one score.
   - Keep `pretrained: false` only for offline smoke tests; use pretrained weights for scientific interpretation.

6. Efficiency join
   - Profile parameters, model size, latency, and optional FLOPs for the same model set.
   - Join efficiency summaries with saliency aggregate outputs.
   - Plot alignment-vs-efficiency for each saliency family.

7. Documentation update after first real data run
   - Record exact dataset paths, manifest generation steps, config names, commands run, and output directories.
   - Add a short results table for baseline vs first real model.
   - Note whether cache hits are working across repeated runs.

This next milestone will move the project from a tested real-benchmark implementation to the first reproducible experimental result set.
