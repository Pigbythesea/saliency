# HMA Project Status And Next Steps

Date: 2026-05-13

## Current Status

The repository is now a working Python benchmark for Phase 1 of the Human-Machine Visual Alignment project. It can load configs, build datasets/models/saliency methods, run dummy and torch-backed static saliency benchmarks, compute metrics, cache saliency maps, write result artifacts, aggregate result CSVs, plot summaries, and run neural-alignment utilities on synthetic data.

The local raw datasets are now present under `data/raw/`, and manifests have been generated for all three prepared behavioral datasets:

- SALICON: `data/manifests/salicon_manifest.csv`, 15,000 rows.
- CAT2000: `data/manifests/cat2000_manifest.csv`, 2,000 rows.
- COCO-Search18: `data/manifests/coco_search18_manifest.csv`, 49,760 rows.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `87 passed, 4 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

The first real-data SALICON debug benchmark now runs successfully:

```powershell
.\.venv\Scripts\python.exe scripts/run_saliency_benchmark.py --config configs/experiments/salicon_resnet50_debug.yaml
```

Output directory:

```text
outputs/salicon_resnet50_debug/
```

Debug-run aggregate metrics over 5 SALICON validation images with `resnet50`, `pretrained: false`, and Grad-CAM:

- `nss`: 0.13321189284324647
- `auc_judd`: 0.6146952838634981
- `cc`: 0.17930675074458122
- `similarity`: 0.45056723356246947
- `kl`: 1.1277051389217376

These numbers prove the real-data path works, but they should not be interpreted scientifically because the model config currently uses `pretrained: false`.

## What Has Been Built

Core infrastructure:

- Python package `hma` with config/path utilities.
- Dataset and model registries.
- Script entrypoints for dummy pipeline, dataset preparation, saliency benchmarking, efficiency profiling, and result aggregation.
- Synced virtual environment with dependencies including `torch`, `timm`, `scipy`, `sklearn`, `pandas`, `matplotlib`, and optional profiling packages.
- Default model-running device policy: `device: auto`, which resolves to GPU when available and CPU otherwise.
- Config defaults for preprocessing and saliency-map caching.

Datasets and manifests:

- Dummy saliency dataset.
- Manifest-based SALICON loader.
- Manifest-based CAT2000 loader with category filtering.
- COCO-Search18 loader with task-driven fixation points and generated fixation maps.
- NSD / Algonauts-style manifest loader for image, subject, and ROI-response data.
- Dataset preparation support for SALICON, CAT2000, and COCO-Search18.
- `scripts/prepare_dataset.py` now handles the local raw layouts:
  - `data/raw/SALICON/images`, `maps`, and `fixations`.
  - `data/raw/CAT2000/trainSet/Stimuli` and `FIXATIONMAPS`.
  - COCO-Search18 fixation JSONs using `name`, `task`, `condition`, `X`, and `Y`.
- Dataset config roots now point to the prepared raw data folders:
  - `data/raw/SALICON`
  - `data/raw/CAT2000`
  - `data/raw/COCO-Search18`

Manifest summaries:

- SALICON: 10,000 train rows and 5,000 validation rows.
- CAT2000: 2,000 train rows across 20 categories.
- COCO-Search18: 42,485 train rows and 7,275 validation rows.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- `auc_judd` is wired into the saliency benchmark config path.
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

- Behavioral fixation/saliency alignment on real SALICON and CAT2000 files.
- Task-driven gaze alignment through COCO-Search18 manifests.
- Multiple saliency definitions: gradients, Integrated Gradients, Grad-CAM, attention rollout, center-bias baseline, random baseline, and dummy routing.
- Explicit saliency families: evidence sensitivity, class localization, internal routing, and baseline controls.
- Neural-alignment skeleton for NSD / Algonauts-style encoding and RSA.
- Efficiency profiling.
- Result aggregation and plotting.

Still missing:

- Baseline comparison result set over the same real-data subsets.
- Pretrained model runs for scientific interpretation.
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

## Completed Recently

The **Real Static Saliency Benchmark V1** milestone has been implemented and moved from synthetic-only validation to a first real SALICON debug run.

Completed additions:

1. Model preprocessing layer
   - Converts PIL, NumPy, and torch tensor images into `BxCxHxW` tensors.
   - Supports resize, RGB/channel handling, float scaling, ImageNet mean/std normalization, batch dimension, and device placement.
   - Uses config-driven defaults under `preprocessing`.

2. Real torch saliency benchmark path
   - Benchmark runner preprocesses images for torch saliency methods.
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

5. Real dataset onboarding
   - Generated SALICON, CAT2000, and COCO-Search18 manifests.
   - Patched dataset preparation for the local CAT2000 and COCO-Search18 layouts.
   - Updated dataset configs to point to `data/raw/...`.
   - Confirmed SALICON real-data benchmark execution.

6. Tests
   - Added preprocessing tests.
   - Added torch benchmark integration tests with fake PIL images and a tiny torch model.
   - Added cache write/reuse/invalidation tests.
   - Added baseline sanity tests showing center bias beats random on synthetic center fixation.
   - Re-ran full suite after manifest-preparation changes: `87 passed, 4 warnings`.

## Recommended Next Steps

The next milestone should be **Real Data Static Benchmark V1 Comparisons**.

This should produce the first interpretable comparison tables across real datasets, baselines, and a small set of model/saliency methods.

Recommended additions:

1. Add baseline experiment configs
   - Create SALICON configs for `center_bias` and `random_saliency` using the same split, `max_items`, metrics, and output structure as `salicon_resnet50_debug.yaml`.
   - Create CAT2000 debug configs for `center_bias`, `random_saliency`, and `resnet50 + gradcam`.
   - Create COCO-Search18 debug configs for `center_bias`, `random_saliency`, and one torch method if the task-driven setup is stable.

2. Run first baseline comparisons
   - Run SALICON `center_bias`, `random_saliency`, and `resnet50 + gradcam` on the same 5-image validation subset.
   - Aggregate those outputs and confirm that baseline rows compare correctly by `dataset`, `model`, `saliency_method`, and `saliency_family`.
   - Increase `max_items` after the debug comparison is stable.

3. Switch scientific model runs to pretrained weights
   - Keep `pretrained: false` for offline smoke tests.
   - Use `pretrained: true` for the first meaningful ResNet/ConvNeXt/ViT comparisons.
   - Confirm whether pretrained weights are already cached; if not, allow `timm`/PyTorch to download them once.

4. Run CAT2000 and COCO-Search18 debug comparisons
   - CAT2000 should use `split: train`, since the prepared manifest has only train rows.
   - COCO-Search18 can use `split: val` or `split: train`; `val` is better for a compact first check.
   - Confirm fixation maps and visualizations look reasonable before scaling up.

5. Add result aggregation artifacts
   - Aggregate all debug outputs into one CSV.
   - Generate model-ranking and alignment-vs-efficiency plots for the debug result set.
   - Record cache hit/write counts after repeat runs.

6. Expand metrics after baseline result sanity
   - Add shuffled AUC to control for dataset center bias.
   - Add AUC-Borji and EMD.
   - Add inter-observer ceiling only when the dataset representation exposes individual observer fixation data or equivalent splits.

7. Expand model matrix
   - Start with `resnet50`, `convnext_tiny`, `vit_base_patch16_224`, `deit_small_patch16_224`, and `swin_tiny_patch4_window7_224`.
   - Compare saliency families separately instead of averaging them into one score.
   - Join saliency aggregates with efficiency profiles for parameter count, model size, latency, and optional FLOPs.

This next milestone will move the project from a validated real-data debug path to the first reproducible comparison result set.
