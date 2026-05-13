# HMA Project Status And Next Steps

Date: 2026-05-13

## Current Status

The repository now has a first real behavioral-saliency comparison result set for Phase 1 of the Human-Machine Visual Alignment project. It can load real SALICON, CAT2000, and COCO-Search18 manifests, run baseline and pretrained `timm` model saliency methods, cache saliency maps, aggregate metrics, plot rankings, and profile model efficiency.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `91 passed, 4 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

## Dataset State

Full manifests:

- SALICON: `data/manifests/salicon_manifest.csv`, 15,000 rows.
- CAT2000: `data/manifests/cat2000_manifest.csv`, 2,000 rows.
- COCO-Search18: `data/manifests/coco_search18_manifest.csv`, 49,760 rows.

Pilot manifests for the first meaningful output:

- SALICON validation pilot: `data/manifests/pilot/salicon_pilot500_manifest.csv`, 500 rows.
- CAT2000 train pilot: `data/manifests/pilot/cat2000_pilot500_manifest.csv`, 500 rows balanced across categories where possible.
- COCO-Search18 validation pilot: `data/manifests/pilot/coco_search18_pilot500_manifest.csv`, 500 rows balanced across target categories where possible.

Pilot manifests are generated deterministically with:

```powershell
.\.venv\Scripts\python.exe scripts/create_pilot_manifests.py --max-rows 500 --seed 123
```

## Completed Model-Matrix Output

The first real model-matrix output is under:

```text
outputs/real_matrix_v1/
```

Generated configs:

- Matrix experiment configs: `configs/experiments/real_matrix_v1/`
- Selected pretrained model config: `configs/models/selected_pretrained_matrix.yaml`

Config generation command:

```powershell
.\.venv\Scripts\python.exe scripts/create_real_matrix_v1_configs.py
```

Completed runs:

- Baselines on all three pilot datasets:
  - `center_bias`
  - `random_saliency`
- Pretrained `vanilla_gradient` on all three pilot datasets for:
  - `resnet50`
  - `convnext_tiny`
  - `vit_base_patch16_224`
  - `deit_small_patch16_224`
  - `swin_tiny_patch4_window7_224`
- Pretrained `resnet50 + gradcam` references on:
  - SALICON pilot
  - CAT2000 pilot

Aggregate output:

```text
outputs/real_matrix_v1/aggregated/results.csv
```

The aggregate table contains 115 rows across:

- 3 pilot datasets.
- 7 model labels, including baselines.
- 4 saliency methods: `center_bias`, `random_saliency`, `vanilla_gradient`, and `gradcam`.
- 5 metrics: `nss`, `auc_judd`, `cc`, `similarity`, and `kl`.

Ranking and efficiency plots were generated under:

```text
outputs/real_matrix_v1/aggregated/
```

Efficiency profile:

```text
outputs/real_matrix_v1/efficiency/model_efficiency.csv
```

The efficiency profile uses pretrained models, input shape `1,3,224,224`, and short latency settings: `warmup=1`, `repeats=3`.

## Headline Pilot Results

Top NSS rows by dataset:

SALICON pilot:

- `center_bias_baseline + center_bias`: 0.4967273755755741
- `resnet50 + gradcam`: 0.3253708058288321
- `resnet50 + vanilla_gradient`: 0.1827250827938551
- `convnext_tiny + vanilla_gradient`: 0.128814539164654
- `swin_tiny_patch4_window7_224 + vanilla_gradient`: 0.02040343557302549

CAT2000 pilot:

- `center_bias_baseline + center_bias`: 0.5205845055580139
- `resnet50 + gradcam`: 0.38764822678710337
- `resnet50 + vanilla_gradient`: 0.23906351370131598
- `convnext_tiny + vanilla_gradient`: 0.15983615024911704
- `random_baseline + random_saliency`: -0.00023580646698610508

COCO-Search18 pilot:

- `center_bias_baseline + center_bias`: 0.013211032435787274
- `resnet50 + vanilla_gradient`: 0.009633028535892664
- `convnext_tiny + vanilla_gradient`: 0.007855317291072424
- `vit_base_patch16_224 + vanilla_gradient`: 0.003890672454595915
- `swin_tiny_patch4_window7_224 + vanilla_gradient`: 0.0007335475623306138

Interpretation caveat:

- These are pilot-scale results, not final scientific claims.
- The strong center-bias baseline on SALICON/CAT2000 confirms the proposal's concern that center bias must be controlled explicitly.
- COCO-Search18 NSS values are much smaller because the task-driven fixation maps are sparse and generated from fixation points.

## What Has Been Built

Core infrastructure:

- Python package `hma` with config/path utilities.
- Dataset and model registries.
- Script entrypoints for dataset preparation, pilot manifest generation, matrix config generation, saliency benchmarking, efficiency profiling, and result aggregation.
- Default model-running device policy: `device: auto`.
- Config defaults for preprocessing and saliency-map caching.

Datasets and manifests:

- Manifest-based SALICON loader.
- Manifest-based CAT2000 loader with category filtering.
- COCO-Search18 loader with task-driven fixation points and generated fixation maps.
- NSD / Algonauts-style manifest loader for image, subject, and ROI-response data.
- Dataset preparation for the local raw layouts under `data/raw/`.
- Deterministic pilot manifest generation with optional stratification.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- Center-bias map utility.
- Efficiency metrics: parameter count, model size, latency, and optional FLOPs.
- Aggregate result tables with `saliency_family` preserved.
- Ranking plots and alignment-vs-efficiency plots.

Models and saliency:

- `timm` image model wrapper.
- Config-driven torch preprocessing for PIL, NumPy, and tensor inputs.
- Gradient saliency.
- Integrated Gradients.
- Minimal Grad-CAM.
- Attention rollout for ViT-like attention tensors.
- First-class `center_bias` and `random_saliency` baselines.
- Model-independent baselines no longer require dummy model wrappers.

## Alignment With Proposal

This milestone directly supports the proposal's Phase 1 direction:

- Behavioral fixation/saliency alignment now runs on real SALICON, CAT2000, and COCO-Search18 data.
- Saliency families remain separated:
  - `baseline`
  - `evidence_sensitivity`
  - `class_localization`
  - `internal_routing`
- The first comparison shows why center-bias and random controls are essential.
- Efficiency profiling is available for alignment-per-computation analysis.

Still missing:

- Shuffled AUC, AUC-Borji, EMD, and inter-observer ceiling.
- DeepGaze-style upper/reference baseline.
- Larger and more stable sample sizes beyond pilot 500.
- More saliency families across model architectures, especially attention rollout and perturbation methods.
- Real fMRI activation extraction over torch dataloaders.
- Brain-Score integration.
- Selective-computation models, token pruning, and foveation.
- Video extension.

## Recommended Next Steps

The next milestone should be **Metric Controls And Scaled Static Benchmark V2**.

Recommended additions:

1. Add center-bias-aware metrics
   - Implement shuffled AUC using other-image fixation locations as negatives.
   - Add AUC-Borji.
   - Add EMD if dependency and performance are acceptable.
   - Keep `auc_judd` for continuity but do not treat it as sufficient.

2. Add inter-observer or upper-bound controls
   - Use individual fixation data where available.
   - Start with SALICON/CAT2000 if observer-level data can be extracted reliably.
   - Add DeepGaze-style baseline only after deciding whether to call a package, import exported predictions, or store precomputed maps.

3. Scale the static benchmark
   - Increase from pilot 500 to larger stable subsets, such as 2,000 rows where available.
   - Keep the pilot manifests for fast regression checks.
   - Reuse saliency caches instead of recomputing unchanged maps.

4. Improve model-family coverage
   - Add attention rollout runs for ViT/DeiT where attention extraction is reliable.
   - Add Grad-CAM or CAM-like references for CNN-compatible models.
   - Keep cross-model rankings separated by saliency family.

5. Improve reporting
   - Add a summary script that extracts top rows per dataset/metric/family from `outputs/real_matrix_v1/aggregated/results.csv`.
   - Add plots that facet by dataset and saliency family.
   - Add lower-is-better handling for KL in all relevant ranking/reporting paths.

6. Update efficiency analysis
   - Re-run latency with more stable settings, such as `warmup=5`, `repeats=20`.
   - Add FLOPs if optional dependencies are available and stable.
   - Report alignment per latency, per parameter, and per model size.

This next milestone will move the project from a first real comparison table to a more scientifically controlled static saliency benchmark.
