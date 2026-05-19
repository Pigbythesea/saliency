# HMA Project Status And Next Steps

Date: 2026-05-18

## Current Status

The repository now has a first real behavioral-saliency comparison result set for Phase 1 of the Human-Machine Visual Alignment project and a stronger V2 benchmark scaffold. It can load real SALICON, CAT2000, and COCO-Search18 manifests, run baseline and pretrained `timm` model saliency methods, cache saliency maps, aggregate metrics, plot rankings, profile model efficiency, parse observer-level fixation files for SALICON/CAT2000, and produce controlled V2 summaries.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `111 passed, 4 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

## Dataset State

Full manifests:

- SALICON: `data/manifests/salicon_manifest.csv`, 15,000 rows.
- CAT2000: `data/manifests/cat2000_manifest.csv`, 2,000 rows.
- COCO-Search18: `data/manifests/coco_search18_manifest.csv`, 49,760 rows.

SALICON and CAT2000 manifests have been regenerated with `fixation_points_path` when raw fixation-location files are available:

- SALICON `.mat` files under `data/raw/SALICON/fixations/`.
- CAT2000 `.mat` files under `data/raw/CAT2000/trainSet/FIXATIONLOCS/`.

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

- Manifest-based SALICON loader with optional `.mat` fixation-point loading.
- Manifest-based CAT2000 loader with category filtering and optional `.mat` fixation-point loading.
- COCO-Search18 loader with task-driven fixation points and generated fixation maps.
- NSD / Algonauts-style manifest loader for image, subject, and ROI-response data.
- Dataset preparation for the local raw layouts under `data/raw/`.
- Deterministic pilot manifest generation with optional stratification.
- Dataset-specific fixation parsers for SALICON `gaze[*].fixations` and CAT2000 `fixLocs`.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- Controlled V2 metrics: `AUC-Borji`, `shuffled AUC`, and lightweight `EMD`.
- Center-bias map utility.
- Dataset-aware observer-control summaries for SALICON, CAT2000, and COCO-style inline fixation manifests.
- Efficiency metrics: parameter count, model size, latency, and optional FLOPs.
- Aggregate result tables with `saliency_family` preserved.
- Ranking plots and alignment-vs-efficiency plots.
- Generated interpretation notes for V2 aggregate summaries.

Models and saliency:

- `timm` image model wrapper.
- Config-driven torch preprocessing for PIL, NumPy, and tensor inputs.
- Gradient saliency.
- Integrated Gradients.
- Minimal Grad-CAM.
- Attention rollout for ViT-like attention tensors.
- First-class `center_bias` and `random_saliency` baselines.
- Precomputed-map / DeepGaze-style reference saliency method for imported prediction maps.
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

Still missing or incomplete:

- Full V2 pilot matrix completion across all three datasets.
- Larger and more stable V2 `static2000` result set after pilot validation.
- Actual DeepGaze prediction maps or another external saliency-reference export.
- Perturbation methods such as occlusion or RISE.
- Real fMRI activation extraction over torch dataloaders.
- Brain-Score integration.
- Selective-computation models, token pruning, and foveation.
- Video extension.

## Previously Recommended V2 Additions

The previously recommended milestone was **Metric Controls And Scaled Static Benchmark V2**. Most of the infrastructure below has now been implemented; the remaining work is mainly full matrix execution, external reference maps, and paper-ready analysis.

Original recommended additions:

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

## V2 Implementation Progress

Implemented after the V1 pilot milestone:

- Added controlled static saliency metrics:
  - `auc_borji`
  - `shuffled_auc`
  - `emd` / `emd_2d`
- Added benchmark metric context so metrics can use per-image fixation points and other-image shuffled negatives.
- Added deterministic caps for fixation samples used by context-aware AUC metrics so dense fixation maps do not make pilot and scaled runs impractically slow.
- Extended SALICON and CAT2000 manifest preparation/loaders to preserve raw fixation-location paths when available.
- Added V2 scaled manifests:
  - `data/manifests/v2/salicon_static2000_manifest.csv`
  - `data/manifests/v2/cat2000_static2000_manifest.csv`
  - `data/manifests/v2/coco_search18_static2000_manifest.csv`
- Added V2 matrix config generation under:
  - `configs/experiments/real_matrix_v2/`
- Added reporting summaries:
  - top rows by dataset / metric / saliency family
  - best non-baseline rows
  - center-bias deltas
  - saliency-family rankings
  - alignment-per-efficiency summaries when an efficiency CSV is provided
- Added a best-effort observer-control script for manifest rows with parseable fixation points:
  - `scripts/summarize_observer_controls.py`
- Updated plots to facet by dataset and saliency family and to treat `kl`, `emd`, and related loss metrics as lower-is-better.
- Re-ran stable model efficiency profiling to:
  - `outputs/real_matrix_v2/efficiency/model_efficiency.csv`
- Added dataset-specific `.mat` fixation parsing:
  - SALICON: `gaze[*].fixations`.
  - CAT2000: `fixLocs`.
- Updated SALICON and CAT2000 loaders so context-aware metrics can use real fixation points when manifests expose `fixation_points_path`.
- Added a faster fixation-map path using `scipy.ndimage.gaussian_filter` when available, with the existing pointwise renderer retained as fallback.
- Upgraded observer-control reporting:
  - SALICON leave-one-observer-out controls, capped deterministically by `--max-observers-per-image` for practical runtime.
  - CAT2000 fixation-location reference controls against fixation-density maps.
- Added precomputed-map saliency support for DeepGaze-style imported references:
  - `precomputed_map`
  - `deepgaze_precomputed`
- Added V2 matrix orchestration:
  - `scripts/run_v2_matrix.py`
  - reliability checks
  - run ledger CSVs
  - resume behavior
  - aggregation, summaries, plots, and generated interpretation notes

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result after implementation: `111 passed, 4 warnings`.

Real V2 smoke run completed:

```powershell
.\.venv\Scripts\python.exe scripts/run_saliency_benchmark.py --config configs/experiments/real_matrix_v2/salicon_pilot500__center_bias_baseline_center_bias.yaml
```

Smoke aggregate:

- `nss`: 0.4967273755755741
- `shuffled_auc`: 0.6066769326782226
- `auc_borji`: 0.6687713580322265
- `auc_judd`: 0.8554065807640752
- `cc`: 0.503242761939764
- `similarity`: 0.5339597517997027
- `kl`: 0.8564417150020599
- `emd`: 0.07618908158377294

V2 smoke outputs:

- `outputs/real_matrix_v2/salicon_pilot500/center_bias_baseline_center_bias/`
- `outputs/real_matrix_v2/aggregated/smoke_results.csv`
- `outputs/real_matrix_v2/aggregated/smoke_summary/`
- `outputs/real_matrix_v2/aggregated/smoke_plots/`

Additional V2 reliability checks completed:

- `convnext_tiny + gradcam` on SALICON pilot succeeded.
- `deit_small_patch16_224 + attention_rollout` on SALICON pilot succeeded.
- `vit_base_patch16_224 + attention_rollout` on SALICON pilot succeeded.
- Current pilot aggregate/report artifacts are under:
  - `outputs/real_matrix_v2/aggregated/pilot_results.csv`
  - `outputs/real_matrix_v2/aggregated/pilot_results_summary/`
  - `outputs/real_matrix_v2/aggregated/pilot_results_plots/`
- Observer-control artifacts are under:
  - `outputs/real_matrix_v2/observer_controls/salicon_pilot500_observer_controls.csv`
  - `outputs/real_matrix_v2/observer_controls/cat2000_pilot500_observer_controls.csv`

Remaining V2 work:

- Run the full pilot V2 matrix before the scaled 2,000-row matrix.
- Run the scaled `static2000` matrix only after reviewing pilot failures and excluding unsupported methods.
- Add actual DeepGaze/reference prediction maps and corresponding configs using the precomputed-map path.
- Add a compact paper-ready V2 results note after the full pilot and scaled matrices are complete.

## Proposal-Aligned Next Steps From Current Progress

The proposal's global goal is a multi-level human-machine visual alignment benchmark: behavioral saliency, neural prediction, representational geometry, Brain-Score-style comparison, and computational efficiency. The current codebase is strongest on behavioral saliency and efficiency, so the next work should finish that layer before expanding.

### Step 1: Finish Controlled Static Benchmark V2

Run the full pilot matrix with resume enabled:

```powershell
.\.venv\Scripts\python.exe scripts/run_v2_matrix.py --phase pilot --resume
```

After reviewing the run ledger and excluding any failed methods, run the scaled matrix:

```powershell
.\.venv\Scripts\python.exe scripts/run_v2_matrix.py --phase static2000 --resume
```

Acceptance criteria:

- Full pilot aggregate table exists at `outputs/real_matrix_v2/aggregated/pilot_results.csv`.
- Full scaled aggregate table exists at `outputs/real_matrix_v2/aggregated/results.csv`.
- Run ledgers identify successful, failed, skipped, and unsupported configs.
- Summary tables and plots exist for pilot and scaled outputs.
- The interpretation note clearly reports center-bias comparison, saliency-family differences, and alignment-per-efficiency.

### Step 2: Add External Reference Baselines

Use the new precomputed-map method to import DeepGaze-style predictions rather than adding package inference first.

Recommended path:

- Store prediction maps in a stable folder such as `data/precomputed/deepgaze/<dataset_label>/`.
- Add reference configs that use `saliency.method: deepgaze_precomputed`.
- Aggregate DeepGaze/reference rows as `saliency_family: reference`.
- Compare reference rows against center bias, model saliency methods, and observer controls.

Acceptance criteria:

- DeepGaze/reference rows appear in aggregate tables.
- Reference maps are reproducible from documented local files.
- The report separates `reference` rows from model-generated saliency families.

### Step 3: Add Perturbation-Based Saliency

The proposal emphasizes that saliency methods can disagree. V2 currently covers gradients, Integrated Gradients, Grad-CAM, and attention rollout; the next method family should be perturbation-based evidence sensitivity.

Recommended implementation:

- Start with occlusion sensitivity before RISE because it is easier to validate.
- Add config controls for patch size, stride, target class, and maximum images.
- Run perturbation methods on pilot subsets first because runtime will be high.

Acceptance criteria:

- Perturbation saliency produces valid maps and cache entries.
- Pilot results can be compared against gradient and Grad-CAM rows.
- Runtime is documented before any scaled run.

### Step 4: Begin Neural Alignment Bootstrap

Once V2 static results are stable, start the proposal's neural-alignment layer with the already present NSD / Algonauts-style loader and neural utilities.

Recommended implementation:

- Add a neural experiment config format for dataset, model, layers, train/test split, ROI, and ridge alpha.
- Implement a script that extracts activations, fits ridge encoding models, and writes ROI-level prediction scores.
- Use a small fixture or local manifest first; do not require overlap with SALICON/CAT2000.
- Add RSA over the same image set using `hma.neural.rsa`.

Acceptance criteria:

- One end-to-end neural smoke run writes activation files and encoding scores.
- Tests cover activation extraction shape, score aggregation, and failure handling for missing ROI responses.
- Neural results can later be joined with behavioral saliency rows by model/family, not necessarily by identical images.

### Step 5: Architecture Expansion Toward The Proposal's Core Hypothesis

After the static and neural layers are stable, expand beyond standard CNN/ViT families toward the proposal's central hypothesis about efficient and adaptive vision.

Priority order:

1. Self-supervised ViT or CLIP/DINO-style encoders, because they are likely to change saliency and neural alignment without requiring custom routing logic.
2. State-space or hybrid vision backbones, if available through `timm` or a stable wrapper.
3. Selective-computation models such as token pruning, foveation, or glimpse models.
4. Video models only after static image results are publishable.

Acceptance criteria:

- Each new architecture family has model metadata, saliency compatibility notes, and efficiency rows.
- Results remain grouped by architecture family and saliency family.
- The analysis can test whether alignment improves with efficiency or selective computation rather than only with model scale.
