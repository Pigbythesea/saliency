# HMA Project Status And Next Steps

Date: 2026-05-18

Latest implementation update: 2026-05-20

## Latest Saliency Evaluation Protocol Fix

Status: **implemented; full corrected static2000 matrix still needs re-execution.**

The behavioral saliency pipeline has been revised to align with academic fixation-benchmark semantics:

- `nss` and `auc_judd` now use discrete fixation coordinates when available, instead of thresholding blurred fixation-density maps.
- The benchmark runner now stores `fixation_protocol` in per-image and aggregate outputs:
  - `points` for free-viewing datasets with observer fixation locations.
  - `task_points` for COCO-Search18 task/scanpath fixation points.
  - `density_fallback` only when raw fixation points are unavailable.
- `auc_borji` and `shuffled_auc` still use coordinate-based positives, with capped sampled positives for runtime.
- CC, SIM, KL, and EMD remain density-map metrics.
- Saliency-map resizing now uses bilinear interpolation instead of nearest-neighbor sampling.
- Prediction maps are resized without forced min-max normalization before all metrics; metric-specific normalization is handled by each metric.
- DeepGaze/precomputed map loading now supports collision-safe `{map_key}` templates.
- DeepGaze reference configs now keep SALICON on `{image_id}.npy` and switch CAT2000 / COCO-Search18 to `{map_key}.npy`, because CAT2000 has 2,000 static rows but only 100 unique `image_id` values.
- The paper inspection pack now includes `fixation_protocol` in the behavioral NSS table and a new benchmark sanity table:
  - `outputs/paper_inspection_v1/tables/table6_benchmark_sanity_ranges.md`

Important interpretation boundary:

- Existing `outputs/real_matrix_v2/aggregated/results.csv` values were produced before this protocol fix and are superseded for NSS/AUC scientific claims.
- The regenerated inspection pack currently exposes the old aggregate with protocol marked `unknown`; this is a warning state, not corrected evidence.
- Corrected claims require rerunning the affected behavioral configs and re-aggregating.

Verification completed for the protocol implementation:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_metrics.py tests\test_saliency_benchmark.py tests\test_aggregate_results.py tests\test_export_deepgaze_maps.py tests\test_paper_inspection_pack.py
```

Result: `46 passed, 2 warnings`.

Required next re-run sequence for corrected behavioral numbers:

1. Re-export collision-safe DeepGaze maps for CAT2000 and COCO-Search18:

```cmd
.\.venv\Scripts\python.exe scripts\export_deepgaze_maps.py --manifest data\manifests\v2\cat2000_static2000_manifest.csv --image-root data\raw\CAT2000 --output-dir data\precomputed\deepgaze\cat2000_static2000 --filename-template "{map_key}.npy"
.\.venv\Scripts\python.exe scripts\export_deepgaze_maps.py --manifest data\manifests\v2\coco_search18_static2000_manifest.csv --image-root data\raw\COCO-Search18 --output-dir data\precomputed\deepgaze\coco_search18_static2000 --filename-template "{map_key}.npy"
```

2. Rerun baselines and DeepGaze references first, then aggregate and inspect whether DeepGaze has the expected point-NSS relationship to center bias:

```cmd
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2\salicon_static2000__center_bias_baseline_center_bias.yaml
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\salicon_static2000__deepgaze_reference_deepgaze_precomputed.yaml
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\cat2000_static2000__deepgaze_reference_deepgaze_precomputed.yaml
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\coco_search18_static2000__deepgaze_reference_deepgaze_precomputed.yaml
```

3. Only after the reference sanity check passes, rerun the model saliency rows and regenerate aggregate summaries / inspection packs.

## Current Status

The repository now has a controlled behavioral-saliency benchmark for Phase 1 of the Human-Machine Visual Alignment project and a scaled first bridge into the proposal's neural layer. It can load real SALICON, CAT2000, COCO-Search18, and one local Algonauts 2023 subject manifest, run baseline and pretrained `timm` model saliency methods, cache saliency maps, aggregate metrics, plot rankings, profile model efficiency, parse observer-level fixation files for SALICON/CAT2000, validate neural manifests, extract named `timm` layer activations, and run neural encoding plus RSA over true PRF visual ROIs.

Current implementation frontier:

- Behavioral layer: frozen V2 static2000 benchmark with center-bias, random, model-saliency, DeepGaze/reference, and pilot occlusion rows.
- Neural layer: ROI500 encoding + RSA completed for `resnet50`, `convnext_tiny`, `deit_small_patch16_224`, `vit_base_patch16_224`, and pretrained `vit_small_patch14_dinov2` across bilateral V1, V2, V3, and hV4 in `subj01`; pretrained V1 debug runs also completed for `vit_base_patch16_clip_224` and `resnet50_clip`.
- Bridge/reporting layer: paper-style behavior-neural analysis tables and an inspection pack generated for frozen static2000 behavioral rows and multi-model ROI500 neural summaries, now including DINOv2 ROI500 neural rows and pretrained SSL/multimodal candidate status.
- Next scientific blocker: add behavioral saliency rows for SSL/multimodal models or expand full ROI500 to CLIP/SigLIP only after deciding which comparison is needed for the next paper claim.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `128 passed, 4 warnings`.

Latest verification after ROI500 summary implementation:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `131 passed, 4 warnings`.

Latest verification after multi-model neural expansion:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `132 passed, 4 warnings`.

Latest verification after behavior-neural analysis V1 and SSL candidate prep:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `134 passed, 4 warnings`.

Latest verification after paper inspection pack generation:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `134 passed, 4 warnings`.

The remaining warnings are not blocking:

- PyTorch Grad-CAM backward-hook warning in tests.
- `.pytest_cache` permission warning on Windows.

## Dataset State

Full manifests:

- SALICON: `data/manifests/salicon_manifest.csv`, 15,000 rows.
- CAT2000: `data/manifests/cat2000_manifest.csv`, 2,000 rows.
- COCO-Search18: `data/manifests/coco_search18_manifest.csv`, 49,760 rows.
- Algonauts / NSD subject 1 smoke manifest: `data/manifests/nsd_algonauts_manifest.csv`, 9,841 train rows for `subj01`, `roi: all_lh_512`.
- Algonauts / NSD PRF visual ROI smoke manifest: `data/manifests/nsd_algonauts_prf_visualrois_manifest.csv`, 256 train rows across `V1`, `V2`, `V3`, and `hV4`.

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

Local neural data now present:

- Algonauts 2023 subject 1 under `data/raw/nsd_algonauts/subj01/`.
- Training images: 9,841 PNG files.
- LH fMRI: `data/raw/nsd_algonauts/subj01/training_split/training_fmri/lh_training_fmri.npy`, shape `(9841, 19004)`.
- RH fMRI: `data/raw/nsd_algonauts/subj01/training_split/training_fmri/rh_training_fmri.npy`, shape `(9841, 20544)`.
- ROI masks under `data/raw/nsd_algonauts/subj01/roi_masks/`.
- Current smoke responses: per-image first-512 LH response vectors under `data/raw/nsd_algonauts/subj01/responses/all_lh_512/`.
- Current true ROI smoke responses:
  - `data/raw/nsd_algonauts/subj01/responses/V1/`, 64 files, bilateral response dimension 2,973.
  - `data/raw/nsd_algonauts/subj01/responses/V2/`, 64 files, bilateral response dimension 2,936.
  - `data/raw/nsd_algonauts/subj01/responses/V3/`, 64 files, bilateral response dimension 2,453.
  - `data/raw/nsd_algonauts/subj01/responses/hV4/`, 64 files, bilateral response dimension 1,296.

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
- Algonauts subject converter for one-subject neural smoke manifests.
- Neural manifest validator for file paths, split / subject / ROI coverage, and response-shape consistency.
- Dataset preparation for the local raw layouts under `data/raw/`.
- Deterministic pilot manifest generation with optional stratification.
- Dataset-specific fixation parsers for SALICON `gaze[*].fixations` and CAT2000 `fixLocs`.

Metrics and analysis:

- Static saliency metrics: `NSS`, `AUC-Judd`, `CC`, `SIM`, `KL`, `MAE`, and Pearson.
- Controlled V2 metrics: `AUC-Borji`, `shuffled AUC`, and lightweight `EMD`.
- Center-bias map utility.
- Dataset-aware observer-control summaries for SALICON, CAT2000, and COCO-style inline fixation manifests.
- Efficiency metrics: parameter count, model size, latency, and optional FLOPs.
- Ridge encoding summaries over neural response vectors.
- RSA summaries comparing model-layer RDMs against ROI-response RDMs.
- Aggregate result tables with `saliency_family` preserved.
- Ranking plots and alignment-vs-efficiency plots.
- Generated interpretation notes for V2 aggregate summaries.

Models and saliency:

- `timm` image model wrapper with named-layer forward-hook extraction for neural runs.
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

- Scaled neural encoding / RSA runs beyond the 64-image PRF visual ROI smoke subsets.
- Multi-ROI result summarization and comparison against behavioral saliency summaries.
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
  - key center-bias / Grad-CAM / gradient / attention comparison rows
  - pilot-to-static ranking stability rows
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
- Added DeepGaze-style reference config generation:
  - `scripts/create_deepgaze_reference_configs.py`
  - output defaults to `configs/experiments/real_matrix_v2_references/`
- Added pilot-only perturbation saliency support:
  - `saliency.method: occlusion`
  - config controls for `patch_size`, `stride`, `baseline_value`, and normal `target_class`
  - V2 config generation adds `resnet50 + occlusion` only for pilot datasets
- Added a neural-alignment smoke runner:
  - `src/hma/experiments/neural_alignment.py`
  - `scripts/run_neural_alignment.py`
  - `configs/experiments/neural_smoke_dummy.yaml`
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

Result after implementation: `121 passed, 4 warnings`.

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

V2 static2000 completion:

- Full aggregate table: `outputs/real_matrix_v2/aggregated/results.csv`.
- Static ledger: `outputs/real_matrix_v2/run_ledgers/static2000_run_ledger.csv`.
- Static ledger status: 33 `static2000` configs succeeded; 3 skipped rows were already-completed pilot reliability checks.
- Summary directory: `outputs/real_matrix_v2/aggregated/results_summary/`.
- Plot directory: `outputs/real_matrix_v2/aggregated/results_plots/`.
- EMD remains pilot-only by config because static EMD is slower; static interpretation should use NSS, shuffled AUC, AUC-Borji, AUC-Judd, CC, SIM, and KL.

Static2000 headline result:

- No non-baseline row beats center bias on NSS for SALICON, CAT2000, or COCO-Search18.
- Best non-baseline NSS rows:
  - SALICON: `resnet50 + gradcam`, mean 0.3421211974733906.
  - CAT2000: `resnet50 + gradcam`, mean 0.39305544706738876.
  - COCO-Search18: `resnet50 + gradcam`, mean 0.6341453774338297.
- Shuffled AUC shows a more nuanced pattern: COCO-Search18 static2000 is led by `deit_small_patch16_224 + attention_rollout`, while SALICON and CAT2000 still favor center bias.
- KL also favors center bias across the three static2000 datasets.

Remaining V2 work:

- Use DeepGaze/reference and pilot occlusion rows in paper-ready behavioral analysis.
- Keep occlusion pilot-only unless a targeted perturbation ablation is needed later.
- Use the new key-comparison and pilot/static-stability summaries in the paper-ready analysis.

## Recent Behavioral Additions

The post-static2000 behavioral session turned the V2 plan into concrete infrastructure:

- Added `docs/v2_static2000_results_note.md` as the compact behavioral-baseline interpretation note.
- Added `occlusion` saliency under `src/hma/saliency/occlusion.py`.
- Registered `occlusion` as `saliency_family: perturbation`.
- Added pilot-only V2 occlusion configs:
  - `configs/experiments/real_matrix_v2/salicon_pilot500__resnet50_occlusion.yaml`
  - `configs/experiments/real_matrix_v2/cat2000_pilot500__resnet50_occlusion.yaml`
  - `configs/experiments/real_matrix_v2/coco_search18_pilot500__resnet50_occlusion.yaml`
- Added DeepGaze/reference config generation:
  - `scripts/create_deepgaze_reference_configs.py`
  - generated configs under `configs/experiments/real_matrix_v2_references/`
- Added DeepGaze map export:
  - `scripts/export_deepgaze_maps.py`
  - supports `--manifest`, `--image-root`, `--output-dir`, `--centerbias`, `--device`, `--max-items`, `--dry-run`, `--overwrite`, and `--save-log-density`
  - exported all V2 static DeepGaze maps under `data/precomputed/deepgaze/`
- Completed DeepGaze/reference static2000 benchmark runs:
  - `outputs/real_matrix_v2/salicon_static2000/deepgaze_reference_deepgaze_precomputed/`
  - `outputs/real_matrix_v2/cat2000_static2000/deepgaze_reference_deepgaze_precomputed/`
  - `outputs/real_matrix_v2/coco_search18_static2000/deepgaze_reference_deepgaze_precomputed/`
- Completed pilot occlusion benchmark runs:
  - `outputs/real_matrix_v2/salicon_pilot500/resnet50_occlusion/`
  - `outputs/real_matrix_v2/cat2000_pilot500/resnet50_occlusion/`
  - `outputs/real_matrix_v2/coco_search18_pilot500/resnet50_occlusion/`
- Added improved summary outputs:
  - `outputs/real_matrix_v2/aggregated/results_summary/key_comparisons.csv`
  - `outputs/real_matrix_v2/aggregated/results_summary/pilot_static_stability.csv`
- Added neural-alignment bootstrap:
  - `src/hma/experiments/neural_alignment.py`
  - `scripts/run_neural_alignment.py`
  - `configs/experiments/neural_smoke_dummy.yaml`
  - smoke outputs under `outputs/neural_smoke_dummy/`
- Expanded tests from 111 to 121 passing tests.

The current codebase now covers the proposal's first layer, behavioral saliency, and has a working bridge into the second layer, neural encoding.

## Latest Behavioral Benchmark Status

The current aggregate table is:

```text
outputs/real_matrix_v2/aggregated/results.csv
```

It now contains 588 aggregate rows:

- `baseline`: 90 rows
- `evidence_sensitivity`: 273 rows
- `class_localization`: 90 rows
- `internal_routing`: 90 rows
- `reference`: 21 rows
- `perturbation`: 24 rows

DeepGaze/reference static2000 results:

- SALICON: NSS 0.43481094856746494, shuffled AUC 0.8252894349874732, CC 0.8018529208600521, KL 0.3562953563779592.
- CAT2000: NSS 0.2749275257792324, shuffled AUC 0.7380476504637372, CC 0.3711839377101278, KL 1.2703175959587096.
- COCO-Search18: NSS 0.5179860137457727, shuffled AUC 0.7382109927269479, CC 0.2832240072847344, KL 1.9249819242060184.

Interpretation:

- DeepGaze is now the strongest non-baseline static row on SALICON for NSS, CC, SIM/KL-style density alignment, and shuffled AUC.
- DeepGaze does not beat `resnet50 + gradcam` on CAT2000 or COCO-Search18 NSS.
- Center bias still remains the main static control and should stay in every headline table.

Pilot occlusion results:

- SALICON: NSS 0.20956744656599768, shuffled AUC 0.6578228844311923, CC 0.19740435802773573, KL 4.163194729328156.
- CAT2000: NSS 0.22179918536139304, shuffled AUC 0.6600552580881113, CC 0.19677008081242092, KL 4.207855820894241.
- COCO-Search18: NSS 0.2771203769161366, shuffled AUC 0.6536168335472635, CC 0.1059631172362715, KL 5.544543475866318.

Interpretation:

- Occlusion is valid and scientifically useful as a perturbation family.
- Occlusion beats ResNet gradients and Grad-CAM on shuffled AUC in all three pilot datasets.
- Occlusion does not beat Grad-CAM on NSS or CC and has weak KL, so it should not be scaled to static2000 yet.

## Proposal-Aligned Roadmap From Here

The proposal's main scientific direction is not just to rank saliency maps. It asks when behavioral fixation alignment, neural predictivity, representational geometry, and computational efficiency agree or dissociate. The implementation sequence should preserve that structure: finish the behavioral layer as a publishable baseline, add a reference upper-bound, validate perturbation methods, then move into real neural data.

The latest research review documents sharpen this into a concrete engineering rule: behavioral saliency should now be treated as one axis in a multi-level alignment matrix, not as the main endpoint. The near-term implementation should therefore prioritize comparable neural ROI summaries, scaled encoding/RSA runs, and a first behavior-neural join before adding more model families.

### Step 1: Freeze And Analyze V2 Static2000

Status: implemented enough for analysis.

Use `outputs/real_matrix_v2/aggregated/results.csv` as the current behavioral baseline. Do not rerun the full `static2000` matrix unless a code/config bug invalidates the outputs.

Immediate analysis tasks:

- Read `docs/v2_static2000_results_note.md`.
- Inspect `outputs/real_matrix_v2/aggregated/results_summary/key_comparisons.csv`.
- Inspect `outputs/real_matrix_v2/aggregated/results_summary/pilot_static_stability.csv`.
- Convert the main findings into a paper-style table and figure shortlist.

Interpretation defaults:

- Main static claims should use `static2000`, not pilot rows.
- Pilot rows are reliability checks and development baselines.
- EMD remains pilot-only unless a later paper table specifically needs a slow static EMD pass.
- Center bias is not a nuisance result; it is a central control and should be reported explicitly.

Acceptance criteria:

- The written analysis names center bias as the strongest NSS/CC/SIM-style static baseline.
- The best non-baseline static row is reported per dataset.
- Shuffled-AUC and KL caveats are reported separately from NSS.
- Pilot/static ranking stability is used before making method-family claims.

### Step 2: DeepGaze Or Equivalent Reference Maps

Status: complete for V2 static2000.

The proposal calls for saliency-prediction upper/reference baselines. The repo should use precomputed maps first rather than adding package inference into the benchmark loop.

Expected local layout:

```text
data/precomputed/deepgaze/salicon_static2000/<image_id>.npy
data/precomputed/deepgaze/cat2000_static2000/<image_id>.npy
data/precomputed/deepgaze/coco_search18_static2000/<image_id>.npy
```

The export and benchmark path is retained for reproducibility:

```powershell
.\.venv\Scripts\python.exe scripts/create_deepgaze_reference_configs.py --precomputed-root data/precomputed/deepgaze
```

Acceptance criteria:

- `deepgaze_reference + deepgaze_precomputed` appears as `saliency_family: reference`.
- Reference rows are compared against center bias, Grad-CAM, attention rollout, and observer controls.
- The report clearly separates learned saliency predictors from model-explanation methods.

### Step 3: Pilot Occlusion / Perturbation Saliency

Status: pilot complete; do not scale yet.

The proposal emphasizes that saliency methods can disagree. Occlusion is the first perturbation method because it is easier to validate than RISE and tests causal evidence sensitivity more directly than gradients.

Acceptance criteria:

- Occlusion writes valid `per_image_metrics.csv`, `aggregate_metrics.json`, saliency cache files, and pilot visualizations.
- Runtime is recorded before any static2000 occlusion config is added.
- Occlusion is compared against `resnet50 + vanilla_gradient` and `resnet50 + gradcam`, not mixed into a generic model ranking.

Current decision:

- Keep occlusion as a pilot-only perturbation family.
- Do not add static2000 occlusion configs now.
- Revisit static occlusion only if a later paper section needs a perturbation-specific ablation.

### Step 4: Move Neural Alignment From Smoke Test To Real Manifest

Status: real-data-ready neural runner infrastructure is implemented, the first `all_lh_512` smoke run has completed, and true PRF visual ROI smoke runs now complete for V1, V2, V3, and hV4. These are still 64-image validation runs, not final neural-alignment claims.

The proposal's second layer asks whether models that match fixations also predict visual-cortex responses. This should be implemented as a parallel neural benchmark, not forced onto SALICON/CAT2000 image IDs.

Completed neural smoke path:

- Local Algonauts data is available under `data/raw/nsd_algonauts/subj01/`.
- `scripts/create_algonauts_manifest.py` generated `data/manifests/nsd_algonauts_manifest.csv`.
- `scripts/validate_neural_manifest.py` validated the 64-item `subj01/all_lh_512` subset.
- `configs/experiments/neural_nsd_algonauts_smoke.yaml` runs ResNet-50 layers `layer1` through `layer4`.
- ROI-mask mode in `scripts/create_algonauts_manifest.py` generated `data/manifests/nsd_algonauts_prf_visualrois_manifest.csv`.
- ROI smoke configs now run bilateral V1, V2, V3, and hV4 subsets.

Run command:

```powershell
.\.venv\Scripts\python.exe scripts/run_neural_alignment.py --config configs/experiments/neural_nsd_algonauts_smoke.yaml
```

Acceptance criteria:

- Real neural run writes `activations.npz`, `encoding_scores.csv`, optional `rsa_scores.csv`, and `metadata.json`.
- Scores are reported by model, layer, ROI, subject, and metric.
- The output can later be joined to saliency summaries by model family, training objective, and architecture class.

Next implementation target:

- Add a compact neural ROI summary table that combines the four `encoding_scores.csv` and `rsa_scores.csv` outputs.
- Scale the true ROI runs beyond 64 images once runtime and disk use are acceptable.
- Join neural ROI summaries to the frozen behavioral benchmark at the model / layer / method-family level where possible.
- Keep one subject until ROI handling and reporting are stable.

### Step 5: Add RSA / Representational Geometry

Status: RSA utilities are now connected to the neural runner behind `neural.rsa.enabled`.

The proposal explicitly asks whether fixation-map similarity and feature-space similarity agree. The next neural-layer extension should compute model RDMs and compare them with brain or ROI-response RDMs.

Implementation target:

- Run the neural runner with `neural.rsa.enabled: true`.
- Compute layer-wise model RDMs from reduced activations.
- Compute ROI-response RDMs from the selected response matrix.
- Save `rsa_scores.csv` with dataset, model, subject, ROI, layer, RDM metrics, comparison method, and score.

Acceptance criteria:

- RSA runs on the same real neural manifest as the encoding smoke.
- RSA scores can be compared against encoding scores and saliency results.
- The report can identify behaviorally aligned but neurally weak models, or the reverse.

## Latest Neural Bootstrap Update

Implemented after the V2 behavioral freeze:

- Extended `NSDAlgonautsDataset` with config-level `subject_id` and `roi` filters while preserving `split`, `max_items`, and `validate_files`.
- Added `scripts/validate_neural_manifest.py` to validate NSD / Algonauts-style manifests, image and response paths, split / subject / ROI coverage, and response-vector shape consistency.
- Upgraded `TimmModelWrapper.get_features()` so `neural.layers` can name dotted `timm` modules captured with forward hooks; `embedding` and omitted layers still use the existing `forward_features` path.
- Added neural feature reduction with `feature_reduction: flatten | spatial_mean`; `flatten` remains the backward-compatible default.
- Extended `scripts/run_neural_alignment.py` / `src/hma/experiments/neural_alignment.py` to write subject/ROI-aware encoding rows and optional `rsa_scores.csv`.
- Added example real-data config:
  - `configs/experiments/neural_nsd_algonauts_smoke.yaml`
- Added tests for dataset filtering, manifest validation, neural RSA output, and timm hook extraction.

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `128 passed, 4 warnings`.

Local real neural-data check:

- `data/raw/nsd_algonauts/subj01`: present with 9,841 training images, LH fMRI shape `(9841, 19004)`, RH fMRI shape `(9841, 20544)`, test images, and ROI masks.
- Added `scripts/create_algonauts_manifest.py` to convert one Algonauts subject into this repo's neural manifest format.
- Generated `data/manifests/nsd_algonauts_manifest.csv` with 9,841 `subj01` training rows for `roi: all_lh_512`.
- Generated per-image response vectors under `data/raw/nsd_algonauts/subj01/responses/all_lh_512/`.
- Validated a 64-row subset:

```powershell
.\.venv\Scripts\python.exe scripts\validate_neural_manifest.py --manifest data\manifests\nsd_algonauts_manifest.csv --root data\raw\nsd_algonauts --split train --subject-id subj01 --roi all_lh_512 --max-items 64
```

Validation result: 64 selected rows, response shape `[512]`.

- Completed the first real neural smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config configs\experiments\neural_nsd_algonauts_smoke.yaml
```

Outputs:

- `outputs/neural_nsd_algonauts_smoke/activations.npz`
- `outputs/neural_nsd_algonauts_smoke/encoding_scores.csv`
- `outputs/neural_nsd_algonauts_smoke/rsa_scores.csv`
- `outputs/neural_nsd_algonauts_smoke/metadata.json`

Encoding smoke scores:

- `layer1`: mean correlation `0.08640751242637634`
- `layer2`: mean correlation `0.07636762410402298`
- `layer3`: mean correlation `0.06536547839641571`
- `layer4`: mean correlation `0.06306871771812439`

RSA smoke scores:

- `layer1`: Spearman RDM score `0.11148366323330737`
- `layer2`: Spearman RDM score `0.10216027262303269`
- `layer3`: Spearman RDM score `0.10045888454575899`
- `layer4`: Spearman RDM score `0.11689677462885366`

Smoke interpretation caveat: this is a pipeline validation run over `max_items: 64` and `all_lh_512`, not a scientific neural-alignment claim.

True PRF visual ROI smoke update:

- Extended `scripts/create_algonauts_manifest.py` with ROI-mask mode:

```powershell
.\.venv\Scripts\python.exe scripts\create_algonauts_manifest.py --root data\raw\nsd_algonauts --subject subj01 --roi-class prf-visualrois --roi-names V1 V2 V3 hV4 --hemispheres lh rh --combine-hemispheres --output-manifest data\manifests\nsd_algonauts_prf_visualrois_manifest.csv --max-items 64
```

- Generated `data/manifests/nsd_algonauts_prf_visualrois_manifest.csv` with 256 rows.
- Bilateral response dimensions:
  - V1: 2,973 targets.
  - V2: 2,936 targets.
  - V3: 2,453 targets.
  - hV4: 1,296 targets.
- Validated 64 rows for each ROI with `scripts/validate_neural_manifest.py`.
- Completed neural encoding + RSA smoke runs for:
  - `outputs/neural_nsd_algonauts_v1_smoke/`
  - `outputs/neural_nsd_algonauts_v2_smoke/`
  - `outputs/neural_nsd_algonauts_v3_smoke/`
  - `outputs/neural_nsd_algonauts_hv4_smoke/`

Best mean encoding correlation by ROI in the 64-image smoke runs:

- V1: `layer1`, `0.008290339261293411`.
- V2: `layer3`, `0.10318402945995331`.
- V3: `layer2`, `0.0902683362364769`.
- hV4: `layer1`, `0.1293710321187973`.

Best RSA score by ROI in the 64-image smoke runs:

- V1: `layer1`, `0.05994080593813508`.
- V2: `layer1`, `0.04640057766107312`.
- V3: `layer1`, `0.06178576422750938`.
- hV4: `layer1`, `0.05507270970848019`.

ROI interpretation caveat: these results use only 64 images and should be treated as an integration check plus early direction, not as a stable hierarchy claim.

## Current Session ROI Progress

This session completed the transition from arbitrary neural smoke responses to true PRF visual ROI smoke runs:

- Extended `scripts/create_algonauts_manifest.py` with ROI-mask export mode while preserving the earlier `all_lh_512` mode.
- Added CLI support for:
  - `--roi-class prf-visualrois`
  - `--roi-names V1 V2 V3 hV4`
  - `--hemispheres lh rh`
  - `--combine-hemispheres`
- Added bilateral ROI response generation:
  - V1 combines LH/RH `V1v` and `V1d`.
  - V2 combines LH/RH `V2v` and `V2d`.
  - V3 combines LH/RH `V3v` and `V3d`.
  - hV4 combines LH/RH `hV4`.
- Added ROI smoke configs:
  - `configs/experiments/neural_nsd_algonauts_v1_smoke.yaml`
  - `configs/experiments/neural_nsd_algonauts_v2_smoke.yaml`
  - `configs/experiments/neural_nsd_algonauts_v3_smoke.yaml`
  - `configs/experiments/neural_nsd_algonauts_hv4_smoke.yaml`
- Added tests for ROI-label resolution and bilateral response-vector concatenation.

Generated manifest and response files:

- `data/manifests/nsd_algonauts_prf_visualrois_manifest.csv`, 256 rows.
- `data/raw/nsd_algonauts/subj01/responses/V1/`, 64 files, response dimension 2,973.
- `data/raw/nsd_algonauts/subj01/responses/V2/`, 64 files, response dimension 2,936.
- `data/raw/nsd_algonauts/subj01/responses/V3/`, 64 files, response dimension 2,453.
- `data/raw/nsd_algonauts/subj01/responses/hV4/`, 64 files, response dimension 1,296.

Completed ROI smoke outputs:

- `outputs/neural_nsd_algonauts_v1_smoke/`
- `outputs/neural_nsd_algonauts_v2_smoke/`
- `outputs/neural_nsd_algonauts_v3_smoke/`
- `outputs/neural_nsd_algonauts_hv4_smoke/`

Each directory contains:

- `activations.npz`
- `encoding_scores.csv`
- `rsa_scores.csv`
- `metadata.json`

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `128 passed, 4 warnings`.

Interpretation boundary:

- These are real ROI-based runs, but still only 64-image smoke runs.
- They validate mask parsing, response extraction, named-layer activations, encoding, and RSA.
- They should not yet be used as stable claims about cortical hierarchy or model ranking.

## Latest ROI500 Neural Summary Update

Implemented after the true ROI smoke milestone:

- Added reusable neural ROI summary tooling:
  - `src/hma/experiments/summarize_neural_roi_results.py`
  - `scripts/summarize_neural_roi_results.py`
- Added ROI500 config generation:
  - `scripts/create_neural_roi500_configs.py`
- Added tests for:
  - combined encoding/RSA summary outputs
  - best-layer selection
  - behavior-neural bridge generation
  - missing-RSA tolerance
  - ROI500 config defaults

Verification:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_validate_neural_manifest.py tests\test_create_algonauts_manifest.py
.\.venv\Scripts\python.exe -m pytest
```

Result: `131 passed, 4 warnings`.

ROI500 manifest and config generation:

```cmd
.\.venv\Scripts\python.exe scripts\create_algonauts_manifest.py --root data\raw\nsd_algonauts --subject subj01 --roi-class prf-visualrois --roi-names V1 V2 V3 hV4 --hemispheres lh rh --combine-hemispheres --output-manifest data\manifests\nsd_algonauts_prf_visualrois_500_manifest.csv --max-items 500
.\.venv\Scripts\python.exe scripts\create_neural_roi500_configs.py
```

Generated:

- `data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv`, 2,000 rows.
- `configs/experiments/neural_nsd_algonauts_v1_500.yaml`
- `configs/experiments/neural_nsd_algonauts_v2_500.yaml`
- `configs/experiments/neural_nsd_algonauts_v3_500.yaml`
- `configs/experiments/neural_nsd_algonauts_hv4_500.yaml`

Validation confirmed 500 selected rows for each ROI:

- V1: response dimension 2,973.
- V2: response dimension 2,936.
- V3: response dimension 2,453.
- hV4: response dimension 1,296.

Completed ROI500 outputs:

- `outputs/neural_nsd_algonauts_v1_500/`
- `outputs/neural_nsd_algonauts_v2_500/`
- `outputs/neural_nsd_algonauts_v3_500/`
- `outputs/neural_nsd_algonauts_hv4_500/`

Each directory contains:

- `activations.npz`
- `encoding_scores.csv`
- `rsa_scores.csv`
- `metadata.json`

Summary command:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_neural_roi_results.py --input-dirs outputs\neural_nsd_algonauts_v1_500 outputs\neural_nsd_algonauts_v2_500 outputs\neural_nsd_algonauts_v3_500 outputs\neural_nsd_algonauts_hv4_500 --output-dir outputs\neural_roi_summary --behavioral-csv outputs\real_matrix_v2\aggregated\results.csv
```

Summary outputs:

- `outputs/neural_roi_summary/combined_encoding_scores.csv`
- `outputs/neural_roi_summary/combined_rsa_scores.csv`
- `outputs/neural_roi_summary/best_layers_by_roi.csv`
- `outputs/neural_roi_summary/behavior_neural_bridge.csv`
- `outputs/neural_roi_summary/neural_roi_summary.md`

Best ROI500 encoding layer by ROI:

- V1: `layer1`, mean correlation `0.20192262530326843`.
- V2: `layer1`, mean correlation `0.16762115061283112`.
- V3: `layer1`, mean correlation `0.13866020739078522`.
- hV4: `layer1`, mean correlation `0.16209737956523895`.

Best ROI500 RSA layer by ROI:

- V1: `layer2`, Spearman RDM score `0.06975722561670442`.
- V2: `layer2`, Spearman RDM score `0.0782044194972823`.
- V3: `layer3`, Spearman RDM score `0.06606586984155853`.
- hV4: `layer3`, Spearman RDM score `0.08617366357734918`.

Smoke-to-ROI500 pattern:

- Encoding partially changes with scale: V1 and hV4 still select `layer1`, while V2 changes from `layer3` to `layer1` and V3 changes from `layer2` to `layer1`.
- RSA changes more clearly with scale: the 64-image smoke runs selected `layer1` for all four ROIs, while ROI500 selects `layer2` for V1/V2 and `layer3` for V3/hV4.
- The bridge table contains 168 descriptive rows linking frozen static2000 ResNet-50 `gradcam` and `vanilla_gradient` behavioral rows to the best ROI500 neural encoding/RSA rows.

Interpretation boundary:

- ROI500 is a stronger integration check than the 64-image smoke runs, but it is still one subject and one architecture.
- The behavior-neural bridge supports side-by-side reporting for ResNet-50, not cross-model correlation claims.
- Do not add DINO/CLIP, Brain-Score, CKA, token pruning, foveation, or video until multi-model neural outputs are available.

## Latest Multi-Model Neural Expansion Update

Implemented after the ResNet-50 ROI500 anchor:

- Generalized ROI500 neural config generation in `scripts/create_neural_roi500_configs.py`.
- Added model-specific layer sets for:
  - `resnet50`: `layer1`, `layer2`, `layer3`, `layer4`.
  - `convnext_tiny`: `stages.0`, `stages.1`, `stages.2`, `stages.3`.
  - `deit_small_patch16_224`: `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, `blocks.11`.
  - `vit_base_patch16_224`: `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, `blocks.11`.
- Added V1 debug config generation under `configs/experiments/neural_roi500_debug/`.
- Extended `summarize_neural_roi_results` so behavior-neural bridge rows are no longer hardcoded to ResNet-50.
- Added multi-model summary outputs:
  - `outputs/neural_roi_summary/best_encoding_by_model_roi.csv`
  - `outputs/neural_roi_summary/best_rsa_by_model_roi.csv`
  - `outputs/neural_roi_summary/behavior_neural_model_summary.csv`
  - `outputs/neural_roi_summary/alignment_per_efficiency.csv`
- Extended tests for multi-model config generation, model-specific layer lists, output naming, multi-model bridge rows, `attention_rollout`, and efficiency joins.

Verification:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_model_wrappers.py
.\.venv\Scripts\python.exe -m pytest
```

Result: `132 passed, 4 warnings`.

Generated multi-model configs:

- Full ROI500 configs: `configs/experiments/neural_roi500/`, 16 configs across 4 models and 4 ROIs.
- Debug configs: `configs/experiments/neural_roi500_debug/`, V1-only configs for `convnext_tiny`, `deit_small_patch16_224`, and `vit_base_patch16_224`.

Debug validation:

```cmd
.\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config configs\experiments\neural_roi500_debug\convnext_tiny_v1_debug.yaml
.\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config configs\experiments\neural_roi500_debug\deit_small_patch16_224_v1_debug.yaml
.\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config configs\experiments\neural_roi500_debug\vit_base_patch16_224_v1_debug.yaml
```

All three debug runs completed with pretrained weights and valid named-layer hooks.

Completed new ROI500 outputs:

- `outputs/neural_roi500/convnext_tiny_v1_500/`
- `outputs/neural_roi500/convnext_tiny_v2_500/`
- `outputs/neural_roi500/convnext_tiny_v3_500/`
- `outputs/neural_roi500/convnext_tiny_hv4_500/`
- `outputs/neural_roi500/deit_small_patch16_224_v1_500/`
- `outputs/neural_roi500/deit_small_patch16_224_v2_500/`
- `outputs/neural_roi500/deit_small_patch16_224_v3_500/`
- `outputs/neural_roi500/deit_small_patch16_224_hv4_500/`
- `outputs/neural_roi500/vit_base_patch16_224_v1_500/`
- `outputs/neural_roi500/vit_base_patch16_224_v2_500/`
- `outputs/neural_roi500/vit_base_patch16_224_v3_500/`
- `outputs/neural_roi500/vit_base_patch16_224_hv4_500/`

Each directory contains `activations.npz`, `encoding_scores.csv`, `rsa_scores.csv`, and `metadata.json`.

Summary command:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_neural_roi_results.py --input-dirs outputs\neural_nsd_algonauts_v1_500 outputs\neural_nsd_algonauts_v2_500 outputs\neural_nsd_algonauts_v3_500 outputs\neural_nsd_algonauts_hv4_500 outputs\neural_roi500\convnext_tiny_v1_500 outputs\neural_roi500\convnext_tiny_v2_500 outputs\neural_roi500\convnext_tiny_v3_500 outputs\neural_roi500\convnext_tiny_hv4_500 outputs\neural_roi500\deit_small_patch16_224_v1_500 outputs\neural_roi500\deit_small_patch16_224_v2_500 outputs\neural_roi500\deit_small_patch16_224_v3_500 outputs\neural_roi500\deit_small_patch16_224_hv4_500 outputs\neural_roi500\vit_base_patch16_224_v1_500 outputs\neural_roi500\vit_base_patch16_224_v2_500 outputs\neural_roi500\vit_base_patch16_224_v3_500 outputs\neural_roi500\vit_base_patch16_224_hv4_500 --output-dir outputs\neural_roi_summary --behavioral-csv outputs\real_matrix_v2\aggregated\results.csv --efficiency-csv outputs\real_matrix_v2\efficiency\model_efficiency.csv
```

Combined summary:

- Input directories: 16.
- Encoding rows: 72.
- RSA rows: 72.
- Behavior-neural bridge rows: 672.

Best mean encoding score averaged across ROIs:

- `deit_small_patch16_224`: `0.26100467145443`.
- `vit_base_patch16_224`: `0.216953299939632`.
- `convnext_tiny`: `0.186267614364624`.
- `resnet50`: `0.167575340718031`.

Best RSA score averaged across ROIs:

- `vit_base_patch16_224`: `0.0875462059288639`.
- `deit_small_patch16_224`: `0.0796801522326127`.
- `resnet50`: `0.0750502946332236`.
- `convnext_tiny`: `0.064057745554811`.

Efficiency-normalized pattern using mean score per latency:

- Encoding: `deit_small_patch16_224` leads among the current four models.
- RSA: `deit_small_patch16_224` also leads by mean score per latency, while `vit_base_patch16_224` has the strongest raw RSA but weaker latency-normalized RSA.

Initial interpretation:

- The ResNet-50 pattern does not fully generalize across model families.
- DeiT selects early transformer block `blocks.0` for best encoding across all ROIs and has the strongest mean encoding score.
- ViT-base has the strongest mean RSA score, with best RSA consistently at `blocks.6`.
- ConvNeXt improves over ResNet-50 on raw mean encoding but is weaker on raw mean RSA.
- These results are now multi-model, but still one subject and one ROI500 subset, so they support descriptive convergence/dissociation analysis rather than final claims.

## Latest Behavior-Neural Analysis V1 And SSL Candidate Prep

Implemented after the supervised multi-model ROI500 expansion:

- Extended `summarize_neural_roi_results` with paper-style derived outputs:
  - `outputs/neural_roi_summary/paper_model_roi_winners.csv`
  - `outputs/neural_roi_summary/neural_model_rankings.csv`
  - `outputs/neural_roi_summary/behavior_neural_alignment_summary.csv`
  - `outputs/neural_roi_summary/behavior_neural_leader_overlap.csv`
  - `outputs/neural_roi_summary/multimodel_interpretation_note.md`
- Added SSL/multimodal dry-inspection support to `scripts/create_neural_roi500_configs.py`.
- Generated pretrained-free V1 debug configs under:
  - `configs/experiments/neural_roi500_ssl_candidates_debug/`
- Generated candidate inventory:
  - `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`
- Added tests for paper winner rows, model rankings, lower-is-better bridge leadership, candidate inventory generation, and pretrained-free debug configs.
- Added paper-style inspection pack generation:
  - `scripts/create_paper_inspection_pack.py`
  - output root: `outputs/paper_inspection_v1/`
  - figures under `outputs/paper_inspection_v1/figures/`
  - tables under `outputs/paper_inspection_v1/tables/`
  - inspection README: `outputs/paper_inspection_v1/README.md`

Verification:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_model_wrappers.py
.\.venv\Scripts\python.exe -m pytest
```

Result: `134 passed, 4 warnings`.

Behavior-neural analysis command:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_neural_roi_results.py --input-dirs outputs\neural_nsd_algonauts_v1_500 outputs\neural_nsd_algonauts_v2_500 outputs\neural_nsd_algonauts_v3_500 outputs\neural_nsd_algonauts_hv4_500 outputs\neural_roi500\convnext_tiny_v1_500 outputs\neural_roi500\convnext_tiny_v2_500 outputs\neural_roi500\convnext_tiny_v3_500 outputs\neural_roi500\convnext_tiny_hv4_500 outputs\neural_roi500\deit_small_patch16_224_v1_500 outputs\neural_roi500\deit_small_patch16_224_v2_500 outputs\neural_roi500\deit_small_patch16_224_v3_500 outputs\neural_roi500\deit_small_patch16_224_hv4_500 outputs\neural_roi500\vit_base_patch16_224_v1_500 outputs\neural_roi500\vit_base_patch16_224_v2_500 outputs\neural_roi500\vit_base_patch16_224_v3_500 outputs\neural_roi500\vit_base_patch16_224_hv4_500 --output-dir outputs\neural_roi_summary --behavioral-csv outputs\real_matrix_v2\aggregated\results.csv --efficiency-csv outputs\real_matrix_v2\efficiency\model_efficiency.csv
```

SSL/multimodal candidate dry-inspection command:

```cmd
.\.venv\Scripts\python.exe scripts\create_neural_roi500_configs.py --inspect-ssl-candidates --write-ssl-debug-configs --candidate-output outputs\neural_roi_summary\ssl_multimodal_candidate_inventory.csv
```

Paper-style inspection pack command:

```cmd
.\.venv\Scripts\python.exe scripts\create_paper_inspection_pack.py
```

Generated inspection figures:

- `outputs/paper_inspection_v1/figures/figure1_behavior_static2000_nss.png`
- `outputs/paper_inspection_v1/figures/figure2_neural_model_rankings.png`
- `outputs/paper_inspection_v1/figures/figure3_roi_heatmaps.png`
- `outputs/paper_inspection_v1/figures/figure4_behavior_neural_leader_overlap.png`

Generated inspection tables:

- `outputs/paper_inspection_v1/tables/table1_behavior_static2000_nss_top.md`
- `outputs/paper_inspection_v1/tables/table2_neural_model_rankings.md`
- `outputs/paper_inspection_v1/tables/table3_roi_winners.md`
- `outputs/paper_inspection_v1/tables/table4_behavior_neural_overlap_summary.md`
- `outputs/paper_inspection_v1/tables/table5_ssl_multimodal_candidates.md`

Inspection-pack readout:

- Report entry point: `outputs/paper_inspection_v1/README.md`.
- Top displayed behavioral NSS row: CAT2000 / center bias / center bias, NSS `0.519`.
- Neural ranking figure shows the current raw / efficiency split: `deit_small_patch16_224` leads raw encoding and latency-normalized neural scores, while `vit_base_patch16_224` leads raw RSA.
- ROI heatmaps show DeiT encoding concentrated at `blocks.0`, ViT-base RSA concentrated at `blocks.6`, and ResNet-50 encoding concentrated at `layer1`.
- Leader-overlap figure makes the current bridge caveat visible: internal-routing behavioral leaders line up with the encoding leader, but class-localization and evidence-sensitivity leaders do not; no current behavioral group matches the raw RSA leader.

Headline analysis:

- Raw mean ROI500 encoding leader: `deit_small_patch16_224`, mean encoding `0.2610046714544296`.
- Raw mean ROI500 RSA leader: `vit_base_patch16_224`, mean RSA `0.0875462059288639`.
- Latency-normalized leader for both encoding and RSA: `deit_small_patch16_224`.
- Behavioral-neural leader overlap is partial for encoding and absent for raw RSA in the current grouped bridge: 21/63 behavioral leaders match the raw encoding leader, and 0/63 match the raw RSA leader.
- The bridge remains descriptive: one subject, ROI500 subset, and frozen static2000 behavioral rows.

Dry-inspected SSL/multimodal candidates:

- DINOv2: `vit_small_patch14_dinov2`, `vit_base_patch14_dinov2`.
- DINOv3: `vit_small_patch16_dinov3`, `vit_base_patch16_dinov3`.
- CLIP: `vit_base_patch16_clip_224`, `resnet50_clip`.
- SigLIP: `vit_base_patch16_siglip_224`.
- EVA-CLIP: `eva02_base_patch16_clip_224`.

Candidate feasibility:

- All 8 candidate names are available in local `timm`.
- All 8 have verified hook layers with `pretrained=False`.
- ViT-like candidates use `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`.
- Local `resnet50_clip` exposes `stages.0`, `stages.1`, `stages.2`, and `stages.3`, not plain `layer1` through `layer4`.
- Superseded by the latest SSL / multimodal pretrained milestone below: pretrained V1 debug runs have now completed for `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, and `resnet50_clip`; `vit_small_patch14_dinov2` has also completed full ROI500.

### Step 6: Architecture Expansion

Status: initial supervised multi-model ROI500 expansion, paper-style behavior-neural analysis, pretrained SSL/VLM debug validation, and full DINOv2 ROI500 are complete. The next architecture step should add model-matched behavioral rows for the pretrained SSL/VLM candidates before expanding to additional neural-only families.

Priority order from the proposal:

1. Self-supervised or multimodal encoders such as DINO/DINOv2/CLIP-style vision backbones.
2. Hierarchical or hybrid backbones available through stable wrappers.
3. Selective-computation models: token pruning, foveation, adaptive patch selection, or glimpse-style models.
4. Video models only after static-image claims are paper-ready.

Acceptance criteria:

- Each architecture family has model metadata, saliency compatibility notes, and efficiency rows.
- Results remain grouped by architecture family and saliency family.
- Claims test alignment per computation, not only raw alignment or model size.

## Latest SSL / Multimodal Pretrained Milestone

Status: **SSL / Multimodal Pretrained Debug Runs V1 completed, with DINOv2 promoted to full ROI500.**

Implemented:

- Added reusable SSL/multimodal config generation in `scripts/create_neural_roi500_configs.py`.
  - Existing pretrained-free debug configs remain under `configs/experiments/neural_roi500_ssl_candidates_debug/`.
  - Pretrained V1 debug configs are under `configs/experiments/neural_roi500_ssl_pretrained_debug/`.
  - Full pretrained SSL ROI500 configs are under `configs/experiments/neural_roi500_ssl/`.
- Extended candidate inventory columns:
  - `pretrained_debug_config_path`
  - `pretrained_output_dir`
  - `pretrained_run_status`
  - `pretrained_weight_status`
  - `pretrained_run_error`
- Added pretrained metadata to neural output `metadata.json`:
  - `model_name`
  - `model_backend`
  - `model_pretrained`
- Updated paper-pack reporting so SSL/multimodal pretrained status is computed from the candidate inventory instead of hardcoded.
- Added `scripts/merge_efficiency_profiles.py` and generated:
  - `outputs/neural_roi500_ssl_pretrained_debug/efficiency/model_efficiency.csv`
  - `outputs/neural_roi_summary/model_efficiency_with_ssl.csv`

Weight policy result:

- `vit_small_patch14_dinov2`: pretrained weights were downloaded with explicit approval, then cached locally.
- `vit_base_patch16_clip_224`: pretrained weights were downloaded with explicit approval, then cached locally.
- `resnet50_clip`: pretrained weights were downloaded with explicit approval, then cached locally.
- No run silently fell back to random weights.

Completed pretrained V1 debug outputs:

- `outputs/neural_roi500_ssl_pretrained_debug/vit_small_patch14_dinov2_v1_pretrained_debug/`
- `outputs/neural_roi500_ssl_pretrained_debug/vit_base_patch16_clip_224_v1_pretrained_debug/`
- `outputs/neural_roi500_ssl_pretrained_debug/resnet50_clip_v1_pretrained_debug/`

Each debug output contains:

- `activations.npz`
- `encoding_scores.csv`
- `rsa_scores.csv`
- `metadata.json`

Full DINOv2 ROI500 outputs:

- `outputs/neural_roi500_ssl/vit_small_patch14_dinov2_v1_500/`
- `outputs/neural_roi500_ssl/vit_small_patch14_dinov2_v2_500/`
- `outputs/neural_roi500_ssl/vit_small_patch14_dinov2_v3_500/`
- `outputs/neural_roi500_ssl/vit_small_patch14_dinov2_hv4_500/`

Layer and preprocessing notes:

- `vit_small_patch14_dinov2` uses layers `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`.
- `vit_base_patch16_clip_224` uses layers `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`.
- `resnet50_clip` uses layers `stages.0`, `stages.1`, `stages.2`, and `stages.3`.
- DINOv2 patch-14 configs use `input_size: [518, 518]`; this was required by the local `timm` model and fixed after the first pretrained run exposed the mismatch.
- CLIP ViT and CLIP ResNet configs use `input_size: [224, 224]`.

Regenerated summaries:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_neural_roi_results.py --input-dirs <16 existing supervised ROI500 dirs> <4 DINOv2 SSL ROI500 dirs> --output-dir outputs\neural_roi_summary --behavioral-csv outputs\real_matrix_v2\aggregated\results.csv --efficiency-csv outputs\neural_roi_summary\model_efficiency_with_ssl.csv
.\.venv\Scripts\python.exe scripts\create_paper_inspection_pack.py
```

Updated inspection-pack readout:

- Report entry point: `outputs/paper_inspection_v1/README.md`.
- SSL/multimodal candidates dry-inspected: 8.
- Pretrained debug runs complete: 3.
- Pretrained status counts: 3 `complete`, 5 `not_run`.

Updated ROI500 neural ranking after adding DINOv2:

- Raw mean encoding leader remains `deit_small_patch16_224`, mean encoding `0.2610046714544296`.
- Raw mean RSA leader remains `vit_base_patch16_224`, mean RSA `0.0875462059288639`.
- New DINOv2 row:
  - model: `vit_small_patch14_dinov2`
  - mean encoding: `0.24179347604513168`, rank 2 of 5.
  - mean RSA: `0.08272091669932777`, rank 2 of 5.
  - latency-normalized encoding and RSA ranks are both 5 of 5 under the short debug efficiency profile.
- Latency-normalized leader remains `deit_small_patch16_224` for both encoding and RSA.
- Behavioral-neural leader overlap remains partial for encoding and absent for raw RSA because current frozen static2000 behavioral rows do not include SSL/multimodal saliency rows.

Verification after implementation:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_model_wrappers.py tests\test_paper_inspection_pack.py tests\test_efficiency_metrics.py
```

Result: `31 passed, 1 warning`.

Full verification:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `142 passed, 4 warnings`.

The remaining warnings are the known Grad-CAM backward-hook warning and Windows `.pytest_cache` permission warning.

## SSL / Multimodal Behavioral Bridge Rows V1 Plan And Progress

Execution status as of 2026-05-20: **DINOv2 behavioral bridge completed; broader CLIP/VLM static matrix remains partial.**

Implemented:

- Added SSL/VLM behavioral config generation:
  - `scripts/create_ssl_behavior_v1_configs.py`
  - configs under `configs/experiments/real_matrix_v2_ssl_behavior/`
  - outputs under `outputs/real_matrix_v2_ssl_behavior/`
- Added reusable behavior-aggregate merging:
  - `scripts/merge_behavioral_aggregates.py`
  - merged output: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Extended `scripts/run_v2_matrix.py` so reliability gates can be supplied through CLI / CSV and so the built-in gates include the SSL/VLM pilot checks.
- Regenerated the behavior-neural summaries and paper inspection pack from the merged behavioral CSV.
- Updated the paper inspection README so it records the behavioral source CSV.

Completed execution:

- Generated 36 SSL/VLM behavioral configs.
- Completed all 18 pilot SSL/VLM behavioral rows.
- Completed the SALICON pilot reliability gates for:
  - `vit_small_patch14_dinov2 + attention_rollout`
  - `vit_base_patch16_clip_224 + attention_rollout`
  - `resnet50_clip + gradcam`
- Completed static2000 rows for:
  - `vit_small_patch14_dinov2 + vanilla_gradient` on SALICON, CAT2000, and COCO-Search18.
  - `vit_small_patch14_dinov2 + attention_rollout` on SALICON, CAT2000, and COCO-Search18.
  - `resnet50_clip + gradcam` on CAT2000.
- The full static SSL/VLM matrix was started but timed out after one hour during `cat2000_static2000 :: resnet50_clip + vanilla_gradient`; resume can continue later.

Generated / regenerated outputs:

- SSL behavior aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Merged behavioral aggregate: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- DINOv2 bridge rows are now present in `outputs/neural_roi_summary/behavior_neural_bridge.csv`:
  - 84 rows for `vit_small_patch14_dinov2 + vanilla_gradient`.
  - 84 rows for `vit_small_patch14_dinov2 + attention_rollout`.
- Paper inspection pack regenerated under `outputs/paper_inspection_v1/`.
- `table1_behavior_static2000_nss_top.csv` now includes DINOv2 static NSS rows for all three static datasets:
  - SALICON: DINOv2 ViT-S/14 + Gradient, NSS `0.220`; Attention rollout, NSS `0.133`.
  - CAT2000: DINOv2 ViT-S/14 + Attention rollout, NSS `0.670`; Gradient, NSS `0.250`.
  - COCO-Search18: DINOv2 ViT-S/14 + Attention rollout, NSS `0.547`; Gradient, NSS `0.324`.
  - In the regenerated inspection pack, CAT2000 / DINOv2 attention rollout is the top displayed behavioral NSS row.

Verification:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_saliency_benchmark.py tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py tests\test_summarize_results.py tests\test_ssl_behavior_configs.py tests\test_merge_behavioral_aggregates.py
```

Result: `31 passed, 2 warnings`.

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `148 passed, 4 warnings`.

Remaining work after the DINOv2 bridge pass:

- Resume and complete the remaining static2000 CLIP behavior rows if a broader VLM comparison is needed:
  - `vit_base_patch16_clip_224 + vanilla_gradient`
  - `vit_base_patch16_clip_224 + attention_rollout`
  - `resnet50_clip + vanilla_gradient`
  - remaining `resnet50_clip + gradcam` datasets
- Re-aggregate and regenerate the bridge / inspection pack after any additional CLIP rows finish.
- The current `results_with_ssl_behavior.csv` is now valid for DINOv2 bridge interpretation, but remains partial for CLIP/VLM behavioral claims.

Original target milestone: **SSL / Multimodal Behavioral Bridge Rows V1**.

Why this is next:

- DINOv2 now has full ROI500 neural encoding/RSA outputs, but the frozen static2000 behavioral aggregate does not contain DINOv2 or CLIP saliency rows.
- `summarize_neural_roi_results.py` only creates behavior-neural bridge rows when the behavioral aggregate has matching `model` values and one of the bridge methods: `vanilla_gradient`, `gradcam`, or `attention_rollout`.
- Because of that, the current paper inspection pack can rank DINOv2 neurally but cannot test whether SSL/VLM behavioral saliency leaders overlap with neural leaders.

Goal:

- Add a small, controlled behavioral matrix for the pretrained SSL/VLM candidates that already passed debug.
- Merge those rows with the frozen V2 static2000 behavioral aggregate without disturbing the original `outputs/real_matrix_v2/aggregated/results.csv`.
- Regenerate neural bridge summaries and the paper inspection pack from the merged behavioral aggregate.

Implementation plan:

1. Add SSL/VLM behavioral config generation.
   - New script: `scripts/create_ssl_behavior_v1_configs.py`.
   - Config root: `configs/experiments/real_matrix_v2_ssl_behavior/`.
   - Output root: `outputs/real_matrix_v2_ssl_behavior/`.
   - Reuse V2 static datasets and metrics from `scripts/create_real_matrix_v2_configs.py`.
   - Keep the original core V2 configs untouched.
   - Candidate rows:
     - `vit_small_patch14_dinov2 + vanilla_gradient`
     - `vit_small_patch14_dinov2 + attention_rollout`
     - `vit_base_patch16_clip_224 + vanilla_gradient`
     - `vit_base_patch16_clip_224 + attention_rollout`
     - `resnet50_clip + vanilla_gradient`
     - `resnet50_clip + gradcam`, with `target_layer: stages.3`
   - Preprocessing:
     - `vit_small_patch14_dinov2`: `input_size: [518, 518]`.
     - `vit_base_patch16_clip_224`: `input_size: [224, 224]`.
     - `resnet50_clip`: `input_size: [224, 224]`.

2. Add reliability-gated execution support.
   - Prefer reusing `scripts/run_v2_matrix.py` with:

```cmd
.\.venv\Scripts\python.exe scripts\run_v2_matrix.py --config-dir configs\experiments\real_matrix_v2_ssl_behavior --output-root outputs\real_matrix_v2_ssl_behavior --phase pilot --resume --progress-interval 50
```

   - Pilot reliability checks should run first on `salicon_pilot500` for:
     - `vit_small_patch14_dinov2 + attention_rollout`
     - `vit_base_patch16_clip_224 + attention_rollout`
     - `resnet50_clip + gradcam`
   - If a candidate method fails on pilot reliability, do not run its static2000 configs.
   - If DINOv2 518x518 static runs are too slow, keep DINOv2 behavioral rows to `static2000` plus no visualizations and document runtime; do not silently lower input size.

3. Run the targeted static2000 behavioral matrix.
   - Static datasets:
     - `salicon_static2000`
     - `cat2000_static2000`
     - `coco_search18_static2000`
   - Static metrics:
     - `nss`
     - `shuffled_auc`
     - `auc_borji`
     - `auc_judd`
     - `cc`
     - `similarity`
     - `kl`
   - Keep `emd` pilot-only if pilot configs are generated; do not add static EMD.
   - Use cache reuse:
     - `cache.enabled: true`
     - `cache.dir: saliency_maps`
     - `cache.reuse: true`

4. Merge behavioral aggregates for bridge reporting.
   - Add or use a small merge script such as `scripts/merge_behavioral_aggregates.py`.
   - Inputs:
     - `outputs/real_matrix_v2/aggregated/results.csv`
     - `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`
   - Output:
     - `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
   - Preserve all original columns.
   - De-duplicate by `dataset`, `model`, `saliency_method`, `saliency_family`, and `metric`, with later SSL rows replacing only identical duplicate keys.

5. Regenerate behavior-neural summaries with the merged behavioral CSV.
   - Use the existing neural input dirs:
     - 16 supervised ROI500 dirs.
     - 4 DINOv2 ROI500 dirs.
   - Use the merged efficiency CSV:
     - `outputs/neural_roi_summary/model_efficiency_with_ssl.csv`
   - Summary command should target the same output directory:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_neural_roi_results.py --input-dirs <16 supervised ROI500 dirs> <4 DINOv2 ROI500 dirs> --output-dir outputs\neural_roi_summary --behavioral-csv outputs\real_matrix_v2\aggregated\results_with_ssl_behavior.csv --efficiency-csv outputs\neural_roi_summary\model_efficiency_with_ssl.csv
```

6. Regenerate the paper inspection pack.

```cmd
.\.venv\Scripts\python.exe scripts\create_paper_inspection_pack.py --behavioral-csv outputs\real_matrix_v2\aggregated\results_with_ssl_behavior.csv
```

   - Update the inspection README and tables so they clearly say the behavioral table now uses `results_with_ssl_behavior.csv`.
   - Confirm `table5_ssl_multimodal_candidates` still reports 3 completed pretrained debug runs.
   - Confirm `behavior_neural_bridge.csv` now contains `vit_small_patch14_dinov2` bridge rows.

7. Update this status document after execution.
   - Record successful pilot reliability rows.
   - Record static2000 SSL/VLM behavioral rows completed.
   - Record any failed or skipped methods.
   - Record whether DINOv2 changes behavioral leader overlap with raw encoding or raw RSA leaders.
   - Record verification result.

Acceptance criteria:

- SSL/VLM behavioral configs are generated under `configs/experiments/real_matrix_v2_ssl_behavior/`.
- At least `vit_small_patch14_dinov2` has completed static2000 behavioral rows for one bridge method, preferably both `vanilla_gradient` and `attention_rollout`.
- `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv` exists and contains the original V2 rows plus SSL/VLM rows.
- `outputs/neural_roi_summary/behavior_neural_bridge.csv` includes DINOv2 bridge rows.
- `outputs/paper_inspection_v1/README.md` and tables are regenerated from the merged behavioral CSV.
- Tests pass:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_saliency_benchmark.py tests\test_paper_inspection_pack.py
.\.venv\Scripts\python.exe -m pytest
```

Non-goals:

- Do not add new neural families in this milestone.
- Do not run full ROI500 for CLIP/SigLIP until the behavioral bridge gap is closed for DINOv2.
- Do not start selective-computation models, video models, Brain-Score, CKA, or saliency-guided training.
