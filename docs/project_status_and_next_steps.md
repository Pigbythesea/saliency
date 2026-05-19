# HMA Project Status And Next Steps

Date: 2026-05-18

Latest implementation update: 2026-05-19

## Current Status

The repository now has a first real behavioral-saliency comparison result set for Phase 1 of the Human-Machine Visual Alignment project and a stronger V2 benchmark scaffold. It can load real SALICON, CAT2000, and COCO-Search18 manifests, run baseline and pretrained `timm` model saliency methods, cache saliency maps, aggregate metrics, plot rankings, profile model efficiency, parse observer-level fixation files for SALICON/CAT2000, and produce controlled V2 summaries.

Latest verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: `121 passed, 4 warnings`.

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

- Real fMRI activation extraction over local NSD / Algonauts manifests.
- RSA / representational-geometry experiment reporting over real neural data.
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

## Current Session Additions

This session turned the post-static2000 plan into concrete infrastructure:

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

The current codebase now covers the proposal's first layer, behavioral saliency, and has a minimal bridge into the second layer, neural encoding.

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

Status: dummy neural smoke runner works; real NSD / Algonauts manifest wiring is next.

The proposal's second layer asks whether models that match fixations also predict visual-cortex responses. This should be implemented as a parallel neural benchmark, not forced onto SALICON/CAT2000 image IDs.

Next implementation target:

- Create or validate a local NSD / Algonauts manifest with:
  - `image_id`
  - `image_path`
  - `split`
  - `subject_id`
  - `roi`
  - `roi_responses` or `roi_response_path`
- Add one real neural config under `configs/experiments/`.
- Run:

```powershell
.\.venv\Scripts\python.exe scripts/run_neural_alignment.py --config configs/experiments/<real_neural_config>.yaml
```

Acceptance criteria:

- Real neural run writes `activations.npz`, `encoding_scores.csv`, and `metadata.json`.
- Scores are reported by model, layer, ROI, subject, and metric.
- The output can later be joined to saliency summaries by model family, training objective, and architecture class.

### Step 5: Add RSA / Representational Geometry

Status: RSA utilities exist; experiment-level reporting is not yet connected.

The proposal explicitly asks whether fixation-map similarity and feature-space similarity agree. The next neural-layer extension should compute model RDMs and compare them with brain or ROI-response RDMs.

Implementation target:

- Extend the neural runner or add a companion script that loads `activations.npz`.
- Compute layer-wise model RDMs.
- Compute ROI-response RDMs.
- Save `rsa_scores.csv` with layer, ROI, metric, and score.

Acceptance criteria:

- RSA runs on the same real neural manifest as the encoding smoke.
- RSA scores can be compared against encoding scores and saliency results.
- The report can identify behaviorally aligned but neurally weak models, or the reverse.

### Step 6: Architecture Expansion

Status: defer until the baseline, reference, perturbation, and first real neural pass are stable.

Priority order from the proposal:

1. Self-supervised or multimodal encoders such as DINO/DINOv2/CLIP-style vision backbones.
2. Hierarchical or hybrid backbones available through stable wrappers.
3. Selective-computation models: token pruning, foveation, adaptive patch selection, or glimpse-style models.
4. Video models only after static-image claims are paper-ready.

Acceptance criteria:

- Each architecture family has model metadata, saliency compatibility notes, and efficiency rows.
- Results remain grouped by architecture family and saliency family.
- Claims test alignment per computation, not only raw alignment or model size.

## What To Do Next

Recommended next action:

1. Build the first real NSD / Algonauts manifest for `scripts/run_neural_alignment.py`.
2. Run one real neural encoding experiment and verify `activations.npz`, `encoding_scores.csv`, and `metadata.json`.
3. Add RSA / representational-geometry reporting over the same real neural manifest.
4. Write a compact behavioral benchmark result table from the current `results.csv`.
5. Only after neural smoke + RSA are stable, add the next architecture family, starting with self-supervised or multimodal encoders.

Do not start selective-computation models, video models, or saliency-guided training yet. The current project now has its reference saliency baseline; it needs one real neural-alignment run before those larger proposal branches will be scientifically interpretable.
