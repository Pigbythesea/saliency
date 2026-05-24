# HMA Project Status And Next Steps

Updated: 2026-05-24

## Purpose

This document is the current handoff for the Human-Machine Visual Alignment project. It should answer:

- what is implemented now,
- which outputs are scientifically usable,
- which reference notes are superseded,
- where the relevant code and artifacts live,
- what the next concrete implementation milestone is.

It is not an experiment diary. Completed rerun logs and stale pilot interpretations have been collapsed into the current state so the next action is easy to find.

## Reference Documents Reviewed

Current steering documents under `docs/`:

- `project_status_and_next_steps.md`: this engineering status file.
- `Literature Review and Research Redesign for the Human-Like Adaptive Visual Attention Project.md`: argues the project should become a multi-axis NeuroAI alignment study, not a saliency-map leaderboard.
- `Deep Research Assessment of the Human-Machine Visual Alignment Project.md`: emphasizes the publishable question as convergence versus dissociation among fixation alignment, neural predictivity, representational geometry, and efficiency.
- `Zhang_Zihuan_zzhan330_proposal.docx`: original proposal; defines behavioral saliency, neural encoding, RSA, Brain-Score-style comparison, and compute efficiency as the core axes.
- `Comparing Human and Machine Visual Saliency_ A Comprehensive Review.pdf`: reinforces that fixation prediction requires strong controls such as center bias, DeepGaze-class references, point-based NSS/AUC, and separate treatment of free-viewing versus task-driven viewing.
- `__Attention and Saliency Map Extraction in Visual AI Models_ A Comprehensive Review__.pdf`: reinforces that gradients, CAMs, attention rollout, perturbation maps, LRP-style methods, and transformer attribution are different explanation objects and should not be collapsed into one "attention" score.
- `v2_static2000_results_note.md`: superseded by corrected point-fixation reruns. Keep it only as historical context for why the protocol fix was needed.

## Current Snapshot

The repository now implements three active layers:

- Behavioral saliency / fixation benchmarking on SALICON, CAT2000, and COCO-Search18.
- Neural encoding/RSA diagnostics on local Algonauts / NSD `subj01` visual ROIs, including ROI500 spatial-mean diagnostics and full-image-count `flatten_pca` PRF ROI baselines.
- Paper-style inspection tables and figures that join corrected behavioral summaries with the current mixed-scope neural summaries.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs:

- Corrected core behavioral aggregate: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Corrected SSL/VLM behavioral aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Neural ROI summary with full `flatten_pca` rows included: `outputs/neural_roi_summary/`.
- Paper inspection pack regenerated from the refreshed neural summary: `outputs/paper_inspection_v1/README.md`.

## Scientific Boundary

The corrected behavioral layer is now usable for diagnostic paper-style analysis. It should still be framed carefully:

- NSS and AUC-style claims are valid only for rows with `fixation_protocol=points` or `fixation_protocol=task_points`.
- CC, SIM, KL, and related map-distribution metrics should be discussed separately from point-fixation metrics.
- DeepGaze and center bias are reference controls. Grad-CAM, gradients, rollout, and similar rows are explanation-map-to-fixation comparisons, not dedicated SOTA fixation-prediction models.
- COCO-Search18 is task-driven search and should not be pooled with free-viewing SALICON/CAT2000 as if all three datasets measure the same behavior.

The neural layer is now a stronger local baseline, but still not a leaderboard result:

- Current neural outputs are one-subject, internal-split `subj01` results. They combine older ROI500 spatial-mean diagnostic rows with full-image-count PRF visual ROI `flatten_pca` rows for `V1`, `V2`, `V3`, and `hV4`.
- They are not Algonauts leaderboard-equivalent scores because the official challenge averages held-out visual-cortex vertices across subjects and hemispheres.
- The full `flatten_pca` rows can be used to evaluate the local strong-ridge baseline and compare methods inside this repo, but should only be compared with Algonauts/SOTA ranges as context, not as equivalent evaluation.
- The bridge tables are descriptive joins, not causal tests or robust cross-model correlations.

## Current Behavioral Status

Corrected merged behavioral aggregate:

- Path: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Rows: `378`
- Dataset rows: `126` each for SALICON, CAT2000, and COCO-Search18
- Protocol rows: `252` with `points`, `126` with `task_points`
- Blank / `unknown` / `density_fallback` protocol rows: none

Corrected NSS headline:

- SALICON: DeepGaze `1.743`, center bias `0.933`, DINOv2 ViT-S/14 gradient `0.736`, ConvNeXt-T Grad-CAM `0.633`, ResNet-50 Grad-CAM `0.598`.
- CAT2000: DeepGaze `1.838`, center bias `1.619`, ResNet-50 Grad-CAM `0.882`, DINOv2 ViT-S/14 gradient `0.810`, ConvNeXt-T Grad-CAM `0.759`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, ResNet-50 Grad-CAM `0.955`, ConvNeXt-T Grad-CAM `0.908`, DINOv2 ViT-S/14 gradient `0.713`.

Current interpretation:

- The old static2000 protocol failure has been resolved for the corrected outputs.
- DeepGaze now beats center bias across all three datasets under the corrected point/task-point protocol.
- DINOv2 gradient is a strong attribution/fixation-similarity row, especially on SALICON and CAT2000.
- The behavioral layer is strong enough to serve as one axis in the broader alignment study. It should not be expanded into a larger leaderboard before the neural axis is upgraded.

## Current Neural Status

Current neural summary:

- Path: `outputs/neural_roi_summary/`
- Input neural directories: `24`
- Encoding rows: `96`
- Encoding target rows: `231792`
- RSA rows: `92`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Feature-reduction rows: `92` spatial-mean diagnostic rows and `4` full-image-count `flatten_pca` rows.
- Benchmark-style per-target encoding scope: mixed because four hV4 targets have `noise_ceiling=0.0`; `231696` rows are `benchmark_style_noise_normalized` and `96` rows are intentionally left `benchmark_style_non_noise_normalized`.

Current noise-normalized neural ranking:

- Mean valid-target noise-normalized encoding leader: `deit_small_patch16_224`, mean `0.556` (`55.57` on x100 scale).
- `vit_small_patch14_dinov2` ranks second by mean valid-target noise-normalized encoding, mean `0.161` (`16.08` x100).
- The current noise-normalized encoding leader is the same model as the raw Pearson leader.
- Each model ranking row aggregates `9654` valid positive-ceiling targets and excludes `4` zero-ceiling hV4 targets from noise-normalized aggregates.
- The `deit_small_patch16_224` ranking is driven by full-image-count `blocks.3` `flatten_pca` runs, while the other current model families remain ROI500 spatial-mean diagnostics.

Current raw neural ranking:

- Mean encoding leader: `deit_small_patch16_224`, mean raw correlation `0.524`.
- Mean RSA leader: `vit_base_patch16_224`, mean Spearman RSA `0.088`.
- `vit_small_patch14_dinov2` ranks second for both raw encoding and noise-normalized encoding; it ranks second by RSA.
- `resnet50` is now included in the regenerated summary and ranks fifth by mean raw encoding, fourth by mean RSA.
- Current ROI set: `V1`, `V2`, `V3`, `hV4` for `subj01`.
- Full-image-count `flatten_pca` runs intentionally have RSA disabled to avoid allocating full `9841 x 9841` RDMs; current RSA rankings still come from ROI500-scale outputs.

Full-image-count `flatten_pca` `deit_small_patch16_224 blocks.3` results:

- V1: `2973` valid targets, mean raw Pearson `0.570`, mean valid-target noise-normalized score `0.592` (`59.15` x100), median noise-normalized score `0.623`.
- V2: `2936` valid targets, mean raw Pearson `0.554`, mean valid-target noise-normalized score `0.584` (`58.43` x100), median noise-normalized score `0.625`.
- V3: `2453` valid targets, mean raw Pearson `0.530`, mean valid-target noise-normalized score `0.560` (`55.97` x100), median noise-normalized score `0.567`.
- hV4: `1292` valid positive-ceiling targets plus `4` zero-ceiling targets, mean raw Pearson `0.444`, mean valid-target noise-normalized score `0.487` (`48.74` x100), median noise-normalized score `0.482`.
- All four extended-alpha runs selected `ridge_alpha=100000.0`, below the maximum tested `10000000.0`; no additional high-alpha pass is currently needed.
- PCA metadata for all four runs records `train_only_fit=true`, `n_train_fit=7873`, `effective_components=512`, and `pca_solver=randomized`.

Current bridge readout:

- Overall behavior-to-encoding leader match rate: `0.079`.
- Overall behavior-to-RSA leader match rate: `0.000`.
- Internal-routing rows account for the observed encoding leader matches; class-localization and evidence-sensitivity rows do not match the encoding or RSA leader in the current descriptive table.

Interpretation:

- The project has moved from a weak ROI500 spatial-mean neural diagnostic to a strong local full-image-count ridge baseline for one model family.
- The `deit_small_patch16_224` `flatten_pca` baseline now clears the Algonauts organizer baseline scale numerically in these local PRF ROI summaries, but this is not leaderboard-equivalent because subject, cortex, split, and target scope differ.
- The strongest current claim is methodological: full image count plus train-only PCA over flattened token features dramatically improves local neural encoding relative to ROI500 spatial-mean probes.
- The next scientific risk is test-set feedback: `blocks.3` was selected from smoke/test observations. Future layer and pooling choices should be made by validation-only selection before launching more model families.

Current methodological gap to SOTA:

- The current strong baseline uses `flatten_pca`, not learned spatial pooling or voxel-specific readouts. It preserves more feature information than spatial mean, but still falls short of SOTA-style spatial heads.
- Current full-image-count runs cover `subj01` PRF visual ROIs only. SOTA scores use broader visual-cortex vertex sets across subjects and hemispheres.
- Current ridge alpha selection is per-layer/ROI, not per-target. Per-target alpha or voxel-specific readouts may improve later but should wait until validation selection is clean.
- Current reporting mixes full `flatten_pca` rows for `deit_small_patch16_224` with ROI500 spatial-mean rows for other models. Cross-model rankings are useful for handoff, but model-family claims should wait until the same full protocol is run for more backbones.
- Current scoring evaluates single layers independently. SOTA methods use validation-selected layers, multi-layer fusion, learned layer selection, subject-specific heads, and ensembles.

Relevant SOTA references:

- Algonauts 2023 challenge evaluation: `https://algonautsproject.com/2023/challenge.html`
- Memory Encoding Model: DINOv2 backbone, voxel-specific RetinaMapper, LayerSelector, memory/task/subject conditioning, random-ROI ensemble; single model around `66.8`, ensemble around `70.85`: `https://ar5iv.labs.arxiv.org/html/2308.01175`
- UARK-UAlbany solution: multi-subject pretraining, subject fine-tuning, ConvNeXt-style backbones, SmoothL1/Pearson/noise-normalized losses, weighted ensemble; baseline around `54.21`, ensemble around `61.56`: `https://arxiv.org/pdf/2308.00262`
- BlobGPT: EVA02 trunk, multi-layer feature tensors, learned spatial pooling, shared and subject-specific transforms, fMRI PCA embedding, end-to-end fine-tuning; score around `60.2`: `https://arxiv.org/pdf/2308.02351`
- Scaling-law report: model size, fMRI sample size, layer/kernel selection, cross-validated ridge, and model averaging materially improve scores: `https://arxiv.org/pdf/2308.00678`

## SSL / Multimodal Status

Current SSL/VLM behavioral rows are corrected and merged into the main behavioral aggregate.

SSL/multimodal candidate inventory:

- Path: `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`
- Dry-inspected compatible candidates: `8`
- Pretrained debug runs complete: `3`
- Complete pretrained debug candidates: `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, `resnet50_clip`
- Not yet run pretrained debug candidates: `vit_base_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_dinov3`, `vit_base_patch16_siglip_224`, `eva02_base_patch16_clip_224`

Do not expand this list again until validation-only layer/pooling selection is implemented for the full `flatten_pca` protocol. More models are useful only after the strong baseline can choose layers without test-set feedback.

## What Is Already Built

Behavioral infrastructure:

- Manifest loaders for SALICON, CAT2000, COCO-Search18, and NSD / Algonauts-style data.
- Fixation parsers for SALICON and CAT2000 `.mat` files.
- Task/scanpath point handling for COCO-Search18.
- Static metrics: NSS, AUC-Judd, AUC-Borji, shuffled AUC, CC, SIM, KL, EMD, MAE, Pearson.
- Saliency methods: center bias, random saliency, gradient, integrated gradients, Grad-CAM, attention rollout, occlusion, and precomputed DeepGaze-style maps.
- Matrix execution, aggregation, summaries, plots, and paper inspection pack generation.

Neural infrastructure:

- `timm` wrappers with named-layer activation extraction.
- Ridge encoding over ROI response vectors.
- Train-only `flatten_pca` feature reduction for full flattened activation tensors.
- Cross-validated ridge alpha selection on an inner split of training images.
- RSA over model and neural response RDMs.
- ROI500 summaries, full-image-count PRF ROI summaries, model rankings, ROI winners, and descriptive behavior-neural bridge tables.

Reporting infrastructure:

- Corrected behavioral aggregate and merged SSL/VLM aggregate.
- Neural ROI summary tables.
- Paper inspection pack with behavior, neural, bridge, SSL/VLM candidate, benchmark sanity tables, and an academic SOTA context section comparing the current figures against MIT/Tuebingen saliency, DeepGaze IIE SALICON, COCO-Search18 task-search, and Algonauts 2023 evaluation references.
- Paper inspection README now explicitly distinguishes ROI500 diagnostics from full-image-count PRF ROI `flatten_pca` baselines.

## Superseded Or Historical Outputs

Historical note:

- `docs/v2_static2000_results_note.md` describes the pre-fix static2000 state from 2026-05-19. Its result interpretation is superseded.

Do not use for current claims:

- Any aggregate row without `fixation_protocol=points` or `task_points`.
- Any pre-2026-05-20 static2000 NSS/AUC result generated before point-fixation scoring was corrected.
- Old claims that center bias beat DeepGaze under the static2000 protocol.

Current corrected outputs have replaced those rows.

## Current Implementation Progress

Updated: 2026-05-24

Neural Benchmark-Equivalent Evaluation V1 is implemented for the current one-subject ROI500 diagnostic scope.

Neural Reliability / Noise-Ceiling Metadata V1 is implemented for `subj01` PRF visual ROI500 data. The earlier local audit found no ceiling files in the Algonauts subset, so NSD `ncsnr` files were added from the full NSD release and converted into target-level ROI noise-ceiling vectors.

Neural Scoring Foundation V1 is implemented for the current ROI500 summary and paper inspection outputs. Neural encoding model rankings now use valid-target mean `noise_normalized_score` as the primary encoding rank when positive noise ceilings are available, while raw Pearson encoding and RSA ranks remain separate diagnostics.

Large NSD/Algonauts Manifest And Run Configs V1 is implemented for `subj01` PRF visual ROIs. The project now has full-image-count manifest plumbing and full-image-count production configs for `V1`, `V2`, `V3`, and `hV4`.

Feature Representation Upgrade V1 is implemented in code, configs, full four-ROI outputs, neural summary, and paper inspection pack. The neural runner now supports train-only `flatten_pca` feature reduction for flattened activation tensors, writes per-layer feature-reduction metadata, and saves reduced activations for PCA runs instead of full raw tensors. The `deit_small_patch16_224` `blocks.3` full `V1`/`V2`/`V3`/`hV4` runs completed successfully and are included in `outputs/neural_roi_summary/`.

Benchmark-equivalent implementation:

- Per-target benchmark-style encoding rows are now written to `encoding_target_scores.csv` by `scripts/run_neural_alignment.py`.
- Layer-level `encoding_scores.csv` remains backward compatible and now includes metric-scope, selected-alpha, alpha-selection-mode, split-seed, feature-reduction metadata, and valid-target noise-normalized aggregate fields.
- Per-target scores include raw Pearson `r`, `r2_score_from_r`, ordinary prediction `r2`, optional noise-ceiling fields, `valid_noise_ceiling`, noise-normalized score when possible, and variance-validity flags.
- Runs without noise-ceiling metadata are explicitly labeled `benchmark_style_non_noise_normalized`; runs with attached NSD-derived ROI ceilings are labeled `benchmark_style_noise_normalized`.
- Optional `neural.ridge_alphas` now enables deterministic inner-validation ridge-alpha selection per layer; configs without `ridge_alphas` keep fixed `neural.ridge_alpha`.
- `summarize_neural_roi_results` now loads optional `encoding_target_scores.csv`, writes `combined_encoding_target_scores.csv` when available, derives noise-normalized layer/model aggregates from target rows, and keeps old neural output directories compatible.

Focused verification run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Latest result after Feature Representation Upgrade V1: `43 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

Full regeneration status:

- Full ROI500 outputs were regenerated for `outputs/neural_roi500/` and `outputs/neural_roi500_ssl/`; the refreshed summary also includes the four full-image-count `flatten_pca` PRF ROI runs.
- Checked neural output directories: `24`.
- Missing `encoding_target_scores.csv`: none.
- Missing `metadata.json`: none.
- Missing `rsa_scores.csv`: four expected missing files from full-image-count `flatten_pca` runs where RSA is intentionally disabled.
- Combined per-target benchmark rows: `231792`.
- Per-target metric scopes: `231696` rows with `benchmark_style_noise_normalized`, `96` rows with `benchmark_style_non_noise_normalized`.
- The `96` non-normalized rows are hV4 targets with `noise_ceiling=0.0`; these rows are intentionally not divided by zero.
- Per-target variance flags: all `valid_prediction_variance=true` and `valid_target_variance=true`.
- `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/` were regenerated from the corrected behavioral aggregate and refreshed neural outputs.
- `outputs/neural_roi_summary/neural_model_rankings.csv` ranks `deit_small_patch16_224` first by mean valid-target noise-normalized score after adding full-image-count `flatten_pca` rows: mean `0.556` (`55.57` x100).
- `outputs/paper_inspection_v1/README.md` now reports the mixed neural scope correctly: one-subject ROI500 diagnostics plus full-image-count PRF visual ROI `flatten_pca` baselines.

Large/full manifest and config status:

- Full PRF visual ROI manifest: `data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv`.
- Full manifest rows: `39364` (`9841` training images x `4` ROIs).
- ROI response sidecars now exist for `9841` images each under `data/raw/nsd_algonauts/subj01/responses/V1`, `V2`, `V3`, and `hV4`.
- Full manifest columns include `noise_ceiling_path`, `noise_ceiling_values`, and `noise_ceiling_source`.
- Attached noise-ceiling source: `nsd_ncsnr_mgh_n_trials_3`.
- Large smoke config: `configs/experiments/neural_large_smoke/deit_small_patch16_224_v1_smoke.yaml`.
- Historical first full-run config: `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_full.yaml`; do not use it as the preferred production baseline because it keeps `spatial_mean` and RSA enabled.
- Preferred full-image-count `flatten_pca` production configs:
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_blocks3_alpha1e7_full.yaml`
- Large smoke output: `outputs/neural_large_smoke/deit_small_patch16_224_v1_smoke/`.
- Large smoke result: `64` images, `51` train / `13` test, `2973` V1 targets, `noise_ceiling_available=true`, all target rows `benchmark_style_noise_normalized`.
- Full `flatten_pca` outputs:
  - `outputs/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_blocks3_alpha1e7_full/`
- Large manifest focused verification: `25 passed` for `tests/test_create_algonauts_manifest.py`, `tests/test_nsd_algonauts_dataset.py`, and `tests/test_neural_roi_summary.py`.
- Full production `flatten_pca` configs intentionally disable RSA to avoid full `9841 x 9841` RDM allocation.

Neural reliability / noise-ceiling implementation:

- `NSDAlgonautsDataset` now accepts optional manifest columns `noise_ceiling_path`, `noise_ceiling_values`, and `noise_ceiling_source`.
- Dataset items emit `metadata.noise_ceiling` when a target-level sidecar or inline vector is available.
- `scripts/run_neural_alignment.py` / `run_neural_alignment` now accepts `neural.noise_ceiling_key`, defaulting to `noise_ceiling`.
- The runner validates that noise-ceiling vectors are target-level, complete across all run items, and consistent across items before passing them into `benchmark_encoding_target_scores`.
- Run metadata now records `noise_ceiling_available`, `noise_ceiling_key`, and `noise_ceiling_source`.
- Audit artifact: `outputs/neural_reliability_audit_v1/README.md`.
- Audit result: `0` candidate reliability / noise-ceiling / repeat-trial / split-half files found under `data/raw/nsd_algonauts`; full NSD `ncsnr` files were therefore added under `data/raw/nsd_full/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/`.
- `scripts/create_nsd_noise_ceiling_manifest.py` converts `lh.ncsnr.mgh` and `rh.ncsnr.mgh` into ROI target-level noise ceilings using `NC = ncsnr^2 / (ncsnr^2 + 1 / n_trials)`.
- `n_trials=3` was verified from `data/raw/nsd_full/experiments/nsd/nsd_expdesign.mat` for the current `subj01` design.
- ROI ceiling files were written to `data/raw/nsd_algonauts/subj01/noise_ceilings/`: `V1.npy`, `V2.npy`, `V3.npy`, and `hV4.npy`.
- The ROI500 manifest `data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv` now includes `noise_ceiling_path`, `noise_ceiling_values`, and `noise_ceiling_source` columns for all `2000` rows.
- Smoke config: `configs/experiments/neural_roi500_noise_ceiling_smoke.yaml`.
- Smoke output: `outputs/neural_roi500_noise_ceiling_smoke/`.
- Smoke result: `2973` V1 target rows with `metric_scope=benchmark_style_noise_normalized` and `noise_ceiling_available=true`.

Full verification:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest result: `177 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

## Next Concrete Milestone

Priority: **Validation-Only Layer/Pooling Selection V1**.

Do this before adding more behavioral datasets, more saliency methods, Brain-Score integration, CKA, adaptive token pruning, foveation, scanpaths, video, or a broad model zoo. The project now has a strong full-image-count `flatten_pca` ridge baseline, but the next methodological weakness is that `blocks.3` was chosen after smoke/test observations. Future layer and pooling choices need a validation-only selection path before running more backbones.

Acceptance target:

- Add a validation-selection mode that evaluates candidate layers and feature reductions on an inner validation split from the training images only.
- Write a selection artifact recording candidate layer, feature mode, PCA components, selected alpha, validation score, and final selected configuration.
- Keep final test scores honest: the test split should be used only after the layer/pooling setting is selected.
- Run a small validation-selection smoke before launching any more full-image-count model-family runs.
- Regenerate summary outputs only after final test rows are produced from validation-selected settings.

Step 1: scoring policy and reporting foundation. **Status: implemented.**

- Add `valid_noise_ceiling = noise_ceiling > 0` to target rows.
- Keep raw `pearson_r`, `r2_score_from_r`, and `prediction_r2` for every target.
- Compute `noise_normalized_score` only for positive finite ceilings.
- Exclude `noise_ceiling <= 0` targets from noise-normalized layer/model aggregates.
- Add aggregate fields: `mean_noise_normalized_score`, `median_noise_normalized_score`, `valid_noise_ceiling_targets`, `zero_noise_ceiling_targets`, `invalid_noise_ceiling_targets`, and `rank_mean_noise_normalized`.
- Update `neural_model_rankings.csv`, ROI winner tables, and paper inspection figures so raw-rank and noise-normalized-rank are separate.
- Regenerate `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/`.

Step 2: full-image-count manifest and run configs. **Status: implemented.**

This is implemented for `subj01` full PRF visual ROIs.

- Build full-subject or large-subject manifests from local NSD/Algonauts files instead of ROI500-only manifests.
- Keep the same ROI targets at first (`V1`, `V2`, `V3`, `hV4`) to isolate the image-count and method changes.
- Preserve deterministic image splits and record exact train/test image IDs in metadata.
- Add smoke configs that run on a small image count, then production configs that use the largest local `subj01` image count available.

Step 3: feature representation upgrade. **Status: implemented.**

- Stop using `feature_reduction: spatial_mean` for competitive neural runs.
- Add a training-only dimensionality-reduction path for flattened feature tensors: PCA or randomized SVD fit on train images only, then applied to validation/test images.
- Store the fitted reduction metadata: number of components, explained variance, input feature shape, layer name, train-only fit flag, and random seed.
- Add configurable feature modes: `spatial_mean` for smoke/debug only, `flatten_pca` for strong ridge baseline, and later `learned_spatial_pooling`.
- Verify no train/test leakage in PCA/SVD fitting.
- Smoke config: `configs/experiments/neural_large_smoke/deit_small_patch16_224_v1_flatten_pca_smoke.yaml`.
- Full config: `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_full.yaml`.
- Preferred first full config after smoke inspection: `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_full.yaml`, because `blocks.3` was the smoke winner by mean valid-target noise-normalized encoding and RSA.
- PCA metadata output: `feature_reduction_metadata.json`.
- Completed preferred V1 full output: `outputs/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_full/`.
- Full V1 `blocks.3` result: `9841` images, `7873` train / `1968` test, `2973` valid V1 targets, mean raw Pearson `0.576`, mean valid-target noise-normalized score `0.602` (`60.23` x100), median noise-normalized score `0.643`, selected ridge alpha `10000.0`.
- Extended-alpha full configs are available for four PRF visual ROIs:
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_blocks3_alpha1e7_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_blocks3_alpha1e7_full.yaml`
- Boundary: this is a strong local V1 PRF-ROI result, not an Algonauts leaderboard-equivalent score because it covers one subject and one ROI rather than all held-out visual-cortex vertices across subjects/hemispheres.
- Extended-alpha four-ROI full runs completed successfully. All selected ridge alpha `100000.0`, below the maximum tested `10000000.0`, so no further high-alpha pass is needed before summary regeneration.
- Extended-alpha full results:
  - V1: `2973` valid targets, mean raw Pearson `0.570`, mean valid-target noise-normalized score `0.592` (`59.15` x100), median noise-normalized score `0.623`.
  - V2: `2936` valid targets, mean raw Pearson `0.554`, mean valid-target noise-normalized score `0.584` (`58.43` x100), median noise-normalized score `0.625`.
  - V3: `2453` valid targets, mean raw Pearson `0.530`, mean valid-target noise-normalized score `0.560` (`55.97` x100), median noise-normalized score `0.567`.
  - hV4: `1292` valid positive-ceiling targets plus `4` zero-ceiling targets, mean raw Pearson `0.444`, mean valid-target noise-normalized score `0.487` (`48.74` x100), median noise-normalized score `0.482`.

Step 4: cross-validated ridge baseline. **Status: implemented for per-layer/ROI alpha selection.**

- Use `neural.ridge_alphas` by default for strong neural runs, spanning several orders of magnitude.
- Select alpha on an inner validation split from the training images only.
- Current full `flatten_pca` runs select alpha per layer/ROI and record the chosen value.
- Consider per-target alpha only if runtime remains manageable and after validation-only layer selection is stable.
- Record selected alpha, validation score, validation split size, and selection mode in outputs.
- Existing tests prove selected alpha changes with data; next tests should cover layer/pooling selection without test leakage.

Step 5: layer and pooling selection. **Status: next concrete implementation step.**

- Evaluate multiple layers and feature-reduction settings with the same train/test split.
- Add a validation-only layer/pooling selection summary before reporting final test scores.
- Keep final test reporting honest: the selected layer/pooling setting must be chosen without using the test set.
- Add optional multi-layer concatenation after single-layer `flatten_pca` is stable.
- Start with `deit_small_patch16_224` and candidate layers from the existing smoke set: `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`.
- Keep the first selection smoke on `V1` with a small image count, then run full four-ROI selection only after the smoke artifact is correct.

Step 6: stronger model backbones.

- Prioritize the SOTA-aligned backbones from the literature: DINOv2, EVA02/CLIP-like vision backbones, ConvNeXt-L/XL if compute allows, and strong ViT variants.
- Do not expand to many models until the strong baseline is methodologically sound.
- Compare small versus large variants only after the feature and ridge pipeline is fixed.

Step 7: learned spatial readout prototype.

- Add a PyTorch encoding-head path after the strong ridge baseline is validated.
- Start with frozen image backbone features and a learned target-specific spatial pooling/readout head.
- Use validation loss based on Pearson or noise-normalized objective, plus regularization and early stopping.
- Keep the first learned-readout scope small: one subject, one ROI, one backbone, then scale.
- This is the local analogue of SOTA components such as RetinaMapper, learned spatial pooling, voxel-specific heads, and subject-specific heads.

Step 8: subject and cortex expansion.

- After `subj01` strong baseline is credible, expand to additional available subjects.
- Add subject-specific metadata and subject-wise summaries.
- Move beyond PRF ROI500 toward the broader Algonauts visual-cortex vertex set if local data permits.
- Keep subject-general claims blocked until at least multiple subjects have the same pipeline.

Step 9: ensemble and SOTA-style reporting.

- Add model/layer/readout ensembling only after single-model baselines are strong and stable.
- Report single-model, selected-layer, multi-layer, learned-readout, and ensemble scores separately.
- Add an explicit comparison table with local score x100, Algonauts organizer baseline `40.42`, strong published baselines around `54-62`, Memory Encoding Model single model around `66.8`, and ensemble around `70.85`.

Step 10: uncertainty and robustness.

- Add bootstrap confidence intervals over images and targets.
- Add split-seed sensitivity checks.
- Add ROI-wise and subject-wise failure diagnostics.
- Keep RSA/CKA as secondary representational analyses, not substitutes for encoding-score improvement.

Implementation order for the next Codex sessions:

1. Add validation-only layer/pooling selection for full-image-count `flatten_pca` runs so future layer/pooling choices are made without test-set feedback.
2. Run a small validation-selection smoke over candidate layers/pooling settings before launching more full runs.
3. If validation selection confirms `blocks.3`, broaden carefully to one stronger model family, likely DINOv2, using the same full-image-count `flatten_pca` protocol.
4. Only then decide whether to implement learned spatial readouts.

## Later Milestones

After validation-selected strong ridge baseline:

1. Add learned spatial pooling / target-specific encoding heads.
2. Expand beyond `subj01`.
3. Expand beyond PRF ROI500 to broader visual-cortex vertices if local data supports it.
4. Add single-model and ensemble comparisons against Algonauts-style SOTA ranges.
5. Add confidence intervals or bootstrap uncertainty over images, targets, and subjects.
6. Run full evaluations for strongest SSL/VLM candidates only after the strong baseline is stable.
7. Add efficiency profiles and alignment-per-compute summaries back into the inspection pack.
8. Add stronger transformer attribution methods such as Chefer-style attribution or AttnLRP before making claims about transformer attention.
9. Add Brain-Score or Brain-Score-style external comparisons.
10. Add CKA or additional representational geometry metrics alongside RSA.
11. Add adaptive/selective-computation models such as token pruning, foveation, adaptive patch selection, or glimpse-style models.
12. Consider scanpath or video analysis only after static-image behavioral and neural claims are stable.

## Code Pointers

Dataset loading and fixation parsing:

- `src/hma/datasets/salicon.py`
- `src/hma/datasets/cat2000.py`
- `src/hma/datasets/coco_search18.py`
- `src/hma/datasets/nsd_algonauts.py`
- `src/hma/datasets/fixation_parsers.py`
- `src/hma/datasets/fixation_utils.py`

Behavioral benchmark:

- `src/hma/experiments/saliency_benchmark.py`
- `src/hma/experiments/aggregate_results.py`
- `src/hma/experiments/summarize_results.py`
- `scripts/run_saliency_benchmark.py`
- `scripts/run_v2_matrix.py`
- `scripts/aggregate_results.py`

Saliency methods:

- `src/hma/saliency/baselines.py`
- `src/hma/saliency/gradients.py`
- `src/hma/saliency/gradcam.py`
- `src/hma/saliency/integrated_gradients.py`
- `src/hma/saliency/attention_rollout.py`
- `src/hma/saliency/occlusion.py`
- `src/hma/saliency/precomputed.py`
- `src/hma/saliency/postprocess.py`

Neural alignment:

- `src/hma/neural/activations.py`
- `src/hma/neural/encoding.py`
- `src/hma/neural/rsa.py`
- `src/hma/experiments/neural_alignment.py`
- `src/hma/experiments/summarize_neural_roi_results.py`
- `scripts/run_neural_alignment.py`
- `scripts/create_nsd_noise_ceiling_manifest.py`
- `scripts/summarize_neural_roi_results.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

For the neural reliability implementation, the focused verification command is:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_nsd_algonauts_dataset.py
```

Latest focused result: `34 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

Audit command:

```cmd
.\.venv\Scripts\python.exe scripts\audit_neural_reliability_metadata.py
```

Latest audit result: `candidate_count=0`, `usable_candidate_count=0`, `usable_metadata_found=False`; the usable ceiling source is now full NSD `ncsnr`, not the local Algonauts subset.

Noise-ceiling conversion command:

```cmd
.\.venv\Scripts\python.exe scripts\create_nsd_noise_ceiling_manifest.py --n-trials 3
```

Smoke verification command:

```cmd
.\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config configs\experiments\neural_roi500_noise_ceiling_smoke.yaml
```

Latest smoke result: `outputs/neural_roi500_noise_ceiling_smoke/encoding_target_scores.csv` contains `2973` rows with `metric_scope=benchmark_style_noise_normalized`.

Current neural summary target-scope verification:

```cmd
.\.venv\Scripts\python.exe -c "import csv,collections; rows=list(csv.DictReader(open('outputs/neural_roi_summary/combined_encoding_target_scores.csv', newline='', encoding='utf-8'))); print(len(rows)); print(dict(collections.Counter(r['metric_scope'] for r in rows)))"
```

Latest regenerated target scope: `231792` rows total; `231696` `benchmark_style_noise_normalized`; `96` `benchmark_style_non_noise_normalized` due to hV4 zero-ceiling targets across ROI500 and full `flatten_pca` rows.

For the next neural method implementation, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```

For broader confidence after code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result: `177 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

After Feature Representation Upgrade V1:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Latest focused result: `43 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result: `177 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

After full `flatten_pca` four-ROI summary regeneration:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_paper_inspection_pack.py tests\test_neural_roi_summary.py
```

Latest focused result: `23 passed`; Windows `.pytest_cache` permission warning remains non-blocking.
