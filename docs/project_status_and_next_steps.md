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
- Input neural directories: `29`
- Encoding rows: `101`
- Encoding target rows: `244423`
- RSA rows: `92`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Feature-reduction rows: `92` spatial-mean diagnostic rows, `8` validation-selected full-image-count `flatten_pca` rows, and `1` full-image-count learned spatial readout row.
- Benchmark-style per-target encoding scope: mixed because four hV4 targets have `noise_ceiling=0.0`; `244323` rows are `benchmark_style_noise_normalized` and `100` rows are intentionally left `benchmark_style_non_noise_normalized`.

Current noise-normalized neural ranking:

- Mean valid-target noise-normalized encoding leader: `vit_small_patch14_dinov2`, mean `0.621` (`62.11` on x100 scale).
- `deit_small_patch16_224` ranks second by mean valid-target noise-normalized encoding, mean `0.562` (`56.17` x100).
- The current noise-normalized encoding leader is the same model as the raw Pearson leader.
- Each model ranking row aggregates `9654` valid positive-ceiling targets and excludes `4` zero-ceiling hV4 targets from noise-normalized aggregates.
- The `vit_small_patch14_dinov2` ranking is now driven by the learned spatial readout V1 row plus validation-selected full-image-count `flatten_pca` rows for the remaining ROIs. The `deit_small_patch16_224` ranking is driven by validation-selected full-image-count `flatten_pca` rows. The other current model families remain ROI500 spatial-mean diagnostics.

Current raw neural ranking:

- Mean encoding leader: `vit_small_patch14_dinov2`, mean raw correlation `0.555`.
- Mean RSA leader: `vit_base_patch16_224`, mean Spearman RSA `0.088`.
- `deit_small_patch16_224` ranks second for both raw encoding and noise-normalized encoding; `vit_small_patch14_dinov2` remains second by ROI500-scale RSA.
- `resnet50` is now included in the regenerated summary and ranks fifth by mean raw encoding, fourth by mean RSA.
- Current ROI set: `V1`, `V2`, `V3`, `hV4` for `subj01`.
- Full-image-count `flatten_pca` runs intentionally have RSA disabled to avoid allocating full `9841 x 9841` RDMs; current RSA rankings still come from ROI500-scale outputs.

Validation-selected full-image-count `flatten_pca` `deit_small_patch16_224` results:

- V1 selected `blocks.0`: `2973` valid targets, mean raw Pearson `0.581`, mean valid-target noise-normalized score `0.611` (`61.14` x100).
- V2 selected `blocks.3`: `2936` valid targets, mean raw Pearson `0.556`, mean valid-target noise-normalized score `0.588` (`58.79` x100).
- V3 selected `blocks.3`: `2453` valid targets, mean raw Pearson `0.531`, mean valid-target noise-normalized score `0.560` (`56.01` x100).
- hV4 selected `blocks.3`: `1292` valid positive-ceiling targets plus `4` zero-ceiling targets, mean raw Pearson `0.444`, mean valid-target noise-normalized score `0.487` (`48.74` x100).
- V1/V2/V3 selected `ridge_alpha=10000.0`; hV4 selected `ridge_alpha=100000.0`, below the maximum tested `10000000.0`; no additional high-alpha pass is currently needed.
- PCA metadata for all four runs records `train_only_fit=true`, `n_train_fit=7873`, `effective_components=512`, and `pca_solver=randomized`.

Validation-selected full-image-count `flatten_pca` `vit_small_patch14_dinov2` results:

- V1 selected `blocks.3`: `2973` valid targets, mean raw Pearson `0.595`, mean valid-target noise-normalized score `0.642` (`64.23` x100), selected `ridge_alpha=10.0`.
- V2 selected `blocks.6`: `2936` valid targets, mean raw Pearson `0.569`, mean valid-target noise-normalized score `0.614` (`61.44` x100), selected `ridge_alpha=1000.0`.
- V3 selected `blocks.6`: `2453` valid targets, mean raw Pearson `0.545`, mean valid-target noise-normalized score `0.591` (`59.06` x100), selected `ridge_alpha=1000.0`.
- hV4 selected `blocks.6`: `1292` valid positive-ceiling targets plus `4` zero-ceiling targets, mean raw Pearson `0.457`, mean valid-target noise-normalized score `0.517` (`51.70` x100), selected `ridge_alpha=1000.0`.
- PCA metadata for all four runs records `train_only_fit=true`, `n_train_fit=7873`, `effective_components=512`, `input_feature_shape=[1370, 384]`, and `pca_solver=randomized`.

Full learned spatial readout `vit_small_patch14_dinov2` result:

- V1 fixed `blocks.3`: `2973` valid targets, mean raw Pearson `0.648`, mean valid-target noise-normalized score `0.762` (`76.25` x100), median raw Pearson `0.691`, median noise-normalized score `0.803`.
- This improves over the validation-selected DINOv2 V1 `flatten_pca` baseline by `+0.052` raw Pearson and `+0.120` valid-target noise-normalized score.
- Training used `6298` inner-train and `1575` validation images inside the `7873` outer-train images; final scoring used `1968` held-out test images.
- Longer diagnostic best epoch was `127`, early-stopped at epoch `142`, validation mean Pearson `0.648`; the gain over the 100-epoch run was negligible (`+0.00021` raw Pearson, `+0.00035` noise-normalized), so the V1 learned-readout training is effectively saturated for this head/config.
- Output path: `C:/saliency_outputs/neural_subj01_full/vit_small_patch14_dinov2_v1_learned_spatial_readout_full_180ep/`.
- The learned row is included in regenerated `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/`.

Current bridge readout:

- Overall behavior-to-encoding leader match rate: `0.587`.
- Overall behavior-to-RSA leader match rate: `0.000`.
- Internal-routing rows account for the observed encoding leader matches; class-localization and evidence-sensitivity rows do not match the encoding or RSA leader in the current descriptive table.

Interpretation:

- The project has moved from a weak ROI500 spatial-mean neural diagnostic to a strong local full-image-count ridge baseline for one model family.
- The `vit_small_patch14_dinov2` learned spatial readout V1 row and the `vit_small_patch14_dinov2` / `deit_small_patch16_224` `flatten_pca` baselines now clear the Algonauts organizer baseline scale numerically in these local PRF ROI summaries, but this is not leaderboard-equivalent because subject, cortex, split, and target scope differ.
- The strongest current claim is methodological: full image count plus feature-preserving readouts dramatically improves local neural encoding relative to ROI500 spatial-mean probes, and learned spatial pooling materially improves V1 over `flatten_pca`.
- The previous test-set feedback risk for layer choice has been addressed for the current one-subject PRF visual ROI baselines by validation-only layer selection.

Current methodological gap to SOTA:

- The current strongest V1 result uses a learned target-wise spatial readout, but only for one ROI and one backbone. The remaining ROIs still use `flatten_pca`, and the readout is simpler than SOTA voxel-specific RetinaMapper-style heads.
- Current full-image-count runs cover `subj01` PRF visual ROIs only. SOTA scores use broader visual-cortex vertex sets across subjects and hemispheres.
- Current ridge alpha selection is per-layer/ROI, not per-target. Per-target alpha or voxel-specific readouts may improve later, after the matched validation-selected backbone comparison is complete.
- Current reporting mixes one learned-readout row, full `flatten_pca` rows for `vit_small_patch14_dinov2` and `deit_small_patch16_224`, and ROI500 spatial-mean rows for other models. Cross-model rankings are useful for handoff, but model-family claims outside matched protocols should wait.
- Current scoring now supports validation-selected single-layer `flatten_pca` and fixed-layer learned spatial readout. SOTA methods go further with multi-layer fusion, learned layer selection, subject-specific heads, voxel-specific spatial readouts, and ensembles.

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

Do not expand this list again before applying the validation-selected full `flatten_pca` protocol to one stronger candidate backbone. The next model-family comparison should be narrow and methodologically matched, not a broad model zoo.

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
- Frozen-backbone learned spatial readout with target-wise spatial pooling, target-wise channel weights, inner-validation early stopping, and summary-compatible output rows.
- Cross-validated ridge alpha selection on an inner split of training images.
- RSA over model and neural response RDMs.
- ROI500 summaries, full-image-count PRF ROI summaries, model rankings, ROI winners, and descriptive behavior-neural bridge tables.

Reporting infrastructure:

- Corrected behavioral aggregate and merged SSL/VLM aggregate.
- Neural ROI summary tables.
- Paper inspection pack with behavior, neural, bridge, SSL/VLM candidate, benchmark sanity tables, and an academic SOTA context section comparing the current figures against MIT/Tuebingen saliency, DeepGaze IIE SALICON, COCO-Search18 task-search, and Algonauts 2023 evaluation references.
- Paper inspection README now explicitly distinguishes ROI500 diagnostics from full-image-count PRF ROI `flatten_pca` baselines and includes the full DINOv2 V1 learned spatial readout row through the neural summary.

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

Feature Representation Upgrade V1 is implemented in code, configs, full four-ROI outputs, neural summary, and paper inspection pack. The neural runner now supports train-only `flatten_pca` feature reduction for flattened activation tensors, writes per-layer feature-reduction metadata, and saves reduced activations for PCA runs instead of full raw tensors. The earlier fixed-layer `deit_small_patch16_224 blocks.3` full runs completed successfully, but the current summary now uses validation-selected full runs instead.

Validation-Only Layer/Pooling Selection V1 is implemented in code, configs, tests, V1 smoke, full four-ROI outputs, neural summary, and paper inspection pack. The neural runner now supports `neural.selection.enabled`, validation-only candidate scoring over layer/feature-reduction settings, selection artifacts, and final held-out test scoring for only the selected candidate. Full selection mode stores raw candidate activations in a temporary disk-backed `feature_cache` instead of stacking multi-GB raw layer tensors in RAM, then removes that cache after final selected outputs are written.

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

- Full ROI500 outputs were regenerated for `outputs/neural_roi500/` and `outputs/neural_roi500_ssl/`; the refreshed summary now includes eight validation-selected full-image-count `flatten_pca` PRF ROI runs: four `deit_small_patch16_224` rows and four `vit_small_patch14_dinov2` rows.
- Checked neural output directories: `28`.
- Missing `encoding_target_scores.csv`: none.
- Missing `metadata.json`: none.
- Missing `rsa_scores.csv`: eight expected missing files from full-image-count `flatten_pca` runs where RSA is intentionally disabled.
- Combined per-target benchmark rows: `241450`.
- Per-target metric scopes: `241350` rows with `benchmark_style_noise_normalized`, `100` rows with `benchmark_style_non_noise_normalized`.
- The `100` non-normalized rows are hV4 targets with `noise_ceiling=0.0`; these rows are intentionally not divided by zero.
- Per-target variance flags: all `valid_prediction_variance=true` and `valid_target_variance=true`.
- `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/` were regenerated from the corrected behavioral aggregate and refreshed neural outputs.
- `outputs/neural_roi_summary/neural_model_rankings.csv` ranks `vit_small_patch14_dinov2` first by mean valid-target noise-normalized score after adding validation-selected full-image-count `flatten_pca` rows: mean `0.591` (`59.11` x100).
- `outputs/paper_inspection_v1/README.md` now reports the mixed neural scope correctly: one-subject ROI500 diagnostics plus validation-selected full-image-count PRF visual ROI `flatten_pca` baselines.

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
- Validation-selection smoke config: `configs/experiments/neural_large_smoke/deit_small_patch16_224_v1_flatten_pca_validation_selection_smoke.yaml`.
- Validation-selection full production config templates:
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_validation_selection_full.yaml`
- Large smoke output: `outputs/neural_large_smoke/deit_small_patch16_224_v1_smoke/`.
- Large smoke result: `64` images, `51` train / `13` test, `2973` V1 targets, `noise_ceiling_available=true`, all target rows `benchmark_style_noise_normalized`.
- Validation-selection smoke output: `outputs/neural_large_smoke/deit_small_patch16_224_v1_flatten_pca_validation_selection_smoke/`.
- Validation-selection smoke result after disk-backed feature-cache fix: `64` images, `51` outer train / `13` held-out test, `41` selection-train / `10` validation, selected `blocks.3` by validation-only mean noise-normalized score `0.416`, final held-out test mean valid-target noise-normalized score `0.374`, wrote `selection_candidates.csv` plus `selection_artifact.json`, and removed temporary `feature_cache`.
- Full validation-selected outputs:
  - `outputs/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_validation_selection_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_validation_selection_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_validation_selection_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_validation_selection_full/`
- Full `flatten_pca` outputs:
  - `outputs/neural_subj01_full/deit_small_patch16_224_v1_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v2_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_v3_flatten_pca_blocks3_alpha1e7_full/`
  - `outputs/neural_subj01_full/deit_small_patch16_224_hv4_flatten_pca_blocks3_alpha1e7_full/`
- Full validation-selection result: V1 selected `blocks.0`; V2, V3, and hV4 selected `blocks.3`; all four outputs have selected-only `encoding_scores.csv` and `encoding_target_scores.csv`, selection artifacts, final PCA metadata, and no remaining `feature_cache`.
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

Latest focused result after full validation-selected summary regeneration: `46 passed` for `tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py`; Windows `.pytest_cache` permission warning remains non-blocking.

## Next Concrete Milestone

Priority: **Learned Spatial Readout ROI Expansion V1**.

Do this before adding more behavioral datasets, more saliency methods, Brain-Score integration, CKA, adaptive token pruning, foveation, scanpaths, video, or a broad model zoo. The project now has two validation-selected full-image-count `flatten_pca` baselines (`vit_small_patch14_dinov2` and `deit_small_patch16_224`) over `subj01` PRF visual ROIs, plus a completed and plateau-checked full learned spatial readout run for DINOv2 V1. The next methodological step is to extend the same learned-readout protocol to `V2`, `V3`, and `hV4` so the project can determine whether learned spatial pooling improves all PRF visual ROIs or mainly V1.

Next acceptance target:

- Create learned-readout configs for `subj01` `V2`, `V3`, and `hV4` using `vit_small_patch14_dinov2`.
- Use the same learned-readout training settings as the accepted V1 180-epoch run: `max_epochs=180`, `patience=15`, `batch_size=32`, `target_batch_size=256`, `lr=0.001`, `weight_decay=0.0001`, progress logging enabled.
- Use the same selected layers as the current validation-selected DINOv2 `flatten_pca` runs unless a separate validation-only learned-layer selection path is implemented first: `blocks.6` for `V2`, `V3`, and `hV4`.
- Run the three ROI configs one at a time and inspect artifacts before summary regeneration.
- Keep output files compatible with the existing neural summary format and continue reporting learned readout separately from `flatten_pca`.
- Do not add more backbones until the learned-readout ROI expansion decision is made.

Completed learned spatial readout prototype status:

- New module: `src/hma/neural/learned_readout.py`.
- Runner support: `neural.encoding_method: learned_spatial_readout` in `scripts/run_neural_alignment.py` / `src/hma/experiments/neural_alignment.py`.
- The prototype normalizes feature tensors to `[n_images, n_positions, n_channels]`, fits target-wise softmax spatial weights plus target-wise channel weights and bias, uses an inner validation split for early stopping, and reports final held-out test scores through the existing benchmark target scoring path.
- Learned-readout training and prediction batch feature reads by image index from the existing disk-backed `feature_cache` path, avoiding materializing full train/test DINOv2 feature blocks in memory.
- Learned-readout rows use `feature_reduction=learned_spatial_readout`, blank `selected_ridge_alpha`, and `alpha_selection_mode=early_stopping_validation`.
- Output compatibility is implemented for `encoding_scores.csv`, `encoding_target_scores.csv`, `metadata.json`, `feature_reduction_metadata.json`, and `learned_readout_metadata.json`.
- Smoke config: `configs/experiments/neural_large_smoke/vit_small_patch14_dinov2_v1_learned_spatial_readout_smoke.yaml`.
- Full config: `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_v1_learned_spatial_readout_full.yaml`.
- Longer V1 diagnostic config: `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_v1_learned_spatial_readout_full_180ep.yaml`; this keeps the same subject, ROI, layer, split seed, optimizer, and output schema, but increases `max_epochs` to `180`, `patience` to `15`, and writes to a separate `C:/saliency_outputs/..._180ep` directory.
- Smoke output: `outputs/neural_large_smoke/vit_small_patch14_dinov2_v1_learned_spatial_readout_smoke/`.
- Smoke result on `64` images: `2973` valid V1 targets, mean raw Pearson `0.172`, mean valid-target noise-normalized score `0.250`, validation best epoch `35`, validation mean Pearson `0.221`, early-stopped after `45` epochs.
- Smoke result is only a wiring and stability check; it should not be compared scientifically against the full `9841`-image `flatten_pca` baseline.
- Full V1 100-epoch result on `9841` images: `2973` valid V1 targets, mean raw Pearson `0.647`, mean valid-target noise-normalized score `0.762`, validation best epoch `100`, validation mean Pearson `0.647`, no early stop.
- Full V1 180-epoch diagnostic result: mean raw Pearson `0.648`, mean valid-target noise-normalized score `0.762`, validation best epoch `127`, early-stopped at epoch `142`, validation mean Pearson `0.648`; improvement over the 100-epoch run is negligible (`+0.00021` raw Pearson and `+0.00035` noise-normalized), so the V1 head/config is effectively saturated.
- Regenerated summary status after accepting the 180-epoch V1 result: `outputs/neural_roi_summary/` has `29` input directories, `101` encoding rows, `244423` target rows, and `1` learned spatial readout row; `outputs/paper_inspection_v1/` was regenerated from this summary.
- Verification after implementation: `185 passed` for `.\.venv\Scripts\python.exe -m pytest`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

Completed DINOv2 setup and run status:

- Coding-agent config setup and all four DINOv2 full PRF visual ROI validation-selection runs are complete.
- New smoke config: `configs/experiments/neural_large_smoke/vit_small_patch14_dinov2_v1_flatten_pca_validation_selection_smoke.yaml`.
- Full configs:
  - `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_v1_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_v2_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_v3_flatten_pca_validation_selection_full.yaml`
  - `configs/experiments/neural_subj01_full/vit_small_patch14_dinov2_hv4_flatten_pca_validation_selection_full.yaml`
- These configs use DINOv2's existing `518 x 518` preprocessing, candidate layers `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`, `flatten_pca`, validation-only layer selection, selected-only final scoring, and RSA disabled for full-image-count PCA runs.
- The four full DINOv2 configs write to `C:/saliency_outputs/neural_subj01_full/...` instead of repo-local `outputs/` so the large temporary `feature_cache` is placed on a disk with enough free space. Summary regeneration must include these absolute C: output directories after the runs complete.
- After the first DINOv2 V1 full attempt, sklearn PCA hit a RAM allocation failure during the final selected-layer full-train PCA fit. The `flatten_pca` implementation was updated to fit PCA without sklearn's extra training-copy and to transform activations in batches, preserving the same PCA method while reducing peak memory. If a run crashes before cleanup, remove the stale `feature_cache` under that run directory before retrying.
- Full DINOv2 outputs passed artifact checks: required score/metadata/selection files exist for all four ROIs, no `feature_cache` remains, and `outputs/neural_roi_summary/` plus `outputs/paper_inspection_v1/` were regenerated with DINOv2 included.

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

Step 5: layer and pooling selection. **Status: implemented for code path, tests, configs, smoke, full four-ROI production runs, neural summary, and paper inspection pack.**

- `neural.selection.enabled` now evaluates candidate layers and feature-reduction settings on a validation split from the outer training images only.
- Selection artifacts record candidate layer, feature mode, PCA settings, selected alpha, validation score, selected candidate, split image IDs, and final test config.
- Final `encoding_scores.csv` and `encoding_target_scores.csv` contain only the selected candidate, so downstream summaries do not rank unselected test rows.
- First V1 smoke selected `blocks.3` from `blocks.0`, `blocks.3`, `blocks.6`, `blocks.9`, and `blocks.11`.
- Full four-ROI production selection selected `blocks.0` for V1 and `blocks.3` for V2, V3, and hV4.

Step 6: stronger model backbones. **Status: implemented for `vit_small_patch14_dinov2`.**

- Prioritize one SOTA-aligned backbone first: `vit_small_patch14_dinov2`.
- The DINOv2 smoke and four full validation-selection runs are complete and summarized.
- Do not expand to many more models until the learned spatial readout prototype clarifies whether feature-preserving spatial heads improve over full `flatten_pca`.
- Compare small versus large variants only after the matched DINOv2 full protocol is stable.

Step 7: learned spatial readout prototype. **Status: implemented for code path, tests, configs, real smoke run, full V1 run, neural summary, and paper inspection pack.**

- Added a PyTorch encoding-head path after the strong ridge baseline.
- Started with frozen `vit_small_patch14_dinov2` `blocks.3` features and a learned target-specific spatial pooling/readout head.
- The first head uses target-wise softmax spatial weights, target-wise channel weights, and target-wise bias.
- Uses an inner validation split from the outer training images for early stopping; final score rows are held-out outer test scores.
- Keeps the first learned-readout scope small: one subject, one ROI, one backbone.
- Full V1 learned readout improves DINOv2 `blocks.3` over `flatten_pca`: raw Pearson `0.648` versus `0.595`; valid-target noise-normalized score `0.762` versus `0.642`.
- The 180-epoch V1 diagnostic early-stopped at epoch `142` with best epoch `127`; improvement over the 100-epoch run was negligible, so V1 is accepted as plateaued for this head/config.
- This is the local analogue of SOTA components such as RetinaMapper, learned spatial pooling, voxel-specific heads, and subject-specific heads.

Step 8: subject and cortex expansion.

- After `subj01` learned-readout ROI coverage is credible, expand to additional available subjects.
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

1. Create matched learned-readout configs for DINOv2 `V2`, `V3`, and `hV4` using the accepted V1 180-epoch training settings.
2. Use fixed selected layers from the current validation-selected `flatten_pca` DINOv2 runs: `blocks.6` for `V2`, `V3`, and `hV4`.
3. Run the three ROI configs one at a time, inspect each `encoding_scores.csv` and `learned_readout_metadata.json`, and compare against the corresponding DINOv2 `flatten_pca` ROI baseline.
4. Regenerate `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/` after accepted ROI-expansion runs.
5. Only after learned-readout ROI coverage is stable, consider validation-only learned-layer selection, multi-layer learned readout, or voxel-specific spatial readout improvements.

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

Latest regenerated target scope: `241450` rows total; `241350` `benchmark_style_noise_normalized`; `100` `benchmark_style_non_noise_normalized` due to hV4 zero-ceiling targets across ROI500 and validation-selected full `flatten_pca` rows.

For the next neural method implementation, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```

For broader confidence after code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result after validation-selected summary regeneration: `180 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

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
