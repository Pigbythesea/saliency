# HMA Project Status And Next Steps

Updated: 2026-05-23

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
- Neural ROI500 encoding and RSA diagnostics on local Algonauts / NSD `subj01` visual ROIs.
- Paper-style inspection tables and figures that join corrected behavioral summaries with the current neural diagnostic summaries.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs:

- Corrected core behavioral aggregate: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Corrected SSL/VLM behavioral aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Neural ROI summary: `outputs/neural_roi_summary/`.
- Paper inspection pack: `outputs/paper_inspection_v1/README.md`.

## Scientific Boundary

The corrected behavioral layer is now usable for diagnostic paper-style analysis. It should still be framed carefully:

- NSS and AUC-style claims are valid only for rows with `fixation_protocol=points` or `fixation_protocol=task_points`.
- CC, SIM, KL, and related map-distribution metrics should be discussed separately from point-fixation metrics.
- DeepGaze and center bias are reference controls. Grad-CAM, gradients, rollout, and similar rows are explanation-map-to-fixation comparisons, not dedicated SOTA fixation-prediction models.
- COCO-Search18 is task-driven search and should not be pooled with free-viewing SALICON/CAT2000 as if all three datasets measure the same behavior.

The neural layer remains diagnostic:

- Current neural outputs are one-subject, ROI500, internal split, raw-correlation encoding/RSA summaries.
- They are not Algonauts leaderboard-equivalent scores.
- They are not noise-normalized and should not be compared numerically with Brain-Score, Algonauts, or NSD SOTA values.
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
- Input neural directories: `20`
- Encoding rows: `92`
- Encoding target rows: `222134`
- RSA rows: `92`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Benchmark-style encoding scope: `benchmark_style_non_noise_normalized`

Current raw neural ranking:

- Mean encoding leader: `deit_small_patch16_224`, mean raw correlation `0.261`.
- Mean RSA leader: `vit_base_patch16_224`, mean Spearman RSA `0.088`.
- `vit_small_patch14_dinov2` ranks second for both raw encoding and RSA.
- `resnet50` is now included in the regenerated summary and ranks fifth by mean raw encoding, fourth by mean RSA.
- Current ROI set: `V1`, `V2`, `V3`, `hV4` for `subj01`.

Current bridge readout:

- Overall behavior-to-encoding leader match rate: `0.079`.
- Overall behavior-to-RSA leader match rate: `0.000`.
- Internal-routing rows account for the observed encoding leader matches; class-localization and evidence-sensitivity rows do not match the encoding or RSA leader in the current descriptive table.

Interpretation:

- These rows are useful for checking data loading, activation extraction, ROI response alignment, rough layer ranking, and basic behavior-neural table generation.
- They are not yet strong enough for claims about model-brain SOTA, architecture superiority, or Algonauts-equivalent prediction quality.

## SSL / Multimodal Status

Current SSL/VLM behavioral rows are corrected and merged into the main behavioral aggregate.

SSL/multimodal candidate inventory:

- Path: `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`
- Dry-inspected compatible candidates: `8`
- Pretrained debug runs complete: `3`
- Complete pretrained debug candidates: `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, `resnet50_clip`
- Not yet run pretrained debug candidates: `vit_base_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_dinov3`, `vit_base_patch16_siglip_224`, `eva02_base_patch16_clip_224`

Do not expand this list before the neural evaluation metric is upgraded. More models will make the bridge look broader but will not fix the current scientific limitation.

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
- RSA over model and neural response RDMs.
- ROI500 summaries, model rankings, ROI winners, and descriptive behavior-neural bridge tables.

Reporting infrastructure:

- Corrected behavioral aggregate and merged SSL/VLM aggregate.
- Neural ROI summary tables.
- Paper inspection pack with behavior, neural, bridge, SSL/VLM candidate, and benchmark sanity tables.

## Superseded Or Historical Outputs

Historical note:

- `docs/v2_static2000_results_note.md` describes the pre-fix static2000 state from 2026-05-19. Its result interpretation is superseded.

Do not use for current claims:

- Any aggregate row without `fixation_protocol=points` or `task_points`.
- Any pre-2026-05-20 static2000 NSS/AUC result generated before point-fixation scoring was corrected.
- Old claims that center bias beat DeepGaze under the static2000 protocol.

Current corrected outputs have replaced those rows.

## Current Implementation Progress

Updated: 2026-05-23

Neural Benchmark-Equivalent Evaluation V1 is implemented for the current one-subject ROI500 diagnostic scope.

Implemented in this session:

- Per-target benchmark-style encoding rows are now written to `encoding_target_scores.csv` by `scripts/run_neural_alignment.py`.
- Layer-level `encoding_scores.csv` remains backward compatible and now includes metric-scope, selected-alpha, alpha-selection-mode, split-seed, and feature-reduction metadata.
- Per-target scores include raw Pearson `r`, `r2_score_from_r`, ordinary prediction `r2`, optional noise-ceiling fields, noise-normalized score when possible, and variance-validity flags.
- Runs without noise-ceiling metadata are explicitly labeled `benchmark_style_non_noise_normalized`.
- Optional `neural.ridge_alphas` now enables deterministic inner-validation ridge-alpha selection per layer; configs without `ridge_alphas` keep fixed `neural.ridge_alpha`.
- `summarize_neural_roi_results` now loads optional `encoding_target_scores.csv`, writes `combined_encoding_target_scores.csv` when available, and keeps old neural output directories compatible.

Verification run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_nsd_algonauts_dataset.py
```

Result: `25 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

Full regeneration status:

- Full ROI500 outputs were regenerated for `outputs/neural_roi500/` and `outputs/neural_roi500_ssl/`.
- Checked neural output directories: `20`.
- Missing `encoding_target_scores.csv`: none.
- Missing `metadata.json`: none.
- Missing `rsa_scores.csv`: none.
- Combined per-target benchmark rows: `222134`.
- Per-target metric scopes: all `benchmark_style_non_noise_normalized`.
- Per-target variance flags: all `valid_prediction_variance=true` and `valid_target_variance=true`.
- `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/` were regenerated from the corrected behavioral aggregate and refreshed neural outputs.

Full verification:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Result: `156 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

## Next Concrete Milestone

Priority: **Neural Reliability / Noise-Ceiling Metadata V1**.

Do this before adding more SSL/VLM models, Brain-Score integration, CKA, adaptive token pruning, foveation, scanpaths, or video. The behavioral side is clean enough and the neural benchmark-style reporting path now works; the blocker is that scores are still non-noise-normalized.

Goal:

- Discover whether local Algonauts / NSD files include reliability, noise-ceiling, repeat-trial, or split metadata that can be linked to each ROI response target.
- If available, add manifest columns or a sidecar loader for target-level noise ceilings.
- Feed target-level ceilings into `benchmark_encoding_target_scores`.
- Regenerate a small smoke run, then the full ROI500 summary, and verify that target rows switch from `benchmark_style_non_noise_normalized` to `benchmark_style_noise_normalized` where ceilings exist.
- If local ceiling metadata is not available, document that explicitly and move to the next highest-value neural method upgrade: cross-validated alpha configs for full ROI500 plus feature dimensionality control.

## Later Milestones

After Neural Benchmark-Equivalent Evaluation V1:

1. Add reliability/noise-ceiling metadata if available from local Algonauts/NSD resources.
2. Expand beyond `subj01` before making subject-general claims.
3. Add feature dimensionality control such as training-only PCA.
4. Add confidence intervals or bootstrap uncertainty over images, targets, and eventually subjects.
5. Run full ROI500 for the strongest SSL/VLM candidates, starting with CLIP or SigLIP only after the neural metric is stable.
6. Add efficiency profiles and alignment-per-compute summaries back into the inspection pack.
7. Add stronger transformer attribution methods such as Chefer-style attribution or AttnLRP before making claims about transformer attention.
8. Add Brain-Score or Brain-Score-style external comparisons.
9. Add CKA or additional representational geometry metrics alongside RSA.
10. Add adaptive/selective-computation models such as token pruning, foveation, adaptive patch selection, or glimpse-style models.
11. Consider scanpath or video analysis only after static-image behavioral and neural claims are stable.

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
- `scripts/summarize_neural_roi_results.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

This edit is documentation-only. No tests are required just to consume this file.

For the next neural metric implementation, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_nsd_algonauts_dataset.py
```

For broader confidence after code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest
```
