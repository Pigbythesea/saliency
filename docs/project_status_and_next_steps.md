# HMA Project Status And Next Steps

Updated: 2026-05-31

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
- `paper1_cross_axis_alignment_roadmap.md`: current publication roadmap for Paper 1. It reframes the project as a cross-axis dissociation study across fixation alignment, neural encoding, representational geometry, and efficiency.
- `paper1_literaturereview.md`: current literature review for Paper 1. It raises the required controls around dataset bias, scanpath/task specificity, subject variability, encoding reliability, representational-geometry metrics, and transformer attribution.
- `Literature Review and Research Redesign for the Human-Like Adaptive Visual Attention Project.md`: argues the project should become a multi-axis NeuroAI alignment study, not a saliency-map leaderboard.
- `Deep Research Assessment of the Human-Machine Visual Alignment Project.md`: emphasizes the publishable question as convergence versus dissociation among fixation alignment, neural predictivity, representational geometry, and efficiency.
- `Zhang_Zihuan_zzhan330_proposal.docx`: original proposal; defines behavioral saliency, neural encoding, RSA, Brain-Score-style comparison, and compute efficiency as the core axes.
- `Comparing Human and Machine Visual Saliency_ A Comprehensive Review.pdf`: reinforces that fixation prediction requires strong controls such as center bias, DeepGaze-class references, point-based NSS/AUC, and separate treatment of free-viewing versus task-driven viewing.
- `__Attention and Saliency Map Extraction in Visual AI Models_ A Comprehensive Review__.pdf`: reinforces that gradients, CAMs, attention rollout, perturbation maps, LRP-style methods, and transformer attribution are different explanation objects and should not be collapsed into one "attention" score.
- `v2_static2000_results_note.md`: historical only. It is superseded by corrected point-fixation reruns and should not guide current planning.

## Current Snapshot

The repository now implements three active layers:

- Behavioral saliency / fixation benchmarking on SALICON, CAT2000, and COCO-Search18.
- Neural encoding and legacy ROI500 RSA diagnostics on local Algonauts / NSD `subj01` visual ROIs, including full-image-count `flatten_pca` PRF ROI baselines and full-image-count learned spatial readout for DINOv2.
- Paper-style inspection tables and figures that join corrected behavioral summaries with the matched full-image neural panel, matched cross-level correlation/regression outputs, and legacy mixed-scope diagnostics.

The repository does not yet implement the missing matched full-image representational-geometry axis. Current RSA rows are not accepted as headline geometry evidence for Paper 1.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs:

- Corrected core behavioral aggregate: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Corrected SSL/VLM behavioral aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Neural ROI summary with full `flatten_pca` and learned-readout rows included: `outputs/neural_roi_summary/`.
- Matched full-image `flatten_pca` panel outputs and audit: `outputs/neural_roi_summary/matched_full_panel_model_rankings.csv`, `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`.
- Matched cross-level analysis outputs: `outputs/neural_roi_summary/matched_cross_level_observations.csv`, `outputs/neural_roi_summary/matched_cross_level_correlations.csv`.
- Paper inspection pack regenerated from the refreshed neural summary: `outputs/paper_inspection_v1/README.md`.

## Scientific Boundary

The corrected behavioral layer is now usable for diagnostic paper-style analysis. It should still be framed carefully:

- NSS and AUC-style claims are valid only for rows with `fixation_protocol=points` or `fixation_protocol=task_points`.
- CC, SIM, KL, and related map-distribution metrics should be discussed separately from point-fixation metrics.
- DeepGaze and center bias are reference controls. Grad-CAM, gradients, rollout, and similar rows are explanation-map-to-fixation comparisons, not dedicated SOTA fixation-prediction models.
- COCO-Search18 is task-driven search and should not be pooled with free-viewing SALICON/CAT2000 as if all three datasets measure the same behavior.

The neural layer is now a stronger local baseline, but still not a leaderboard result:

- Current neural outputs are one-subject, internal-split `subj01` results. They include older ROI500 spatial-mean diagnostics, a complete six-model matched full-image-count PRF visual ROI `flatten_pca` panel, and four full-image-count learned-readout provenance rows for DINOv2.
- They are not Algonauts leaderboard-equivalent scores because the official challenge averages held-out visual-cortex vertices across subjects and hemispheres.
- The matched `flatten_pca` panel is the primary evidence for cross-model neural comparisons. The four-ROI DINOv2 learned-readout rows are the strongest local single-backbone method result and should be treated as method provenance, not as matched-panel ranking rows.
- The matched cross-level correlation tables are now the primary descriptive cross-axis evidence, but they are still small-n one-subject model-level analyses, not causal tests.
- The legacy bridge and leader-overlap tables remain descriptive continuity diagnostics.

Paper 1 should be held to these publication gates before strong top-venue claims:

- Full-image matched representational geometry must exist for the same six-model x four-ROI panel.
- Cross-axis results must report uncertainty and sensitivity, especially bootstrap intervals, leave-one-model-out behavior, and exact model counts.
- Claims must be framed as descriptive convergence/dissociation, not causal attention intervention.
- At least one nontrivial dissociation or convergence pattern must survive sensitivity checks; otherwise Paper 1 should be framed as a measurement framework, workshop paper, thesis chapter, or methods note.

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
- The behavioral layer is strong enough to serve as one axis in the broader alignment study. It should not be expanded into a larger leaderboard before the matched geometry and uncertainty axes are implemented.
- Later behavioral upgrades should prioritize human/interobserver ceilings, DeepGaze MSDB or another modern fixation reference, and a task-specific COCO-Search18 baseline. Broad scanpath/video expansion belongs after Paper 1 unless the paper explicitly shifts away from static-image dissociation.

## Current Neural Status

Current neural summary:

- Path: `outputs/neural_roi_summary/`
- Input neural directories: `48`
- Encoding rows: `120`
- Encoding target rows: `289740`
- RSA rows: `92`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Feature-reduction rows: `92` spatial-mean diagnostic rows, `24` validation-selected full-image-count matched-panel `flatten_pca` rows, and `4` full-image-count learned spatial readout rows.
- Matched cross-level analysis rows: `315` correlation/regression groups, with `210` complete and `105` marked `insufficient_models`.
- Matched cross-level datasets remain separate: `105` groups each for SALICON, CAT2000, and COCO-Search18.
- New paper-pack artifacts: `outputs/paper_inspection_v1/tables/table9_matched_cross_level_correlations.md` and `outputs/paper_inspection_v1/figures/figure5_matched_cross_level_correlations.png`.
- Benchmark-style per-target encoding scope: mixed because four hV4 targets have `noise_ceiling=0.0`; `289620` rows are `benchmark_style_noise_normalized` and `120` rows are intentionally left `benchmark_style_non_noise_normalized`.
- Matched-panel reporting is now implemented separately from the mixed-scope neural ranking. The full-image-count validation-selected `flatten_pca` panel is now complete for all six planned model families across all four PRF visual ROIs: `resnet50`, `convnext_tiny`, `deit_small_patch16_224`, `vit_base_patch16_224`, `vit_small_patch14_dinov2`, and `vit_base_patch16_clip_224`.
- Completed `resnet50` full matched-panel results:
  - V1 selected `layer2`, `ridge_alpha=0.001`, mean raw Pearson `0.585`, mean valid-target noise-normalized score `0.622`, `2973` valid targets.
  - V2 selected `layer2`, `ridge_alpha=100000.0`, mean raw Pearson `0.562`, mean valid-target noise-normalized score `0.600`, `2936` valid targets.
  - V3 selected `layer2`, `ridge_alpha=100000.0`, mean raw Pearson `0.532`, mean valid-target noise-normalized score `0.561`, `2453` valid targets.
  - hV4 selected `layer3`, `ridge_alpha=1000000.0`, mean raw Pearson `0.466`, mean valid-target noise-normalized score `0.540`, `1292` valid positive-ceiling targets plus `4` zero-ceiling targets.
- Completed `convnext_tiny` full matched-panel results: V1 `0.599`, V2 `0.527`, V3 `0.474`, hV4 `0.438` mean valid-target noise-normalized score; mean across four ROIs `0.510`.
- Completed `vit_base_patch16_224` full matched-panel results: V1 `0.578`, V2 `0.534`, V3 `0.528`, hV4 `0.495` mean valid-target noise-normalized score; mean across four ROIs `0.534`.
- Completed `vit_base_patch16_clip_224` full matched-panel results: V1 `0.626`, V2 `0.607`, V3 `0.577`, hV4 `0.513` mean valid-target noise-normalized score; mean across four ROIs `0.581`.
- Final matched-panel mean noise-normalized ranking: `vit_small_patch14_dinov2` `0.591`, `vit_base_patch16_clip_224` `0.581`, `resnet50` `0.581`, `deit_small_patch16_224` `0.562`, `vit_base_patch16_224` `0.534`, `convnext_tiny` `0.510`.
- Artifact audit path: `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`; current status is `24` complete cells and `0` missing/skipped cells.
- No stale `feature_cache` directories remain under the matched full-image-count output directories.

Current noise-normalized neural ranking:

- Mean valid-target noise-normalized encoding leader: `vit_small_patch14_dinov2`, mean `0.681` (`68.06` on x100 scale).
- In the mixed-scope ranking, `vit_base_patch16_clip_224` now ranks second (`0.581`), `resnet50` third (`0.581`), `deit_small_patch16_224` fourth (`0.562`), `vit_base_patch16_224` fifth (`0.534`), and `convnext_tiny` sixth (`0.510`).
- The current noise-normalized encoding leader is the same model as the raw Pearson leader.
- Each matched-panel model ranking row aggregates `9654` valid positive-ceiling targets and excludes `4` zero-ceiling hV4 targets from noise-normalized aggregates.
- The mixed-scope `vit_small_patch14_dinov2` ranking is driven by learned spatial readout rows for all four PRF visual ROIs. The separate matched-panel ranking uses only validation-selected full-image-count `flatten_pca` rows for all six model families.

Current raw neural ranking:

- Mean encoding leader: `vit_small_patch14_dinov2`, mean raw correlation `0.581`.
- Mean RSA leader: `vit_base_patch16_224`, mean Spearman RSA `0.088`.
- In the matched-panel raw encoding ranking, `vit_small_patch14_dinov2` remains first (`0.541`), followed by `vit_base_patch16_clip_224` (`0.538`), `resnet50` (`0.536`), `deit_small_patch16_224` (`0.528`), `vit_base_patch16_224` (`0.513`), and `convnext_tiny` (`0.502`).
- RSA rankings remain ROI500-scale and should not be interpreted as matched full-image representational-geometry evidence.
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
- V2 fixed `blocks.6`: `2936` valid targets, mean raw Pearson `0.607`, mean valid-target noise-normalized score `0.700` (`69.99` x100), median raw Pearson `0.654`, median noise-normalized score `0.735`.
- V3 fixed `blocks.6`: `2453` valid targets, mean raw Pearson `0.583`, mean valid-target noise-normalized score `0.674` (`67.37` x100), median raw Pearson `0.611`, median noise-normalized score `0.703`.
- hV4 fixed `blocks.6`: `1292` valid positive-ceiling targets plus `4` zero-ceiling targets, mean raw Pearson `0.488`, mean valid-target noise-normalized score `0.586` (`58.62` x100), median raw Pearson `0.518`, median noise-normalized score `0.588`.
- Learned readout improves over validation-selected DINOv2 `flatten_pca` for all four ROIs: V1 `+0.053` raw / `+0.120` noise-normalized, V2 `+0.038` raw / `+0.086` noise-normalized, V3 `+0.038` raw / `+0.083` noise-normalized, hV4 `+0.031` raw / `+0.069` noise-normalized.
- Training used `6298` inner-train and `1575` validation images inside the `7873` outer-train images; final scoring used `1968` held-out test images.
- Best epochs / early stopping: V1 best `127` and stopped at `142`; V2 best `65` and stopped at `80`; V3 best `63` and stopped at `78`; hV4 best `59` and stopped at `74`.
- Output paths: `C:/saliency_outputs/neural_subj01_full/vit_small_patch14_dinov2_{v1,v2,v3,hv4}_learned_spatial_readout_full_180ep/`.
- The learned rows are included in regenerated `outputs/neural_roi_summary/` and `outputs/paper_inspection_v1/`.

Matched cross-level readout:

- The new matched cross-level table uses only the six-model full-image `flatten_pca` panel and excludes learned-readout, ROI500, spatial-mean, and rejected voxel-specific rows.
- Across-ROI mean NSS versus mean noise-normalized encoding Spearman correlations:
  - CAT2000 attention rollout: `0.400` across `4` matched transformer rows; OLS noise-normalized R2 `0.588`.
  - CAT2000 vanilla gradient: `0.486` across `6` matched rows; OLS noise-normalized R2 `0.132`.
  - COCO-Search18 attention rollout: `0.800` across `4` matched transformer rows; OLS noise-normalized R2 `0.281`.
  - COCO-Search18 vanilla gradient: `0.314` across `6` matched rows; OLS noise-normalized R2 `0.061`.
  - SALICON attention rollout: `0.000` across `4` matched transformer rows; OLS noise-normalized R2 `0.020`.
  - SALICON vanilla gradient: `0.486` across `6` matched rows; OLS noise-normalized R2 `0.138`.
- Grad-CAM across-ROI NSS groups are marked `insufficient_models` because only `resnet50` and `convnext_tiny` have matched Grad-CAM behavioral rows in the six-model panel.
- Lower-is-better behavioral metrics such as KL are sign-aligned before correlation while retaining the raw behavioral mean in the observation table.

Interpretation:

- The project has moved from weak ROI500 spatial-mean neural diagnostics to a complete full-image-count matched `flatten_pca` panel for six model families, plus a stronger DINOv2 learned-readout method result.
- The matched-panel ranking is now the accepted basis for cross-model neural comparisons: `vit_small_patch14_dinov2` first, `vit_base_patch16_clip_224` second, `resnet50` third, `deit_small_patch16_224` fourth, `vit_base_patch16_224` fifth, and `convnext_tiny` sixth by mean valid-target noise-normalized score.
- The DINOv2 learned spatial readout materially improves all four PRF visual ROIs over DINOv2 `flatten_pca`, but it is not method-matched to the other backbones and should not be used as the primary cross-model row.
- The previous test-set feedback risk for layer choice has been addressed for the current one-subject PRF visual ROI baselines by validation-only layer selection.

## Global Direction Rationale

The project direction remains a multi-axis NeuroAI alignment study, not a saliency-map leaderboard or a local Algonauts score chase. The target scientific question is now explicit:

> Do models that better match human fixation behavior also better align with human visual cortex in encoding and representational geometry, or do these alignment axes dissociate across architecture, attribution family, ROI, and viewing regime?

The current behavioral layer is mature enough to serve as the fixation-alignment axis, but it is not the main claim. The current neural layer now has a complete matched full-image-count local encoding panel for six model families across `subj01` PRF visual ROIs, and the first matched behavior-versus-encoding correlation/regression tables are implemented. It is still local and one-subject, but it is now suitable for the first descriptive convergence/dissociation analysis. The failed full V1 voxel-specific decision run closes the current DINOv2-only readout search. The next implementation work should add a matched representational-geometry axis instead of adding more single-backbone capacity.

The immediate engineering priority is now scalable representational geometry for the same matched full-image panel. The voxel-specific low-rank branch was scientifically useful as a control: it improved the small smoke but failed on full V1, so it should be treated as a rejected DINOv2-only readout variant, not as a protocol to expand. The project should now optimize validity of the central convergence/dissociation matrix, not the absolute DINOv2 local score.

The global milestone order for Paper 1 is:

1. Freeze the accepted behavioral and matched neural encoding assets.
2. Add matched full-image representational geometry for the same six-model, four-ROI panel.
3. Add uncertainty and sensitivity analyses across images, targets, geometry subsets, ROIs, and model leave-one-out splits.
4. Add essential controls: human/interobserver fixation ceilings if data allow, DeepGaze MSDB or comparable modern fixation reference, task-specific COCO-Search18 baseline if feasible, and Chefer/AttnLRP-style transformer attribution before attention-specific claims.
5. Add subject robustness before broad model-zoo expansion. If full subject expansion is too expensive, use a reduced confirmatory panel with DINOv2, CLIP ViT, ResNet-50, and one weaker transformer baseline.
6. Add efficiency profiles only after the central behavior-encoding-geometry matrix is stable.
7. Split causal adaptive-attention interventions into Paper 2 unless Paper 1 exposes a very clear intervention target.

Broad model-zoo expansion, Brain-Score-style positioning, scanpath/video analysis, and alignment-per-compute claims should wait until the matched behavior, encoding, and geometry matrix has uncertainty estimates and a defensible dissociation story.

Current methodological gap to SOTA:

- The current strongest local results use a learned target-wise spatial readout for all four PRF visual ROIs, but only for one subject and one backbone. A low-rank voxel-specific DINOv2 variant was tested and rejected on full V1, so stronger RetinaMapper-style heads remain a SOTA gap but are no longer the next local milestone.
- Current full-image-count runs cover `subj01` PRF visual ROIs only. SOTA scores use broader visual-cortex vertex sets across subjects and hemispheres.
- Current ridge alpha selection is per-layer/ROI, not per-target. Per-target alpha and richer voxel-specific heads may improve later, but should wait until after matched representational geometry and uncertainty estimates are working.
- Current reporting keeps a mixed-scope ranking for continuity, but matched-panel tables now separately restrict to `subj01`, `9841` images, `flatten_pca`, validation-selected final rows, and PRF ROIs `V1`/`V2`/`V3`/`hV4`. Cross-model neural claims should use the matched-panel tables, not the mixed-scope ranking.
- Current scoring now supports validation-selected single-layer `flatten_pca` and fixed-layer learned spatial readout. SOTA methods go further with multi-layer fusion, learned layer selection, subject-specific heads, voxel-specific spatial readouts, and ensembles.
- Current RSA is still ROI500-scale for most reporting; the strongest full-image-count encoding rows do not yet have matched scalable RSA, CKA, or other representational-geometry metrics.
- Current matched cross-level tables now report model-level Spearman correlations and simple OLS regressions between corrected behavioral rows and the matched neural encoding panel.
- Legacy bridge tables still report leader overlap only and should be treated as continuity diagnostics, not the main cross-axis evidence.
- Current behavioral SOTA controls include DeepGaze IIE and center bias, but do not yet include DeepGaze MSDB, scanpath-level references, human inter-observer ceilings, or task-trained COCO-Search18 search models.
- Current transformer attention evidence relies mostly on gradients and attention rollout. Add Chefer-style attribution or AttnLRP before making claims about transformer attention as human-like evidence.
- Current cross-axis correlations are descriptive because they use `n=6` matched models or `n=4` transformer-only rows. Treat p-values as weak, report exact `n`, and prefer rank stability, sign consistency, permutation checks, and leave-one-model-out sensitivity.

Relevant SOTA references:

- Algonauts 2023 challenge evaluation: `https://algonautsproject.com/2023/index.html`
- NSD dataset paper: `https://www.nature.com/articles/s41593-021-00962-x`
- Brain-Score platform: `https://www.brain-score.org/`
- DeepGaze III scanpath modeling: `https://bethgelab.org/publication/2022_04_kummerer/`
- DeepGaze MSDB / saliency dataset bias: `https://openaccess.thecvf.com/content/ICCV2025/html/Kummerer_Modeling_Saliency_Dataset_Bias_ICCV_2025_paper.html`
- Memory Encoding Model: DINOv2 backbone, voxel-specific RetinaMapper, LayerSelector, memory/task/subject conditioning, random-ROI ensemble; single model around `66.8`, ensemble around `70.8`: `https://github.com/huzeyann/MemoryEncodingModel`
- UARK-UAlbany solution: multi-subject pretraining, subject fine-tuning, ConvNeXt-style backbones, SmoothL1/Pearson/noise-normalized losses, weighted ensemble; baseline around `54.21`, ensemble around `61.56`: `https://arxiv.org/pdf/2308.00262`
- BlobGPT: EVA02 trunk, multi-layer feature tensors, learned spatial pooling, shared and subject-specific transforms, fMRI PCA embedding, end-to-end fine-tuning; score around `60.2`: `https://arxiv.org/pdf/2308.02351`
- Scaling-law report: model size, fMRI sample size, layer/kernel selection, cross-validated ridge, and model averaging materially improve scores: `https://arxiv.org/pdf/2308.00678`
- Controlled model-brain comparison / inductive biases: `https://www.nature.com/articles/s41467-024-53147-y`
- Neural encoding with visual attention: `https://neuroml.wiki/publication/neurips2020/`
- AttnLRP transformer attribution: `https://icml.cc/virtual/2024/poster/33480`

## SSL / Multimodal Status

Current SSL/VLM behavioral rows are corrected and merged into the main behavioral aggregate.

SSL/multimodal candidate inventory:

- Path: `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`
- Dry-inspected compatible candidates: `8`
- Pretrained debug runs complete: `3`
- Complete pretrained debug candidates: `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, `resnet50_clip`
- Not yet run pretrained debug candidates: `vit_base_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_dinov3`, `vit_base_patch16_siglip_224`, `eva02_base_patch16_clip_224`

The voxel-specific readout decision is complete and the matched neural panel is complete. Do not expand this inventory again before matched full-image geometry, uncertainty, and the first dissociation analysis are implemented and interpreted. The next model-family comparison should remain narrow and methodologically matched, not a broad model zoo.

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
- ROI500 summaries, full-image-count PRF ROI summaries, model rankings, ROI winners, matched cross-level behavior-neural tables, and legacy descriptive behavior-neural bridge tables.

Reporting infrastructure:

- Corrected behavioral aggregate and merged SSL/VLM aggregate.
- Neural ROI summary tables.
- Paper inspection pack with behavior, neural, matched cross-level, legacy bridge, SSL/VLM candidate, benchmark sanity tables, and an academic SOTA context section comparing the current figures against MIT/Tuebingen saliency, DeepGaze IIE SALICON, COCO-Search18 task-search, and Algonauts 2023 evaluation references.
- Paper inspection README now explicitly distinguishes mixed-scope diagnostics from the complete six-model matched full-image-count PRF ROI `flatten_pca` panel, and includes the four DINOv2 learned spatial readout rows only as method-provenance context.

## Superseded Or Historical Outputs

Historical note:

- `docs/v2_static2000_results_note.md` describes the pre-fix static2000 state from 2026-05-19. Its result interpretation is superseded.

Do not use for current claims:

- Any aggregate row without `fixation_protocol=points` or `task_points`.
- Any pre-2026-05-20 static2000 NSS/AUC result generated before point-fixation scoring was corrected.
- Old claims that center bias beat DeepGaze under the static2000 protocol.
- Legacy behavior-to-encoding or behavior-to-RSA leader-overlap rates as evidence for the main paper claim. They are continuity diagnostics only and are superseded for cross-axis reasoning by matched model-level tables.
- ROI500 RSA rankings as headline representational-geometry evidence. Full-image matched CKA/subset RSA is required before making a geometry claim.
- Any mixed-scope neural ranking as a cross-model headline. Use matched full-image `flatten_pca` rows for cross-model claims and learned-readout rows only as method provenance.
- `C:/saliency_outputs/neural_subj01_full/vit_small_patch14_dinov2_v1_learned_spatial_readout_layer_selection_full_180ep/` should be treated as diagnostic provenance only. It selected the same layer and produced the same held-out score as the accepted fixed-layer V1 learned-readout run, so it should not be added as an extra accepted baseline row in the current neural summary.

Current corrected outputs have replaced those rows.

## Current Implementation Progress

Updated: 2026-05-31

Benchmark-Style Neural Scoring V1 is implemented for the current one-subject local neural scope. It is reliability-aware and Algonauts-inspired, but it is not official leaderboard-equivalent because subject, split, target scope, and cortex coverage differ.

Neural Reliability / Noise-Ceiling Metadata V1 is implemented for `subj01` PRF visual ROI data. The earlier local audit found no ceiling files in the Algonauts subset, so NSD `ncsnr` files were added from the full NSD release and converted into target-level ROI noise-ceiling vectors.

Large NSD/Algonauts Manifest And Run Configs V1 is implemented for `subj01` PRF visual ROIs. The project has full-image-count manifest plumbing for `V1`, `V2`, `V3`, and `hV4`.

Feature Representation Upgrade V1 is implemented. The neural runner supports train-only `flatten_pca` for flattened activation tensors, writes per-layer feature-reduction metadata, and uses batch transforms to avoid keeping full raw tensors in memory.

Validation-Only Layer/Pooling Selection V1 is implemented. The neural runner supports `neural.selection.enabled`, validation-only candidate scoring over layer/feature-reduction settings, selection artifacts, and final held-out test scoring for only the selected candidate.

Matched Full-Image-Count Neural Panel V1 is complete. All `24` expected cells are complete for `resnet50`, `convnext_tiny`, `deit_small_patch16_224`, `vit_base_patch16_224`, `vit_small_patch14_dinov2`, and `vit_base_patch16_clip_224` across `V1`, `V2`, `V3`, and `hV4`.

Benchmark-equivalent implementation:

- Per-target benchmark-style encoding rows are written to `encoding_target_scores.csv` by `scripts/run_neural_alignment.py`.
- Layer-level `encoding_scores.csv` remains backward compatible and includes metric-scope, selected-alpha, alpha-selection-mode, split-seed, feature-reduction metadata, and valid-target noise-normalized aggregate fields.
- Per-target scores include raw Pearson `r`, `r2_score_from_r`, ordinary prediction `r2`, optional noise-ceiling fields, `valid_noise_ceiling`, noise-normalized score when possible, and variance-validity flags.
- Runs without positive noise ceilings are explicitly labeled `benchmark_style_non_noise_normalized`; runs with attached NSD-derived positive ROI ceilings are labeled `benchmark_style_noise_normalized`.
- Optional `neural.ridge_alphas` enables deterministic inner-validation ridge-alpha selection per layer; configs without `ridge_alphas` keep fixed `neural.ridge_alpha`.
- `summarize_neural_roi_results` loads optional `encoding_target_scores.csv`, writes `combined_encoding_target_scores.csv` when available, derives noise-normalized layer/model aggregates from target rows, and keeps old neural output directories compatible.
- Matched-panel outputs are written separately from the mixed-scope ranking so cross-model claims can use only matched rows.

Full regeneration status:

- Neural summary path: `outputs/neural_roi_summary/`.
- Paper inspection pack path: `outputs/paper_inspection_v1/`.
- Checked neural output directories: `48`.
- Encoding rows: `120`.
- Encoding target rows: `289740`.
- RSA rows: `92`.
- Combined per-target metric scopes: `289620` rows with `benchmark_style_noise_normalized`, `120` rows with `benchmark_style_non_noise_normalized`.
- The non-normalized rows are hV4 targets with `noise_ceiling=0.0`; these rows are intentionally not divided by zero.
- Feature-reduction rows: `92` spatial-mean diagnostic rows, `24` validation-selected full-image-count matched-panel `flatten_pca` rows, and `4` full-image-count learned spatial readout rows.
- `outputs/neural_roi_summary/neural_model_rankings.csv` keeps the mixed-scope ranking for continuity and ranks DINOv2 learned spatial readout first by mean valid-target noise-normalized score.
- `outputs/neural_roi_summary/matched_full_panel_model_rankings.csv` is the accepted cross-model neural ranking table for this milestone.
- `outputs/neural_roi_summary/matched_cross_level_observations.csv` and `outputs/neural_roi_summary/matched_cross_level_correlations.csv` are the accepted matched cross-axis evidence tables for behavior-versus-encoding analysis.
- `outputs/paper_inspection_v1/README.md` uses the complete matched panel for the neural encoding headline, surfaces matched cross-level correlations, and keeps DINOv2 learned-readout rows as method provenance.

Matched panel final ranking:

- `vit_small_patch14_dinov2`: mean raw Pearson `0.541`, mean valid-target noise-normalized score `0.591`.
- `vit_base_patch16_clip_224`: mean raw Pearson `0.538`, mean valid-target noise-normalized score `0.581`.
- `resnet50`: mean raw Pearson `0.536`, mean valid-target noise-normalized score `0.581`.
- `deit_small_patch16_224`: mean raw Pearson `0.528`, mean valid-target noise-normalized score `0.562`.
- `vit_base_patch16_224`: mean raw Pearson `0.513`, mean valid-target noise-normalized score `0.534`.
- `convnext_tiny`: mean raw Pearson `0.502`, mean valid-target noise-normalized score `0.510`.

Matched panel audit:

- Audit artifact: `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`.
- Current status: `24` complete cells, `0` missing cells, `0` incomplete cells, and `0` explicitly skipped cells.
- Each complete cell has `encoding_scores.csv`, `encoding_target_scores.csv`, `metadata.json`, `feature_reduction_metadata.json`, `selection_candidates.csv`, and `selection_artifact.json`.
- All matched cells use `feature_reduction=flatten_pca`, `num_items=9841`, train-only PCA metadata, validation-selected final rows, and RSA disabled.
- No stale `feature_cache` directories remain under matched full-image-count output directories.

Large/full manifest and config status:

- Full PRF visual ROI manifest: `data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv`.
- Full manifest rows: `39364` (`9841` images x `4` ROIs).
- ROI response sidecars exist for `9841` images each under `data/raw/nsd_algonauts/subj01/responses/V1`, `V2`, `V3`, and `hV4`.
- Full manifest columns include `noise_ceiling_path`, `noise_ceiling_values`, and `noise_ceiling_source`.
- Attached noise-ceiling source: `nsd_ncsnr_mgh_n_trials_3`.
- Matched full-panel config generation is implemented in `scripts/create_neural_roi500_configs.py` via `--full-subject --name-suffix flatten_pca_validation_selection_full` or `--flatten-pca-validation-selection`.
- Audit helper: `scripts/audit_matched_neural_panel.py`.
- Full DINOv2 outputs live under `C:/saliency_outputs/neural_subj01_full/...`; summary regeneration must include these absolute C: directories.
- Full production `flatten_pca` and learned-readout configs intentionally disable RSA to avoid full `9841 x 9841` RDM allocation.

Neural reliability / noise-ceiling implementation:

- `NSDAlgonautsDataset` accepts optional manifest columns `noise_ceiling_path`, `noise_ceiling_values`, and `noise_ceiling_source`.
- Dataset items emit `metadata.noise_ceiling` when a target-level sidecar or inline vector is available.
- `scripts/run_neural_alignment.py` / `run_neural_alignment` accepts `neural.noise_ceiling_key`, defaulting to `noise_ceiling`.
- The runner validates that noise-ceiling vectors are target-level, complete across all run items, and consistent across items before passing them into `benchmark_encoding_target_scores`.
- Run metadata records `noise_ceiling_available`, `noise_ceiling_key`, and `noise_ceiling_source`.
- Audit artifact: `outputs/neural_reliability_audit_v1/README.md`.
- Audit result: `0` candidate reliability / noise-ceiling / repeat-trial / split-half files found under `data/raw/nsd_algonauts`; full NSD `ncsnr` files were therefore added under `data/raw/nsd_full/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/`.
- `scripts/create_nsd_noise_ceiling_manifest.py` converts `lh.ncsnr.mgh` and `rh.ncsnr.mgh` into ROI target-level noise ceilings using `NC = ncsnr^2 / (ncsnr^2 + 1 / n_trials)`.
- `n_trials=3` was verified from `data/raw/nsd_full/experiments/nsd/nsd_expdesign.mat` for the current `subj01` design.
- ROI ceiling files were written to `data/raw/nsd_algonauts/subj01/noise_ceilings/`: `V1.npy`, `V2.npy`, `V3.npy`, and `hV4.npy`.

Learned spatial readout status:

- Learned Spatial Readout V1 is implemented in `src/hma/neural/learned_readout.py` and exposed through `neural.encoding_method: learned_spatial_readout`.
- The accepted DINOv2 fixed-layer learned-readout protocol covers `V1`, `V2`, `V3`, and `hV4` for `vit_small_patch14_dinov2`.
- Learned-readout rows use `feature_reduction=learned_spatial_readout`, blank `selected_ridge_alpha`, and `alpha_selection_mode=early_stopping_validation`.
- Output compatibility is implemented for `encoding_scores.csv`, `encoding_target_scores.csv`, `metadata.json`, `feature_reduction_metadata.json`, and `learned_readout_metadata.json`.
- Full four-ROI learned-readout results improve over corresponding DINOv2 `flatten_pca` baselines for every ROI: V1 `0.762` versus `0.642`, V2 `0.700` versus `0.614`, V3 `0.674` versus `0.591`, and hV4 `0.586` versus `0.517` by mean valid-target noise-normalized score.
- V1 learned-layer selection selected the same `blocks.3` layer and exactly matched the accepted fixed-layer V1 held-out score, so V2/V3/hV4 learned-layer selection is not planned.
- Multi-layer learned-readout smoke was wired but inconclusive; do not prioritize a full multi-layer run now.
- Voxel-specific low-rank learned-readout smoke improved slightly, but the full V1 decision run was worse than the accepted fixed-layer learned readout. Treat `voxel_specific_spatial_readout` outputs as rejected method provenance and do not include them in accepted summaries.

Scalable representational geometry status:

- Frozen Paper 1 scope artifact: `configs/paper1_config.yaml`.
- Geometry implementation: `src/hma/neural/geometry.py` exposes scalable `linear_cka` and deterministic `subset_rsa` helpers. `linear_cka` avoids full `9841 x 9841` image-kernel/RDM allocation by using centered cross-products.
- Posthoc generation script: `scripts/compute_matched_geometry.py`.
- Future-run compatibility: `neural.geometry.enabled` is supported in `src/hma/experiments/neural_alignment.py` and writes `geometry_scores.csv` when enabled.
- Accepted matched geometry rows: `24` valid `linear_cka` rows, covering all six matched models x four PRF ROIs.
- Summary artifacts:
  - `outputs/neural_roi_summary/matched_geometry_scores.csv`
  - `outputs/neural_roi_summary/matched_geometry_model_rankings.csv`
  - `outputs/neural_roi_summary/matched_geometry_roi_rankings.csv`
- Paper inspection artifacts:
  - `outputs/paper_inspection_v1/tables/table10_matched_geometry_model_rankings.csv`
  - `outputs/paper_inspection_v1/tables/table10_matched_geometry_model_rankings.md`
- Matched geometry model ranking by across-ROI mean `linear_cka`:
  1. `vit_small_patch14_dinov2`: `0.229035`
  2. `convnext_tiny`: `0.210549`
  3. `resnet50`: `0.209893`
  4. `deit_small_patch16_224`: `0.202786`
  5. `vit_base_patch16_clip_224`: `0.187853`
  6. `vit_base_patch16_224`: `0.103863`
- Cross-level reporting now carries geometry columns for behavior-versus-geometry and encoding-versus-geometry summaries under the same six-model panel. Current regenerated `matched_cross_level_correlations.csv` has `315` rows, with `210` complete rows containing `linear_cka` geometry fields where model counts are sufficient.
- Deterministic subset RSA is implemented as a utility and metadata-compatible scoring path, but full-panel subset RSA at sizes `512`, `1024`, and `2048` is too slow for the current posthoc milestone. Treat it as the next sensitivity task, not a blocker for the accepted scalable geometry V1 result.

Session progress summary:

- This session implemented the first matched full-image geometry axis and moved the project from "missing representational geometry" to "has one valid but preliminary geometry metric."
- New code/artifacts added:
  - `src/hma/neural/geometry.py` with `linear_cka`, deterministic subset index selection, and `subset_rsa`.
  - `scripts/compute_matched_geometry.py` for posthoc geometry generation from existing matched `flatten_pca` activations.
  - `configs/paper1_config.yaml` freezing the Paper 1 model panel, ROIs, behavior CSV, matched neural output dirs, geometry methods, and exclusions.
  - Optional `neural.geometry.enabled` support in `src/hma/experiments/neural_alignment.py`.
  - Geometry loading, matched geometry rankings, and geometry cross-level columns in `src/hma/experiments/summarize_neural_roi_results.py`.
  - Paper inspection table 10 in `scripts/create_paper_inspection_pack.py`.
- Current accepted geometry output quality:
  - `24/24` matched cells have valid full-image `linear_cka` rows.
  - Every ROI has six valid model rows: `V1`, `V2`, `V3`, and `hV4`.
  - The geometry result is internally matched to the accepted six-model x four-ROI encoding panel.
- Current scientific quality:
  - The result is not SOTA and not paper-headline proof.
  - It is a valid internal geometry baseline that exposes a useful pattern: DINOv2 leads both encoding and CKA, but CKA-versus-encoding model-rank agreement is weak across all six models.
  - This creates a plausible cross-axis dissociation target, but it needs RSA/CKA agreement checks and sensitivity analysis before it can be trusted.
- Verification:
  - Focused geometry/reporting tests passed with `83 passed` using a fresh `--basetemp`.
  - The ordinary `.pytest_tmp` path can be locked by Windows permissions; use a fresh `--basetemp` when rerunning focused tests.

Outcome assessment:

- Validity: the result is a valid matched-panel geometry baseline, not a complete representational-geometry claim. It uses the same six models, same four `subj01` PRF ROIs, same full-image count, and same validation-selected `flatten_pca` feature rows as the accepted encoding panel. This makes it internally comparable to the matched encoding results.
- Method status: linear CKA is a standard representational-comparison method and the implementation avoids the memory failure mode that blocked full `9841 x 9841` RSA. The current implementation is therefore standardizable as a scalable first-pass geometry axis.
- Main limitation: the current geometry result has only one geometry method, one subject, one feature-reduction policy, and no uncertainty. It should not be described as definitive representational alignment or as a novel metric.
- Novelty status: the method itself is not novel. The publishable contribution, if it survives sensitivity checks, is the matched cross-axis dissociation analysis: fixation alignment, neural encoding, and full-image representational geometry evaluated on the same controlled panel.
- Current signal: DINOv2 leads the across-ROI `linear_cka` ranking and also leads the matched noise-normalized encoding ranking, but the full six-model geometry-versus-encoding Spearman correlation is only about `0.257`. This suggests partial convergence at the leader level but weak rank-level equivalence across the panel.
- Interpretation: this is scientifically useful because it prevents the paper from relying only on encoding and legacy ROI500 RSA. It is not yet strong enough for a headline claim until subset-RSA/rank-stability and leave-one-out sensitivity are implemented.
- Standardization path: freeze the Paper 1 config, keep `geometry_scores.csv` as the per-run contract, keep summary-level geometry tables separate from legacy `rsa_scores.csv`, and require every future geometry method to report method, centering, subset/image count, seed, feature source, response source, validity, and status.

Figure and SOTA-alignment assessment:

- Bottom line: the current figures do not match academic SOTA numeric performance or main-paper evidence standards. They are useful diagnostic figures for a controlled internal analysis, not SOTA result figures.
- Local behavioral numbers are below saliency SOTA. `figure1_behavior_static2000_nss.png` reports local DeepGaze IIE NSS of `1.838` on CAT2000, `1.743` on SALICON, and `1.745` on COCO-Search18. Current MIT/Tuebingen CAT2000 reference values are higher: official CAT2000 DeepGaze IIE NSS `2.1122`, DeepGaze MSDB NSS `2.5127`, leave-one-subject-out gold standard NSS `2.4878`, and joint gold standard NSS `2.7429`. Therefore the local CAT2000 figure is below current saliency benchmark standard and should not be interpreted as SOTA.
- Local center-bias numbers also indicate benchmark mismatch. Local CAT2000 center-bias NSS is `1.619`, while the MIT/Tuebingen CAT2000 center-bias row reports NSS `2.0870`. This means the local evaluation protocol, preprocessing, reference maps, or image handling are not identical to the official benchmark, even though the relative sanity ordering DeepGaze > center bias > attribution maps is reasonable.
- Local attribution/fixation rows are far below dedicated fixation models. The strongest local attribution rows are CAT2000 ResNet-50 Grad-CAM NSS `0.882`, SALICON DINOv2 gradient NSS `0.736`, and COCO-Search18 ResNet-50 Grad-CAM NSS `0.955`. These are not competitive with dedicated saliency/search models and should be described only as explanation-map-to-fixation similarity.
- COCO-Search18 is especially far from task-specific SOTA. Local COCO-Search18 DeepGaze IIE NSS is `1.745` and the best local attribution row is `0.955`, while a task-trained COCO-Search18 CNN report gives NSS `4.64`, AUC-Judd `0.95`, sAUC `0.84`, CC `0.72`, SIM `0.54`, and IG `2.59`. The local COCO figure is therefore a diagnostic free-viewing/reference transfer result, not task-search SOTA.
- Local neural encoding numbers are not Algonauts SOTA-comparable. `figure2_neural_model_rankings.png` reports a best matched-panel mean noise-normalized score of `0.591` (`59.11` on x100 scale) for DINOv2 over one subject and four PRF ROIs. Algonauts 2023 leaderboard scores are mean noise-normalized encoding accuracy across held-out vertices of all subjects and hemispheres; the public top score is `70.8473`, with many leaderboard entries in the `58-63` range. Because the scope differs, the local `59.11` should not be called SOTA even though the magnitude overlaps mid-leaderboard values.
- The local matched geometry numbers have no direct SOTA benchmark. `table10_matched_geometry_model_rankings.md` reports DINOv2 `linear_cka=0.229`, ConvNeXt-T `0.211`, ResNet-50 `0.210`, DeiT-S `0.203`, CLIP ViT-B/16 `0.188`, and ViT-B/16 `0.104`. CKA is standard for representation comparison, but absolute CKA values are not portable SOTA scores across datasets, feature reductions, ROIs, dimensionalities, and centering policies. These numbers are valid only as within-study relative rankings.
- The current cross-level figure/table is not main-paper standard yet. `table9_matched_cross_level_correlations.md` includes many high Spearman values, but these are small-`n` model-level correlations (`n=4` or `n=6`) without leave-one-model-out, permutation, bootstrap, or subject replication. The academic-standard version should show model-labeled scatter plots and sensitivity intervals, not only rank-correlation tables.
- `figure3_roi_heatmaps.png` is field-compatible as a model x ROI heatmap for encoding, but its RSA panels should be treated as legacy continuity diagnostics. The accepted geometry axis is now full-image `linear_cka` in `table10`, not ROI500 RSA. A paper-facing update should replace or supplement the RSA heatmap with a matched CKA/RSA geometry heatmap.
- `figure4_behavior_neural_leader_overlap.png` is diagnostic rather than field-standard headline evidence. Leader-overlap rates are too coarse for a paper claim and should move to supplement unless paired with model-level scatter plots and uncertainty.
- `figure5_matched_cross_level_correlations.png` is closest to the intended Paper 1 claim, but it still does not meet the normal standard for a NeuroAI/vision paper result figure. The paper needs labeled scatter plots for behavior-vs-encoding, behavior-vs-geometry, and encoding-vs-geometry, with exact `n`, leave-one-model-out sensitivity, and uncertainty/permutation context.
- CKA/RSA standard: CKA is a recognized representation-comparison method, but a paper-quality geometry result should either show agreement between CKA and subset RSA or explicitly report metric dependence. Legacy ROI500 RSA does not satisfy this requirement because its item scope is not matched to the full-image CKA/encoding panel.

Full verification:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result before matched cross-level implementation: `210 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

Latest focused reporting result:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Result after matched cross-level implementation: `36 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

Latest geometry-focused reporting result:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py --basetemp=D:\Git\saliency\.pytest_tmp_fresh
```

Result after scalable representational geometry V1 implementation: `83 passed`.

Latest broader neural/reporting result:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```

Result after matched cross-level implementation: `85 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

## Next Concrete Milestone

Priority: **Geometry Sensitivity And Cross-Axis Uncertainty V1**.

Scalable Representational Geometry V1 is now implemented for the accepted six-model x four-ROI matched panel using full-image linear CKA. Do not broaden the behavioral datasets, model zoo, subject set, scanpath/video scope, or adaptive-attention interventions before checking whether the new geometry axis is stable enough to support a Paper 1 dissociation claim.

Planning improvements over the previous plan:

- Prioritize output quality over adding more outputs. The current figures are diagnostic and below SOTA standards; the next work should improve evidence quality, not merely add more rows.
- Treat saliency, encoding, and geometry as three different benchmark regimes. Do not compare local numbers to SOTA without matching protocol, subject/image scope, and metric definitions.
- Separate "field-standard method" from "SOTA result." CKA and RSA are standard methods, but current absolute CKA/RSA values are not SOTA scores.
- Make RSA/CKA agreement the immediate geometry gate. If subset RSA disagrees with full-image CKA, the paper must report metric dependence rather than claim a single representational-geometry ranking.
- Replace or supplement inspection-pack figures with paper-facing figures: behavior benchmark context, model-labeled cross-axis scatter plots, matched CKA/RSA geometry heatmaps, and sensitivity panels.
- Add a decision gate before subject expansion or model-zoo expansion. If leave-one-model-out or RSA/CKA checks collapse the story, Paper 1 should be framed as a measurement framework or workshop/thesis chapter rather than a strong top-venue dissociation claim.

Next acceptance target:

- Add paper-facing figure upgrades before further science expansion:
  - benchmark-context behavioral figure with local DeepGaze/center-bias values shown against official SOTA/reference values where comparison is valid;
  - matched geometry heatmap by model x ROI for `linear_cka`;
  - model-labeled cross-axis scatter matrix for behavior-vs-encoding, behavior-vs-CKA, and encoding-vs-CKA;
  - placeholders or completed panels for leave-one-model-out and RSA/CKA agreement.
- Add a tractable subset-RSA sensitivity pass. Start with one model x one ROI profiling at subset sizes `128`, `256`, and `512`; scale to all `24` cells only after runtime is measured. Record subset size, seed, feature source, response source, validity, and runtime.
- Add geometry rank-stability summaries comparing full-image linear CKA against subset RSA when subset RSA is available. Report Spearman/Kendall agreement by ROI and across-ROI mean.
- Freeze explicit method labels before running subset RSA:
  - current CKA rows should be reported as `linear_cka_full9841` in paper-facing tables/figures;
  - subset RSA rows should use labels such as `subset_rsa_corr_rdm_spearman_size128_seed123`.
- For subset RSA, define the RDM policy in `configs/paper1_config.yaml`: model RDM metric `correlation`, neural RDM metric `correlation`, RDM comparison `spearman`, deterministic sorted subset indices, subset sizes, and seeds.
- Add leave-one-model-out and leave-one-ROI-out sensitivity for behavior-versus-encoding, behavior-versus-geometry, and encoding-versus-geometry correlations. Treat any result that flips sign or collapses under one omitted model as descriptive only.
- Add model-label permutation checks for cross-axis correlations. With `n=6`, use these as calibration and robustness diagnostics, not as strong null-hypothesis claims.
- Add uncertainty where it is cheap and well-defined: target bootstrap for encoding summaries and subset/image bootstrap for geometry. Do not delay the project on expensive full-image bootstrap if subset-based uncertainty is enough to decide whether the story is stable.
- Add a compact decision table that classifies each cross-axis relationship as `stable_convergence`, `stable_dissociation`, `unstable`, or `insufficient_models`.
- Keep COCO-Search18 separate from SALICON/CAT2000 in all behavior-geometry reporting.
- Regenerate `outputs/neural_roi_summary/`, regenerate `outputs/paper_inspection_v1/`, and update this file with stability results and whether the geometry axis strengthens or weakens the dissociation story.

Completed milestone sequence:

- Scoring policy and reporting foundation: implemented target-level raw Pearson, R2 fields, noise-ceiling metadata, valid-ceiling filtering, and noise-normalized layer/model aggregates.
- Full-image-count manifest and run configs: implemented for `subj01` `V1`, `V2`, `V3`, and `hV4` with deterministic splits and NSD-derived target-level noise ceilings.
- Feature representation upgrade: implemented train-only `flatten_pca` with metadata and batch transforms; `spatial_mean` remains useful for ROI500 diagnostics and smoke/debug runs.
- Cross-validated ridge baseline: implemented per-layer/ROI ridge-alpha selection from the outer training split only.
- Validation-only layer/pooling selection: implemented for `flatten_pca`; final score files contain only the selected candidate.
- Learned spatial readout: completed DINOv2 four-ROI fixed-layer learned-readout runs; this is the strongest local single-backbone method result.
- Learned-readout diagnostics: V1 learned-layer selection matched the fixed-layer result, multi-layer smoke was inconclusive, and full V1 voxel-specific low-rank readout was rejected.
- Matched small-model neural panel: completed all `24` full-image-count validation-selected `flatten_pca` cells for the six planned model families and included them in the refreshed summaries and paper inspection pack.
- Matched cross-level analysis: implemented model-level Spearman correlations and simple OLS regressions between corrected behavioral rows and the matched full-image `flatten_pca` neural panel; regenerated neural summary and paper inspection outputs.
- Scalable representational geometry V1: implemented full-image linear CKA for all `24` matched cells, regenerated matched geometry summaries and paper inspection table 10, and added geometry fields to matched cross-level reporting. Subset RSA remains a sensitivity follow-up because full-panel subset sizes were too slow for the current posthoc pass.

Implementation order for the next Codex sessions:

1. Fix the paper inspection pack to report figure-quality evidence rather than only diagnostic rankings:
   - add local-versus-reference behavioral benchmark table/plot;
   - add matched `linear_cka` model x ROI heatmap;
   - add model-labeled cross-axis scatter matrix.
2. Add runtime profiling to `scripts/compute_matched_geometry.py` or a companion script, then benchmark subset RSA on `vit_small_patch14_dinov2` and `resnet50` for `V1` at subset sizes `128`, `256`, and `512`.
3. Freeze the subset-RSA method contract in `configs/paper1_config.yaml`: RDM metric, comparison metric, subset sizes, seeds, and naming convention.
4. Choose one feasible subset-RSA protocol and run it across all `24` matched cells with at least three deterministic seeds if runtime allows; otherwise run one seed and record the limitation.
5. Extend `summarize_neural_roi_results.py` with geometry-method agreement tables, including CKA-versus-subset-RSA rank correlation by ROI and across-ROI mean.
6. Add leave-one-model-out and leave-one-ROI-out cross-axis sensitivity tables for encoding, geometry, and behavior relationships.
7. Add model-label permutation summaries for the small-`n` model-level correlations.
8. Extend `scripts/create_paper_inspection_pack.py` with compact sensitivity and decision-gate tables.
9. Regenerate outputs, run focused tests with a fresh `--basetemp`, and update this file with a decision: proceed toward Paper 1 dissociation claim, frame as measurement framework, or pause Paper 1 expansion.

## Later Milestones

Proceed in phases that map directly to the research question.

1. **Freeze and geometry.** Freeze the accepted Paper 1 scope, keep ROI500 RSA for continuity, and add tractable full-image CKA plus subset RSA so the representational-space claim is not dependent on one legacy Spearman RSA implementation.
2. **Uncertainty and sensitivity.** Estimate intervals over images, neural targets, and geometry subsets; add leave-one-model-out, leave-one-ROI-out, Kendall tau, and model-label permutation checks. Cross-level claims should report uncertainty, not only rankings.
3. **Cross-axis decision gate.** Decide whether Paper 1 has a robust dissociation/convergence story. Continue toward a top venue only if geometry changes or clarifies the story and sensitivity checks survive; otherwise frame the work as a measurement framework or workshop/thesis chapter and shift effort toward Paper 2.
4. **Subject expansion.** Add more NSD/Algonauts subjects before broadening the model zoo. If full replication is too expensive, run a reduced confirmatory panel with DINOv2, CLIP ViT, ResNet-50, and one weaker transformer baseline.
5. **Behavioral SOTA controls.** Add human inter-observer ceilings, DeepGaze MSDB or a comparable modern fixation reference, and a task-trained COCO-Search18 baseline before writing strong fixation-SOTA comparisons. Keep scanpath/video analysis later unless the paper explicitly shifts to adaptive sequential attention.
6. **Transformer attribution depth.** Add Chefer-style attribution or AttnLRP before making claims about transformer attention, and keep gradients, Grad-CAM, rollout, perturbation, LRP-style methods, and token routing as distinct explanation families.
7. **Target-scope expansion.** Move beyond PRF V1/V2/V3/hV4 to broader visual-cortex vertices and higher-level visual ROIs only after the Paper 1 matrix and robustness checks are stable.
8. **Efficiency.** Add FLOPs, latency, token count, retained-patch statistics, and memory footprint for the matched model panel, then regenerate alignment-per-compute summaries. Keep efficiency exploratory unless it produces a clean dissociation or tradeoff.
9. **Brain-Score or Brain-Score-style external positioning.** Use it as context and sanity checking, not as a substitute for the local fixation/fMRI/geometry cross-level tests.
10. **Publication split.** Treat Paper 1 as a static-image cross-level convergence/dissociation study. Treat causal adaptive attention, foveation, adaptive token routing, scanpaths, video, or recurrent policies as Paper 2 unless Paper 1 exposes a sharply defined intervention target.

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
- `src/hma/neural/learned_readout.py`
- `src/hma/neural/rsa.py`
- `src/hma/experiments/neural_alignment.py`
- `src/hma/experiments/summarize_neural_roi_results.py`
- `scripts/run_neural_alignment.py`
- `scripts/create_nsd_noise_ceiling_manifest.py`
- `scripts/summarize_neural_roi_results.py`
- `scripts/audit_matched_neural_panel.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

For the current neural/reporting implementation, the focused verification command is:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Latest focused result after matched cross-level implementation: `36 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

For broader confidence after neural/reporting changes, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```

Latest broader result after matched cross-level implementation: `85 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

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

Matched panel audit command:

```cmd
.\.venv\Scripts\python.exe scripts\audit_matched_neural_panel.py
```

Latest matched panel audit result: `24` complete cells, `0` missing cells, `0` incomplete cells, and `0` explicitly skipped cells.

Current neural summary target-scope verification:

```cmd
.\.venv\Scripts\python.exe -c "import csv,collections; rows=list(csv.DictReader(open('outputs/neural_roi_summary/combined_encoding_target_scores.csv', newline='', encoding='utf-8'))); print(len(rows)); print(dict(collections.Counter(r['metric_scope'] for r in rows)))"
```

Latest regenerated target scope: `289740` rows total; `289620` `benchmark_style_noise_normalized`; `120` `benchmark_style_non_noise_normalized` due to hV4 zero-ceiling targets across ROI500, validation-selected full `flatten_pca`, and learned-readout rows.

For the next scalable representational-geometry implementation, run the focused neural/reporting tests above plus any new geometry-specific tests.

Current reporting compatibility check:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Latest focused reporting result after matched cross-level reporting implementation: `36 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

For full confidence after broad code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result before matched cross-level implementation: `210 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.
