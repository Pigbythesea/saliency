# HMA Project Status And Next Steps

Updated: 2026-05-27

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
- Neural encoding/RSA diagnostics on local Algonauts / NSD `subj01` visual ROIs, including ROI500 spatial-mean diagnostics, full-image-count `flatten_pca` PRF ROI baselines, and full-image-count learned spatial readout for DINOv2.
- Paper-style inspection tables and figures that join corrected behavioral summaries with the current mixed-scope neural summaries.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs:

- Corrected core behavioral aggregate: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Corrected SSL/VLM behavioral aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Neural ROI summary with full `flatten_pca` and learned-readout rows included: `outputs/neural_roi_summary/`.
- Matched full-image `flatten_pca` panel outputs and audit: `outputs/neural_roi_summary/matched_full_panel_model_rankings.csv`, `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`.
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
- Input neural directories: `48`
- Encoding rows: `120`
- Encoding target rows: `289740`
- RSA rows: `92`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Feature-reduction rows: `92` spatial-mean diagnostic rows, `24` validation-selected full-image-count matched-panel `flatten_pca` rows, and `4` full-image-count learned spatial readout rows.
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

Current bridge readout:

- Overall behavior-to-encoding leader match rate: `0.587`.
- Overall behavior-to-RSA leader match rate: `0.000`.
- Evidence-sensitivity rows and most internal-routing rows account for the observed encoding leader matches; class-localization rows do not match the encoding or RSA leader in the current descriptive table, and no behavioral family matches the current RSA leader.

Interpretation:

- The project has moved from weak ROI500 spatial-mean neural diagnostics to a complete full-image-count matched `flatten_pca` panel for six model families, plus a stronger DINOv2 learned-readout method result.
- The matched-panel ranking is now the accepted basis for cross-model neural comparisons: `vit_small_patch14_dinov2` first, `vit_base_patch16_clip_224` second, `resnet50` third, `deit_small_patch16_224` fourth, `vit_base_patch16_224` fifth, and `convnext_tiny` sixth by mean valid-target noise-normalized score.
- The DINOv2 learned spatial readout materially improves all four PRF visual ROIs over DINOv2 `flatten_pca`, but it is not method-matched to the other backbones and should not be used as the primary cross-model row.
- The previous test-set feedback risk for layer choice has been addressed for the current one-subject PRF visual ROI baselines by validation-only layer selection.

## Global Direction Rationale

The project direction remains a multi-axis NeuroAI alignment study, not a saliency-map leaderboard or a local Algonauts score chase. The target scientific question is now explicit: test whether human-like fixation alignment, model attribution/routing, neural encoding, representational geometry, and computational efficiency converge or dissociate across model families, ROIs, subjects, and viewing regimes.

The current behavioral layer is mature enough to serve as the fixation-alignment axis, but it is not the main claim. The current neural layer now has a complete matched full-image-count local encoding panel for six model families across `subj01` PRF visual ROIs. It is still local and one-subject, but it is finally suitable for the first model-level convergence/dissociation analysis. The failed full V1 voxel-specific decision run closes the current DINOv2-only readout search. The next implementation work should build matched cross-model and cross-metric tests instead of adding more single-backbone capacity.

The immediate engineering priority is now matched cross-level analysis. The voxel-specific low-rank branch was scientifically useful as a control: it improved the small smoke but failed on full V1, so it should be treated as a rejected DINOv2-only readout variant, not as a protocol to expand. The project should now optimize validity of the central convergence/dissociation test, not the absolute DINOv2 local score.

The next priority is to make the central correlation test valid: model-level correlations/regressions between corrected behavioral fixation metrics and the matched neural panel, then scalable representational geometry, uncertainty estimates, subject expansion, and eventual adaptive-attention intervention. Broad model-zoo expansion, Brain-Score-style positioning, and alignment-per-compute claims should wait until the matched cross-level analysis is implemented.

Current methodological gap to SOTA:

- The current strongest local results use a learned target-wise spatial readout for all four PRF visual ROIs, but only for one subject and one backbone. A low-rank voxel-specific DINOv2 variant was tested and rejected on full V1, so stronger RetinaMapper-style heads remain a SOTA gap but are no longer the next local milestone.
- Current full-image-count runs cover `subj01` PRF visual ROIs only. SOTA scores use broader visual-cortex vertex sets across subjects and hemispheres.
- Current ridge alpha selection is per-layer/ROI, not per-target. Per-target alpha and richer voxel-specific heads may improve later, but should wait until after the matched backbone comparison and cross-level analysis are working.
- Current reporting keeps a mixed-scope ranking for continuity, but matched-panel tables now separately restrict to `subj01`, `9841` images, `flatten_pca`, validation-selected final rows, and PRF ROIs `V1`/`V2`/`V3`/`hV4`. Cross-model neural claims should use the matched-panel tables, not the mixed-scope ranking.
- Current scoring now supports validation-selected single-layer `flatten_pca` and fixed-layer learned spatial readout. SOTA methods go further with multi-layer fusion, learned layer selection, subject-specific heads, voxel-specific spatial readouts, and ensembles.
- Current RSA is still ROI500-scale for most reporting; the strongest full-image-count encoding rows do not yet have matched scalable RSA, CKA, or other representational-geometry metrics.
- Current bridge tables report leader overlap only. They are useful diagnostics, but they are not yet a statistically defensible fixation-versus-fMRI correlation analysis.
- Current behavioral SOTA controls include DeepGaze IIE and center bias, but do not yet include DeepGaze MSDB, scanpath-level references, human inter-observer ceilings, or task-trained COCO-Search18 search models.
- Current transformer attention evidence relies mostly on gradients and attention rollout. Add Chefer-style attribution or AttnLRP before making claims about transformer attention as human-like evidence.

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

The voxel-specific readout decision is complete and the matched neural panel is complete. Do not expand this inventory again before matched cross-level analysis is implemented and interpreted. The next model-family comparison should remain narrow and methodologically matched, not a broad model zoo.

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
- Paper inspection README now explicitly distinguishes mixed-scope diagnostics from the complete six-model matched full-image-count PRF ROI `flatten_pca` panel, and includes the four DINOv2 learned spatial readout rows only as method-provenance context.

## Superseded Or Historical Outputs

Historical note:

- `docs/v2_static2000_results_note.md` describes the pre-fix static2000 state from 2026-05-19. Its result interpretation is superseded.

Do not use for current claims:

- Any aggregate row without `fixation_protocol=points` or `task_points`.
- Any pre-2026-05-20 static2000 NSS/AUC result generated before point-fixation scoring was corrected.
- Old claims that center bias beat DeepGaze under the static2000 protocol.
- `C:/saliency_outputs/neural_subj01_full/vit_small_patch14_dinov2_v1_learned_spatial_readout_layer_selection_full_180ep/` should be treated as diagnostic provenance only. It selected the same layer and produced the same held-out score as the accepted fixed-layer V1 learned-readout run, so it should not be added as an extra accepted baseline row in the current neural summary.

Current corrected outputs have replaced those rows.

## Current Implementation Progress

Updated: 2026-05-27

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
- `outputs/paper_inspection_v1/README.md` uses the complete matched panel for the neural encoding headline and keeps DINOv2 learned-readout rows as method provenance.

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

Full verification:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result after matched-panel implementation: `210 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

Latest focused reporting result:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Result: `31 passed`; Windows `.pytest_cache` permission warning remains non-blocking.

## Next Concrete Milestone

Priority: **Matched Cross-Level Analysis V1**.

Do this before adding more behavioral datasets, more saliency methods, Brain-Score integration, adaptive token pruning, foveation, scanpaths, video, subject expansion, or a broad model zoo. The matched full-image-count neural panel is complete, so the next implementation should replace leader-overlap bridge summaries with model-level correlations/regressions between corrected behavioral fixation metrics and matched neural encoding metrics.

Next acceptance target:

- Add cross-level analysis outputs that join corrected behavioral rows to the matched full-panel neural rows only.
- Use only neural rows with `feature_reduction=flatten_pca`, `metadata_num_items=9841`, `subject_id=subj01`, ROIs `V1`/`V2`/`V3`/`hV4`, validation-selected final rows, and the six completed model families.
- Keep `vit_small_patch14_dinov2` learned spatial readout rows as method-provenance rows only. Do not use learned-readout rows in the matched cross-model correlation.
- Keep COCO-Search18 separate from free-viewing SALICON/CAT2000 in cross-level analysis.
- Report behavior-neural relationships by dataset, behavioral metric, saliency method/family, ROI, and across-ROI model mean.
- Include rank-based and value-based analyses, at minimum Spearman correlations and simple OLS-style regressions across the matched models where the small sample size is stated clearly.
- Regenerate `outputs/neural_roi_summary/`, regenerate `outputs/paper_inspection_v1/`, and update this file with correlation/regression results and remaining limitations.

Completed milestone sequence:

- Scoring policy and reporting foundation: implemented target-level raw Pearson, R2 fields, noise-ceiling metadata, valid-ceiling filtering, and noise-normalized layer/model aggregates.
- Full-image-count manifest and run configs: implemented for `subj01` `V1`, `V2`, `V3`, and `hV4` with deterministic splits and NSD-derived target-level noise ceilings.
- Feature representation upgrade: implemented train-only `flatten_pca` with metadata and batch transforms; `spatial_mean` remains useful for ROI500 diagnostics and smoke/debug runs.
- Cross-validated ridge baseline: implemented per-layer/ROI ridge-alpha selection from the outer training split only.
- Validation-only layer/pooling selection: implemented for `flatten_pca`; final score files contain only the selected candidate.
- Learned spatial readout: completed DINOv2 four-ROI fixed-layer learned-readout runs; this is the strongest local single-backbone method result.
- Learned-readout diagnostics: V1 learned-layer selection matched the fixed-layer result, multi-layer smoke was inconclusive, and full V1 voxel-specific low-rank readout was rejected.
- Matched small-model neural panel: completed all `24` full-image-count validation-selected `flatten_pca` cells for the six planned model families and included them in the refreshed summaries and paper inspection pack.

Implementation order for the next Codex sessions:

1. Inspect the existing bridge generation in `src/hma/experiments/summarize_neural_roi_results.py` and the behavior summary tables used by `scripts/create_paper_inspection_pack.py`.
2. Add a matched cross-level table that starts from `matched_full_panel_model_rankings.csv` and ROI-level `matched_full_panel_encoding_scores.csv`, then joins to behavioral model rows from `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
3. Normalize model identifiers carefully so matched neural models join to behavioral rows for comparable backbones and saliency method families.
4. Compute model-level Spearman correlations and simple regression summaries by dataset, behavioral metric, saliency method/family, ROI, and across-ROI neural mean.
5. Keep mixed-scope leader-overlap bridge tables for continuity, but label them as descriptive diagnostics and make the new matched cross-level tables the headline cross-axis evidence.
6. Update `scripts/create_paper_inspection_pack.py` so the README and tables expose the matched cross-level results without implying official Algonauts or saliency-leaderboard equivalence.
7. Run focused reporting tests, regenerate outputs, and update this file with the new tables and interpretation.

## Later Milestones

Proceed in phases that map directly to the research question.

1. **Matched cross-level analysis.** Replace leader-overlap-only bridge summaries with correlations/regressions across models and saliency families: fixation metrics versus noise-normalized encoding, fixation metrics versus RSA/CKA when available, encoding versus representational geometry, and all axes by ROI and dataset. Treat COCO-Search18 separately from free-viewing SALICON/CAT2000.
2. **Scalable representational geometry.** Keep ROI500 RSA for continuity, then add tractable full-image subset RSA plus CKA or a Procrustes-style metric so the representational-space claim is not dependent on one Spearman RSA implementation.
3. **Uncertainty.** Bootstrap or otherwise estimate confidence intervals over images, targets, and eventually subjects. Cross-level claims should report uncertainty, not only rankings.
4. **Subject expansion.** Add more NSD/Algonauts subjects before broadening the model zoo. Subject replication is higher priority than another saliency method now that the matched model panel exists.
5. **Target-scope expansion.** Move beyond PRF V1/V2/V3/hV4 to broader visual-cortex vertices and higher-level visual ROIs if local data supports it.
6. **Transformer attribution depth.** Add Chefer-style attribution or AttnLRP before making claims about transformer attention, and keep gradients, Grad-CAM, rollout, perturbation, LRP-style methods, and token routing as distinct explanation families.
7. **Behavioral SOTA controls.** Add human inter-observer ceilings, DeepGaze MSDB or a comparable modern fixation reference, and a task-trained COCO-Search18 baseline before writing strong fixation-SOTA comparisons. Keep scanpath/video analysis later unless the paper explicitly shifts to adaptive sequential attention.
8. **Efficiency.** Add FLOPs, latency, token count, and retained-patch statistics for the matched model panel, then regenerate alignment-per-compute summaries.
9. **Brain-Score or Brain-Score-style external positioning.** Use it as context and sanity checking, not as a substitute for the local fixation/fMRI cross-level tests.
10. **One causal adaptive-attention intervention.** After the descriptive cross-level matrix is stable, test one controlled intervention such as gaze-guided token masking, foveated input, adaptive patch selection, or fixation-regularized token routing. The intervention should ask whether changing computation allocation changes fixation alignment, neural encoding, representational geometry, and efficiency together or separately.
11. **Publication split.** Treat the first paper as a static-image cross-level convergence/dissociation study. Treat adaptive scanpaths, video, or foveated recurrent policies as a second paper unless the first matrix exposes a very clear intervention target.

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

For the next reporting or cross-level-analysis implementation, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```

For broader confidence after code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Latest full result after matched-panel implementation: `210 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.

Current reporting compatibility check:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Latest focused reporting result after matched-panel reporting implementation: `31 passed`; Windows `.pytest_cache` permission warning remains non-blocking.
