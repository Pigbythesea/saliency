# HMA Project Status And Next Steps

Updated: 2026-06-11

## Purpose And Codex Operating Contract

This document is the primary implementation handoff for Codex sessions on the Human-Machine Visual Alignment project.

Its purpose is to steer implementation toward publication-grade scientific evidence, not toward self-reassuring engineering progress. A Codex session should use this file to decide:

* what scientific claim the project is currently trying to test;
* which outputs are accepted as evidence and which are only diagnostics;
* which implementation task most directly strengthens the paper claim;
* which tasks should be avoided because they only expand the codebase, leaderboard size, or logging surface without improving the publication argument;
* what exact artifact must exist at the end of the session.

The current Paper 1 claim to test is:

> Human-like fixation alignment, neural encoding, and latent representational geometry are related but non-equivalent axes of visual alignment; dissociation and convergence patterns may emerge out of different axis, giving insight on brain decoding via model architecture. Then, do models that better match human visual attention behavior also better align with human visual cortex in encoding and representational geometry, or do these alignment axes dissociate across architecture, attribution family, ROI, and viewing regime?

Codex must treat every implementation task as subordinate to this claim. The project should not be optimized for a larger saliency leaderboard, a local Algonauts score chase, broad model-zoo accumulation, or cosmetic paper-pack expansion unless the task directly improves the cross-axis dissociation/convergence evidence.

### What counts as progress

A change counts as project progress only if it produces at least one of the following:

* a new accepted evidence table used by the Paper 1 claim;
* a robustness or uncertainty table that changes how strongly a result can be trusted;
* a reviewer-facing control that protects the paper from a known methodological attack;
* a cleaner separation between accepted evidence and diagnostic/provenance artifacts;
* a figure/table that directly supports or falsifies the cross-axis dissociation claim;
* a documented decision that stops an unproductive branch.

Engineering achievements, successful smoke runs, code reorganization, new logs, new configs, and larger result matrices do not count as scientific progress unless they produce one of the outputs above. 

Do not apply publication hardening to a smoke-run object. First build the paper-grade object, then harden it.

### What Codex should refuse or deprioritize

Codex should avoid:

* expanding generic saliency rows before human ceilings, stronger fixation baselines, and attribution-family controls are handled;
* adding more model families before the existing six-model matched panel has stronger robustness analysis;
* treating DINOv2 learned spatial readout as a cross-model ranking result;
* treating attention rollout as evidence of human-like transformer attention;
* mixing free-viewing SALICON/CAT2000 claims with task-driven COCO-Search18 claims;
* reporting model-level correlations without exact `n`, leave-one-model-out sensitivity, and uncertainty;
* polishing paper-pack summaries before the underlying accepted evidence changes;
* optimizing toward local benchmark scores without connecting the result to the cross-axis claim.

### Required end-of-session report

At the end of each Codex session, update this file with:

1. `Scientific change`: what changed in the evidence, not merely the code.
2. `Accepted artifact`: exact path(s) to the new table/figure/config.
3. `Claim impact`: whether the result strengthens, weakens, or fails to affect the Paper 1 dissociation claim.
4. `Reviewer risk reduced`: which concrete reviewer objection the change addresses.
5. `Next decisive step`: the next task most likely to improve the publication argument.


## Reference Documents Reviewed

Current steering documents under `docs/`:

- `project_status_and_next_steps.md`: this engineering status file.
- `paper1_cross_axis_alignment_roadmap.md`: current publication roadmap for Paper 1. It reframes the project as a cross-axis dissociation study across fixation alignment, neural encoding, representational geometry, and efficiency.
- `paper1_literaturereview.md`: current literature review for Paper 1. It raises the required controls around dataset bias, scanpath/task specificity, subject variability, encoding reliability, representational-geometry metrics, and transformer attribution.
- `Literature Review and Research Redesign for the Human-Like Adaptive Visual Attention Project.md`: argues the project should become a multi-axis NeuroAI alignment study, not a saliency-map leaderboard.
- `Deep Research Assessment of the Human-Machine Visual Alignment Project.md`: emphasizes the publishable question as convergence versus dissociation among fixation alignment, neural predictivity, representational geometry, and efficiency.
- `hma_project_publication_critique_handoff.md`: current publication-readiness critique. It is a read-only reference for claim hygiene, stale-output cleanup, and top-venue risk assessment.
- `Zhang_Zihuan_zzhan330_proposal.docx`: original proposal; defines behavioral saliency, neural encoding, RSA, Brain-Score-style comparison, and compute efficiency as the core axes.
- `Comparing Human and Machine Visual Saliency_ A Comprehensive Review.pdf`: reinforces that fixation prediction requires strong controls such as center bias, DeepGaze-class references, point-based NSS/AUC, and separate treatment of free-viewing versus task-driven viewing.
- `__Attention and Saliency Map Extraction in Visual AI Models_ A Comprehensive Review__.pdf`: reinforces that gradients, CAMs, attention rollout, perturbation maps, LRP-style methods, and transformer attribution are different explanation objects and should not be collapsed into one "attention" score.

## Current Snapshot

The repository now implements three active layers:

- Behavioral saliency / fixation benchmarking on SALICON, CAT2000, and COCO-Search18.
- Neural encoding on local Algonauts / NSD `subj01` visual ROIs, including the complete six-model full-image-count `flatten_pca` PRF ROI diagnostic panel, full-image-count learned spatial readout provenance for DINOv2, and the V1 four-model x ten-ROI discovery matrix.
- Paper-style inspection tables and figures that join corrected behavioral summaries with the matched full-image neural panel, matched geometry sensitivity outputs, V1 ROI-expanded geometry, V1 geometry-method sensitivity decisions, and matched cross-level correlation/regression outputs.

The repository now implements the diagnostic PRF-only matched full-image representational-geometry axis, the V1 ROI-expanded `subj01` geometry axis, the reduced `subj02`-`subj04` subject-robustness panel, and the DINOv2-vs-ResNet uncertainty/margin interpretation for that panel. The rank-only subject-robustness gate remains **partial**, but the uncertainty-aware decision is more specific: geometry replication is robust, while encoding is subject-sensitive and aggregate-labeled `geometry_replicated_encoding_ambiguous`.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs are classified as follows:

Accepted diagnostic evidence:

- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Matched full-image `flatten_pca` neural panel: `outputs/neural_roi_summary/matched_full_panel_model_rankings.csv`.
- Matched full-image geometry rankings and sensitivity summaries: `outputs/neural_roi_summary/matched_geometry_model_rankings.csv`, `outputs/neural_roi_summary/matched_geometry_method_agreement.csv`.
- Matched cross-level analysis outputs: `outputs/neural_roi_summary/matched_cross_level_observations.csv`, `outputs/neural_roi_summary/matched_cross_level_correlations.csv`.
- V1 ROI-expanded discovery evidence: `outputs/paper1_experiment_v1/summary/roi_expanded_encoding_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_agreement.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_sensitivity_decisions.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_failure_gate_summary.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_correlations.csv`, and `outputs/paper1_experiment_v1/summary/roi_expanded_cross_axis_decisions.csv`.

Robustness/control artifacts:

- Matched panel audit: `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`.
- Cross-axis sensitivity and decision diagnostics: `outputs/neural_roi_summary/matched_cross_axis_sensitivity.csv`, `outputs/neural_roi_summary/matched_cross_axis_decisions.csv`.
- V1 ROI-expanded artifact audit: `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`; current status is all checks passing after `40` geometry cells x `10` geometry rows per cell, plus sensitivity-decision and failure-gate output checks.
- V1 geometry-method sensitivity and failure-gate outputs: `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_sensitivity_decisions.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_failure_gate_summary.csv`.
- V1 subject-robustness outputs: `outputs/paper1_experiment_v1/summary/subject_robustness_decisions.csv`, `outputs/paper1_experiment_v1/summary/subject_robustness_encoding_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/subject_robustness_geometry_model_rankings.csv`, and `outputs/paper1_experiment_v1/summary/subject_robustness_geometry_method_sensitivity_decisions.csv`.
- V1 subject-robustness uncertainty outputs: `outputs/paper1_experiment_v1/summary/subject_robustness_encoding_margin_uncertainty.csv`, `outputs/paper1_experiment_v1/summary/subject_robustness_geometry_margin_summary.csv`, and `outputs/paper1_experiment_v1/summary/subject_robustness_uncertainty_decisions.csv`.
- V1 paper-facing synthesis outputs: `outputs/paper1_experiment_v1/summary/subject_robustness_paper_interpretation.csv` and `outputs/paper1_experiment_v1/summary/behavioral_observer_control_summary.csv`.
- Behavioral-control hardening outputs: `outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv`, `outputs/real_matrix_v2_task_search_baseline/aggregated/results.csv`, `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv`, `outputs/real_matrix_v2/coco_search18_static2000/coco_search18_task_prior_baseline_coco_search18_task_prior/aggregate_metrics.json`, and the merged accepted behavioral aggregate `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Free-viewing reference feasibility output: `outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv`.
- Observer-control outputs: `outputs/observer_controls_v2/coco_search18_static2000_observer_controls.csv`, `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`.

Diagnostics/provenance:

- Core behavioral aggregate before SSL/VLM merge: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected SSL/VLM behavioral aggregate before merge: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Full neural ROI summary directory, including learned-readout provenance rows: `outputs/neural_roi_summary/`.
- Paper inspection pack: `outputs/paper_inspection_v1/README.md`, including `outputs/paper_inspection_v1/tables/table14_subject_robustness_interpretation.csv` and `outputs/paper_inspection_v1/tables/table15_observer_control_summary.csv`.

Experiment-definition artifacts:

- Paper-grade experiment specification V1: `docs/paper1_experiment_spec_v1.md`.
- Forward-looking experiment contract: `configs/paper1_experiment_v1.yaml`.
- Scope decision table: `outputs/planning/paper1_experiment_scope_decisions.csv`.

## Scientific Boundary

The corrected behavioral layer is now usable for diagnostic paper-style analysis. It should still be framed carefully:

- NSS and AUC-style claims are valid only for rows with `fixation_protocol=points` or `fixation_protocol=task_points`.
- CC, SIM, KL, and related map-distribution metrics should be discussed separately from point-fixation metrics.
- DeepGaze and center bias are reference controls. Grad-CAM, gradients, rollout, and similar rows are explanation-map-to-fixation comparisons, not dedicated SOTA fixation-prediction models.
- COCO-Search18 is task-driven search and should not be pooled with free-viewing SALICON/CAT2000 as if all three datasets measure the same behavior.

The neural layer is now a stronger local baseline, but still not a leaderboard result:

- Current neural outputs include one-subject diagnostic `subj01` results plus a reduced `subj02`-`subj04` PRF-ROI subject-robustness panel. The accepted diagnostic cross-model comparison object remains the complete six-model matched full-image-count PRF visual ROI `flatten_pca` panel. Four full-image-count learned-readout rows for DINOv2 are method provenance only.
- They are not Algonauts leaderboard-equivalent scores because the official challenge averages held-out visual-cortex vertices across subjects and hemispheres.
- The matched `flatten_pca` panel is the primary evidence for cross-model neural comparisons. The four-ROI DINOv2 learned-readout rows are the strongest local single-backbone method result and should be treated as method provenance, not as matched-panel ranking rows.
- The matched cross-level correlation tables are now primary descriptive cross-axis evidence, but they remain small-n model-level analyses and are not causal tests.

Paper 1 should be held to these publication gates before strong top-venue claims:

- Full-image matched representational geometry now exists for the same six-model x four-ROI panel, but geometry claims still require method/seed stability and explicit CKA/subset-RSA sensitivity.
- Cross-axis results must report uncertainty and sensitivity, especially bootstrap intervals, leave-one-model-out behavior, and exact model counts.
- Claims must be framed as descriptive convergence/dissociation, not causal attention intervention.
- At least one nontrivial dissociation or convergence pattern must survive sensitivity checks; otherwise Paper 1 should be framed as a measurement framework, workshop paper, thesis chapter, or methods note.

## Publication Claim State

### Current paper status

Paper 1 is not yet top-venue ready. The current repository supports a serious diagnostic study, but the accepted evidence is still too narrow for a strong conference claim because it is mainly:

* one discovery subject plus a reduced three-subject PRF-ROI robustness panel with geometry replication but subject-sensitive encoding;
* small model-level correlations (`n=4` for the V1 discovery matrix, `n=6` for the older PRF diagnostic panel);
* one completed ROI-expanded discovery pass with subject robustness only on PRF visual ROIs, not the full stream-ROI scope;
* limited behavioral SOTA controls;
* limited transformer attribution coverage;
* geometry-method-dependent cross-axis analysis.

The current project should therefore be treated as a publication-directed evidence-building pipeline, not as a finished paper.

### Current strongest claim

The strongest currently defensible claim is:

> In the current V1 PRF-ROI subject-robustness scope, representational geometry replicates more cleanly than neural encoding: DINOv2-vs-ResNet geometry margins support DINOv2 across confirmatory subjects, while encoding support is subject-sensitive and reverses to ResNet-50 in `subj04`.

This is now a claim-shaped but still not top-venue-complete pattern: full-image CKA and deterministic subset-RSA margins favor DINOv2 across `subj02`-`subj04`, while encoding is split by subject. `subj02` and `subj03` support DINOv2 encoding, but `subj04` supports ResNet-50 encoding, so the accepted aggregate interpretation is geometry-replicated and encoding-ambiguous.

The current results should be used to drive behavioral-control hardening around the paper-facing geometry-first dissociation claim, not to support a universal best-model conclusion.

### Current weakest links

These are reviewer risks in order of severity, not the implementation order. The subject-robustness decision gate, DINOv2-vs-ResNet margin/uncertainty interpretation, paper-facing geometry-first synthesis, observer-control integration, task-search baseline, and modern free-viewing DeepGaze MSDB control are now complete, so the next milestone should strengthen attribution-family controls before efficiency or model expansion.

1. **Small model-level ****`n`****:** behavior-encoding-geometry correlations over `n=4` in the V1 matrix remain descriptive. This is the highest reviewer risk, but generic model expansion is deferred until behavioral controls are stronger.
2. **Small attribution-family coverage:** Grad-CAM, vanilla gradients, integrated gradients, occlusion, and attention rollout exist, but transformer-specific attention/relevance remains weak. Attention rollout must not support attention-specific claims without a stronger transformer relevance control.
3. **Paper-facing claim framing:** the paper inspection pack now explicitly supports geometry-first dissociation / measurement evidence and avoids a universal DINOv2 encoding-win narrative, but main-paper figures/tables still need later hardening.
4. **Attribution ambiguity:** Grad-CAM, vanilla gradients, and attention rollout are different explanation objects. Attention rollout must not be treated as human-like attention.
5. **Geometry metric dependence:** V1 full-image CKA and subset-RSA now have an explicit sensitivity synthesis, so this risk is reduced but not gone. Some relationships are stable across methods, while direction conflicts remain and must be reported.
6. **Efficiency missing:** compute and alignment-per-compute remain untested, so the original efficient-attention axis is not yet active in Paper 1.
7. **Causality absent:** Paper 1 is observational. It cannot claim that human-like saliency causes neural alignment.

### Evidence acceptance rules

Use these rules when deciding whether a result enters the Paper 1 headline:

* Headline neural model comparisons must use the matched full-image `flatten_pca` panel unless the same readout protocol exists for all compared models.
* DINOv2 learned spatial readout can be reported only as method provenance or an upper-bound/proof-of-readout-sensitivity result.
* Behavioral claims must keep free-viewing datasets and task-driven search datasets separate.
* NSS/AUC-style point-fixation metrics must be interpreted separately from map-distribution metrics such as CC, SIM, and KL.
* Transformer claims require method labels such as `attention rollout attribution`, `gradient attribution`, or `transformer relevance`, not the generic word `attention`.
* Cross-axis correlations must report exact model count, Spearman, Kendall, leave-one-model-out sensitivity, and uncertainty where available.
* A result is publication-relevant only if it clarifies whether fixation alignment, neural encoding, and representational geometry converge or dissociate.

### Paper 1 acceptance gate

Paper 1 can be treated as main-conference-targetable only if all of the following are true:

* corrected behavioral results remain stable under accepted point/task-point protocols;
* matched neural encoding results remain stable under uncertainty or subject/ROI robustness checks;
* matched geometry results are available and stable enough across CKA and subset-RSA sensitivity;
* at least one dissociation or convergence pattern survives leave-one-model-out analysis;
* human/interobserver ceiling or a stronger fixation baseline is added, or the absence is explicitly framed as a limitation;
* attribution-family language is cleaned so the paper never equates explanation maps with operational attention;
* every headline figure has an associated accepted table and reproducible config.

If this gate fails, Paper 1 should be framed as a workshop paper, thesis chapter, or measurement framework, while the project shifts toward Paper 2’s causal adaptive-attention intervention.


## Current Behavioral Status

Corrected merged behavioral aggregate:

- Path: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Rows: `399`
- Dataset rows: `133` each for SALICON, CAT2000, and COCO-Search18
- Protocol rows: `266` with `points`, `133` with `task_points`
- Blank / `unknown` / `density_fallback` protocol rows: none

Corrected NSS headline:

- SALICON: DeepGaze MSDB `1.760`, DeepGaze IIE `1.743`, center bias `0.933`, DINOv2 ViT-S/14 gradient `0.736`, ConvNeXt-T Grad-CAM `0.633`, ResNet-50 Grad-CAM `0.598`.
- CAT2000: DeepGaze MSDB `1.979`, DeepGaze IIE `1.838`, center bias `1.619`, ResNet-50 Grad-CAM `0.882`, DINOv2 ViT-S/14 gradient `0.810`, ConvNeXt-T Grad-CAM `0.759`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, ResNet-50 Grad-CAM `0.955`, ConvNeXt-T Grad-CAM `0.908`, DINOv2 ViT-S/14 gradient `0.713`.

Current interpretation:

- Corrected outputs have valid point/task-point protocol labels.
- DeepGaze MSDB is now the accepted modern free-viewing reference for SALICON/CAT2000 and improves over the earlier DeepGaze IIE row in both datasets.
- DeepGaze IIE remains a useful historical/reference control and COCO-Search18 DeepGaze IIE remains diagnostic because it is a free-viewing reference on task-search data.
- DINOv2 gradient is a strong attribution/fixation-similarity row, especially on SALICON and CAT2000.
- The behavioral layer is strong enough to serve as one axis in the broader alignment study. It should not be expanded into a larger leaderboard before the paper-grade matrix is defined.
- Behavioral controls are now sufficient for the current static-image Paper 1 scope: observer controls, center bias, DeepGaze IIE, DeepGaze MSDB for free-viewing, and the COCO-Search18 task prior are represented. Broad scanpath/video expansion belongs after Paper 1 unless the paper explicitly shifts away from static-image dissociation.

## Current Neural Status

Current neural summary:

- Path: `outputs/neural_roi_summary/`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Efficiency CSV: not provided in the latest summary.
- Summary scope: `120` encoding rows, `289740` encoding target rows, and `92` RSA rows across accepted, diagnostic, and provenance outputs.
- Accepted matched-panel scope: `24` validation-selected full-image-count `flatten_pca` rows for six model families across four `subj01` PRF visual ROIs.
- Method-provenance scope: `4` full-image-count DINOv2 learned spatial readout rows.
- Matched cross-level analysis rows: `315` correlation/regression groups, with `210` complete and `105` marked `insufficient_models`.
- Matched cross-level datasets remain separate: `105` groups each for SALICON, CAT2000, and COCO-Search18.
- New paper-pack artifacts: `outputs/paper_inspection_v1/tables/table9_matched_cross_level_correlations.md` and `outputs/paper_inspection_v1/figures/figure5_matched_cross_level_correlations.png`.
- Benchmark-style per-target encoding scope: mixed because four hV4 targets have `noise_ceiling=0.0`; `289620` rows are `benchmark_style_noise_normalized` and `120` rows are intentionally left `benchmark_style_non_noise_normalized`.
- Matched-panel reporting is now implemented separately from the mixed-scope neural ranking. The full-image-count validation-selected `flatten_pca` panel is now complete for all six planned model families across all four PRF visual ROIs: `resnet50`, `convnext_tiny`, `deit_small_patch16_224`, `vit_base_patch16_224`, `vit_small_patch14_dinov2`, and `vit_base_patch16_clip_224`.
- Final matched-panel mean noise-normalized ranking: `vit_small_patch14_dinov2` `0.591`, `vit_base_patch16_clip_224` `0.581`, `resnet50` `0.581`, `deit_small_patch16_224` `0.562`, `vit_base_patch16_224` `0.534`, `convnext_tiny` `0.510`.
- Artifact audit path: `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`; current status is `24` complete cells and `0` missing/skipped cells.

Current accepted neural ranking:

- Use the six-model matched full-image `flatten_pca` ranking for cross-model claims: `vit_small_patch14_dinov2` first, then `vit_base_patch16_clip_224`, `resnet50`, `deit_small_patch16_224`, `vit_base_patch16_224`, and `convnext_tiny`.
- Each matched-panel model ranking row aggregates `9654` valid positive-ceiling targets and excludes `4` zero-ceiling hV4 targets from noise-normalized aggregates.
- Current ROI set: `V1`, `V2`, `V3`, `hV4` for `subj01`.
- Detailed per-ROI/layer/alpha numbers belong in `docs/project_results_numbers.md`, not in this steering file.
- The DINOv2 learned spatial readout materially improves all four PRF visual ROIs over DINOv2 `flatten_pca`, but it is not method-matched to the other backbones and should not be used as the primary cross-model row.

Matched cross-level readout:

- The matched cross-level table uses only the six-model full-image `flatten_pca` panel and excludes learned-readout and other non-matched provenance rows.
- Across-ROI behavior-vs-encoding correlations are small-`n` descriptive results (`n=4` transformer-only groups or `n=6` full matched groups). They are useful for designing the paper-grade experiment, not for making a final claim.
- Grad-CAM across-ROI NSS groups are marked `insufficient_models` because only `resnet50` and `convnext_tiny` have matched Grad-CAM behavioral rows in the six-model panel.
- Lower-is-better behavioral metrics such as KL are sign-aligned before correlation while retaining the raw behavioral mean in the observation table.

V1 ROI-expanded discovery matrix:

- Path: `outputs/paper1_experiment_v1/summary/`.
- Artifact audit: `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`; all `9` audit checks pass.
- Encoding scope: `40` completed full-image `flatten_pca` cells across four models x ten ROIs, with `148144` target rows.
- Geometry scope: `40` completed cells, each with `10` valid geometry rows: one `linear_cka_full9841` row and nine deterministic `subset_rsa` rows from subset sizes `512`, `1024`, and `2048` with seeds `123`, `456`, and `789`.
- V1 encoding ranking across ten ROIs: `vit_small_patch14_dinov2` `0.556`, `resnet50` `0.537`, `vit_base_patch16_clip_224` `0.521`, and `vit_base_patch16_224` `0.502`.
- V1 full-image CKA ranking: `vit_small_patch14_dinov2` `0.194`, `resnet50` `0.187`, `vit_base_patch16_clip_224` `0.099`, and `vit_base_patch16_224` `0.089`.
- V1 subset-RSA rankings are not identical to CKA: most subset-RSA variants rank `resnet50` first, `vit_small_patch14_dinov2` second, `vit_base_patch16_224` third, and `vit_base_patch16_clip_224` fourth; two subset seeds rank DINOv2 first.
- CKA/subset-RSA rank agreement at `across_roi_mean` is complete but moderate: Spearman is mostly `0.6`, occasionally `0.8`; Kendall is mostly `0.333`, occasionally `0.667`.

Interpretation:

- The project has a complete full-image-count matched `flatten_pca` panel for six model families, plus a stronger DINOv2 learned-readout method result.
- The matched-panel ranking is now the accepted basis for cross-model neural comparisons: `vit_small_patch14_dinov2` first, `vit_base_patch16_clip_224` second, `resnet50` third, `deit_small_patch16_224` fourth, `vit_base_patch16_224` fifth, and `convnext_tiny` sixth by mean valid-target noise-normalized score.
- The previous test-set feedback risk for layer choice has been addressed for the current one-subject PRF visual ROI baselines by validation-only layer selection.
- The V1 discovery, robustness, and uncertainty matrices change the scientific state: geometry robustly favors DINOv2 across confirmatory subjects, while encoding is subject-sensitive enough that the accepted aggregate label is `geometry_replicated_encoding_ambiguous`.

## Global Direction Rationale

The project direction is a multi-axis NeuroAI alignment study. The central goal is to test whether fixation alignment, neural encoding, and representational geometry measure the same underlying “human-likeness” factor or whether they dissociate across architecture, attribution family, ROI, and viewing regime.

The project should be shaped around this question:

> Which models are aligned behaviorally, neurally, and geometrically, and where do those axes fail to agree?

The behavioral layer is an alignment axis, not the main paper by itself. The neural encoding layer is a local brain-alignment axis, not an Algonauts leaderboard claim. The geometry layer is a latent representational axis, not a replacement for encoding. The paper becomes interesting only when these axes are analyzed jointly.

### Current interpretation of existing results

The current behavioral results show that the corrected scoring protocol is sane: DeepGaze beats center bias, and generic classifier explanation maps remain below dedicated fixation references. This supports the pipeline, but it is not central novelty.

The current matched neural panel shows a plausible local ranking over six models and four PRF ROIs, with DINOv2, CLIP ViT, and ResNet-50 near the top. This is useful diagnostic evidence, while the reduced subject-robustness panel is now the claim-relevant check for the four-model V1 PRF-ROI scope.

The DINOv2 learned spatial readout is a strong method-provenance result because it improves all four PRF visual ROIs over DINOv2 `flatten_pca`. It should motivate later readout/adaptive-sampling work, but it should not enter cross-model headline rankings until the same readout protocol exists for all compared models.

The matched geometry axis is the current most important addition because it can expose cases where encoding and representational geometry disagree. Geometry should be used to test dissociation, not to create a separate leaderboard.

### Current priority

The immediate priority is attribution-family hardening for Paper 1. Behavioral controls are now strong enough for the current static-image scope: SALICON/CAT2000 have center bias, DeepGaze IIE, DeepGaze MSDB, and observer context; COCO-Search18 has center bias, observer context, a task-conditioned prior, and DeepGaze IIE only as a diagnostic free-viewing reference.

The current V1 behavioral, neural, geometry, geometry-method sensitivity, subject-robustness, margin-uncertainty, and behavioral-control outputs are claim-relevant. Future Codex sessions should not expand the generic model zoo or efficiency axis before adding at least one stronger transformer attribution/relevance family and updating the cross-axis reporting to keep attribution families separate.

The next Codex work should therefore focus on:

1. **Transformer Attribution-Family Control**

   Add one stronger transformer relevance method for ViT/DINO/CLIP-style models, preferably Chefer-style transformer attribution or AttnLRP-style relevance propagation. The implementation must label it as a relevance/attribution object, not operational attention.

2. **Scoped Behavioral Evaluation**

   Evaluate the new attribution family only on the existing static benchmark scope needed for Paper 1. Keep free-viewing SALICON/CAT2000 separate from COCO-Search18 task search and avoid broad leaderboard expansion.

3. **Cross-Axis Reporting Hygiene**

   Update aggregate/reporting code so Grad-CAM, gradients, rollout, perturbation, and relevance-style maps remain separate saliency families in cross-axis summaries.

4. **Outcome-first implementation**

   Each Codex session should produce an artifact that changes the paper’s scientific state.

   Good session outputs include:

   * a stronger transformer relevance attribution row with accepted configs and tests;
   * a reporting update that prevents attention rollout from being interpreted as human-like attention;
   * a documented decision that rejects an attribution method as infeasible or scientifically mismatched.

   Poor session outputs include:

   * more logging;
   * more paper-pack formatting;
   * more plots of the current diagnostic matrix;
   * small improvements to stale summaries;
   * broad model-zoo expansion without a claim-driven design.

5. **Robustness before polishing**

   Subject-robustness uncertainty, paper-facing synthesis, observer-control integration, task-search baseline, and DeepGaze MSDB free-viewing control are summarized. The next cleanup should be attribution-family hardening, not cosmetic plotting or model expansion.

   The correct order is:

   1. inspect existing saliency-method registry and transformer model wrappers;
   2. add one scoped transformer relevance method with tests and debug configs;
   3. run a small static benchmark smoke pass before any full matrix expansion.

   Do not reverse this order.

### Explicit non-priorities

Do not prioritize:

* broad model-zoo expansion before a paper-grade scope, ROI expansion, and controls;
* larger behavioral leaderboards before human ceilings or stronger fixation baselines;
* video, scanpath, recurrent policies, or adaptive attention before Paper 1’s static-image dissociation claim is stabilized;
* new DINOv2-only readout variants unless they test a clearly defined Paper 2 intervention;
* paper-pack polishing unless accepted evidence changes;
* leaderboard-style claims about best model, best architecture, or best attention method.

### Milestone order

The current milestone order is:

1. Freeze the current six-model/subj01/V1–hV4 results as diagnostic validation outputs, not paper-grade evidence.
2. Complete the V1 four-model x ten-ROI discovery matrix for `subj01`.
3. Apply geometry-method sensitivity and the V1 failure gate. **Complete.**
4. Run subject robustness on the reduced `subj02`-`subj04` panel. **Complete:** aggregate decision is `partial`; geometry replicates across all confirmatory subjects, while encoding flips to ResNet-50 in `subj04`.
5. Quantify DINOv2-vs-ResNet subject-robustness uncertainty and margins. **Complete:** aggregate decision is `geometry_replicated_encoding_ambiguous`.
6. Build paper-facing geometry-first dissociation framing and integrate generated observer controls. **Complete.**
7. Harden behavioral controls with a modern free-viewing fixation reference if feasible and a task-specific COCO-Search18 baseline before stronger task-search interpretation. **Complete:** the behavioral-control audit accepts DeepGaze MSDB for SALICON/CAT2000, observer controls are integrated, and the COCO-Search18 task prior is accepted.
8. Add attribution-family controls before any attention-specific interpretation. **Current next milestone.**
9. If the V1 matrix does not produce a defensible claim after those controls, demote Paper 1 to a methods/workshop paper and shift the main publication effort toward Paper 2’s causal adaptive-attention intervention.

### Decision rule

Do not treat the V1 full-image CKA convergence as a universal model-ranking result. The accepted reading is narrower: geometry replicates for DINOv2 across confirmatory PRF-ROI subjects, but encoding remains subject-sensitive.

Robustness can strengthen a decision-clean pattern. It cannot rescue an interpretation that depends on silently choosing one geometry method.

Continue Paper 1 toward a top venue only if the upgraded experiment can say more than:

> DINOv2 leads under encoding and CKA in one subject.

The desired paper-level statement is:

> Under a controlled, sufficiently broad cross-axis experiment, fixation alignment, neural encoding, and representational geometry converge or dissociate in identifiable model/ROI/task regimes.

That statement now requires stronger behavioral controls around the already synthesized `geometry_replicated_encoding_ambiguous` result before broader controls or model expansion.

### Relevant SOTA references:

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

The voxel-specific readout decision is complete and the matched neural panel is complete. Do not expand this inventory again before the paper-grade experiment spec defines the model panel, ROI scope, and accepted attribution families. The next model-family comparison should remain narrow and methodologically matched, not a broad model zoo.

## What Is Already Built

Behavioral infrastructure:

- Manifest loaders for SALICON, CAT2000, COCO-Search18, and NSD / Algonauts-style data.
- Fixation parsers for SALICON and CAT2000 `.mat` files.
- SALICON official JSON annotation conversion to worker-level observer manifests via `scripts/create_salicon_observer_manifest.py`.
- Task/scanpath point handling for COCO-Search18, including target-present and target-absent train/validation annotations.
- Inter-observer control summaries for inline JSON fixation rows and `.mat` fixation rows; inline fixation rows are scaled from original image size to evaluation map size.
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
- Full-image-count PRF ROI summaries, model rankings, ROI winners, and matched cross-level behavior-neural tables.
- V1 ROI-expanded full-image `flatten_pca` summaries for four models x ten ROIs.
- V1 ROI-expanded geometry scoring and summaries for `linear_cka_full9841` plus deterministic subset-RSA.
- Additional Algonauts subject full PRF visual ROI manifests are prepared for `subj02`, `subj03`, and `subj04`, with per-image `V1`/`V2`/`V3`/`hV4` response files.

Reporting infrastructure:

- Corrected behavioral aggregate and merged SSL/VLM aggregate.
- Neural ROI summary tables.
- Paper inspection pack with behavior, neural, matched cross-level, SSL/VLM candidate, benchmark sanity, subject-robustness interpretation, observer-control summary tables, and an academic SOTA context section comparing the current figures against MIT/Tuebingen saliency, DeepGaze IIE SALICON, COCO-Search18 task-search, and Algonauts 2023 evaluation references.
- Paper inspection README now explicitly distinguishes mixed-scope diagnostics from the complete six-model matched full-image-count PRF ROI `flatten_pca` panel, includes the four DINOv2 learned spatial readout rows only as method-provenance context, states the geometry-first dissociation / measurement framing, and labels observer-control rows as human/interobserver context rather than model performance.

## Current Implementation Progress

Updated: 2026-06-10

Current implementation state:

- Benchmark-style neural scoring, NSD-derived noise-ceiling metadata, full-image-count manifests, train-only `flatten_pca`, validation-only layer/pooling selection, learned spatial readout, matched geometry scoring, and paper inspection reporting are implemented.
- The matched neural panel is complete for all `24` expected six-model x four-ROI `subj01` cells and is the accepted cross-model neural comparison object.
- The matched geometry axis is complete for all `24` expected cells using full-image `linear_cka`; it is useful diagnostic evidence but still not a complete geometry claim without method/seed stability.
- The DINOv2 learned spatial readout rows are retained as method provenance because the readout is not method-matched across all backbones.
- The current paper inspection figures are diagnostic figures, not final main-paper figures. They should be replaced or supplemented after the paper-grade experiment spec defines the actual accepted evidence tables.
- Paper-Grade Experiment Definition V1 is now complete as a planning milestone. The new spec fixes the next falsifiable matrix as `subj01`, four method-matched models, PRF ROIs plus stream ROIs, full-image `flatten_pca` encoding, and `linear_cka_full9841` plus deterministic `subset_rsa` geometry.
- The V1 config is a forward-looking experiment contract, not a replacement for `configs/paper1_config.yaml`, which remains the scope file for the current diagnostic PRF-only results.
- The new validation test `tests/test_paper1_experiment_spec.py` locks required model names, ROI names, accepted artifact paths, exclusion rules, and cmd-only verification-command documentation.
- The `subj01` V1 stream-ROI manifest/config readiness path is now implemented. The stream manifest exists, the ROI-expanded config directory contains the expected `40` full-image `flatten_pca` validation-selection configs, and the readiness audit passes.
- The V1 readiness generator is `scripts/create_paper1_v1_roi_expanded_configs.py`; it reads `configs/paper1_experiment_v1.yaml`, generates the stream manifest/configs, and writes `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`.
- Stream ROI noise ceilings are now generated from NSD ncsnr files and attached to `data/manifests/nsd_algonauts_subj01_streams_full_manifest.csv`. The `resnet50`/`midventral` V1 smoke rerun has `noise_ceiling_available=true`, `1913` valid noise-ceiling targets, `4` zero-ceiling targets, and downstream summaries use the populated noise-normalized score.
- The full V1 ROI-expanded `subj01` encoding matrix is now complete for all `40` expected cells: four models x ten ROIs. The generated summary is in `outputs/paper1_experiment_v1/summary/`.
- V1 ROI-expanded mean noise-normalized encoding ranking across ten ROIs is: `vit_small_patch14_dinov2` `0.556`, `resnet50` `0.537`, `vit_base_patch16_clip_224` `0.521`, and `vit_base_patch16_224` `0.502`.
- The V1 encoding summary has `40` encoding rows and `148144` target rows. Target-level noise-ceiling scope is mostly normalized: `148108` normalized target rows and `36` non-normalized target rows caused by zero ceilings.
- The full V1 ROI-expanded `subj01` geometry matrix is now complete for all `40` expected cells. Each cell has `10` valid geometry rows: `linear_cka_full9841` plus deterministic `subset_rsa` at sizes `512`, `1024`, and `2048` with seeds `123`, `456`, and `789`.
- V1 ROI-expanded full-image CKA ranking across ten ROIs is: `vit_small_patch14_dinov2` `0.194`, `resnet50` `0.187`, `vit_base_patch16_clip_224` `0.099`, and `vit_base_patch16_224` `0.089`.
- V1 deterministic subset-RSA rankings are method-sensitive: most subset-RSA variants rank `resnet50` first and `vit_small_patch14_dinov2` second, while two subset-RSA seeds rank DINOv2 first. CKA/subset-RSA across-ROI rank agreement is complete but moderate.
- The V1 CKA-primary cross-axis summaries exist: `693` cross-level correlation rows and `2079` cross-axis decision rows. Current CKA-primary decision rows are dominated by `insufficient_models` (`1386`), with `627` `stable_convergence` and `66` `unstable` rows. The stable across-ROI rows are concentrated in `vanilla_gradient` behavior and `linear_cka_full9841` geometry.
- The V1 geometry-method sensitivity synthesis now exists: `2079` rows in `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_sensitivity_decisions.csv`. Labels are `908` `stable_across_geometry_methods`, `693` `not_tested`, `462` `insufficient_models`, and `16` `direction_conflict`. The table preserves free-viewing versus task-search separation with `1386` free-viewing rows and `693` task-search rows, and includes `linear_cka_full9841` plus all nine deterministic subset-RSA variants.
- The V1 failure-gate summary now exists at `outputs/paper1_experiment_v1/summary/roi_expanded_failure_gate_summary.csv`. The current gate decision is `paper_pack_geometry_first_dissociation` because at least one geometry relationship is stable across CKA and subset-RSA, so Paper 1 should proceed as geometry-first dissociation or measurement evidence rather than universal model ranking.
- The reduced V1 subject-robustness scaffold is implemented for `subj02`-`subj04`: `configs/paper1_experiment_v1.yaml` has an executable `confirmatory_matrix`, `scripts/create_paper1_v1_subject_robustness_configs.py` generates `48` subject/model/ROI configs under `configs/experiments/paper1_experiment_v1/neural_subject_robustness/`, and `outputs/paper1_experiment_v1/summary/subject_robustness_artifact_audit.csv` passes all scaffold checks.
- Local subject-robustness progress: `subj02`, `subj03`, and `subj04` each have `16/16` expected encoding output directories and completed confirmatory geometry. The regenerated rank-only decision table marks `subj02` and `subj03` as `replicated`, `subj04` as `partial`, and `all_confirmatory_subjects` as `partial`.
- The subject-robustness geometry runner now prints progress by default at subject, cell, and geometry-method/subset granularity, with `--no-progress` available for quiet runs.
- The geometry result is robust: `vit_small_patch14_dinov2` is the full-image CKA leader and the subset-RSA leader for all `30/30` subject x geometry-method rankings across `subj02`-`subj04`. The encoding result is ambiguous: `vit_small_patch14_dinov2` leads raw mean encoding by tiny margins in `subj02` and `subj03`, while `resnet50` leads in `subj04`.
- Subject-robustness uncertainty is now implemented. `outputs/paper1_experiment_v1/summary/subject_robustness_uncertainty_decisions.csv` labels `subj02` and `subj03` as `geometry_replicated_encoding_supported`, `subj04` as `geometry_replicated_encoding_resnet_supported`, and the aggregate as `geometry_replicated_encoding_ambiguous`. The aggregate label is intentionally conservative because the subject-level encoding direction conflicts even though the pooled target-level DINOv2-minus-ResNet encoding margin is positive.
- Paper-facing geometry-first synthesis and observer-control integration are now implemented. `outputs/paper1_experiment_v1/summary/subject_robustness_paper_interpretation.csv` preserves the `subj04` ResNet-50 encoding reversal and aggregate `geometry_replicated_encoding_ambiguous` decision, while `outputs/paper1_experiment_v1/summary/behavioral_observer_control_summary.csv` separates SALICON free-viewing observer context from COCO-Search18 task-search observer context.

Latest session report:

1. `Scientific change`: the modern free-viewing fixation-reference control is now implemented and accepted for the current Paper 1 static-image scope. DeepGaze MSDB rows are scored and merged for SALICON/CAT2000 with `2000` images per dataset and `7` metrics per dataset. MSDB improves the free-viewing NSS reference from DeepGaze IIE `1.743` to MSDB `1.760` on SALICON and from DeepGaze IIE `1.838` to MSDB `1.979` on CAT2000.
2. `Accepted artifact`: `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv`, merged `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`, refreshed `outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv`, and `outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv`.
3. `Claim impact`: strengthens behavioral-control claim hygiene by adding a modern dedicated free-viewing fixation reference before stronger fixation-alignment claims. It raises the behavioral baseline for SALICON/CAT2000, but it does not by itself change the neural/geometry dissociation result.
4. `Reviewer risk reduced`: addresses the objection that free-viewing behavioral controls relied only on center bias and older DeepGaze IIE. The audit now marks the modern free-viewing reference as `accepted`, while COCO-Search18 DeepGaze IIE remains diagnostic and the COCO-Search18 task prior remains the accepted task-search baseline.
5. `Next decisive step`: add an attribution-family control, preferably a stronger transformer relevance method, before any attention-specific interpretation or efficiency/model-zoo expansion.

Implementation history was moved to `docs/project_status_changelog.md`.

## Cluster Workflow Guidance

Use the JHU DSAI cluster for long GPU or high-I/O jobs, including full-dataset saliency map export, full-image neural encoding reruns, large geometry regeneration, broad benchmark scoring, or other runs that would tie up the laptop for hours. Keep smoke tests, audit scripts, config generation, small unit tests, and result inspection local unless a local dependency or device issue blocks progress.

Cluster account and workspace:

- Login: `zzhan330@dsailogin.arch.jhu.edu`.
- Project workspace: `/scratch/tshu2/zzhan330/saliency`.
- Use git for tracked source/config/test changes whenever possible.
- Use WSL `rsync` for large or untracked data, generated maps, model caches, and output directories.
- Do not rely on git alone for raw datasets, precomputed artifacts, or generated outputs.

Recommended laptop-to-cluster pattern:

1. Commit/push tracked code changes when appropriate, then `git pull` on the cluster.
2. If changes are not ready to commit, sync the working tree from WSL.
3. Sync only the data roots required by the specific job.
4. Run a small smoke job on the cluster before launching the full Slurm job.
5. Copy only the required outputs back to the laptop.
6. Run local audits/summaries after outputs return, then update this status file.

Generic working-tree sync from Windows `cmd.exe` through WSL:

```cmd
wsl -e bash -lc "cd /mnt/d/Git/saliency && rsync -av --delete --exclude '.git/' --exclude '.venv/' --exclude '.pytest_tmp/' --exclude '__pycache__/' --exclude 'outputs/' ./ zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/"
```

Generic data sync template:

```cmd
wsl -e bash -lc "cd /mnt/d/Git/saliency && rsync -av <LOCAL_DATA_OR_ARTIFACT_PATH>/ zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/<REMOTE_DATA_OR_ARTIFACT_PATH>/"
```

Generic output return template:

```cmd
wsl -e bash -lc "cd /mnt/d/Git/saliency && rsync -av zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/<REMOTE_OUTPUT_PATH>/ <LOCAL_OUTPUT_PATH>/"
```

Cluster-side Slurm policy:

- Put Slurm scripts and logs under `/scratch/tshu2/zzhan330/saliency/slurm_logs/` or another project-local log directory.
- Prefer `sbatch` for long jobs and `srun --pty` only for debugging.
- Use a small `--max-items` or reduced-scope command before the full job whenever the code path has not already run on that environment.
- Monitor with `squeue -u zzhan330` and `tail -f` on the relevant Slurm log.
- After cluster completion, verify expected file counts before copying outputs back.

## Next Concrete Milestone

Priority: **Attribution-Family Control For Transformer Relevance**.

The paper-pack geometry-first framing, subject robustness, observer-control integration, COCO-Search18 task prior, and SALICON/CAT2000 DeepGaze MSDB control milestones are complete. Do not rerun `subj01` encoding, confirmatory subject encoding, confirmatory geometry, observer-control generation, task-prior scoring, or MSDB export/scoring unless an audit regresses or a concrete data-integrity blocker is discovered.

The next decisive task is to reduce attribution ambiguity before any attention-specific interpretation: add one stronger transformer relevance/attribution method for the existing transformer models, evaluate it in a narrow static-image benchmark scope, and update reporting so rollout, gradients, CAM, perturbation, and relevance-style maps remain separate evidence families.

### Required outcome

By the end of the next milestone, the project should have a reviewer-facing attribution-family control that prevents attention rollout from carrying attention-specific claims:

- Done: paper-facing subject-robustness interpretation at `outputs/paper1_experiment_v1/summary/subject_robustness_paper_interpretation.csv`.
- Done: V1 observer-control summary at `outputs/paper1_experiment_v1/summary/behavioral_observer_control_summary.csv`.
- Done: paper inspection tables `outputs/paper_inspection_v1/tables/table14_subject_robustness_interpretation.csv` and `outputs/paper_inspection_v1/tables/table15_observer_control_summary.csv`.
- Done: a behavioral-control gap audit at `outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv`.
- Done: a task-specific COCO-Search18 baseline, kept separate from SALICON/CAT2000 free-viewing claims, at `outputs/real_matrix_v2_task_search_baseline/aggregated/results.csv`.
- Done: a feasibility decision table for adding DeepGaze MSDB or another modern free-viewing fixation reference without turning the project into a saliency leaderboard at `outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv`.
- Done: SALICON/CAT2000-only DeepGaze MSDB precomputed-map configs under `configs/experiments/real_matrix_v2_references_msdb/`.
- Done: SALICON/CAT2000-only DeepGaze MSDB scoring and merged behavioral aggregate at `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv` and `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- New next target: one scoped transformer relevance method with tests, debug configs, benchmark smoke output, and updated attribution-family labels in reporting.

### Completed behavioral-control implementation work

The behavioral-control audit script is implemented at `scripts/audit_behavioral_controls.py` and writes `outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv`. The COCO-Search18 task-specific baseline is implemented as `coco_search18_task_prior` and configured at `configs/experiments/real_matrix_v2/coco_search18_static2000__coco_search18_task_prior_baseline_coco_search18_task_prior.yaml`. DeepGaze MSDB is implemented through precomputed-map export/scoring and merged into the accepted behavioral aggregate.

The audit separates accepted controls, diagnostic controls, and missing controls:

- Accepted: DeepGaze MSDB, DeepGaze IIE, and center bias for SALICON/CAT2000 free-viewing; center bias for COCO-Search18 task search; SALICON observer controls; COCO-Search18 observer controls; the task-specific COCO-Search18 target/task-conditioned prior; and the metric-boundary separation between point-fixation metrics and map-distribution metrics.
- Diagnostic: COCO-Search18 DeepGaze IIE, because it is a free-viewing reference used on task-search data rather than a task-specific search baseline.

### Next Codex session implementation plan

Implement a scoped transformer relevance control, not neural reruns:

1. Inspect `src/hma/saliency/attention_rollout.py`, `src/hma/saliency/gradients.py`, `src/hma/saliency/registry.py`, model wrapper support, and benchmark config patterns.
2. Choose one implementable transformer relevance method, preferably Chefer-style transformer attribution or AttnLRP-style relevance propagation. If neither is feasible without large dependency churn, write an explicit infeasibility decision and choose the smaller scientifically defensible relevance-style control.
3. Add the method under `src/hma/saliency/` with a distinct method/family label such as `transformer_relevance`, not `attention`.
4. Add focused synthetic/unit tests for tensor shape, normalization compatibility, unsupported-model behavior, and registry/config dispatch.
5. Add a debug config for a tiny SALICON or CAT2000 subset and run a smoke benchmark before any full static2000 run.
6. Update reporting/cross-axis summaries only as needed to keep attribution families separate and prevent attention rollout from being interpreted as operational attention.
7. Do not add broad model-zoo rows, efficiency profiling, neural reruns, or COCO-Search18-specific attribution expansion before the scoped transformer relevance control is working and documented.

### Subject-robustness runbook

Generate or refresh the config scaffold and audit:

```cmd
.\.venv\Scripts\python.exe scripts\create_paper1_v1_subject_robustness_configs.py --config configs\paper1_experiment_v1.yaml
```

Local encoding already completed for `subj02`, `subj03`, and `subj04`. Keep these commands for reproducibility, but do not rerun them unless an audit fails:

```cmd
for %F in (configs\experiments\paper1_experiment_v1\neural_subject_robustness\subj02\*.yaml) do .\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config "%F"
```

```cmd
for %F in (configs\experiments\paper1_experiment_v1\neural_subject_robustness\subj03\*.yaml) do .\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config "%F"
```

Historical local `subj04` encoding command, retained for reproducibility only:

```cmd
for %F in (configs\experiments\paper1_experiment_v1\neural_subject_robustness\subj04\*.yaml) do .\.venv\Scripts\python.exe scripts\run_neural_alignment.py --config "%F"
```

The JHU DSAI cluster path, local geometry regeneration, and subject-robustness synthesis are no longer active blockers for the current milestone. Do not rerun them unless an audit regresses.

Historical geometry regeneration command, retained for reproducibility only:

```cmd
.\.venv\Scripts\python.exe scripts\compute_paper1_v1_subject_robustness_geometry.py --config configs\paper1_experiment_v1.yaml --skip-existing
```

Subject-robustness summary and uncertainty regeneration command, retained for reproducibility:

```cmd
.\.venv\Scripts\python.exe scripts\summarize_paper1_v1_subject_robustness_results.py --config configs\paper1_experiment_v1.yaml
```

### Scope decision rules

- Keep `configs/paper1_config.yaml` unchanged as the diagnostic PRF-only result scope.
- Keep V1 fLOC category ROIs out of scope unless `docs/paper1_experiment_spec_v1.md` is explicitly revised.
- Do not add generic new models, efficiency, or neural reruns before the scoped transformer relevance attribution control is completed or explicitly rejected as infeasible.
- Use cmd-form commands in any docs or handoff text.

### Acceptance criteria

The behavioral-control audit and task-search baseline milestone is complete because:

- A behavioral-control gap audit exists under `outputs/paper1_experiment_v1/summary/` and separates `free_viewing` from `task_search`.
- The audit marks current DeepGaze MSDB, DeepGaze IIE, center bias, observer-control summaries, and the task-specific COCO-Search18 prior as `accepted`, while keeping COCO-Search18 DeepGaze IIE `diagnostic`.
- The audit identifies DeepGaze MSDB as the accepted modern free-viewing fixation reference for SALICON/CAT2000.
- The audit identifies the task-specific COCO-Search18 baseline as accepted and keeps COCO-Search18 DeepGaze IIE diagnostic.
- No SALICON/CAT2000 free-viewing row is pooled with COCO-Search18 task-search rows in a single behavioral headline.
- Any long-running commands are split into copy-pastable `cmd` batches for the user.

The free-viewing-reference feasibility milestone is complete because:

- `outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv` exists with rows for DeepGaze MSDB, current DeepGaze IIE, and a comparable modern free-viewing reference option.
- The decision table explicitly marks whether each candidate is `feasible_now`, `requires_download_or_dependency`, or `defer_or_document_limitation`.
- DeepGaze MSDB is marked `feasible_now`; the current DeepGaze IIE reference is marked `feasible_now`; unnamed comparable references are marked `defer_or_document_limitation`.
- The behavioral-control audit marks the modern free-viewing reference row as `accepted`.

The MSDB export/evaluation milestone is complete because:

- SALICON MSDB maps exist locally under `data/precomputed/deepgaze_msdb/salicon_static2000/`; CAT2000 MSDB maps were scored on the cluster and the per-image metrics were copied back rather than all raw maps.
- SALICON/CAT2000-only configs continue to exist under `configs/experiments/real_matrix_v2_references_msdb/`.
- `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv` exists and contains only SALICON/CAT2000 free-viewing rows for `deepgaze_msdb_reference`.
- The merged accepted behavioral aggregate contains `14` DeepGaze MSDB rows and the behavioral-control audit accepts the modern free-viewing reference.

The next attribution-family milestone is complete only if:

- one transformer relevance method exists with a distinct method/family label;
- focused tests cover registry/config dispatch and unsupported-model behavior;
- a small benchmark smoke output exists for the new method;
- reporting text or tables keep rollout and relevance-style attribution separate;
- the status file records whether the method is accepted evidence, diagnostic only, or infeasible.

## Data/control readiness update:

- Algonauts `subj02`, `subj03`, and `subj04` raw training images, training fMRI arrays, and ROI masks are present under `data/raw/nsd_algonauts/`.
- Full PRF visual ROI response manifests are generated:
  - `data/manifests/nsd_algonauts_subj02_prf_visualrois_full_manifest.csv`: `39364` rows; response dimensions V1 `2737`, V2 `2779`, V3 `2615`, hV4 `1262`.
  - `data/manifests/nsd_algonauts_subj03_prf_visualrois_full_manifest.csv`: `36328` rows; response dimensions V1 `2676`, V2 `2991`, V3 `2418`, hV4 `887`.
  - `data/manifests/nsd_algonauts_subj04_prf_visualrois_full_manifest.csv`: `35116` rows; response dimensions V1 `2328`, V2 `2474`, V3 `2146`, hV4 `1190`.
- Combined four-subject audit manifest: `data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv`, `150172` rows.
- Per-ROI bounded validation passed for `subj02`-`subj04` using `25` sampled rows per subject/ROI. Full all-row validation is intentionally avoided because it touches more than `100k` response files and is slow on Windows.
- `subj01` has local ROI masks for PRF visual ROIs, streams, and fLOC category maps under `data/raw/nsd_algonauts/subj01/roi_masks/`. V1 now uses PRF visual ROIs plus streams; fLOC category maps are deferred.
- COCO-Search18 target-absent data is integrated. `data/manifests/coco_search18_manifest.csv` now has `74646` rows: `49760` target-present and `24886` target-absent rows. The V2 static manifest now has `2000` validation rows: `1338` target-present and `662` target-absent.
- SALICON official JSON annotations are integrated. `data/manifests/salicon_observer_annotations_manifest.csv` has `893854` worker-level rows with explicit `worker_id`; the V2 subset `data/manifests/v2/salicon_static2000_observer_annotations_manifest.csv` has `125874` rows over `2000` images.
- Observer-control outputs are generated:
  - `outputs/observer_controls_v2/coco_search18_static2000_observer_controls.csv`: `1867` rows.
  - `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`: `1384614` leave-one-observer-out rows over `2000` images; the paper-facing summary streams this file instead of loading it wholly into memory.
- COCO-Search18 task-search baseline output is generated:
  - `outputs/real_matrix_v2_task_search_baseline/aggregated/results.csv`: `7` metric rows for `coco_search18_task_prior_baseline`.
  - `outputs/real_matrix_v2/coco_search18_static2000/coco_search18_task_prior_baseline_coco_search18_task_prior/aggregate_metrics.json`: `2000` validation rows; NSS `2.199`, AUC-Judd `0.838`, shuffled-AUC `0.674`, CC `0.448`, SIM `0.338`, KL `1.538`.
- DeepGaze MSDB free-viewing reference output is generated and merged:
  - `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv`: `14` metric rows for `deepgaze_msdb_reference` across SALICON/CAT2000.
  - `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`: `399` merged behavioral rows after task-prior and MSDB integration.
  - SALICON MSDB NSS `1.760`; CAT2000 MSDB NSS `1.979`.

Current post-spec implementation priorities:

The V1 spec and config now define the falsifiable paper-grade matrix, the ROI-expanded encoding, geometry, and geometry-method sensitivity axes are complete for `subj01`, the reduced subject-robustness panel is complete with an uncertainty-aware aggregate decision of `geometry_replicated_encoding_ambiguous`, and the paper pack now includes geometry-first dissociation framing plus observer-control, task-search baseline, and modern free-viewing DeepGaze MSDB context. The immediate work is transformer attribution-family hardening before broader model or efficiency expansion.

- Use the completed additional subjects as a robustness follow-up:
  - interpret the reduced confirmatory panel: all three confirmatory subjects have local encoding and geometry, with an aggregate uncertainty decision of `geometry_replicated_encoding_ambiguous`;
  - keep the completed `subj01` ROI-expanded encoding and geometry matrix as the discovery reference;
  - do not add new models, attribution methods, or efficiency before behavioral-control hardening.
- Keep behavioral controls fixed before broader behavioral expansion:
  - use the generated SALICON and COCO-Search18 observer-control summaries as accepted reviewer-facing context;
  - treat the next behavioral control as targeted claim hardening, not as a broad saliency leaderboard expansion;
  - use the accepted SALICON/CAT2000 DeepGaze MSDB reference before stronger free-viewing claims;
  - use the implemented task-specific COCO-Search18 baseline before interpreting task-search alignment.
- Improve transformer attribution coverage:
  - add one stronger transformer relevance method, preferably Chefer-style transformer attribution or AttnLRP-style relevance propagation;
  - keep gradients, rollout, perturbation, and relevance-style maps as separate saliency families.
- Use paper-grade uncertainty now that the subject-robustness decision exists:
  - report target-level bootstrap intervals for DINOv2-vs-ResNet matched encoding margins;
  - report deterministic subset-RSA geometry margins across subjects, ROIs, subset sizes, and seeds;
  - keep leave-one-subject, leave-one-ROI, and Kendall tau sensitivity as robustness diagnostics.
- Add efficiency only after the above robustness pass:
  - collect FLOPs, latency, memory, token count, and retained-patch statistics for the matched panel;
  - regenerate alignment-per-compute summaries as exploratory diagnostics.

Completed milestones are archived in `docs/project_status_changelog.md`.

## Later Milestones

Proceed in phases that map directly to the research question.

Completed before this list: the V1 `subj01` geometry-method decision exists, the reduced `subj02`-`subj04` subject-robustness gate is complete with a rank-only `partial` aggregate decision, and the DINOv2-vs-ResNet uncertainty decision is complete with aggregate `geometry_replicated_encoding_ambiguous`. The phase order below is the execution order; it is deliberately different from the reviewer-risk severity list above.

1. **Subject robustness.** Complete. The reduced `subj02`-`subj04` panel shows robust geometry replication and ambiguous encoding replication.
2. **Uncertainty and sensitivity.** Complete for the DINOv2-vs-ResNet subject-robustness gate. Keep leave-one-subject, leave-one-ROI, Kendall tau, and model-label permutation checks as later diagnostics rather than current claim filters.
3. **Paper-facing synthesis and observer-control integration.** Complete. The paper pack now exposes the geometry-first dissociation claim and integrates generated observer-control summaries.
4. **Behavioral controls.** Complete for the current Paper 1 static-image scope. The task-trained COCO-Search18 baseline is implemented, observer controls are integrated, and DeepGaze MSDB is accepted for SALICON/CAT2000 free-viewing.
5. **Attribution-family controls.** Current next milestone. Add stronger transformer attribution before any attention-specific interpretation, while keeping gradient, Grad-CAM, rollout, perturbation, and relevance-style maps as separate evidence families.
6. **Cross-axis decision gate.** Use the decision table to identify weak implementation areas, not to assert paper-ready conclusions. Revisit publication framing only after ROI-expanded results, subject robustness, uncertainty, and observer controls materially improve the evidence base.
7. **Efficiency.** Add FLOPs, latency, token count, retained-patch statistics, and memory footprint for the matched model panel, then regenerate alignment-per-compute summaries. Keep efficiency exploratory unless it produces a clean dissociation or tradeoff.
8. **Brain-Score or Brain-Score-style external positioning.** Use it as context and sanity checking, not as a substitute for the local fixation/fMRI/geometry cross-level tests.
9. **Publication split.** Defer publication split decisions until the static-image cross-axis result and controls clarify whether Paper 1 is strong enough. Keep causal adaptive attention, foveation, adaptive token routing, scanpaths, video, or recurrent policies out of the immediate implementation path unless the robustness/control pass exposes a sharply defined intervention target.

## Code Pointers

Dataset loading and fixation parsing:

- `src/hma/datasets/salicon.py`
- `src/hma/datasets/cat2000.py`
- `src/hma/datasets/coco_search18.py`
- `src/hma/datasets/nsd_algonauts.py`
- `src/hma/datasets/fixation_parsers.py`
- `src/hma/datasets/fixation_utils.py`
- `scripts/create_salicon_observer_manifest.py`
- `scripts/summarize_observer_controls.py`

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
- `coco_search18_task_prior` in `src/hma/saliency/baselines.py` is the accepted task-specific COCO-Search18 behavioral-control baseline.

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
- `scripts/compute_matched_geometry.py`
- `scripts/summarize_paper1_v1_roi_expanded_results.py`
- `scripts/create_paper1_v1_subject_robustness_configs.py`
- `scripts/compute_paper1_v1_subject_robustness_geometry.py`
- `scripts/summarize_paper1_v1_subject_robustness_results.py`
- `scripts/audit_matched_neural_panel.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_behavioral_controls.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

For the latest behavioral-control audit and COCO-Search18 task-prior baseline implementation, focused tests passed:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_saliency_extraction.py tests\test_saliency_benchmark.py tests\test_audit_behavioral_controls.py tests\test_fixation_parsers_and_observer_controls.py tests\test_merge_behavioral_aggregates.py tests\test_paper_inspection_pack.py
```

Last known result for this session: `49 passed` with existing non-blocking PyTorch Grad-CAM hook warnings.

For the latest paper-pack geometry-first framing and observer-control integration, focused tests passed:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_paper_inspection_pack.py tests\test_paper1_v1_subject_robustness.py tests\test_fixation_parsers_and_observer_controls.py
```

Last known result for this session: `31 passed`.

Failure-gate reporting verification also passed:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_paper1_v1_geometry_and_summary.py
```

Last known result for this session: `5 passed`.

For the latest subject-robustness uncertainty implementation, focused tests passed:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_paper1_v1_subject_robustness.py tests\test_paper1_v1_geometry_and_summary.py
```

Last known result for this session: `13 passed`.

For the latest subject-robustness scaffold implementation, focused tests passed:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_paper1_v1_subject_robustness.py tests\test_paper1_v1_roi_expanded_generation.py tests\test_paper1_v1_geometry_and_summary.py tests\test_neural_roi_summary.py --basetemp=.pytest_tmp_subject_robustness
```

Last known subject-robustness scaffold result: `35 passed`.

For future neural/reporting implementation changes, use this focused verification command:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper1_experiment_spec.py tests\test_paper1_v1_geometry_and_summary.py tests\test_paper1_v1_subject_robustness.py
```

Last known focused result: `87 passed`.

For broader confidence after neural/reporting changes, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py tests\test_paper1_v1_roi_expanded_generation.py tests\test_paper1_experiment_spec.py tests\test_paper1_v1_geometry_and_summary.py
```

Last known broader result: `105 passed` with `--basetemp=.tmp_pytest_v1_broader`.

V1 ROI-expanded artifact audit command:

```cmd
.\.venv\Scripts\python.exe -c "import csv; from pathlib import Path; rows=list(csv.DictReader(Path('outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv').open(newline='',encoding='utf-8'))); print([(r['check'], r['status']) for r in rows])"
```

Last known V1 audit result: all `11` checks pass, including `40` encoding cells, `40` geometry cells, `10` geometry rows per cell, geometry-method sensitivity decisions present, and failure-gate summary present.

Matched panel audit command:

```cmd
.\.venv\Scripts\python.exe scripts\audit_matched_neural_panel.py
```

Last known matched panel audit result: `24` complete cells, `0` missing cells, `0` incomplete cells, and `0` explicitly skipped cells.

For full confidence after broad code changes, run:

```cmd
.\.venv\Scripts\python.exe -m pytest
```

Last known full result: `210 passed`; known non-blocking warnings remain PyTorch Grad-CAM hook warnings and Windows `.pytest_cache` permission warnings.
