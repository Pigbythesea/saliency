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

> Human-like fixation alignment, neural encoding, representational geometry, cortical stream structure, and computational efficiency are separable axes of visual alignment. Paper 1 should test whether these axes converge or dissociate across modern vision systems. The central question is whether models that look more human-like behaviorally also predict visual-cortex responses and neural representational geometry better, or whether behavioral attention, neural encoding, latent geometry, stream selectivity, and efficiency come apart in systematic ways.

The paper should be organized around a cross-axis outcome grid:

| fixation / behavioral alignment | neural encoding / geometry alignment | intended interpretation                                                                |
| ------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| high                            | high                                 | overt human-like selection may track brain-like representation                         |
| low                             | high                                 | representation convergence may emerge without human-like gaze or saliency              |
| high                            | low                                  | saliency-map mimicry or human-like output may arise from non-human internal processing |
| low                             | low                                  | weak alignment on both behavioral and neural axes                                      |

Codex must treat every implementation task as subordinate to building this cross-axis matrix. A task counts only if it directly clarifies the relationship among behavioral fixation alignment, neural encoding, representational geometry, ROI/stream specificity, model family, and efficiency.

### What counts as progress

A change counts as project progress only if it advances **Paper 1 Matrix V2**, meaning a model × axis × ROI/stream × efficiency matrix that can test convergence and dissociation among:

- behavioral fixation / saliency alignment;
- task-search or scanpath alignment where feasible;
- neural encoding across early, ventral, lateral/dorsal, and parietal/stream ROIs;
- representational geometry using CKA plus subset RSA or another metric family;
- computational efficiency, including parameters, FLOPs/MACs, latency, memory, token count, and retained-token fraction where applicable;
- model-family contrasts between CNN, plain ViT, self-supervised dense-feature models, vision-language models, efficient/hybrid models, and adaptive/foveated/selective-computation models.

A change counts as progress only if it produces at least one of the following:

- a new Matrix V2 accepted evidence table;
- a new model-family feasibility audit for modern frontier-relevant models;
- a new ROI/stream grouping table or neural-manifest expansion that enables ventral/dorsal/where-vs-what analysis;
- a behavioral-to-neural cross-axis table that can classify models into high/high, low/high, high/low, or low/low alignment quadrants;
- an efficiency profile that can be merged into alignment-per-compute analysis;
- a scanpath, foveation, token-selection, or adaptive-computation axis that tests the original attention-as-resource-allocation hypothesis;
- a negative decision that stops a stale branch, especially claim-table generation from the superseded interpretation.

Engineering achievements, successful smoke runs, code reorganization, new logs, new configs, and paper-pack updates do not count as scientific progress unless they directly produce one of the Matrix V2 outputs above.

Do not apply publication hardening before actual scientific discovery. First build the stronger Matrix V2 scientific object, then decide what the paper can claim.

### What Codex should prioritize

Codex should prioritize:

- modern model-family coverage over generic model count;
- stream/ROI structure over another flat PRF-only score table;
- efficiency and alignment-per-compute over decorative paper artifacts;
- adaptive/foveated/selective-computation mechanisms over more post-hoc heatmap variants;
- direct quadrant classification of models across behavioral, neural, geometry, and efficiency axes.

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
- `project_results_numbers.md`: current numerical result source of truth. Use it to inspect actual outcomes, margins, sample sizes, and confidence intervals before making any claim.
- `paper1_cross_axis_alignment_roadmap.md`: useful background, but partially superseded. Keep its cross-axis dissociation framing; revise its implementation priorities toward Matrix V2: modern model families, stream-level neural analysis, adaptive/foveated mechanisms, and efficiency.
- `paper1_literaturereview.md`: current literature review for Paper 1. It raises the required controls around dataset bias, scanpath/task specificity, subject variability, encoding reliability, representational-geometry metrics, and transformer attribution.
- `Literature Review and Research Redesign for the Human-Like Adaptive Visual Attention Project.md`: argues the project should become a multi-axis NeuroAI alignment study, not a saliency-map leaderboard.
- `Deep Research Assessment of the Human-Machine Visual Alignment Project.md`: emphasizes the publishable question as convergence versus dissociation among fixation alignment, neural predictivity, representational geometry, and efficiency.
- `hma_project_publication_critique_handoff.md`: current publication-readiness critique. It is a read-only reference for claim hygiene, stale-output cleanup, and top-venue risk assessment.
- `Zhang_Zihuan_zzhan330_proposal.docx`: original proposal; defines behavioral saliency, neural encoding, RSA, Brain-Score-style comparison, and compute efficiency as the core axes.
- `Comparing Human and Machine Visual Saliency_ A Comprehensive Review.pdf`: reinforces that fixation prediction requires strong controls such as center bias, DeepGaze-class references, point-based NSS/AUC, and separate treatment of free-viewing versus task-driven viewing.
- `__Attention and Saliency Map Extraction in Visual AI Models_ A Comprehensive Review__.pdf`: reinforces that gradients, CAMs, attention rollout, perturbation maps, LRP-style methods, and transformer attribution are different explanation objects and should not be collapsed into one "attention" score.

## Current Snapshot

The repository currently contains a strong diagnostic scaffold, not the final Paper 1 scientific matrix.

Completed scaffold layers:

- behavioral saliency / fixation benchmarking on SALICON, CAT2000, and COCO-Search18;
- accepted behavioral controls, including DeepGaze MSDB for SALICON/CAT2000, a COCO-Search18 task prior, center bias, observer controls, and transformer relevance as a separated attribution family;
- local Algonauts / NSD neural encoding for `subj01`, including a complete six-model PRF visual ROI `flatten_pca` panel and a four-model x ten-ROI discovery matrix;
- representational geometry using full-image CKA and deterministic subset RSA;
- reduced confirmatory subject robustness for `subj02`-`subj04` over PRF visual ROIs;
- DINOv2 learned spatial readout provenance;
- paper inspection and audit infrastructure.

Current diagnostic outcome:

- behavioral scoring is sane: dedicated fixation/task references outperform generic classifier attribution maps;
- DINOv2 is strong in current encoding and CKA results, but the ten-ROI subset-RSA top rank is method-sensitive and often favors ResNet-50;
- confirmatory-subject geometry favors DINOv2, while encoding is subject-sensitive and reverses to ResNet-50 in `subj04`;
- transformer relevance improves over rollout and vanilla gradients behaviorally on SALICON/CAT2000, but it remains a post-hoc attribution family rather than operational attention;
- the current evidence is useful as a pilot scaffold, but it is too narrow to define the final paper story.

Active missing layers for Paper 1 Matrix V2:

- modern model-family coverage beyond the current conservative anchors;
- explicit ventral/dorsal or stream-level neural analysis beyond PRF-only confirmatory robustness;
- efficiency and alignment-per-compute;
- adaptive, foveated, token-pruning, scanpath, or selective-computation models/mechanisms;
- a direct quadrant-style cross-axis analysis that classifies models by behavioral alignment versus neural/geometry alignment.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs are classified as follows:

Accepted diagnostic evidence:

- Corrected behavioral aggregate merged with SSL/VLM rows and accepted transformer relevance control: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`.
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
- Behavioral-control hardening outputs: `outputs/paper1_experiment_v1/summary/behavioral_control_gap_audit.csv`, `outputs/paper1_experiment_v1/summary/behavioral_bridge_integration_audit.csv`, `outputs/real_matrix_v2_task_search_baseline/aggregated/results.csv`, `outputs/real_matrix_v2_msdb_reference/aggregated/results.csv`, `outputs/real_matrix_v2_transformer_relevance/aggregated/results.csv`, `outputs/paper1_experiment_v1/summary/transformer_relevance_control_audit.csv`, `outputs/real_matrix_v2/coco_search18_static2000/coco_search18_task_prior_baseline_coco_search18_task_prior/aggregate_metrics.json`, and the merged accepted behavioral aggregate `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`.
- Free-viewing reference feasibility output: `outputs/paper1_experiment_v1/summary/free_viewing_reference_feasibility_decision.csv`.
- Observer-control outputs: `outputs/observer_controls_v2/coco_search18_static2000_observer_controls.csv`, `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`.

Diagnostics/provenance:

- Core behavioral aggregate before SSL/VLM merge: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected SSL/VLM behavioral aggregate before merge: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Full neural ROI summary directory, including learned-readout provenance rows: `outputs/neural_roi_summary/`.
- Paper inspection pack: `outputs/paper_inspection_v1/README.md`, including `outputs/paper_inspection_v1/tables/table14_subject_robustness_interpretation.csv` and `outputs/paper_inspection_v1/tables/table15_observer_control_summary.csv`.
- Paper 1 outcome interpretation: `docs/paper1_outcome_interpretation_v1.md`.

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
* behavioral controls that are strong for the current static-image scope but still scoped rather than exhaustive;
* transformer attribution coverage that is now sufficient for Paper 1 reporting but not a broad attribution-method inventory;
* geometry-method-dependent cross-axis analysis.

The current project should therefore be treated as a publication-directed evidence-building pipeline, not as a finished paper.

### Current strongest result

The strongest current result is a **pilot scaffold result**, not the final Paper 1 claim:

> Current diagnostics show that behavioral fixation alignment, neural encoding, and representational geometry are measurable in one pipeline and already show non-identical behavior across model family, attribution family, subject, ROI, and geometry method.

Important current numerical facts:

- behavioral references dominate generic attribution maps: DeepGaze MSDB is strongest on SALICON/CAT2000, and the COCO-Search18 task prior is strongest for task search;
- transformer relevance improves over rollout and vanilla gradients on SALICON/CAT2000 but remains below DeepGaze MSDB and is not operational attention;
- DINOv2 leads the six-model PRF encoding panel and the four-model ten-ROI discovery encoding panel;
- DINOv2 leads ten-ROI full-image CKA, but ResNet-50 leads most ten-ROI subset-RSA settings, with small top-rank margins;
- confirmatory-subject geometry favors DINOv2 in `3/3` subjects, while encoding favors DINOv2 in `2/3` and ResNet-50 in `subj04`.

Interpretation:

These results justify building Matrix V2. They should not be hardened into a final geometry-first story. The next claim must be tested across a deliberately designed model-family panel, stream/ROI grouping, and efficiency axis.

### Current weakest links

These are reviewer risks and scientific design risks in priority order.

1. **Wrong abstraction risk:** the project may over-harden a narrow DINOv2-vs-ResNet geometry/encoding observation instead of testing the broader human-machine alignment question.
2. **Modern model coverage risk:** the current panel has useful anchors, but it does not yet cover enough current frontier-relevant families such as newer SSL dense-feature models, stronger VLMs, efficient/hybrid backbones, or adaptive/foveated/selective-computation models.
3. **Attention-mechanism risk:** most current behavioral rows are post-hoc explanation maps. The project still lacks a strong operational attention, scanpath, foveation, token-selection, or adaptive-computation axis.
4. **Stream/anatomy risk:** current subject robustness is PRF-ROI limited. A paper about attention and “where/what” alignment needs stream-level analysis, especially ventral versus lateral/dorsal/parietal groupings.
5. **Efficiency missing:** compute and alignment-per-compute remain untested even though efficiency is central to the original proposal’s attention-as-resource-allocation hypothesis.
6. **Small model-level `n`:** current cross-axis correlations over `n=4` or `n=6` are descriptive. Matrix V2 needs a designed compact panel with enough model-family contrast to make quadrant classification meaningful.
7. **Geometry method sensitivity:** full-image CKA and subset-RSA do not always agree at the top rank. Geometry must remain a multi-method axis.
8. **Attribution ambiguity:** Grad-CAM, gradients, rollout, transformer relevance, perturbation maps, internal routing, scanpaths, and retained-token masks are different objects and must remain separated.
9. **Causality absent:** Paper 1 remains observational. Causal gaze/adaptive-attention intervention belongs to Paper 2 unless Matrix V2 explicitly adds a small controlled intervention.


## Current Behavioral Status

Corrected merged behavioral aggregate:

- Path: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`
- Rows: `455`
- Dataset rows: `161` each for SALICON and CAT2000, `133` for COCO-Search18
- Protocol rows: `322` with `points`, `133` with `task_points`
- Blank / `unknown` / `density_fallback` protocol rows: none

Accepted scoped transformer-relevance control:

- Path: `outputs/real_matrix_v2_transformer_relevance/aggregated/results.csv`
- Rows: `56`
- Scope: `2` free-viewing datasets x `4` transformer models x `7` metrics; datasets are SALICON/CAT2000 only, with `points` protocol only.
- Evidence-gate audit: `outputs/paper1_experiment_v1/summary/transformer_relevance_control_audit.csv`; all `7` scope/method/family/metric/cell checks pass and `evidence_decision=accepted_evidence_ready`.
- Current integration status: merged into the accepted behavioral bridge candidate at `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`; `outputs/paper1_experiment_v1/summary/behavioral_bridge_integration_audit.csv` has `12/12` checks passing.

Corrected NSS headline:

- SALICON: DeepGaze MSDB `1.760`, DeepGaze IIE `1.743`, center bias `0.933`, DINOv2 ViT-S/14 gradient `0.736`, ConvNeXt-T Grad-CAM `0.633`, ResNet-50 Grad-CAM `0.598`.
- CAT2000: DeepGaze MSDB `1.979`, DeepGaze IIE `1.838`, center bias `1.619`, ResNet-50 Grad-CAM `0.882`, DINOv2 ViT-S/14 gradient `0.810`, ConvNeXt-T Grad-CAM `0.759`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, ResNet-50 Grad-CAM `0.955`, ConvNeXt-T Grad-CAM `0.908`, DINOv2 ViT-S/14 gradient `0.713`.
- Transformer relevance NSS, accepted scoped control and now merged into reporting: SALICON DINOv2 `1.033`, CLIP ViT `0.981`, DeiT `0.931`, ViT-B `0.851`; CAT2000 DINOv2 `1.141`, CLIP ViT `0.940`, DeiT `0.886`, ViT-B `0.733`.

Current interpretation:

- Corrected outputs have valid point/task-point protocol labels.
- DeepGaze MSDB is now the accepted modern free-viewing reference for SALICON/CAT2000 and improves over the earlier DeepGaze IIE row in both datasets.
- DeepGaze IIE remains a useful historical/reference control and COCO-Search18 DeepGaze IIE remains diagnostic because it is a free-viewing reference on task-search data.
- DINOv2 gradient is a strong attribution/fixation-similarity row, especially on SALICON and CAT2000.
- Transformer relevance is now the strongest tested transformer attribution family on SALICON/CAT2000 within the scoped four-model panel. Across the `8` matched dataset/model cells, it improves over attention rollout and vanilla gradients for every checked metric; mean NSS gain is `+0.426` over rollout and `+0.715` over vanilla gradients.
- Transformer relevance does **not** overturn the reference-control story. DeepGaze MSDB / DeepGaze IIE remain the strongest free-viewing fixation references, and center bias remains stronger than transformer relevance on CAT2000 and remains competitive on SALICON map-distribution metrics.
- The behavioral layer is strong enough to serve as one axis in the broader alignment study. It should not be expanded into a larger leaderboard before the paper-grade matrix is defined.
- Behavioral controls are now sufficient for the current static-image Paper 1 scope: observer controls, center bias, DeepGaze IIE, DeepGaze MSDB for free-viewing, the COCO-Search18 task prior, and a separate transformer-relevance attribution family are represented. Broad scanpath/video expansion belongs after Paper 1 unless the paper explicitly shifts away from static-image dissociation.

## Current Neural Status

Current neural summary:

- Path: `outputs/neural_roi_summary/`
- Behavioral bridge CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`
- Efficiency CSV: not provided in the latest summary.
- Summary scope: `120` encoding rows, `289740` encoding target rows, and `92` RSA rows across accepted, diagnostic, and provenance outputs.
- Accepted matched-panel scope: `24` validation-selected full-image-count `flatten_pca` rows for six model families across four `subj01` PRF visual ROIs.
- Method-provenance scope: `4` full-image-count DINOv2 learned spatial readout rows.
- Matched cross-level analysis rows: `385` correlation/regression groups, with `280` complete and `105` marked `insufficient_models`; transformer relevance contributes `70` SALICON/CAT2000 groups and remains separate from `internal_routing`.
- Matched cross-level datasets remain separate: SALICON/CAT2000 include the merged transformer relevance rows, while COCO-Search18 remains task-search only with no transformer relevance expansion.
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

## SSL / Multimodal Status

Current SSL/VLM behavioral rows are corrected and merged into the main behavioral aggregate.

SSL/multimodal candidate inventory:

- Path: `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`
- Dry-inspected compatible candidates: `8`
- Pretrained debug runs complete: `3`
- Complete pretrained debug candidates: `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, `resnet50_clip`
- Not yet run pretrained debug candidates: `vit_base_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_dinov3`, `vit_base_patch16_siglip_224`, `eva02_base_patch16_clip_224`



## Global Direction Rationale

The project direction is a multi-axis NeuroAI alignment study. The central goal is to test whether behavioral fixation alignment, neural encoding, representational geometry, cortical stream alignment, and computational efficiency measure the same underlying human-likeness factor or dissociate across model families and viewing regimes.

The project should now be shaped around this question:

> Which models align with humans behaviorally, neurally, geometrically, anatomically, and computationally, and where do those axes fail to agree?

### Matrix V2 target

Paper 1 Matrix V2 should be a compact, deliberate model × axis × ROI/stream × efficiency matrix.

Required axes:

- behavioral fixation/saliency alignment:
  - SALICON and CAT2000 as free-viewing;
  - COCO-Search18 as task search;
  - scanpath/task-search metrics if feasible;
- neural encoding:
  - early visual PRF ROIs;
  - ventral stream / “what” ROIs;
  - lateral/dorsal/parietal or stream ROIs relevant to spatial selection / “where” processing;
- representational geometry:
  - full-image CKA;
  - deterministic subset RSA;
  - explicit method-agreement and disagreement reporting;
- efficiency:
  - parameters;
  - FLOPs/MACs;
  - latency;
  - memory;
  - visual token count;
  - retained-token fraction or selected-glimpse count where applicable;
- attention/resource-allocation mechanism:
  - post-hoc attribution families remain controls;
  - adaptive/foveated/token-pruning/scanpath models are needed to test the original attention hypothesis.

Required model categories:

- CNN anchor: e.g. ResNet-50 or ConvNeXt;
- plain ViT anchor: e.g. ViT-B or DeiT;
- self-supervised dense-feature model: current DINOv2 plus a newer available DINO-family candidate if feasible;
- VLM / semantic model: CLIP plus a stronger current SigLIP-like or comparable candidate if feasible;
- efficient/hybrid sequence model: MambaVision-like or comparable candidate if feasible;
- hierarchical/multiscale transformer if feasible;
- adaptive/foveated/token-pruning/scanpath model or mechanism if feasible;
- dedicated fixation/scanpath model as behavioral reference only, not as neural backbone unless features are extractable and scientifically justified.

### Current interpretation of existing results

The current behavioral results show that scoring is sane and that dedicated fixation/task references outperform generic classifier attribution maps. This supports the pipeline but is not central novelty.

The current neural and geometry results show that the pipeline can produce plausible local alignment signals. DINOv2 is strong in several current scores, but the result is not enough to support a universal DINOv2 claim.

The DINOv2 learned spatial readout is method provenance. It suggests spatial readout and adaptive sampling matter, which should motivate Matrix V2 and Paper 2, but it cannot be used as a matched cross-model headline.

The reduced subject-robustness result is useful pilot evidence. It should be treated as motivation for stream-level and model-family expansion rather than as the final paper story.

### Current priority

The immediate priority is **Paper 1 Matrix V2 Redesign And Feasibility Audit**.

Do not generate a claim-decision table from the superseded outcome interpretation. The next Codex session should produce an actionable Matrix V2 plan and, if feasible, the first machine-readable config/audit files that define:

- model categories and candidate models;
- which candidates are feasible with current dependencies and GPU constraints;
- required model wrappers/features/attribution objects;
- ROI/stream groups and which subjects have required data;
- efficiency metrics and how they will be measured;
- accepted behavioral, neural, geometry, and efficiency artifacts for Matrix V2;
- the exact next run order.

### Explicit non-priorities

Do not prioritize:

- formalizing `geometry_first_dissociation_candidate`;
- generating `paper1_claim_decision_table.csv` from the old interpretation;
- new paper inspection packs before Matrix V2 exists;
- broad saliency leaderboard expansion;
- broad timm model-zoo accumulation without designed model-family contrast;
- DINOv2-only readout variants unless used to test adaptive sampling or readout as a specific mechanism;
- COCO-Search18 transformer relevance expansion unless the Matrix V2 design requires it;
- manuscript polishing.

### Decision rule

Continue Paper 1 only if Matrix V2 can say more than:

> DINOv2 leads under some encoding and geometry settings.

The desired paper-level statement is:

> Under a controlled cross-axis experiment, behavioral fixation alignment, neural encoding, representational geometry, stream specificity, and computational efficiency converge or dissociate in identifiable model-family and viewing-regime patterns.

If Matrix V2 cannot produce this, demote Paper 1 to a methods/workshop paper and shift main effort toward Paper 2’s causal adaptive-attention or foveated-computation intervention.

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


## What Is Already Built

Behavioral infrastructure:

- Manifest loaders for SALICON, CAT2000, COCO-Search18, and NSD / Algonauts-style data.
- Fixation parsers for SALICON and CAT2000 `.mat` files.
- SALICON official JSON annotation conversion to worker-level observer manifests via `scripts/create_salicon_observer_manifest.py`.
- Task/scanpath point handling for COCO-Search18, including target-present and target-absent train/validation annotations.
- Inter-observer control summaries for inline JSON fixation rows and `.mat` fixation rows; inline fixation rows are scaled from original image size to evaluation map size.
- Static metrics: NSS, AUC-Judd, AUC-Borji, shuffled AUC, CC, SIM, KL, EMD, MAE, Pearson.
- Saliency methods: center bias, random saliency, gradient, integrated gradients, Grad-CAM, attention rollout, transformer relevance, occlusion, and precomputed DeepGaze-style maps.
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

1. `Scientific change`: no new experiment was run; the session corrected the project handoff away from implementation-success framing. The accepted outputs already contain enough claim-facing evidence to force a scientific interpretation pass. The next step is to decide what Paper 1 can honestly claim, not to produce another engineering success marker.
2. `Accepted artifact`: this updated status file, plus the already accepted attribution-family interpretation outputs: `outputs/paper1_experiment_v1/summary/attribution_family_cross_axis_interpretation.csv`, `outputs/paper_inspection_v1/tables/table16_attribution_family_cross_axis_interpretation.csv`, and regenerated `outputs/paper_inspection_v1/README.md`.
3. `Claim impact`: clarifies that the current evidence most plausibly supports a constrained geometry-first dissociation / multi-axis measurement story. It does **not** support a strong claim that human-like visual attention predicts cortical alignment, that transformer relevance is operational attention, or that DINOv2 universally wins neural encoding.
4. `Reviewer risk reduced`: reduces the risk of self-referential engineering claims by forcing the next session to classify scientific outcomes as `supported`, `partially_supported`, `diagnostic_only`, or `not_supported`.
5. `Next decisive step`: write the actual Paper 1 outcome interpretation: strongest defensible claim, unsupported claims, remaining reviewer objections, and recommended paper framing.

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

Priority: **Paper 1 Matrix V2 Redesign And Feasibility Audit**.

The previous immediate milestone, “Paper 1 Outcome Interpretation And Claim Decision,” is superseded. Do not generate claim-decision tables from the archived outcome interpretation. Do not treat the current geometry-first framing as the active destination.

### Required outcome

Create a Matrix V2 planning artifact set:

- `configs/paper1_matrix_v2.yaml`
- `outputs/planning/paper1_matrix_v2_model_feasibility.csv`
- `outputs/planning/paper1_matrix_v2_axis_scope.csv`
- `outputs/planning/paper1_matrix_v2_next_run_order.md`

These artifacts must define the next real experiment before any new large run.

### Matrix V2 feasibility audit must answer

1. Which current artifacts are retained as diagnostic scaffold?
2. Which model families are required?
3. Which exact model candidates are feasible in the current repo?
4. Which model candidates require new dependencies, wrappers, checkpoints, or cluster execution?
5. Which candidates expose usable feature tensors for encoding/geometry?
6. Which candidates expose usable attribution, routing, retained-token, scanpath, foveation, or adaptive-computation outputs?
7. Which ROIs can be grouped into early, ventral, lateral/dorsal, and parietal/stream categories?
8. Which subjects can support those ROI groups?
9. Which efficiency metrics can be measured now?
10. What is the first minimal Matrix V2 run that can falsify or support the revised paper story?

### Required model-category table

The feasibility audit must include at least these categories:

| category | role | examples / candidates | required decision |
| --- | --- | --- | --- |
| CNN anchor | local hierarchy baseline | ResNet-50, ConvNeXt | keep one or two |
| plain ViT anchor | standard transformer baseline | ViT-B, DeiT | keep one |
| SSL dense feature | modern self-supervised representation | DINOv2, newer DINO-family if feasible | include current plus audit newer candidate |
| VLM / semantic | language/semantic training axis | CLIP, SigLIP-like if feasible | include current plus audit stronger candidate |
| efficient/hybrid | nonstandard efficient sequence modeling | MambaVision-like if feasible | audit feasibility |
| hierarchical/multiscale | multiscale visual hierarchy | Swin/Hiera-like if feasible | audit feasibility |
| adaptive/foveated/selective | actual attention/resource-allocation mechanism | token-pruning, foveated, scanpath, glimpse model | highest-priority feasibility audit |
| dedicated fixation/scanpath reference | behavioral upper/reference model | DeepGaze / scanpath model | behavioral reference only unless features are justified |

### Required axis-scope table

The axis-scope artifact must include:

| axis | required status |
| --- | --- |
| free-viewing fixation alignment | already has SALICON/CAT2000 scaffold; keep separate |
| task-search alignment | already has COCO-Search18 scaffold; keep separate |
| scanpath/sequential alignment | audit feasibility |
| neural encoding | expand from PRF-only robustness toward stream grouping |
| representational geometry | keep CKA and subset RSA separate |
| efficiency | implement minimal profile now, not after another paper artifact |
| adaptive computation | audit retained tokens, foveation, scanpath, glimpse, or routing availability |

### Acceptance criteria

The milestone is complete only if:

- the archived outcome interpretation is no longer referenced as active steering;
- `configs/paper1_matrix_v2.yaml` exists;
- a model feasibility table exists and separates `ready_now`, `needs_wrapper`, `needs_dependency`, `needs_checkpoint`, `defer`, and `reject`;
- an axis-scope table exists and explicitly includes efficiency and stream/ROI grouping;
- the next run order begins with the smallest experiment that tests the revised cross-axis quadrant story;
- no new paper-facing claim table is generated before Matrix V2 exists.

### First Codex implementation plan

1. Remove `docs/paper1_outcome_interpretation_v1.md` from active docs references.
2. Create `configs/paper1_matrix_v2.yaml`.
3. Write `scripts/audit_paper1_matrix_v2_feasibility.py`.
4. Inspect existing model wrappers, manifests, outputs, and dependencies.
5. Emit the model feasibility table and axis-scope table.
6. Propose the smallest next executable Matrix V2 run.
7. Update this status file with the scientific change and next decisive step.

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
- Transformer relevance attribution-family control output is generated and audit-accepted:
  - `outputs/real_matrix_v2_transformer_relevance/aggregated/results.csv`: `56` metric rows for `transformer_relevance` across SALICON/CAT2000 and four transformer models.
  - `outputs/paper1_experiment_v1/summary/transformer_relevance_control_audit.csv`: all scope/method/family/metric/cell checks pass; `evidence_decision=accepted_evidence_ready`.
  - `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`: `455` merged behavioral rows after accepted transformer relevance integration.
  - `outputs/paper1_experiment_v1/summary/behavioral_bridge_integration_audit.csv`: `12/12` integration checks pass.
  - SALICON transformer relevance NSS: DINOv2 `1.033`, CLIP ViT `0.981`, DeiT `0.931`, ViT-B `0.851`; CAT2000 transformer relevance NSS: DINOv2 `1.141`, CLIP ViT `0.940`, DeiT `0.886`, ViT-B `0.733`.

Current post-spec implementation priorities:

The V1 outputs are now retained as diagnostic scaffold. They should no longer define the active paper destination.

Active priority:

- build Paper 1 Matrix V2 before any additional paper-facing interpretation;
- audit modern model-family candidates instead of expanding generic model count;
- add stream/ROI grouping as a first-class axis;
- add minimal efficiency profiling as a first-class axis;
- audit adaptive/foveated/token-pruning/scanpath mechanisms as the strongest route back to the original proposal;
- preserve existing behavioral controls, attribution-family separation, CKA/subset-RSA sensitivity, and subject-robustness outputs as controls/provenance.

Do not:

- generate the old claim-decision table;
- harden the `geometry_replicated_encoding_ambiguous` result into the main story;
- rerun old subject robustness, observer controls, transformer relevance, or MSDB scoring unless an audit fails;
- add new summary tables whose only purpose is to restate the old result.

Completed milestones are archived in `docs/project_status_changelog.md`.

## Later Milestones

Proceed in phases that map directly to the revised Matrix V2 research question:

> Do behavioral fixation alignment, neural encoding, representational geometry, stream specificity, adaptive/resource-allocation mechanisms, and efficiency converge or dissociate across modern vision systems?

The V1 milestones are now treated as completed scaffold work. They should not determine the next paper direction. The old `geometry_first_dissociation_candidate` framing and the old generated claim-decision-table milestone are superseded.

### Phase 0 — Archive stale steering and freeze V1 as scaffold

Status: immediate cleanup.

Purpose:

* remove the superseded outcome interpretation from active steering;
* preserve V1 outputs as diagnostic scaffold;
* prevent Codex from hardening the narrow geometry-first story.

Required actions:

1. Move `docs/paper1_outcome_interpretation_v1.md` to `docs/archive_stale/`.
2. Remove all active references to `geometry_first_dissociation_candidate` as the Paper 1 destination.
3. Remove `paper1_claim_decision_table.csv` as the next milestone.
4. Mark V1 subject robustness, transformer relevance, DeepGaze MSDB, observer controls, CKA/RSA, and paper-pack outputs as completed scaffold/provenance.
5. Do not rerun V1 outputs unless an audit fails.

Completion artifact:

* updated `docs/project_status_and_next_steps.md`;
* archived stale interpretation file.

### Phase 1 — Paper 1 Matrix V2 redesign and feasibility audit

Status: current next milestone.

Purpose:

Define the real next experiment before any additional large run.

Required outputs:

* `configs/paper1_matrix_v2.yaml`
* `outputs/planning/paper1_matrix_v2_model_feasibility.csv`
* `outputs/planning/paper1_matrix_v2_axis_scope.csv`
* `outputs/planning/paper1_matrix_v2_next_run_order.md`

The model feasibility table must classify candidates as:

* `ready_now`
* `needs_wrapper`
* `needs_dependency`
* `needs_checkpoint`
* `needs_cluster`
* `defer`
* `reject`

Required model categories:

1. CNN/local hierarchy anchor.
2. Plain ViT anchor.
3. self-supervised dense-feature model.
4. VLM / semantic model.
5. efficient or hybrid sequence model.
6. hierarchical or multiscale transformer.
7. adaptive/foveated/token-pruning/scanpath/selective-computation model or mechanism.
8. dedicated fixation or scanpath model as behavioral reference.

Required axis categories:

1. free-viewing fixation alignment;
2. task-search alignment;
3. scanpath or sequential gaze alignment if feasible;
4. neural encoding by ROI/stream group;
5. representational geometry with CKA and subset RSA separated;
6. efficiency and alignment-per-compute;
7. adaptive computation / resource allocation.

Acceptance rule:

Phase 1 is complete only when Codex can name the smallest executable Matrix V2 run that tests the revised paper story.

### Phase 2 — Stream/ROI grouping and neural-scope upgrade

Status: after Matrix V2 audit.

Purpose:

Move beyond PRF-only robustness and make the “where versus what” axis explicit.

Required actions:

1. Define ROI groups:

   * early visual / PRF;
   * ventral or “what” stream;
   * lateral/dorsal/parietal or “where/spatial selection” stream.
2. Audit which subjects support each ROI group.
3. Decide whether Matrix V2 uses:

   * `subj01` stream-expanded discovery plus reduced confirmatory PRF subjects;
   * stream-expanded replication on additional subjects;
   * or a smaller model panel that makes stream replication feasible.
4. Produce model × ROI-group encoding and geometry summaries rather than only flat ROI rankings.

Completion artifacts:

* `outputs/planning/paper1_matrix_v2_roi_groups.csv`
* updated `configs/paper1_matrix_v2.yaml`

Acceptance rule:

The paper must be able to ask whether fixation-like or adaptive mechanisms align more strongly with spatial/dorsal/parietal regions than with ventral/semantic regions, or whether the reverse pattern appears.

### Phase 3 — Modern model-family expansion with designed contrasts

Status: after Phase 1 feasibility audit.

Purpose:

Replace generic model-zoo accumulation with a compact, scientifically designed model panel.

The panel should compare model families, not simply add rows.

Required contrast families:

* classical CNN/local hierarchy;
* plain transformer;
* self-supervised dense representation;
* vision-language / semantic training;
* efficient or hybrid sequence architecture;
* adaptive/foveated/token-selective computation where feasible.

Rules:

* Do not add many generic `timm` models.
* Do not add a model unless it contributes a specific contrast.
* Every added model must expose usable features for encoding and geometry.
* If a model exposes routing, retained-token masks, scanpaths, foveation, or adaptive compute, record that as a first-class axis.

Completion artifact:

* `outputs/planning/paper1_matrix_v2_model_feasibility.csv`
* first executable Matrix V2 model panel in `configs/paper1_matrix_v2.yaml`

Acceptance rule:

The model panel must make at least one of these dissociations testable:

* high fixation alignment but low neural/geometry alignment;
* low fixation alignment but high neural/geometry alignment;
* high neural alignment but poor efficiency;
* lower raw alignment but stronger alignment-per-compute;
* stream-specific alignment reversal.

### Phase 4 — Efficiency and alignment-per-compute

Status: no longer deferred behind old claim-table work.

Purpose:

Restore the original proposal’s attention-as-resource-allocation axis.

Required metrics:

* parameter count;
* FLOPs or MACs;
* measured latency under fixed image resolution and batch size;
* peak memory if feasible;
* visual token count;
* retained-token fraction, selected-glimpse count, or foveated high-resolution area where applicable;
* accuracy or reference task performance where available;
* alignment-per-compute for behavioral, encoding, and geometry axes.

Required output:

* `outputs/efficiency_profiles/paper1_matrix_v2_efficiency.csv`

Acceptance rule:

Efficiency must be merged into the Matrix V2 cross-axis table. It should not appear as a decorative supplement.

### Phase 5 — Behavioral axis upgrade: scanpath/task/adaptive attention where feasible

Status: after model feasibility audit, can run in parallel with Phase 3 if lightweight.

Purpose:

Move beyond static post-hoc heatmap comparison where possible.

Required actions:

1. Keep SALICON/CAT2000 free-viewing separate from COCO-Search18 task search.
2. Preserve DeepGaze MSDB, center bias, observer controls, and COCO-Search18 task prior as controls.
3. Audit feasibility of adding:

   * scanpath models;
   * sequential gaze metrics;
   * foveated models;
   * retained-token masks;
   * token-pruning maps;
   * adaptive-glimpse trajectories.
4. Label every behavioral object by type:

   * dedicated fixation model;
   * scanpath model;
   * post-hoc attribution;
   * internal routing;
   * adaptive compute allocation;
   * task prior;
   * center prior;
   * observer/human context.

Acceptance rule:

Do not claim “human-like attention” from Grad-CAM, gradients, rollout, or transformer relevance. Human-like attention claims require either human fixation/scanpath alignment, explicit routing/computation allocation, or foveated/adaptive mechanism evidence.

### Phase 6 — Matrix V2 cross-axis quadrant analysis

Status: after the first Matrix V2 evidence table exists.

Purpose:

Make the paper story explicit.

Required output:

* `outputs/paper1_matrix_v2/summary/matrix_v2_cross_axis_quadrants.csv`

Each model or model-method pair should be classified into one of four descriptive quadrants:

| behavioral fixation alignment | neural/geometry alignment | interpretation                                                 |
| ----------------------------- | ------------------------- | -------------------------------------------------------------- |
| high                          | high                      | behavioral attention may track brain-like representation       |
| low                           | high                      | representation convergence without human-like overt attention  |
| high                          | low                       | superficial saliency mimicry or different internal computation |
| low                           | low                       | weak alignment across axes                                     |

Required safeguards:

* exact model `n`;
* uncertainty or bootstrap where available;
* leave-one-model or leave-one-family sensitivity where feasible;
* separate free-viewing and task-search analyses;
* separate CKA and subset-RSA geometry;
* separate ROI/stream groups.

Acceptance rule:

The paper cannot advance to manuscript hardening until Matrix V2 produces at least one interpretable quadrant pattern or a scientifically meaningful null result.

### Phase 7 — External positioning and Brain-Score-style context

Status: after Matrix V2 has a stable internal result.

Purpose:

Position the result against the broader NeuroAI field.

Required actions:

* compare model-family conclusions against Brain-Score-style expectations where feasible;
* clarify that local Algonauts/NSD encoding scores are not leaderboard-equivalent;
* explain whether Matrix V2 supports convergence, dissociation, or measurement pluralism.

Acceptance rule:

External positioning is context. It cannot substitute for the local Matrix V2 behavioral/neural/geometry/efficiency analysis.

### Phase 8 — Paper split decision

Status: after Matrix V2 cross-axis results.

Decision options:

1. **Paper 1 main-track attempt:** only if Matrix V2 shows a robust, interpretable dissociation or convergence pattern across model family, ROI/stream, and efficiency.
2. **Paper 1 workshop/thesis/methods paper:** if Matrix V2 is useful but underpowered.
3. **Shift main effort to Paper 2:** if observational Matrix V2 cannot produce a strong story.
4. **Paper 2 causal adaptive-attention intervention:** use human gaze, foveation, token pruning, adaptive readout, or saliency-guided computation to test whether attention/resource allocation changes alignment, efficiency, or neural predictivity.

Do not decide the publication split from V1 alone.


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
- `src/hma/saliency/transformer_relevance.py`
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

- `scripts/create_attribution_family_interpretation.py`
- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_behavioral_controls.py`
- `scripts/audit_transformer_relevance_control.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`
