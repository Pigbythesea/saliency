# HMA Project Status And Next Steps

Updated: 2026-06-07

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
- Paper-style inspection tables and figures that join corrected behavioral summaries with the matched full-image neural panel, matched geometry sensitivity outputs, V1 ROI-expanded geometry, and matched cross-level correlation/regression outputs.

The repository now implements both the diagnostic PRF-only matched full-image representational-geometry axis and the V1 ROI-expanded `subj01` geometry axis. The V1 artifact gate passes, but interpretation is not yet paper-grade because the main result is geometry-method dependent.

Main package: `src/hma/`.

Main scripts: `scripts/`.

Current generated outputs are classified as follows:

Accepted diagnostic evidence:

- Corrected behavioral aggregate merged with SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Matched full-image `flatten_pca` neural panel: `outputs/neural_roi_summary/matched_full_panel_model_rankings.csv`.
- Matched full-image geometry rankings and sensitivity summaries: `outputs/neural_roi_summary/matched_geometry_model_rankings.csv`, `outputs/neural_roi_summary/matched_geometry_method_agreement.csv`.
- Matched cross-level analysis outputs: `outputs/neural_roi_summary/matched_cross_level_observations.csv`, `outputs/neural_roi_summary/matched_cross_level_correlations.csv`.
- V1 ROI-expanded discovery evidence: `outputs/paper1_experiment_v1/summary/roi_expanded_encoding_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_agreement.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_correlations.csv`, and `outputs/paper1_experiment_v1/summary/roi_expanded_cross_axis_decisions.csv`.

Robustness/control artifacts:

- Matched panel audit: `outputs/neural_roi_summary/matched_full_panel_artifact_audit.csv`.
- Cross-axis sensitivity and decision diagnostics: `outputs/neural_roi_summary/matched_cross_axis_sensitivity.csv`, `outputs/neural_roi_summary/matched_cross_axis_decisions.csv`.
- V1 ROI-expanded artifact audit: `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`; current status is all checks passing after `40` geometry cells x `10` geometry rows per cell.
- Observer-control outputs: `outputs/observer_controls_v2/coco_search18_static2000_observer_controls.csv`, `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`.

Diagnostics/provenance:

- Core behavioral aggregate before SSL/VLM merge: `outputs/real_matrix_v2/aggregated/results.csv`.
- Corrected SSL/VLM behavioral aggregate before merge: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Full neural ROI summary directory, including learned-readout provenance rows: `outputs/neural_roi_summary/`.
- Paper inspection pack: `outputs/paper_inspection_v1/README.md`.

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

- Current neural outputs are one-subject, internal-split `subj01` results. The accepted cross-model comparison object is the complete six-model matched full-image-count PRF visual ROI `flatten_pca` panel. Four full-image-count learned-readout rows for DINOv2 are method provenance only.
- They are not Algonauts leaderboard-equivalent scores because the official challenge averages held-out visual-cortex vertices across subjects and hemispheres.
- The matched `flatten_pca` panel is the primary evidence for cross-model neural comparisons. The four-ROI DINOv2 learned-readout rows are the strongest local single-backbone method result and should be treated as method provenance, not as matched-panel ranking rows.
- The matched cross-level correlation tables are now the primary descriptive cross-axis evidence, but they are still small-n one-subject model-level analyses, not causal tests.

Paper 1 should be held to these publication gates before strong top-venue claims:

- Full-image matched representational geometry now exists for the same six-model x four-ROI panel, but geometry claims still require method/seed stability and explicit CKA/subset-RSA sensitivity.
- Cross-axis results must report uncertainty and sensitivity, especially bootstrap intervals, leave-one-model-out behavior, and exact model counts.
- Claims must be framed as descriptive convergence/dissociation, not causal attention intervention.
- At least one nontrivial dissociation or convergence pattern must survive sensitivity checks; otherwise Paper 1 should be framed as a measurement framework, workshop paper, thesis chapter, or methods note.

## Publication Claim State

### Current paper status

Paper 1 is not yet top-venue ready. The current repository supports a serious diagnostic study, but the accepted evidence is still too narrow for a strong conference claim because it is mainly:

* one-subject neural evidence;
* small model-level correlations (`n=4` for the V1 discovery matrix, `n=6` for the older PRF diagnostic panel);
* one completed ROI-expanded discovery pass without subject replication;
* limited behavioral SOTA controls;
* limited transformer attribution coverage;
* geometry-method-dependent cross-axis analysis.

The current project should therefore be treated as a publication-directed evidence-building pipeline, not as a finished paper.

### Current strongest claim

The strongest currently defensible claim is:

> The current pipeline can produce a complete ROI-expanded `subj01` discovery matrix, and the first V1 results suggest that encoding and full-image CKA converge while subset-RSA weakens that geometry conclusion.

This is not yet a paper claim. It is a candidate method-dependent convergence/dissociation pattern that needs explicit failure-gate synthesis before subject replication.

The current results should be used to decide whether the V1 pattern is worth confirming, not to support a top-venue conclusion.

### Current weakest links

The weakest links, in order of reviewer severity, are:

1. **Small model-level ****`n`****:** behavior-encoding-geometry correlations over `n=4` in the V1 matrix remain descriptive.
2. **One-subject neural evidence:** current ROI-expanded encoding and geometry results are still local `subj01` evidence and need subject robustness only after the V1 pattern is decision-clean.
3. **Insufficient behavioral SOTA controls:** DeepGaze IIE and center bias are useful, but human ceilings, DeepGaze MSDB/comparable modern fixation baseline, and task-specific COCO-Search18 controls remain important.
4. **Attribution ambiguity:** Grad-CAM, vanilla gradients, and attention rollout are different explanation objects. Attention rollout must not be treated as human-like attention.
5. **Geometry metric dependence:** V1 full-image CKA converges with encoding, while subset-RSA gives a weaker or different ranking. This is now the immediate scientific issue, not a future concern.
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
- Rows: `378`
- Dataset rows: `126` each for SALICON, CAT2000, and COCO-Search18
- Protocol rows: `252` with `points`, `126` with `task_points`
- Blank / `unknown` / `density_fallback` protocol rows: none

Corrected NSS headline:

- SALICON: DeepGaze `1.743`, center bias `0.933`, DINOv2 ViT-S/14 gradient `0.736`, ConvNeXt-T Grad-CAM `0.633`, ResNet-50 Grad-CAM `0.598`.
- CAT2000: DeepGaze `1.838`, center bias `1.619`, ResNet-50 Grad-CAM `0.882`, DINOv2 ViT-S/14 gradient `0.810`, ConvNeXt-T Grad-CAM `0.759`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, ResNet-50 Grad-CAM `0.955`, ConvNeXt-T Grad-CAM `0.908`, DINOv2 ViT-S/14 gradient `0.713`.

Current interpretation:

- Corrected outputs have valid point/task-point protocol labels.
- DeepGaze now beats center bias across all three datasets under the corrected point/task-point protocol.
- DINOv2 gradient is a strong attribution/fixation-similarity row, especially on SALICON and CAT2000.
- The behavioral layer is strong enough to serve as one axis in the broader alignment study. It should not be expanded into a larger leaderboard before the paper-grade matrix is defined.
- Later behavioral upgrades should prioritize human/interobserver ceilings, DeepGaze MSDB or another modern fixation reference, and a task-specific COCO-Search18 baseline. Broad scanpath/video expansion belongs after Paper 1 unless the paper explicitly shifts away from static-image dissociation.

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
- The V1 discovery matrix changes the scientific state: encoding and full-image CKA converge on DINOv2 as the top model, but subset-RSA partially dissociates from that story. The next accepted evidence object must therefore synthesize geometry-method sensitivity, not simply report the CKA ranking.

## Global Direction Rationale

The project direction is a multi-axis NeuroAI alignment study. The central goal is to test whether fixation alignment, neural encoding, and representational geometry measure the same underlying “human-likeness” factor or whether they dissociate across architecture, attribution family, ROI, and viewing regime.

The project should be shaped around this question:

> Which models are aligned behaviorally, neurally, and geometrically, and where do those axes fail to agree?

The behavioral layer is an alignment axis, not the main paper by itself. The neural encoding layer is a local brain-alignment axis, not an Algonauts leaderboard claim. The geometry layer is a latent representational axis, not a replacement for encoding. The paper becomes interesting only when these axes are analyzed jointly.

### Current interpretation of existing results

The current behavioral results show that the corrected scoring protocol is sane: DeepGaze beats center bias, and generic classifier explanation maps remain below dedicated fixation references. This supports the pipeline, but it is not central novelty.

The current matched neural panel shows a plausible local ranking over six models and four PRF ROIs, with DINOv2, CLIP ViT, and ResNet-50 near the top. This is useful local evidence, but it is one-subject and should not be described as SOTA neural alignment.

The DINOv2 learned spatial readout is a strong method-provenance result because it improves all four PRF visual ROIs over DINOv2 `flatten_pca`. It should motivate later readout/adaptive-sampling work, but it should not enter cross-model headline rankings until the same readout protocol exists for all compared models.

The matched geometry axis is the current most important addition because it can expose cases where encoding and representational geometry disagree. Geometry should be used to test dissociation, not to create a separate leaderboard.

### Current priority

The immediate priority is to move from a completed V1 discovery matrix to a decision-clean scientific interpretation.

The current V1 behavioral, neural, and geometry outputs are the first claim-relevant discovery matrix, but they are not yet a paper result. Future Codex sessions should not expand subjects, models, behavioral baselines, or efficiency before the project decides whether the V1 pattern is a stable convergence, a method-dependent dissociation, or an insufficient/ambiguous result.

The next Codex work should therefore focus on:

1. **V1 geometry-method sensitivity synthesis**

   Build an accepted table that compares the V1 cross-axis conclusions under `linear_cka_full9841` against all deterministic `subset_rsa` variants. This table should explicitly mark whether behavior-geometry and encoding-geometry conclusions are `stable_across_geometry_methods`, `cka_only`, `subset_rsa_only`, `direction_conflict`, or `insufficient_models`.

2. **Failure-gate decision**

   Apply the V1 failure criteria before any expansion. The decision should answer whether the current result is:

   * a stable convergence pattern;
   * a method-dependent dissociation pattern worth confirming;
   * an ambiguous result requiring uncertainty/geometry repair before replication;
   * or a failed Paper 1 discovery matrix that should demote the paper framing.

3. **Outcome-first implementation**

   Each Codex session should produce an artifact that changes the paper’s scientific state.

   Good session outputs include:

   * a geometry-method sensitivity decision table;
   * a V1 failure-gate synthesis note;
   * a targeted robustness table that changes whether the V1 pattern is trusted;
   * a documented decision to proceed to subject robustness or pause for method repair.

   Poor session outputs include:

   * more logging;
   * more paper-pack formatting;
   * more plots of the current diagnostic matrix;
   * small improvements to stale summaries;
   * broad model-zoo expansion without a claim-driven design.

4. **Robustness after decision**

   Uncertainty, subject replication, behavioral-control integration, and efficiency should be applied after the V1 decision is clear.

   The correct order is:

   1. finish the V1 geometry-method sensitivity synthesis;
   2. decide whether the V1 pattern is stable, method-dependent, or insufficient;
   3. then choose the next robustness target.

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
3. Apply geometry-method sensitivity and the V1 failure gate.
4. If a coherent method-dependent convergence/dissociation pattern remains, run subject robustness on the reduced `subj02`-`subj04` panel.
5. If the pattern is ambiguous, add uncertainty/geometry repair before replication.
6. Only after the V1 decision is clean, add behavioral controls, stronger attribution, and efficiency.
7. If the V1 matrix does not produce a defensible claim after sensitivity, demote Paper 1 to a methods/workshop paper and shift the main publication effort toward Paper 2’s causal adaptive-attention intervention.

### Decision rule

Do not treat the V1 full-image CKA convergence as paper-grade evidence until subset-RSA sensitivity is explicitly included in the decision table.

Robustness can strengthen a decision-clean pattern. It cannot rescue an interpretation that depends on silently choosing one geometry method.

Continue Paper 1 toward a top venue only if the upgraded experiment can say more than:

> DINOv2 leads under encoding and CKA in one subject.

The desired paper-level statement is:

> Under a controlled, sufficiently broad cross-axis experiment, fixation alignment, neural encoding, and representational geometry converge or dissociate in identifiable model/ROI/task regimes.

That statement now requires V1 geometry-method synthesis first, then robustness second.

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
- Paper inspection pack with behavior, neural, matched cross-level, SSL/VLM candidate, benchmark sanity tables, and an academic SOTA context section comparing the current figures against MIT/Tuebingen saliency, DeepGaze IIE SALICON, COCO-Search18 task-search, and Algonauts 2023 evaluation references.
- Paper inspection README now explicitly distinguishes mixed-scope diagnostics from the complete six-model matched full-image-count PRF ROI `flatten_pca` panel, and includes the four DINOv2 learned spatial readout rows only as method-provenance context.

## Current Implementation Progress

Updated: 2026-06-07

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
- The V1 cross-axis summaries exist: `693` cross-level correlation rows and `2079` cross-axis decision rows. Current decision rows are dominated by `insufficient_models` (`1386`), with `627` `stable_convergence` and `66` `unstable` rows. The stable across-ROI rows are concentrated in `vanilla_gradient` behavior and `linear_cka_full9841` geometry.

Latest session report:

1. `Scientific change`: the V1 ROI-expanded `subj01` discovery matrix is now complete across encoding and geometry. Encoding and full-image CKA converge on DINOv2 as the top model, but deterministic subset-RSA weakens or partially reverses that geometry story by usually ranking ResNet first. This is a candidate geometry-method dissociation, not a clean model-ranking result.
2. `Accepted artifact`: `outputs/paper1_experiment_v1/neural_subj01_roi_expanded/`, `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_encoding_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_model_rankings.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_agreement.csv`, `outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_correlations.csv`, and `outputs/paper1_experiment_v1/summary/roi_expanded_cross_axis_decisions.csv`.
3. `Claim impact`: positive for the Paper 1 dissociation claim, but still not paper-ready. The result provides a concrete convergence/dissociation candidate: encoding aligns with full-image CKA, while subset-RSA exposes geometry-method dependence.
4. `Reviewer risk reduced`: reduces the "PRF-only anatomical scope" and "missing geometry axis" risks by completing method-matched PRF-plus-stream encoding and geometry for the four-model V1 panel. It also surfaces the geometry-metric-dependence risk instead of hiding it.
5. `Next decisive step`: implement a V1 geometry-method sensitivity and failure-gate synthesis table that compares CKA-based cross-axis conclusions against all subset-RSA variants before any subject replication, model expansion, or new behavioral controls.

Implementation history was moved to `docs/project_status_changelog.md`.

## Next Concrete Milestone

Priority: **Add V1 Geometry-Method Sensitivity And Failure-Gate Synthesis**.

The previous milestone, Paper 1 V1 ROI-Expanded Geometry, is complete. Do not rerun encoding or geometry unless an audit regresses or a concrete data-integrity blocker is discovered.

The next decisive implementation task is to synthesize whether the apparent V1 convergence/dissociation pattern survives geometry-method sensitivity. The current cross-axis decision table uses `linear_cka_full9841` as the primary geometry method, while subset-RSA is summarized mostly through method agreement. The next session must make the CKA-vs-subset-RSA implication explicit.

### Required outcome

By the end of this milestone, the project should produce a reviewer-facing decision artifact for the completed V1 discovery matrix:

- A V1 geometry-method sensitivity table under `outputs/paper1_experiment_v1/summary/`, preferably named `roi_expanded_geometry_method_sensitivity_decisions.csv`.
- A compact V1 failure-gate synthesis table or Markdown note under `outputs/paper1_experiment_v1/summary/`, preferably named `roi_expanded_failure_gate_summary.csv` or `roi_expanded_failure_gate_summary.md`.
- Explicit labels for each behavior/encoding/geometry relationship: `stable_across_geometry_methods`, `cka_only`, `subset_rsa_only`, `direction_conflict`, `insufficient_models`, or `not_tested`.
- A clear decision on whether the V1 result should move to subject robustness, geometry/uncertainty repair, or downgraded Paper 1 framing.

### Required implementation work

The next Codex session should implement:

1. **Load completed V1 evidence**

   Read `configs/paper1_experiment_v1.yaml`, `roi_expanded_encoding_model_rankings.csv`, `roi_expanded_geometry_model_rankings.csv`, `roi_expanded_geometry_method_agreement.csv`, `roi_expanded_cross_level_correlations.csv`, and `roi_expanded_cross_axis_decisions.csv`.

2. **Build geometry-method sensitivity summaries**

   Compare cross-model rankings and correlation conclusions for `linear_cka_full9841` versus every deterministic `subset_rsa` method. Report per-method and aggregated subset-RSA agreement with CKA for model mean and ROI-specific rows.

3. **Extend cross-axis decisions beyond primary CKA**

   The existing `roi_expanded_cross_axis_decisions.csv` is CKA-primary. Add a companion decision table that states whether behavior-vs-geometry and encoding-vs-geometry conclusions remain stable when geometry is represented by subset-RSA variants.

4. **Apply the V1 failure criteria**

   Mark whether the V1 pattern depends on one model, one ROI, or one geometry method. Treat "DINOv2 leads under CKA but not subset-RSA" as a possible method-dependent dissociation, not as a failure by itself.

5. **Update status and next step**

   Update this file with the scientific decision. If the method-dependent pattern is coherent, the next step becomes subject robustness on `subj02`-`subj04`; if it is ambiguous, the next step becomes uncertainty/geometry repair before replication.

### Scope decision rules

The implementation should follow these rules:

- Prefer a small reusable generator over hand-writing `40` configs.
- Keep `configs/paper1_config.yaml` unchanged as the diagnostic PRF-only result scope.
- Keep V1 fLOC category ROIs out of scope unless `docs/paper1_experiment_spec_v1.md` is explicitly revised.
- Do not run subject replication before the V1 geometry-method sensitivity decision exists.
- Do not add new models, new behavioral baselines, efficiency, or stronger attribution before the V1 failure gate is applied.
- Use cmd-form commands in any docs or handoff text.

### Acceptance criteria

This milestone is complete only if:

- The new sensitivity table includes all four V1 models, all ten ROIs, `linear_cka_full9841`, and all nine subset-RSA variants.
- The synthesis explicitly marks whether encoding-geometry convergence is stable across geometry methods or CKA-only.
- The synthesis explicitly preserves SALICON/CAT2000 free-viewing versus COCO-Search18 task-search separation.
- The failure-gate output states whether to proceed to subject robustness, add uncertainty/geometry repair, or demote Paper 1 framing.
- Focused tests verify that subset-RSA rows are not silently excluded from the sensitivity decision.

### Stop condition

If the existing summary tables do not contain enough information to compare CKA and subset-RSA cross-axis conclusions, stop and add the missing summary fields instead of falling back to subject replication, model expansion, or PRF-only robustness.

Suggested focused inspection command before sensitivity work:

```cmd
.\.venv\Scripts\python.exe -c "import csv; from pathlib import Path; audit=list(csv.DictReader(Path('outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv').open(newline='',encoding='utf-8'))); print([(r['check'], r['status']) for r in audit]); geom=list(csv.DictReader(Path('outputs/paper1_experiment_v1/summary/roi_expanded_geometry_model_rankings.csv').open(newline='',encoding='utf-8'))); print('geometry_methods', len(set(r['geometry_method'] for r in geom))); print('models', sorted(set(r['model'] for r in geom)))"
```

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
  - `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`: `20000` rows using at most `10` workers per image.

Current post-spec implementation priorities:

The V1 spec and config now define the falsifiable paper-grade matrix, and the ROI-expanded encoding plus geometry axes are complete. The immediate work is to decide whether the observed CKA/subset-RSA method dependence is a coherent cross-axis dissociation pattern or an ambiguity that needs repair before replication.

- Apply V1 geometry-method sensitivity before expanding the model zoo or prioritizing subject replication:
  - keep the completed `subj01` ROI-expanded encoding matrix as the V1 discovery encoding axis;
  - keep the completed `subj01` ROI-expanded geometry matrix as the V1 discovery geometry axis;
  - explicitly compare CKA-primary conclusions against deterministic subset-RSA conclusions before adding subjects or models.
- Use the prepared additional subjects as a robustness follow-up:
  - after the V1 failure gate identifies a coherent candidate pattern, run the reduced confirmatory panel on `subj02`-`subj04`;
  - do not let subject replication precede geometry-method sensitivity synthesis.
- Strengthen behavioral controls:
  - consume the generated SALICON and COCO-Search18 observer-control outputs in behavioral summaries and paper-pack diagnostics;
  - add DeepGaze MSDB or another modern fixation reference if feasible;
  - add a task-specific COCO-Search18 baseline before interpreting task-search alignment.
- Improve transformer attribution coverage:
  - add one stronger transformer relevance method, preferably Chefer-style transformer attribution or AttnLRP-style relevance propagation;
  - keep gradients, rollout, perturbation, and relevance-style maps as separate saliency families.
- Add paper-grade uncertainty only after the V1 failure-gate decision exists:
  - add target bootstrap intervals for matched encoding summaries;
  - add subset/image bootstrap or additional deterministic seeds for subset RSA;
  - keep leave-one-model and leave-one-ROI sensitivity, but use them as robustness diagnostics.
- Add efficiency only after the above robustness pass:
  - collect FLOPs, latency, memory, token count, and retained-patch statistics for the matched panel;
  - regenerate alignment-per-compute summaries as exploratory diagnostics.

Completed milestones are archived in `docs/project_status_changelog.md`.

## Later Milestones

Proceed in phases that map directly to the research question.

1. **V1 geometry-method decision.** Resolve whether the completed ROI-expanded `subj01` matrix supports a stable convergence, a method-dependent dissociation, or an ambiguous result.
2. **Behavioral and attribution controls.** Integrate observer ceilings, add a modern DeepGaze reference if feasible, add a task-trained COCO-Search18 baseline, and add stronger transformer attribution before any attention-specific interpretation.
3. **Subject robustness.** Use `subj02`-`subj04` after the V1 geometry-method decision to check whether candidate patterns replicate. Subject expansion is a robustness layer, not the main discovery path.
4. **Uncertainty and sensitivity.** Estimate intervals over images, neural targets, and geometry subsets only after the paper-grade matrix exists; add leave-one-model-out, leave-one-ROI-out, Kendall tau, and model-label permutation checks as diagnostics rather than claim filters.
5. **Cross-axis decision gate.** Use the decision table to identify weak implementation areas, not to assert paper-ready conclusions. Revisit publication framing only after ROI-expanded results, controls, and subject robustness materially improve the evidence base.
6. **Efficiency.** Add FLOPs, latency, token count, retained-patch statistics, and memory footprint for the matched model panel, then regenerate alignment-per-compute summaries. Keep efficiency exploratory unless it produces a clean dissociation or tradeoff.
7. **Brain-Score or Brain-Score-style external positioning.** Use it as context and sanity checking, not as a substitute for the local fixation/fMRI/geometry cross-level tests.
8. **Publication split.** Defer publication split decisions until the robustness expansion clarifies whether static-image cross-axis results are reliable. Keep causal adaptive attention, foveation, adaptive token routing, scanpaths, video, or recurrent policies out of the immediate implementation path unless the robustness pass exposes a sharply defined intervention target.

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
- `scripts/audit_matched_neural_panel.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

For current status-file cleanup, no tests are required.

For future neural/reporting implementation changes, use this focused verification command:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_alignment.py tests\test_neural_roi_summary.py tests\test_paper1_experiment_spec.py tests\test_paper1_v1_geometry_and_summary.py
```

Last known focused result: `78 passed` with `--basetemp=.tmp_pytest_v1_focused`.

For broader confidence after neural/reporting changes, run:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py tests\test_paper1_v1_roi_expanded_generation.py tests\test_paper1_experiment_spec.py tests\test_paper1_v1_geometry_and_summary.py
```

Last known broader result: `105 passed` with `--basetemp=.tmp_pytest_v1_broader`.

V1 ROI-expanded artifact audit command:

```cmd
.\.venv\Scripts\python.exe -c "import csv; from pathlib import Path; rows=list(csv.DictReader(Path('outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv').open(newline='',encoding='utf-8'))); print([(r['check'], r['status']) for r in rows])"
```

Last known V1 audit result: all `9` checks pass, including `40` encoding cells, `40` geometry cells, and `10` geometry rows per cell.

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
