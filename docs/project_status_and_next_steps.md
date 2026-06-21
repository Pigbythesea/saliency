# HMA Project Status And Next Steps

Updated: 2026-06-18

## Purpose And Codex Operating Contract

This document is the primary implementation handoff for Codex sessions on the Human-Machine Visual Alignment project.

Its purpose is to steer implementation toward publication-grade scientific evidence, not toward self-reassuring engineering progress. A Codex session should use this file to decide:

* what scientific claim the project is currently trying to test;
* which outputs are accepted as evidence and which are only diagnostics;
* which implementation task most directly strengthens the paper claim;
* which tasks should be avoided because they only expand the codebase, leaderboard size, or logging surface without improving the publication argument;
* what exact artifact must exist at the end of the session.

The current Paper 1 claim to test is:

> Human fixation behavior, visual-cortex neural encoding, representational geometry, cortical stream structure, model role, and computational efficiency are separable axes of visual alignment. Paper 1 should test whether these axes converge or dissociate across modern vision systems. The primary behavioral axis is no longer native saliency-map or scanpath-output similarity. It is a standardized latent-to-fixation encoding probe: frozen model latent features are mapped through a matched linear probabilistic readout to held-out human fixation density. The central question is whether models whose latent spaces make human fixation behavior linearly decodable also predict visual-cortex responses and neural representational geometry better, or whether fixation decodability, neural encoding, latent geometry, stream selectivity, model role, and efficiency come apart systematically.

The paper should be organized around a cross-axis outcome grid:

| primary behavioral latent-to-fixation decodability | neural encoding / geometry alignment | intended interpretation |
| --- | --- | --- |
| high | high | human fixation structure is linearly accessible from representations that also align with visual-cortex responses or neural geometry |
| low | high | brain-predictive representations may emerge without linearly decodable human fixation behavior |
| high | low | human-fixation-decodable representations may arise without strong neural encoding or geometry alignment |
| low | low | weak alignment on both behavioral-decoding and neural/geometry axes |

## Publication Evidence Contract Override

This file now prioritizes Paper 1 Publication Matrix V0 over all previous Matrix V2 and V1 outputs.

Whenever older sections, paths, or result summaries conflict with the publication-scope reset, Codex must follow the publication-scope reset.

The publication evidence root is:

`outputs/paper1_publication_v0/`

No result outside this root is final Paper 1 evidence unless it is explicitly regenerated under the frozen publication contract or certified by an equivalence audit.

No publication-root full rerun is allowed unless the relevant behavioral, neural, geometry, adapter-comparability, efficiency, and cross-axis method gates are passed or explicitly accepted with limitations.

The current active task is **Updated V0 Full Cluster Rerun With Primary Behavioral Latent-To-Fixation Evidence**. It is not interpretation, manuscript writing, blocked-model repair, adapter-only installation, legacy behavioral-map rerunning, or another incremental planning loop. The primary behavioral latent-to-fixation pipeline now exists, has a bounded local smoke output, and is routed into V0. The remaining work is to execute the generated full clean cluster package, copy back the outputs, rerun import validation, and keep the legacy behavioral-map/saliency/scanpath lane excluded from primary V0 evidence.


The `Current Implementation Progress` section must summarize publication-readiness state only; it must not re-list historical runs, smoke passes, old Matrix V2 outputs, old result numbers, or debugging milestones.

### Paper 1 scope refinement

The current Paper 1 claim remains valid. It should sharpen the neural-alignment axis by explicitly separating:

- brain-region / stream structure:
  - early visual / retinotopic ROIs;
  - dorsal, lateral, and parietal spatial-selection ROIs;
  - ventral and semantic/object-related ROIs;
- model-role categories:
  - human gaze-prediction models;
  - human-gaze-inspired efficient or foveated models;
  - generic efficient-computation models;
  - general visual representation models;
  - spatial-prior and task-prior controls;
- representation type:
  - latent-feature neural encoding and latent-feature representational geometry as mandatory neural/geometry evidence;
  - primary behavioral latent-to-fixation encoding as mandatory behavioral evidence;
  - native saliency maps, Grad-CAM maps, attention rollout maps, routing maps, token-retention maps, generated scanpaths, and model-native fixation outputs as legacy/diagnostic behavioral objects unless explicitly reintroduced as secondary controls after V0;
  - output-map-to-fMRI encoding only as a secondary diagnostic/control, not the primary neural-comparison scheme.

Paper 1 should remain centered on whether latent-to-fixation decodability, latent neural encoding, latent representational geometry, cortical stream structure, model role, and efficiency converge or dissociate. The publication-facing version must test whether these axes dissociate systematically by cortical stream and by model role. The behavioral comparison must be based on the standardized learned linear fixation readout from frozen latent features, not on heterogeneous native saliency, attribution, routing, or scanpath outputs.

The model universe is fixed by the research question, not by convenience. Feasibility determines implementation status and adapter order; it must not shrink the scientific scope before a model-role and latent-feature audit is complete.

All publication-facing central evidence must be regenerated under a frozen Paper 1 publication contract and written under a new publication output root. Existing behavioral, neural, geometry, efficiency, and cross-axis outputs are scaffold/provenance unless explicitly regenerated under the publication contract or certified by an equivalence audit.

Codex must treat every implementation task as subordinate to building the publication evidence matrix. A task counts only if it clarifies the relationship among behavioral fixation/scanpath alignment, latent-feature neural encoding, latent-feature representational geometry, ROI/stream specificity, model role, and efficiency under the frozen publication contract.

### What counts as progress

A change counts as progress only if it advances **Paper 1 Publication Matrix V0** by improving either:

1. the scientific scope of the publication matrix; or
2. the academic validity of the methods used to generate publication-root evidence.

A change counts as progress only if it produces at least one of the following:

- a frozen publication contract or publication-scope audit;
- a literature-grounded method audit comparing the project pipeline against relevant SOTA methods;
- a method acceptance gate decision for behavioral evaluation, neural encoding, geometry, adapter comparability, efficiency/resource allocation, or cross-axis inference;
- a model-role matrix covering every required candidate model;
- an adapter-certification result proving which models expose behavioral outputs, latent features, scanpaths/glimpses, efficiency metadata, and controlled input conditions;
- a clean behavioral rerun artifact under the publication output root;
- a clean latent-feature neural encoding artifact under the publication output root;
- a clean primary behavioral latent-to-fixation encoding artifact under the publication output root;
- a negative audit marking the old native-map / scanpath / saliency-metric behavioral rerun as legacy and excluded from V0 evidence;
- a stream/ROI grouping table tied to actual neural manifests and publication configs;
- an efficiency/resource-allocation table merged into the publication matrix;
- a cross-axis table using only publication-root evidence;
- a negative, audited decision that marks a method, model, or result family as behavior-only, diagnostic-only, accepted-with-limitations, or rejected for publication evidence.

Engineering achievements, successful smoke runs, code reorganization, log expansion, paper-pack updates, and summaries of legacy outputs do not count as scientific progress unless they directly produce one of the publication-root, method-gate, or scope-reset artifacts above.

A full publication-root rerun is not allowed until the relevant method acceptance gates are passed or explicitly marked as accepted limitations.

### What Codex should prioritize

Codex should prioritize:

- implementation that removes current method-gate blockers, not more audit prose;
- full required model-role coverage before convenience-based model reduction;
- adapter certification and setup scaffolding across the full required model universe, not only local anchors;
- reusable method infrastructure before publication-root evidence generation;
- completing the primary behavioral latent-to-fixation encoding pipeline before any V0 rerun;
- debiased geometry and geometry resampling before clean geometry reruns;
- sequential/adaptive total-cost accounting before efficiency comparisons;
- family-aware cross-axis sensitivity before paper-facing quadrant interpretation;
- latent-feature neural encoding and latent-feature geometry over output-map neural controls;
- stream/ROI structure over flat ROI averages;
- latent-feature behavioral encoding over post-hoc heatmap variants, native saliency maps, scanpath-output scoring, or routing-map similarity;
- explicit paper-evidence status for every model, artifact, and result table.

Codex should avoid:

- treating scaffold outputs as paper evidence;
- producing audit-only, planning-only, or Markdown-only sessions;
- certifying only local anchors and calling the milestone complete;
- generating interpretation from the three-model adaptive pilot;
- beginning adaptive sweep, manuscript work, or publication-root full reruns before method gates are unblocked;
- shrinking the model universe before setup, checkpoint, feature-hook, and behavioral-output requirements are made explicit;
- reporting smoke tests as progress unless they directly certify a model, unblock a method gate, or produce reusable publication-run infrastructure.

### Required end-of-session report

Until publication-root evidence exists, Codex must report publication-readiness changes, not claim changes.

At the end of each Codex session, update this file with:

1. `Publication-contract change`: what changed in the frozen scope, method contract, model-role matrix, adapter certification, or clean rerun plan.
2. `Accepted artifact`: exact path(s) under `outputs/paper1_scope_reset/`, `configs/`, or `outputs/paper1_publication_v0/`.
3. `Method gate status change`: which method gate moved among `not_started`, `audit_required`, `method_gap_found`, `accepted_with_limitations`, or `accepted_for_publication_rerun`.
4. `Paper evidence status change`: which model/artifact moved among `not_started`, `adapter_in_progress`, `adapter_certified`, `publication_rerun_ready`, `publication_rerun_complete`, `accepted_publication_evidence`, `diagnostic_only`, or `rejected_for_paper_evidence`.
5. `Reviewer risk reduced`: which concrete risk was reduced, such as SOTA-method mismatch, legacy-output contamination, missing latent features, behavioral-regime mixing, adapter incomparability, stream/ROI ambiguity, or weak uncertainty.

Do not report smoke tests, debugging fixes, old-result summaries, legacy audits, or paper-pack generation as progress unless they directly change one of the publication-contract or method-gate artifacts above.


## Reference Documents Reviewed

Current steering documents under `docs/`:

- `project_status_and_next_steps.md`: this engineering status file.
- `project_results_numbers.md`: historical numerical context for expected ranges and regression checks. It is not the source of final Paper 1 evidence.
- `paper1_cross_axis_alignment_roadmap.md`: useful background, but superseded for implementation. Keep only the cross-axis dissociation framing; Publication Matrix V0 is the active execution target.
- `paper1_literaturereview.md`: current literature review for Paper 1. It raises the required controls around dataset bias, scanpath/task specificity, subject variability, encoding reliability, representational-geometry metrics, and transformer attribution.
- `Literature Review and Research Redesign for the Human-Like Adaptive Visual Attention Project.md`: argues the project should become a multi-axis NeuroAI alignment study, not a saliency-map leaderboard.
- `Deep Research Assessment of the Human-Machine Visual Alignment Project.md`: emphasizes the publishable question as convergence versus dissociation among fixation alignment, neural predictivity, representational geometry, and efficiency.
- `hma_project_publication_critique_handoff.md`: current publication-readiness critique. It is a read-only reference for claim hygiene, stale-output cleanup, and top-venue risk assessment.
- `Zhang_Zihuan_zzhan330_proposal.docx`: original proposal; defines behavioral saliency, neural encoding, RSA, Brain-Score-style comparison, and compute efficiency as the core axes.
- `Comparing Human and Machine Visual Saliency_ A Comprehensive Review.pdf`: reinforces that fixation prediction requires strong controls such as center bias, DeepGaze-class references, point-based NSS/AUC, and separate treatment of free-viewing versus task-driven viewing.
- `__Attention and Saliency Map Extraction in Visual AI Models_ A Comprehensive Review__.pdf`: reinforces that gradients, CAMs, attention rollout, perturbation maps, LRP-style methods, and transformer attribution are different explanation objects and should not be collapsed into one "attention" score.
- Method SOTA audit sources must be treated as active methodological constraints, not background citations. Codex must compare the project pipeline against actual methods and reported numbers before authorizing publication-root full reruns.

## Current Snapshot

The repository currently contains a large scaffold/provenance base, including repaired Matrix V2 outputs, older behavioral aggregates, older neural/geometry summaries, and methodology-trace audits. These artifacts validate code paths and reveal prior failure modes, but they are not the final Paper 1 evidence matrix.

Current scaffold/provenance assets:

- repaired behavior-map routing infrastructure with collision-safe `map_key` / `row_key` semantics;
- metric-layer constant-map handling;
- external-model artifact infrastructure for feature export, routing resources, efficiency metadata, provenance, hashes, and validation;
- local NSD/Algonauts-style neural encoding and representational-geometry pipelines;
- behavioral benchmark infrastructure for SALICON, CAT2000, and COCO-Search18;
- existing adapters or registry scaffolds for static DeiT-S, DynamicViT, ToMe, DINOv3, SigLIP, MambaVision, Hiera, Swin, HAT, ScanDiff, and related external models;
- cluster workflow templates and scratch-isolated environment setup;
- older diagnostic outputs that can be used for expected-range checks and regression testing only.

Current publication-facing status:

- No final Paper 1 publication evidence matrix is accepted yet.
- No legacy output is accepted as final paper evidence by default.
- The V0 rerun path is authorized and cluster-tested, but the behavioral lane is now methodologically stale.
- The old behavioral fixation/saliency/scanpath/map-metric rerun is demoted to `legacy_behavioral_pipeline` and must not enter the V0 rerun.
- The new primary behavioral evidence lane is `primary_behavioral_latent_to_fixation_encoding`.
- The next accepted scientific artifact is the completed primary behavioral latent-to-fixation pipeline plus a regenerated V0 rerun under `outputs/paper1_publication_v0/`.
- Static DeiT-S, DynamicViT, ToMe, DINOv3, SigLIP, MambaVision, DeepGaze, HAT, ScanDiff, AdaptiveNN, certified anchors, and controls must enter the behavioral lane only through frozen latent features or explicitly audited latent-feature artifacts.
- Native behavioral outputs from DeepGaze, HAT, ScanDiff, AdaptiveNN, Grad-CAM, attention rollout, routing maps, token/glimpse maps, task priors, and legacy saliency maps are diagnostic-only for V0 unless explicitly used as baselines/controls outside the primary behavioral score.
- SemBA/SemBA-FAST exclusion remains explicit in audit/config artifacts.
- Final V0 evidence must contain primary behavioral latent-to-fixation encoding, latent-feature neural encoding, latent-feature geometry, efficiency/resource, and cross-axis outputs regenerated under the publication root and audited to exclude legacy/admission contamination.

Publication output root:

- `outputs/paper1_publication_v0/`

Main package: `src/hma/`.

Main scripts: `scripts/`.

Required scaffold/provenance rule:

- Existing outputs may guide debugging, expected-value sanity checks, and adapter regression tests.
- Existing outputs must not be merged into publication summaries unless their `paper_evidence_status` is upgraded by publication-rerun completion or equivalence audit.

## Scientific Boundary

Paper 1 is now governed by a publication evidence contract, not by legacy output availability.

### Publication evidence rule

All final Paper 1 central evidence must be regenerated under the frozen publication contract and written under `outputs/paper1_publication_v0/`.

This applies to:

- behavioral fixation / saliency / scanpath metrics;
- latent-feature neural encoding;
- latent-feature representational geometry;
- efficiency and resource-allocation metrics;
- cross-axis and quadrant summaries.

Legacy behavioral, neural, geometry, efficiency, and cross-axis outputs remain scaffold/provenance. They are useful for method tracing, expected-range checks, and debugging. They are not final Paper 1 evidence unless regenerated under the publication contract or certified by an explicit equivalence audit.

### Representation rule

Latent-feature neural encoding and latent-feature representational geometry are mandatory for every central model class.

A model can carry the neural-alignment claim only if it exposes latent features that can be passed through the same neural encoding and geometry scheme as the other central models.

Output-map-to-fMRI encoding is allowed only as secondary diagnostic/control evidence and must be labeled:

`output_map_neural_control`

It must not be labeled:

`latent_feature_neural_encoding`

### Behavioral rule

The primary V0 behavioral pipeline is `primary_behavioral_latent_to_fixation_encoding`.

Its scientific question is:

> After freezing each model and giving it the same constrained fixation readout, how much held-out human fixation density is linearly decodable from the model's latent representation?

The primary behavioral pipeline must use:

- frozen model latent features, not native saliency/attention/routing maps;
- deterministic one-tensor-per-image or one-audited-condition-per-image feature artifacts;
- architecture-normalized spatial feature handling:
  - CNN feature maps stay spatial;
  - ViT/DeiT/DINO/CLIP/SigLIP tokens are reshaped to a 2D patch grid;
  - DynamicViT/ToMe token-pruning or token-merging states are reconstructed or masked into an audited rectangular grid;
  - DeepGaze/HAT/ScanDiff/AdaptiveNN states are used only under explicit deterministic condition labels;
- a common spatial output grid, default `28x28`;
- train-only feature normalization and dimensionality reduction;
- a fixed primary readout family: linear L2-regularized log-density readout;
- explicit center-bias baseline estimated only from training data;
- held-out fixation log-likelihood and information gain above center bias as primary metrics;
- NSS, CC, KL, AUC, and native map scores only as secondary or diagnostic metrics.

SALICON and CAT2000 remain free-viewing datasets. COCO-Search18 remains task-search. They must remain separated if included, but V0 primary behavioral completion should prioritize the free-viewing latent-to-fixation lane unless task-conditioned latent fixtures are already certified.

The following are legacy/diagnostic for V0 and must not define the primary behavioral axis:

- Grad-CAM and Grad-CAM variants;
- vanilla/integrated gradients;
- attention rollout and transformer relevance maps;
- native DeepGaze output maps;
- HAT/ScanDiff/AdaptiveNN native scanpaths or glimpse outputs;
- token-retention, routing, merge, or resource-allocation maps;
- old point-fixation/map-distribution/scanpath metric aggregates from Matrix V1/V2 or admission-panel outputs.

Legacy behavioral outputs may be used only for expected-range checks, diagnostic plots, or optional secondary controls after the primary behavioral latent-to-fixation encoding rerun exists and passes audit.

### Model inclusion rule

The model universe is not reduced before audit. DINOv3, SigLIP, MambaVision, Hiera, Swin, HAT, ScanDiff, SemBA/SemBA-FAST, AdaptiveNN, DynamicViT, ToMe, DeepGaze, CNN anchors, ViT anchors, SSL models, VLM/semantic models, spatial priors, and task priors must all receive model-role and adapter-certification inspection.

Feasibility produces implementation status. It does not silently remove a model from the scientific scope.

### Publication-gate rule

Paper-facing interpretation begins only after:

1. the full model-role matrix exists;
2. every required candidate has adapter-certification status;
3. clean behavioral outputs are regenerated under `outputs/paper1_publication_v0/`;
4. clean latent-feature neural encoding outputs are regenerated under `outputs/paper1_publication_v0/`;
5. clean latent-feature geometry outputs are regenerated under `outputs/paper1_publication_v0/`;
6. efficiency/resource-allocation metrics are regenerated or certified under the same contract;
7. cross-axis summaries use publication-root evidence only.

## Publication Method Acceptance Gates

Paper 1 is not allowed to proceed from scope definition to publication-root full reruns until the core methods pass explicit acceptance gates.

The current repository method is a controlled baseline pipeline, not a SOTA leaderboard pipeline. This is acceptable only if the paper claim is framed as a controlled multi-axis NeuroAI alignment benchmark rather than a claim of SOTA fMRI prediction, SOTA saliency prediction, or causal attention mechanism.

### Method position rule

The publication method must be described as:

> A controlled frozen-feature benchmark for comparing model roles across primary behavioral latent-to-fixation encoding, latent-feature neural encoding, representational geometry, cortical stream structure, and efficiency/resource allocation.

The publication method must not be described as:

* an Algonauts leaderboard-equivalent fMRI model;
* a SOTA saliency model or native saliency-map leaderboard;
* a comparison of heterogeneous Grad-CAM, attention-rollout, routing-map, native-saliency, or scanpath objects as if they were the same behavioral representation;
* a causal attention-intervention study;
* proof that fixation alignment causes neural alignment;
* proof that output-map alignment replaces latent-feature neural encoding.

### Gate 1 — Primary behavioral latent-to-fixation encoding acceptance

Status: `implemented_smoke_tested_routed_full_cluster_pending`.

The old behavioral evaluation gate is superseded for V0. Native map, saliency, attribution, routing, scanpath, and point-metric behavioral scoring is now `legacy_behavioral_pipeline_excluded_from_v0` and must not enter the V0 rerun as the primary behavioral evidence.

The primary behavioral latent-to-fixation pipeline has been implemented, smoke-tested locally, routed through the V0 runner/preflight/cross-axis path, and audited. The remaining V0 blocker is full clean cluster execution, copy-back, and import validation.

Required implementation artifacts:

* `configs/paper1_primary_behavioral_latent_fixation.yaml`
* `scripts/run_paper1_primary_behavioral_latent_fixation.py`
* `src/hma/behavioral/latent_fixation.py`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_image_scores.csv`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/feature_reduction_metadata.json`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/readout_selection_artifact.json`
* `outputs/paper1_publication_v0/audits/primary_behavioral_latent_fixation_audit.csv`
* `outputs/paper1_publication_v0/audits/legacy_behavioral_pipeline_exclusion_audit.csv`
* updated `outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv`
* updated `outputs/paper1_publication_v0/preflight/expected_outputs_manifest.csv`
* updated `outputs/paper1_publication_v0/cross_axis/model_axis_scores.csv` after the rerun finishes.

Required method contract:

* input: frozen model latent tensors from certified adapters/artifacts;
* model families: use the same eligible latent-feature model universe as the neural/geometry lane wherever fixation-dataset image coverage exists;
* feature handling:
  * CNN maps remain spatial;
  * ViT-family tokens are reshaped to patch grids;
  * token-pruned/merged models are reconstructed or masked into an audited grid;
  * gaze/foveated/scanpath models use deterministic condition labels;
* spatial grid: default `28x28`;
* feature reduction: train-only channel PCA or equivalent train-only low-rank bottleneck, default `512` components when feasible;
* readout: linear L2-regularized log-density readout;
* center bias: train-split empirical center prior, never fit on validation/test fixations;
* probability normalization: spatial softmax over the output grid;
* layer and regularization selection: inner validation only;
* held-out test split: untouched by PCA fitting, layer selection, readout selection, and center-prior fitting;
* primary score: fixation log-likelihood and information gain above center bias;
* secondary scores: NSS, CC, KL, AUC only as supporting diagnostics;
* forbidden primary evidence: Grad-CAM, attention rollout, transformer relevance, native DeepGaze maps, routing maps, token maps, scanpath metrics, and old behavioral aggregate files.

Acceptance rule:

This gate can move to `accepted_for_publication_rerun` after the full clean cluster rerun finishes, outputs are copied back, import validation passes, and the cross-axis publication-root outputs are regenerated from the new primary behavioral lane.

Current limitation:

The updated V0 rerun is authorized and cluster-ready, but full cluster execution is still pending. The old behavioral outputs are marked `legacy_behavioral_pipeline_excluded_from_v0`.

### Gate 2 — Latent-feature neural encoding acceptance

Status: `accepted_with_limitations`.

Before clean neural reruns, Codex must produce:

* `outputs/paper1_scope_reset/method_neural_encoding_sota_audit.md`
* `outputs/paper1_scope_reset/neural_encoding_acceptance_table.csv`

The audit must decide:

* whether the primary method is a controlled frozen-feature baseline;
* whether a learned spatial readout is required as a sensitivity analysis;
* which score is primary: raw Pearson, noise-normalized score, or both separately;
* how zero, negative, unavailable, or non-finite noise ceilings are handled;
* whether noise-normalized and non-noise-normalized rows may enter the same aggregate;
* how layer selection, ridge-alpha selection, dimensionality reduction, and validation splits are prevented from leaking test information;
* how subject-level and ROI-level robustness are reported;
* what uncertainty unit is used: images, targets, ROIs, subjects, or model families.

A neural rerun can proceed only after this gate is `accepted_for_publication_rerun` or `accepted_with_limitations`.

### Gate 3 — Representational geometry acceptance

Status: `accepted_for_publication_rerun`.

Before clean geometry reruns, Codex must produce:

* `outputs/paper1_scope_reset/method_geometry_sota_audit.md`
* `outputs/paper1_scope_reset/geometry_acceptance_table.csv`

The audit must decide:

* whether full-image CKA remains primary, secondary, or diagnostic;
* whether debiased CKA or another bias-corrected geometry metric must be implemented;
* how subset RSA sizes and seeds are selected;
* how response-permutation controls are used;
* how image-resampling uncertainty is reported;
* how CKA/RSA agreement and disagreement are interpreted;
* how geometry avoids becoming the sole evidence for neural alignment.

A geometry rerun can proceed only after this gate is `accepted_for_publication_rerun` or `accepted_with_limitations`.

### Gate 4 — Model adapter comparability acceptance

Status: `accepted_with_limitations`.

Before any model enters publication-root evidence generation, Codex must produce:

* `outputs/paper1_scope_reset/model_adapter_comparability_audit.md`
* `outputs/paper1_scope_reset/model_adapter_comparability_table.csv`

The audit must certify for every required candidate:

* deterministic input condition;
* preprocessing path;
* checkpoint and environment provenance;
* behavioral output type;
* latent-feature tensor availability;
* layer or block candidates;
* gaze-history, task, text, foveation, or stochastic conditioning;
* resource-allocation output;
* efficiency metadata;
* whether the model is central, behavior-only, diagnostic-only, or rejected after audit.

This gate applies especially to DeepGaze, HAT, ScanDiff, SemBA/SemBA-FAST, and AdaptiveNN.

### Gate 5 — Efficiency/resource-allocation acceptance

Status: `accepted_with_limitations`.

Before efficiency results enter cross-axis analysis, Codex must produce:

* `outputs/paper1_scope_reset/method_efficiency_sota_audit.md`
* `outputs/paper1_scope_reset/efficiency_acceptance_table.csv`

The audit must decide:

* which efficiency metrics are comparable across static, token-pruning, token-merging, foveated, scanpath, and active-vision models;
* how FLOPs/MACs, latency, memory, token count, retained-token fraction, selected-glimpse count, fixation count, scanpath length, stopping behavior, and foveated high-resolution area are reported;
* whether efficiency is measured under matched image resolution, batch size, hardware, and preprocessing;
* whether sequential models report total cost per image/task rather than per-glimpse cost only.

### Gate 6 — Cross-axis inference acceptance

Status: `accepted_with_limitations`.

Before paper-facing interpretation, Codex must produce:

* `outputs/paper1_scope_reset/method_cross_axis_inference_audit.md`
* `outputs/paper1_scope_reset/cross_axis_inference_acceptance_table.csv`

The audit must decide:

* minimum model count by role;
* whether leave-one-model and leave-one-family sensitivity are required;
* whether bootstrap or permutation uncertainty is required;
* how free-viewing, task-search, and scanpath analyses remain separate;
* how ROI/stream-specific claims are reported;
* when a quadrant label is descriptive only;
* what language is allowed for convergence, dissociation, measurement pluralism, and causality.

### Method gate status values

Every method gate must be assigned one of:

* `not_started`
* `audit_required`
* `method_gap_found`
* `accepted_with_limitations`
* `accepted_for_publication_rerun`
* `rejected_for_publication_claim`

A publication-root full rerun may begin only for axes whose gates are `accepted_for_publication_rerun` or `accepted_with_limitations`.


## Global Direction Rationale

Paper 1 is a multi-axis NeuroAI alignment study. The central question is:

> Do human gaze-prediction models, human-gaze-inspired adaptive/foveated models, generic efficient-computation models, general representation models, VLM/semantic models, hierarchical/hybrid models, and spatial/task-prior controls converge or dissociate across behavioral fixation/scanpath alignment, latent-feature neural encoding, latent representational geometry, stream specificity, and efficiency/resource allocation?

### Paper 1 Publication Matrix V0 target

Publication Matrix V0 is the required paper-facing evidence object.

It must contain:

- model-role matrix;
- adapter-certification matrix;
- clean primary behavioral latent-to-fixation encoding rerun;
- clean latent-feature neural encoding rerun;
- clean latent-feature geometry rerun;
- stream/ROI grouping;
- efficiency/resource-allocation profiling;
- cross-axis publication analysis using publication-root evidence only.

Required candidate families:

- ResNet + ConvNeXt anchors;
- ViT + DeiT anchors;
- DINOv2 and DINOv3 family;
- CLIP + SigLIP-family candidates;
- MambaVision;
- Hiera;
- Swin + SwinV2;
- DynamicViT;
- ToMe;
- DeepGaze;
- HAT;
- ScanDiff;
- SemBA / SemBA-FAST where applicable;
- AdaptiveNN;
- center, random, spatial-prior, and task-prior controls.

### Decision rule

Paper interpretation is allowed only after Publication Matrix V0 exists under `outputs/paper1_publication_v0/`.

If Publication Matrix V0 produces a robust convergence or dissociation pattern across model role, ROI/stream, and efficiency, Paper 1 can pursue a main-track submission.

If the result is mainly a measurement-framework contribution, Paper 1 should become a methods/workshop/thesis paper.

If observational cross-axis results remain weak after the full contract, shift main effort toward Paper 2’s causal adaptive-attention or foveated-computation intervention.

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

## Current Implementation Progress

Updated: 2026-06-21

Current implementation state is classified by **publication readiness**, not by smoke tests, local convenience runs, or legacy Matrix V2 progress.

### Publication-readiness classification

Current implementation readiness is:

| component | current classification | publication-facing meaning |
| --- | --- | --- |
| publication contract | frozen_v0 | `configs/paper1_publication_contract.yaml` remains the governing contract |
| publication output root | v0_authorized_primary_behavioral_lane_replaced_full_cluster_pending | `outputs/paper1_publication_v0/` remains the publication root; the primary behavioral latent-fixation bounded output exists, while full neural/geometry/efficiency/cross-axis outputs are still pending |
| first clean rerun authorization | authorized_verified_after_behavioral_replacement | regenerated preflight and strict verification report `ready`; full cluster job package has been regenerated after replacing the stale behavioral lane |
| primary behavioral latent-to-fixation method | implemented_smoke_tested_routed | `configs/paper1_primary_behavioral_latent_fixation.yaml`, `scripts/run_paper1_primary_behavioral_latent_fixation.py`, and `src/hma/behavioral/latent_fixation.py` implement train-only reduction, validation-selected ridge readouts, held-out fixation scoring, and legacy exclusion audit |
| latent-feature neural encoding method | accepted_with_limitations_export_validated_for_deepgaze | controlled frozen-feature PCA/ridge is allowed for certified tensors; DeepGaze IIE/III/MSDB bounded tensor exports validate deterministically |
| representational geometry method | accepted_for_publication_rerun_after_tensor_validation | debiased CKA/RSA machinery is ready for certified tensors; geometry execution must wait until model tensor exports are actually validated |
| efficiency/resource-allocation method | accepted_with_limitations_clean_runner_ready | static/token-pruning/token-merging schemas and total-cost/resource hooks are preflight-certified where available; clean efficiency/resource config and dry-run runner exist |
| cross-axis inference method | accepted_with_limitations_no_interpretation_primary_behavior_routed | cross-axis assembly now reads `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`; convergence/dissociation interpretation remains blocked until all publication-root outputs are validated |
| model adapter comparability | repaired_preflight_certified | current regenerated preflight has no non-diagnostic uncertified model rows |
| easy public image encoders | preflight_certified | SigLIP and MambaVision have adapter-certification rows and checkpoint locks in the regenerated preflight |
| DINOv3 | preflight_certified | DINOv3 has an adapter-certification row and checkpoint lock in the regenerated preflight |
| empirical spatial prior | implemented_ready | leakage-safe train-split empirical spatial prior is registered, tested, and included in method readiness |
| DeepGaze IIE/III/MSDB | deterministic_tensor_exports_validated | IIE latent, III conditional/scanpath, and MSDB latent bounded exports validate deterministically |
| scanpath/foveated/adaptive candidates | preflight_certified | DeepGaze III, AdaptiveNN, HAT, ScanDiff free-view, and ScanDiff visual-search are certified scanpath/foveated candidates |
| efficient-computation models | keep_in_scope_not_main_bottleneck | DynamicViT and ToMe stay in scope, but they are not the central next-session target |
| neural ROI/stream scope | partial_ready | early retinotopic and subj01 stream manifests exist; stream/category coverage limits must remain explicit |
| cluster execution | generated_full_package_ready_full_run_pending | regenerated cluster tables contain 63 behavioral latent export cells, 462 neural cells, 25 efficiency cells, and zero unsupported behavioral rows; Codex has not run remote Slurm jobs |
| legacy outputs | excluded | no existing empirical root is accepted or equivalence-certified as final Paper 1 evidence; old behavioral-map/saliency/scanpath outputs are explicitly `legacy_behavioral_pipeline_excluded_from_v0` |

### Current checks before real paper progress

The regenerated preflight, strict artifact-level verification, and cluster job generation authorize the updated V0 rerun execution path. Model/data/method/cache/DeepGaze checks pass, clean configs are enabled, dedicated runners exist, runner dry-runs pass, and the primary behavioral latent-to-fixation lane has a bounded local smoke output at `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`. Before remote execution, verify:

1. `outputs/paper1_publication_v0/preflight/user_action_checklist.csv` contains no blocked execution-gate rows;
2. `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv` reports `first_clean_rerun_authorization=authorized`;
3. `model_certification_summary.csv`, `model_setup_attempts.csv`, and `model_rerun_eligibility_table.csv` agree on which models enter behavioral, latent neural/geometry, efficiency/resource, and scanpath/task-search axes;
4. DINOv3, SigLIP, MambaVision, HAT, ScanDiff, AdaptiveNN, DeepGaze IIE/III/MSDB, DynamicViT, ToMe, local anchors, priors, random baseline, and task prior have actual source/checkpoint/environment/adapter/certification records where applicable;
5. DeepGaze tensor exports and conditional/scanpath outputs actually exist and pass deterministic validation, rather than only having manifests;
6. SemBA/SemBA-FAST exclusion is explicit in registry/config/audit artifacts and does not silently remove the model family from the model-universe record;
7. clean-rerun configs include only eligible certified rows, have `execution_enabled: true`, route behavioral evidence through `primary_behavioral_latent_to_fixation_encoding`, and do not reuse admission-panel or legacy behavioral outputs as clean evidence;
8. dedicated clean runner scripts or commands exist and pass `--dry-run` for behavioral, latent neural encoding, geometry, efficiency/resource, and cross-axis assembly;
9. expected-output manifests define output paths, row/file expectations, and import validation;
10. no primary behavioral latent-to-fixation, latent neural encoding, geometry, efficiency, or cross-axis artifact is accepted as Paper 1 publication evidence until the updated V0 rerun finishes and passes audit.


### Current implementation priority

Active priority:

> Run the generated full clean cluster package, copy back `outputs/paper1_publication_v0/`, regenerate preflight and authorization verification, and confirm every expected V0 output exists. Do not use admission-panel runners, legacy saliency-map aggregates, native scanpath outputs, Grad-CAM/rollout maps, routing maps, or old behavioral metric tables as clean behavioral evidence. Do not interpret results until the updated V0 rerun and post-run clean evidence audit pass.

This is not another blocked-model repair session, DynamicViT/ToMe-only session, cluster-only session, Markdown session, smoke-test session, audit-only session, config-only session, or paper-interpretation session. The next Codex session must finish code, config, runner, dry-run, rerun execution path, output generation, and audit for the primary behavioral latent-to-fixation lane in one session.

### Required end-of-session report

Use the publication-readiness report format in `Current Implementation Progress`. Do not use the older scientific-claim report format until publication-root evidence exists.

Do not report smoke tests, debugging fixes, old-result summaries, or legacy audits as progress unless they directly change one of the publication-contract artifacts above.

Implementation history is archived in `docs/project_status_changelog.md`.


### End-of-session report - Clean Rerun Execution Infrastructure

1. **Publication-contract change:** no full clean rerun was launched. Dedicated clean runner infrastructure now exists for behavioral, latent neural encoding, geometry, efficiency/resource, and cross-axis assembly. `configs/paper1_publication_contract.yaml` has `execution_authorized: true` after runner dry-runs and strict verification passed.
2. **Accepted artifact:** accepted current infrastructure artifacts are `configs/paper1_primary_behavioral_latent_fixation.yaml`, `configs/paper1_latent_neural_matrix.yaml`, `configs/paper1_efficiency_resource_rerun.yaml`, `configs/paper1_cross_axis_assembly.yaml`, `scripts/paper1_clean_rerun_common.py`, `scripts/run_paper1_primary_behavioral_latent_fixation.py`, `scripts/run_paper1_clean_latent_neural_encoding.py`, `scripts/run_paper1_clean_geometry.py`, `scripts/run_paper1_clean_efficiency_resource.py`, `scripts/assemble_paper1_clean_cross_axis.py`, `scripts/verify_paper1_clean_rerun_authorization.py`, `outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv`, and `outputs/paper1_publication_v0/preflight/user_run_cluster_commands.md`. `configs/paper1_clean_behavioral_rerun.yaml` and `scripts/run_paper1_clean_behavioral_rerun.py` are retained only as legacy behavioral diagnostics and are excluded from V0 primary evidence.
3. **Method gate status change:** method gates remain pass/accepted-with-limitations for pre-rerun readiness. The execution-path gate moved from blocked to ready after config, runner, dry-run, and strict verification checks passed.
4. **Paper evidence status change:** the bounded local primary behavioral smoke output exists at `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`; neural encoding, geometry, efficiency, and final cross-axis clean outputs remain pending full cluster execution, copy-back, and import validation.
5. **Reviewer risk reduced:** reduced false-authorization and legacy-contamination risk by requiring every clean lane to reject legacy/admission paths, write only under `outputs/paper1_publication_v0/`, preserve model/role/source/checkpoint and ROI/stream/regime fields, emit dry-run plans, and pass strict verification before execution.

### End-of-session report - Authorization Verification And Execution-Path Block

Historical status retained for provenance only. This blocked state was superseded by the later Clean Rerun Execution Infrastructure report.

1. **Publication-contract change:** no clean rerun was launched. The earlier authorization claim was repaired by adding strict contract/config/runner gates to regenerated preflight. `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv` now blocks `first_clean_rerun_authorization` on `publication_contract_execution_flag`, `clean_rerun_configs_execution_enabled`, and `clean_rerun_runner_scripts`.
2. **Accepted artifact:** accepted verification artifacts are `outputs/paper1_publication_v0/preflight/authorization_verification_table.csv`, `outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv`, `outputs/paper1_publication_v0/preflight/user_run_cluster_commands.md`, `outputs/paper1_publication_v0/preflight/postrun_import_validation_plan.md`, `outputs/paper1_publication_v0/audits/clean_rerun_audit.csv`, regenerated `outputs/paper1_publication_v0/preflight/expected_outputs_manifest.csv`, regenerated `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`, and `scripts/verify_paper1_clean_rerun_authorization.py`.
3. **Method gate status change:** behavioral, latent neural encoding, geometry, efficiency/resource, and cross-axis method gates remain pass/accepted-with-limitations for pre-rerun readiness. A separate execution-path gate is now blocked; this is not a reopened blocked-model repair.
4. **Paper evidence status change:** this earlier blocked-state report has been superseded. A bounded local primary behavioral smoke output now exists at `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`; final neural encoding, geometry, efficiency, and cross-axis clean outputs still require full cluster execution and import validation.
5. **Reviewer risk reduced:** reduced false-authorization risk by requiring the frozen contract, clean configs, dedicated runners, checkpoint locks, DeepGaze deterministic tensor exports, and expected-output manifest to agree before any cluster or local clean run.

### End-of-session report - Blocked-Model Repair And Rerun Authorization

1. **Publication-contract change:** no clean rerun was launched. `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv` now blocks `first_clean_rerun_authorization` through the `blocked_model_repair` gate.
2. **Accepted artifact:** regenerated readiness artifacts include `model_setup_attempts.csv`, `model_certification_summary.csv`, `user_action_checklist.csv`, `method_rerun_readiness_table.csv`, `rerun_readiness_table.csv`, `first_clean_rerun_plan.md`, DeepGaze IIE/III deterministic validation JSONs, and the DeepGaze MSDB timeout validation JSON.
3. **Method gate status change:** empirical spatial prior is implemented and method leftovers are ready; behavioral, latent, geometry, efficiency, and cross-axis methods remain pre-rerun methods only, not final evidence.
4. **Model setup status change:** AdaptiveNN is now adapter-certified with source, checkpoint, environment, and smoke evidence. DINOv3, SigLIP, MambaVision, HAT, ScanDiff, SemBA, and SemBA-FAST remain evidence-backed setup or user-action blockers, not vague placeholders.
5. **Reviewer risk reduced:** model-universe missingness is now explicit and machine-readable; first clean rerun authorization cannot be confused with adapter smoke success or partial local setup.

### End-of-session report - Paper 1 Publication Matrix V0 Preflight
Historical note: this pre-repair report is retained as provenance only. Its earlier suspended authorization was superseded by the later Blocked-Model Repair And Rerun Authorization report. Do not use this older section to block the currently authorized bounded rerun; use the regenerated readiness artifacts and current `Next Concrete Milestone` instead.

1. **Publication-contract change:** the first clean rerun gate now uses machine-readable method/model/data/checkpoint/environment/ROI/expected-output/readiness artifacts under `outputs/paper1_publication_v0/preflight/` and `outputs/paper1_publication_v0/roi_stream/`. DeepGaze III is the certified scanpath-capable model for the initial clean rerun.
2. **Accepted artifact:** accepted preflight artifacts include `outputs/paper1_publication_v0/preflight/model_certification_summary.csv`, `outputs/paper1_publication_v0/preflight/model_setup_attempts.csv`, `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`, `outputs/paper1_publication_v0/preflight/method_rerun_readiness_table.csv`, `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv`, `outputs/paper1_publication_v0/roi_stream/stream_roi_grouping_spec.csv`, `outputs/paper1_publication_v0/roi_stream/subject_roi_availability.csv`, `outputs/paper1_publication_v0/external/scanpath_or_foveated_certified/manifest.json`, and `configs/paper1_latent_neural_stream_admission.yaml`.
3. **Method gate status change:** model adapter comparability and efficiency/resource allocation remain `accepted_with_limitations`; the prior `first_clean_rerun_authorization=authorized` state is superseded and must be regenerated after blocked-model repair.
4. **Paper evidence status change:** ConvNeXt, SwinV2, Hiera, DeepGaze III, DeepGaze MSDB, DynamicViT, and ToMe remain preflight-certified or setup-preflighted, but their status does not authorize rerun execution. DINOv3, SigLIP, MambaVision, HAT, ScanDiff, AdaptiveNN, SemBA/SemBA-FAST, and empirical spatial prior require a blocked-model repair pass. No clean rerun output is `accepted_publication_evidence`.
5. **Reviewer risk reduced:** reduced missing latent-feature risk for DeepGaze, missing scanpath-gate risk via DeepGaze III, adapter-incomparability risk through full-universe certification rows, stream/ROI ambiguity through explicit availability manifests, and stale-runbook risk through generated rerun-readiness and user-action tables.

## Cluster Workflow Guidance

Use the JHU DSAI cluster for long GPU or high-I/O jobs, including full-dataset saliency map export, full-image neural encoding reruns, large geometry regeneration, broad benchmark scoring, or other runs that would tie up the laptop for hours. 

DO NOT ACCESS THE CLUSTER YOURSELF-ALL CLUSTER INTERACTION MUST BE DONE WITH COMMANDS GIVEN TO USER!!!

Cluster account and workspace:

- Login: `zzhan330@dsailogin.arch.jhu.edu`.
- Project workspace: `/scratch/tshu2/zzhan330/saliency`.
- Use git for tracked source/config/test changes whenever possible.
- Use WSL `rsync` for large or untracked data, generated maps, model caches, and output directories.
- Do not rely on git alone for raw datasets, precomputed artifacts, or generated outputs.
- Observed partitions: use `l40s` for external-model inference and `cpu` for
  PCA/ridge/geometry and behavioral scoring.
- Generated requests: one L40S, `14` CPUs, and `48 GiB` for export tasks;
  `32` CPUs and `120 GiB` for each model's sequential four-ROI analysis task.

Recommended laptop-to-cluster pattern:

1. Commit/push tracked code changes when appropriate, then `git pull` on the cluster.
2. If changes are not ready to commit, sync the working tree from WSL.
3. Sync only the data roots required by the specific job.
4. Run a small smoke job on the cluster before launching the full Slurm job.
5. Copy only the required outputs back to the laptop.
6. Run local audits/summaries after outputs return, then update this status file.

Working-tree sync from Windows `cmd.exe` through WSL:

```cmd
wsl -e bash -lc "cd /mnt/d/Git/saliency && rsync -az --delete --exclude '/.git/' --exclude '/.venv/' --exclude '/external/' --exclude '/data/' --exclude '/outputs/' --exclude '/logs/' --exclude '/.pytest_tmp/' --exclude '__pycache__/' ./ zzhan330@dsailogin.arch.jhu.edu:/scratch/tshu2/zzhan330/saliency/"
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
- Use reduced-scope cluster commands only when they produce an adapter-certification artifact, method-gate artifact, publication-contract audit, or environment-readiness result required by `Next Concrete Milestone`.
- Monitor with `squeue -u zzhan330` and `tail -f` on the relevant Slurm log.
- After cluster completion, verify expected file counts before copying outputs back.

## Completed Milestone — Publication Contract And Full Adapter Audit

Status: completed on 2026-06-14.

This milestone produced the publication contract, evidence-reset plan, method SOTA audits, method acceptance gates, model availability audit, adapter comparability audit, clean rerun plans, stream/ROI grouping spec, efficiency/resource-allocation plan, and publication-root plan.

The milestone does **not** authorize full publication reruns, adaptive sweeps, paper interpretation, or manuscript writing.

Current superseding gate state:

- behavioral evaluation: `accepted_with_limitations`;
- latent-feature neural encoding: `accepted_with_limitations`;
- representational geometry: `accepted_for_publication_rerun`;
- model adapter comparability: `accepted_with_limitations`;
- efficiency/resource allocation: `accepted_with_limitations`;
- cross-axis inference: `accepted_with_limitations`.

Current evidence state:

- no empirical artifact is accepted as final Paper 1 publication evidence;
- legacy outputs remain scaffold/provenance only;
- bounded admission-panel artifacts exist but remain `admission_panel_not_final_paper_result`;
- expanded certified rows and blocked setup rows are recorded in the preflight artifacts; final empirical evidence is still absent until the bounded clean rerun is completed.

Next milestone:

> Verify the completed rerun authorization, build or verify the bounded clean-rerun execution path, and execute the bounded first clean rerun if verification passes. Do not reopen blocked-model repair unless artifact-level verification fails.

## Next Concrete Milestone

Priority: **Execute Updated V0 Full Cluster Rerun With Primary Behavioral Latent-Fixation Evidence**.

The V0 rerun path is authorized, verified, and regenerated after replacing the old behavioral lane. The remaining blocker is execution: the full cluster package must run, outputs must be copied back, and import validation must pass. The old behavioral fixation/saliency/scanpath/map-metric lane is marked `legacy_behavioral_pipeline_excluded_from_v0` and must stay out of primary V0 evidence.

This milestone is not another blocked-model repair pass, audit-only pass, config-only pass, runner-only pass, or incremental planning loop. It must execute the generated cluster package or document an execution blocker with the exact failed command, log path, missing artifact, and next repair command.

### Stage A — Freeze the updated behavioral evidence contract

Status: completed for configuration/routing; full rerun execution pending.

The V0 evidence contract now sets the primary behavioral lane to `primary_behavioral_latent_to_fixation_encoding`.

Required checks:

1. old behavioral configs/runners are marked `legacy_behavioral_pipeline` or disabled for V0;
2. `configs/paper1_clean_behavioral_rerun.yaml` is not used as the V0 primary behavioral lane unless it is rewritten to call the latent-to-fixation pipeline;
3. no V0 clean behavioral output path points to old SALICON/CAT2000/COCO-Search18 map-metric aggregate files;
4. expected-output manifests contain the new behavioral latent-to-fixation outputs;
5. cross-axis assembly reads behavioral evidence from `outputs/paper1_publication_v0/behavioral_latent_fixation/`, not from legacy behavioral aggregate paths;
6. native behavioral outputs are labeled diagnostic-only if retained anywhere.

Required artifact:

- `outputs/paper1_publication_v0/audits/legacy_behavioral_pipeline_exclusion_audit.csv`
- `outputs/paper1_publication_v0/audits/primary_behavioral_latent_fixation_audit.csv`

Acceptance rule:

If any old behavioral-map/saliency/scanpath output still enters the V0 primary behavioral axis, Codex must repair the config/runner/cross-axis path before running anything.

### Stage B — Implement the primary behavioral latent-to-fixation pipeline

Status: implemented and smoke-tested.

Implemented behavior:

1. `src/hma/behavioral/latent_fixation.py` provides dataset loading, external latent-feature artifact loading, image-order validation, deterministic splits, train-only flattening/reduction, validation-selected ridge readouts, spatial probability normalization, held-out fixation scoring, and aggregate/image-level output rows.
2. `scripts/run_paper1_primary_behavioral_latent_fixation.py` runs the lane, supports dry-run and `local_smoke`, writes the required behavioral latent-fixation outputs, and writes the legacy pipeline exclusion audit.
3. `configs/paper1_primary_behavioral_latent_fixation.yaml` defines the full and local-smoke lane inputs.
4. The clean rerun/common runner, expected-output manifest, preflight verification, cluster job generator, and cross-axis assembly route behavioral evidence through `latent_fixation_information_gain`.

Required outputs:

- `configs/paper1_primary_behavioral_latent_fixation.yaml`
- `scripts/run_paper1_primary_behavioral_latent_fixation.py`
- `src/hma/behavioral/latent_fixation.py`
- updated `outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv`
- updated `outputs/paper1_publication_v0/preflight/expected_outputs_manifest.csv`
- `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`
- `outputs/paper1_publication_v0/audits/legacy_behavioral_pipeline_exclusion_audit.csv`

Acceptance rule:

A local smoke test has trained and scored an eligible model/layer on a bounded subset, written the required lane files, and produced the legacy exclusion audit. Full publication V0 still requires the generated cluster package to run and be imported.

### Stage C — Updated V0 execution path

The V0 rerun path is authorized and regenerated so the behavioral lane is the primary latent-to-fixation encoding lane. The bounded local smoke output exists; full cluster execution remains pending.

Required command/update package:

- sync/update commands if cluster execution is needed;
- environment activation commands;
- feature-artifact/cache verification commands;
- one small verification command for `run_paper1_primary_behavioral_latent_fixation.py`;
- full V0 rerun command or `sbatch` script using the updated behavioral lane;
- `squeue` and log-monitoring commands if using Slurm;
- expected-output verification commands;
- output-copy-back commands if outputs are generated on cluster;
- local import/merge validation commands after outputs return;
- explicit check that old behavioral aggregate paths are absent from cross-axis input.

Required artifact:

- updated `outputs/paper1_publication_v0/preflight/user_run_cluster_commands.md`

Acceptance rule:

Codex must not close the session with only instructions. It must update the command package, run every feasible local verification command, and leave the exact full rerun command ready for execution. If the session environment can execute the lane locally, Codex must execute the bounded rerun locally. If cluster execution is required, the command package must be exact and tied to expected outputs.

### Stage D — Bounded first clean rerun

After Stages A-C pass, Codex may execute local lanes and provide user-run cluster commands for cluster lanes.

Required outputs:

- `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`
- `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_image_scores.csv`
- `outputs/paper1_publication_v0/behavioral_latent_fixation/feature_reduction_metadata.json`
- `outputs/paper1_publication_v0/behavioral_latent_fixation/readout_selection_artifact.json`
- `outputs/paper1_publication_v0/neural_encoding/encoding_scores.csv`
- `outputs/paper1_publication_v0/geometry/geometry_scores.csv`
- `outputs/paper1_publication_v0/efficiency/efficiency_profiles.csv`
- `outputs/paper1_publication_v0/efficiency/resource_allocation_profiles.csv`
- `outputs/paper1_publication_v0/cross_axis/model_axis_scores.csv`
- `outputs/paper1_publication_v0/audits/primary_behavioral_latent_fixation_audit.csv`
- `outputs/paper1_publication_v0/audits/clean_rerun_audit.csv`

Acceptance rule:
Every output must be labeled as clean publication-root evidence or rejected/diagnostic evidence according to the publication contract. No legacy behavioral-map/saliency/scanpath output, admission-panel output, Matrix V1 output, or Matrix V2 output may be silently merged. The cross-axis behavioral score must come from `latent_fixation_information_gain` or an explicitly named primary latent-fixation score produced by the new pipeline.

### Stage E — Post-run audit and status update

After the bounded rerun completes, Codex must audit outputs before reporting scientific results.

Required checks:

1. output files exist at expected paths;
2. row counts and model/axis coverage match the eligibility table;
3. no blocked/excluded model is silently included;
4. no eligible certified model is silently omitted;
5. no admission/provenance output is merged as clean evidence;
6. metrics are finite or failures are explicitly logged;
7. behavioral/free-viewing/task-search/scanpath regimes remain separated;
8. neural/geometry outputs retain ROI/stream/subject scope;
9. efficiency rows retain matched-cost and total-cost fields;
10. cross-axis table reports availability and scores without causal or paper-facing interpretation.

Required artifacts:

- `outputs/paper1_publication_v0/audits/clean_rerun_audit.csv`
- `outputs/paper1_publication_v0/audits/publication_contract_compliance.md`
- updated `docs/project_status_and_next_steps.md`

Acceptance rule:

The session succeeds only if clean publication-root evidence exists and passes audit, or if execution fails with exact failed command, error/log path, missing artifact, and next repair command.

### Non-goals

Do not reopen blocked-model repair unless verification fails.

Do not run a broader matrix than the regenerated authorization allows.

Do not produce paper interpretation.

Do not collapse free-viewing, task-search, and scanpath metrics.

Do not collapse neural evidence into V1-only summaries.

Do not treat smoke tests, preflight tables, or admission artifacts as clean paper evidence.

Do not claim cluster execution unless the user ran the commands and outputs were imported/validated.

### Acceptance rule

The milestone succeeds only if:

1. authorization verification passes or failed claims are repaired;
2. execution paths exist for behavioral, neural, geometry, efficiency/resource, and cross-axis lanes;
3. cluster/user-run commands are produced where needed;
4. clean publication-root outputs are generated or the exact execution blocker is recorded;
5. post-run audit proves outputs follow the publication contract;
6. project status is updated with evidence status, not scientific interpretation.

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

Generated observer-control, task-prior, DeepGaze, transformer-relevance, and merged behavioral outputs are scaffold/provenance only. They should be listed in `evidence_reset_manifest.csv`, not in the data-readiness section.

## Later Milestones

Proceed in phases that map directly to the revised publication question:

> Do human gaze-prediction models, human-gaze-inspired adaptive/foveated models, generic efficient-computation models, general representation models, VLM/semantic models, hierarchical/hybrid models, and spatial/task-prior controls converge or dissociate across behavioral fixation/scanpath alignment, latent-feature neural encoding, latent representational geometry, stream specificity, and efficiency/resource allocation?

### Phase 0 — Evidence reset and publication root freeze

Status: complete. Contract and root are frozen; no empirical evidence exists.

Purpose:

* prevent legacy outputs from shaping paper claims;
* preserve old outputs as scaffold/provenance only;
* create a clean publication evidence root.

Required outputs:

* `outputs/paper1_scope_reset/evidence_reset_manifest.csv`
* `outputs/paper1_scope_reset/publication_output_root_plan.md`
* `configs/paper1_publication_contract.yaml`
* `outputs/paper1_publication_v0/`

Acceptance rule:

No old output can enter publication summaries unless regenerated under the publication contract or certified by equivalence audit.

### Phase 1 — Literature-grounded method acceptance

Status: audit complete. Behavioral evaluation, neural encoding, adapter comparability, efficiency/resource allocation, and cross-axis inference are accepted with limitations; representational geometry is accepted for publication rerun. Remaining work is implementation preflight and explicitly listed limitations, not a general method-gap state.

Purpose:

* verify that the planned method is academically valid before full publication reruns;
* position the project as a controlled benchmark method rather than a SOTA leaderboard method;
* identify method gaps that would make future full runs scientifically weak.

Required outputs:

* `outputs/paper1_scope_reset/method_sota_alignment_audit.md`
* `outputs/paper1_scope_reset/method_acceptance_gates.md`
* `outputs/paper1_scope_reset/behavioral_metric_acceptance_table.csv`
* `outputs/paper1_scope_reset/neural_encoding_acceptance_table.csv`
* `outputs/paper1_scope_reset/geometry_acceptance_table.csv`
* `outputs/paper1_scope_reset/efficiency_acceptance_table.csv`
* `outputs/paper1_scope_reset/cross_axis_inference_acceptance_table.csv`

Acceptance rule:

No publication-root full rerun begins until the relevant method gates are `accepted_for_publication_rerun` or `accepted_with_limitations`.

### Phase 2 — Full model-role matrix and adapter certification
Status: blocked-model repair pass complete for the current regenerated preflight. Model/data/method/cache/DeepGaze verification passes, and clean execution infrastructure is now strictly verified.

Purpose:

* inspect every required model family;
* record behavioral-output, latent-feature, geometry, efficiency, and resource-allocation eligibility;
* assign `paper_evidence_status`;
* avoid convenience-based model reduction.

Required candidate families:

* CNN/local hierarchy anchors;
* ViT/DeiT anchors;
* DINOv2/DINOv3 SSL dense-feature models;
* CLIP/SigLIP VLM/semantic models;
* MambaVision, Hiera, Swin/SwinV2 hierarchical/hybrid/efficient models;
* DynamicViT and ToMe generic efficient-computation models;
* DeepGaze, HAT, ScanDiff, SemBA/SemBA-FAST gaze or scanpath models;
* AdaptiveNN human-gaze-inspired adaptive/foveated model;
* center, random, spatial-prior, and task-prior controls.

Required outputs:

* `outputs/paper1_scope_reset/model_role_matrix.csv`
* `outputs/paper1_scope_reset/model_availability_audit.md`
* `outputs/paper1_scope_reset/model_adapter_comparability_table.csv`
* `outputs/paper1_scope_reset/latent_feature_adapter_requirements.csv`
* `outputs/paper1_scope_reset/behavioral_output_adapter_requirements.csv`

Acceptance rule:

Every required candidate has a status. A model may be behavior-only, diagnostic-only, or rejected after audit, but it must not disappear before audit.

### Phase 2.5 — Gate-conditioned admission panel
Status: bounded local admission object generated on 2026-06-15 and retained as provenance only. Blocked-model repair has now superseded the earlier suspended authorization. Phase 2.5 feeds into authorization verification and bounded clean-rerun execution, not paper interpretation.

Purpose:

Produce the first bounded publication-root evidence object without treating it as the final paper result.

Required outputs:

- `outputs/paper1_publication_v0/behavioral/aggregate_admission_panel.csv`
- `outputs/paper1_publication_v0/behavioral/uncertainty_admission_panel.csv`
- `outputs/paper1_publication_v0/neural_encoding/encoding_scores_admission_panel.csv`
- `outputs/paper1_publication_v0/geometry/geometry_scores_admission_panel.csv`
- `outputs/paper1_publication_v0/efficiency/efficiency_profiles_admission_panel.csv`
- `outputs/paper1_publication_v0/efficiency/resource_allocation_profiles_admission_panel.csv`
- `outputs/paper1_publication_v0/cross_axis/model_axis_scores_admission_panel.csv`
- `outputs/paper1_publication_v0/cross_axis/admission_panel_preflight.md`
- `outputs/paper1_publication_v0/audits/admission_panel_audit.csv`

Acceptance rule:

Admission-panel outputs may diagnose readiness, coverage, missingness, and role/family completeness. They must not be interpreted as final paper evidence or used for manuscript claims.

### Phase 3 — Clean primary behavioral latent-to-fixation rerun
Status: implemented, smoke-tested, routed, and cluster-ready; full V0 cluster execution remains pending.

Purpose:

Regenerate behavioral evidence under the repaired publication contract using a neural-encoding-like latent-to-fixation pipeline. The primary behavioral score is not native map similarity or scanpath similarity. It is held-out human fixation information linearly decodable from frozen model latent features under a matched probabilistic readout.

Required outputs:

* `configs/paper1_primary_behavioral_latent_fixation.yaml`
* `scripts/run_paper1_primary_behavioral_latent_fixation.py`
* `src/hma/behavioral/latent_fixation.py`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_image_scores.csv`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/feature_reduction_metadata.json`
* `outputs/paper1_publication_v0/behavioral_latent_fixation/readout_selection_artifact.json`
* `outputs/paper1_publication_v0/audits/primary_behavioral_latent_fixation_audit.csv`
* `outputs/paper1_publication_v0/audits/legacy_behavioral_pipeline_exclusion_audit.csv`

Acceptance rule:

The phase is accepted for bounded local smoke because the pipeline trains, scores, writes the required publication-root behavioral latent-fixation files, is routed into V0 preflight/cross-axis infrastructure, and audits legacy behavioral-map/saliency/scanpath outputs as `legacy_behavioral_pipeline_excluded_from_v0`. Full publication V0 acceptance still requires cluster execution, copy-back, and import validation.

### Phase 4 — Clean latent-feature neural and geometry rerun

Status: authorized for bounded clean rerun execution after strict dry-run verification; DeepGaze bounded tensor-export validation passes; not yet executed.

Purpose:

Regenerate latent-feature neural encoding and latent-feature representational geometry under the publication contract.

Required outputs:

* `configs/paper1_latent_neural_matrix.yaml`
* `outputs/paper1_publication_v0/neural_encoding/encoding_scores.csv`
* `outputs/paper1_publication_v0/geometry/geometry_scores.csv`
* `outputs/paper1_publication_v0/audits/neural_geometry_audit.csv`

Acceptance rule:

Central model classes must expose latent features and pass through the same neural/geometry pipeline. Models that cannot do this after audit may remain behavioral or diagnostic models, but they cannot carry the neural-alignment claim.

### Phase 5 — Stream/ROI and subject-robustness analysis

Status: after Phase 4 outputs exist.

Purpose:

Move from flat ROI tables to early / dorsal-lateral-parietal / ventral-semantic conclusions and subject-aware robustness.

Required outputs:

* `outputs/paper1_publication_v0/roi_stream/stream_roi_grouping_spec.csv`
* `outputs/paper1_publication_v0/roi_stream/model_by_stream_encoding.csv`
* `outputs/paper1_publication_v0/roi_stream/model_by_stream_geometry.csv`
* `outputs/paper1_publication_v0/roi_stream/subject_robustness_summary.csv`

Acceptance rule:

Every neural and geometry claim must state its ROI/stream and subject scope.

### Phase 6 — Efficiency and resource allocation

Status: authorized for bounded clean rerun execution after strict dry-run verification; not yet executed.

Purpose:

Make resource allocation central rather than decorative.

Required outputs:

* `outputs/paper1_publication_v0/efficiency/efficiency_profiles.csv`
* `outputs/paper1_publication_v0/efficiency/resource_allocation_profiles.csv`
* `outputs/paper1_publication_v0/efficiency/alignment_per_compute.csv`

Acceptance rule:

Efficiency metrics must be connected to model role and behavioral/neural/geometry axes. Sequential and active-vision models must report total task/image cost, not only per-glimpse cost.

### Phase 7 — Cross-axis publication analysis

Status: assembly/audit may run after clean behavioral, neural, geometry, and efficiency outputs exist; interpretation remains blocked until post-run audit passes.

Purpose:

Build the actual paper result.

Required outputs:

* `outputs/paper1_publication_v0/cross_axis/model_axis_scores.csv`
* `outputs/paper1_publication_v0/cross_axis/stream_specific_quadrants.csv`
* `outputs/paper1_publication_v0/cross_axis/sensitivity_leave_one_model.csv`
* `outputs/paper1_publication_v0/cross_axis/sensitivity_leave_one_family.csv`
* `outputs/paper1_publication_v0/cross_axis/bootstrap_or_uncertainty_intervals.csv`
* `outputs/paper1_publication_v0/cross_axis/paper_claim_decision.md`

Acceptance rule:

Only publication-root evidence can enter cross-axis analysis. The primary behavioral column must be derived from `outputs/paper1_publication_v0/behavioral_latent_fixation/fixation_encoding_scores.csv`, using held-out fixation information gain or an explicitly named equivalent primary latent-fixation score. Legacy behavioral-map/saliency/scanpath metrics may appear only as diagnostic secondary columns and must not determine quadrant labels. Quadrant labels are descriptive unless the cross-axis inference gate allows stronger language.

### Phase 8 — External positioning and paper split decision

Status: after cross-axis results.

Purpose:

Position the findings against broader NeuroAI alignment work and decide the paper shape.

Decision options:

1. **Main-track Paper 1:** if Publication Matrix V0 shows robust, interpretable convergence or dissociation across model role, ROI/stream, and efficiency.
2. **Workshop/thesis/methods Paper 1:** if the measurement framework is strong but the result is underpowered or mainly methodological.
3. **Shift main effort to Paper 2:** if observational cross-axis results remain weak after the full contract.
4. **Paper 2 causal intervention:** use human gaze, foveation, token pruning, adaptive readout, or saliency-guided computation to test whether changing attention/resource allocation changes alignment, efficiency, or neural predictivity.

Do not decide the publication split from scaffold outputs.


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

Primary behavioral latent-to-fixation encoding:

- `src/hma/behavioral/latent_fixation.py`
- `scripts/run_paper1_primary_behavioral_latent_fixation.py`
- `configs/paper1_primary_behavioral_latent_fixation.yaml`
- outputs under `outputs/paper1_publication_v0/behavioral_latent_fixation/`

Legacy behavioral benchmark:

- `src/hma/experiments/saliency_benchmark.py`
- `src/hma/experiments/aggregate_results.py`
- `src/hma/experiments/summarize_results.py`
- `scripts/run_saliency_benchmark.py`
- `scripts/run_v2_matrix.py`
- `scripts/aggregate_results.py`

Legacy behavioral benchmark code remains usable for diagnostics and expected-range checks, but it is excluded from V0 primary behavioral evidence.

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

External model integration and publication adapter scaffolds:

- `configs/paper1_matrix_v2.yaml`
- `configs/external_models/registry.yaml`
- `configs/external_models/environments/`
- `src/hma/external/registry.py`
- `src/hma/external/artifacts.py`
- `src/hma/external/adapters.py`
- `scripts/setup_external_model.py`
- `scripts/run_external_model.py`
- `scripts/apply_external_patches.py`
- `scripts/export_external_routing_maps.py`
- `scripts/create_paper1_matrix_v2_configs.py`
- `scripts/audit_paper1_matrix_v2.py`
- `scripts/run_paper1_matrix_v2_scientific64.py`
- `tests/test_external_model_integration.py`

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

- `scripts/summarize_paper1_matrix_v2_full.py`
- `scripts/create_attribution_family_interpretation.py`
- `scripts/create_paper_inspection_pack.py`
- `scripts/audit_behavioral_controls.py`
- `scripts/audit_transformer_relevance_control.py`
- `scripts/audit_neural_reliability_metadata.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

Adaptive/foveated/gaze-model adapters:

- `src/hma/external/` for publication-contract adapters;
- `configs/external_models/registry.yaml` for required candidate registration;
- future AdaptiveNN / HAT / ScanDiff / SemBA adapter modules should expose both behavioral outputs and latent-feature hooks where available.

Blocked-model repair and preflight regeneration:

- `scripts/generate_paper1_publication_preflight.py`
- `outputs/paper1_publication_v0/preflight/model_setup_attempts.csv`
- `outputs/paper1_publication_v0/preflight/model_certification_summary.csv`
- `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`
- `outputs/paper1_publication_v0/preflight/method_rerun_readiness_table.csv`
- `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv`
- future repair scripts should explicitly target empirical spatial prior, SigLIP, MambaVision, DINOv3, DeepGaze tensor export, ScanDiff, HAT, SemBA/SemBA-FAST, and AdaptiveNN.

Authorized rerun verification and execution:

- `outputs/paper1_publication_v0/preflight/authorization_verification_table.csv`
- `outputs/paper1_publication_v0/preflight/execution_path_verification_table.csv`
- `outputs/paper1_publication_v0/preflight/expected_outputs_manifest.csv`
- `outputs/paper1_publication_v0/preflight/user_run_cluster_commands.md`
- `outputs/paper1_publication_v0/preflight/postrun_import_validation_plan.md`
- `outputs/paper1_publication_v0/audits/clean_rerun_audit.csv`
- next implementation should verify or create runner commands for behavioral, latent neural encoding, geometry, efficiency/resource allocation, and cross-axis assembly under `outputs/paper1_publication_v0/`.
