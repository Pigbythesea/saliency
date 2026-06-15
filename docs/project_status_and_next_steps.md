# HMA Project Status And Next Steps

Updated: 2026-06-14

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

## Publication Evidence Contract Override

This file now prioritizes Paper 1 Publication Matrix V0 over all previous Matrix V2 and V1 outputs.

Whenever older sections, paths, or result summaries conflict with the publication-scope reset, Codex must follow the publication-scope reset.

The publication evidence root is:

`outputs/paper1_publication_v0/`

No result outside this root is final Paper 1 evidence unless it is explicitly regenerated under the frozen publication contract or certified by an equivalence audit.

No publication-root full rerun is allowed unless the relevant behavioral, neural, geometry, adapter-comparability, efficiency, and cross-axis method gates are passed or explicitly accepted with limitations.

The current active task is **Gate-Conditioned Publication Admission Panel V1**. It is not interpretation, manuscript writing, a full final publication-root run, adapter-only installation, metric-only implementation, or model-universe reduction. The task is to resolve the minimum remaining behavioral blocker, admit a role-structured model panel under the frozen method gates, and generate the first bounded publication-root admission evidence object across behavioral, neural, geometry, efficiency, and cross-axis-preflight axes.

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
  - latent-feature neural encoding and latent-feature representational geometry as mandatory main evidence;
  - behavioral fixation/scanpath output as behavioral evidence;
  - output-map-to-fMRI encoding only as a secondary diagnostic/control, not the primary neural-comparison scheme.

Paper 1 should remain centered on whether fixation alignment, latent neural encoding, latent representational geometry, cortical stream structure, model role, and efficiency converge or dissociate. The publication-facing version must test whether these axes dissociate systematically by cortical stream and by model role.

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
- a clean latent-feature geometry artifact under the publication output root;
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
- behavioral uncertainty and sequence/task metrics before clean behavioral reruns;
- debiased geometry and geometry resampling before clean geometry reruns;
- sequential/adaptive total-cost accounting before efficiency comparisons;
- family-aware cross-axis sensitivity before paper-facing quadrant interpretation;
- latent-feature neural encoding and latent-feature geometry over output-map neural controls;
- stream/ROI structure over flat ROI averages;
- adaptive/foveated/scanpath/selective-computation mechanisms over additional post-hoc heatmap variants;
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

- No final Paper 1 publication evidence matrix exists yet.
- No legacy output is accepted as final paper evidence by default.
- The next accepted scientific artifact must be a gate-conditioned admission panel: behavioral distribution metrics admitted after log-likelihood/information-gain implementation, model rows admitted only after certification, and bounded publication-root evidence rows generated across behavioral, latent neural encoding, corrected geometry, efficiency/resource allocation, and cross-axis preflight.
- The repaired static DeiT-S / DynamicViT / ToMe run is a method-validation artifact for the generic efficient-computation role. It must not define the final model scope or paper interpretation.
- All final behavioral, latent-feature neural, latent-feature geometry, efficiency, and cross-axis evidence must be regenerated under a new publication output root unless explicitly certified by an equivalence audit.

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

SALICON and CAT2000 are free-viewing datasets. COCO-Search18 is task-search. They must remain separated in all publication-root summaries.

Point-fixation metrics, map-distribution metrics, task-search metrics, and scanpath/sequence metrics must be reported as distinct behavioral objects.

DeepGaze, HAT, ScanDiff, SemBA/SemBA-FAST, AdaptiveNN, center priors, random baselines, task priors, attribution maps, routing maps, and token/glimpse maps must each retain their model-role and behavioral-object labels.

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

> A controlled frozen-feature and behavioral-output benchmark for comparing model roles across behavioral fixation/scanpath alignment, latent-feature neural encoding, representational geometry, cortical stream structure, and efficiency/resource allocation.

The publication method must not be described as:

* an Algonauts leaderboard-equivalent fMRI model;
* a SOTA saliency model;
* a causal attention-intervention study;
* proof that fixation alignment causes neural alignment;
* proof that output-map alignment replaces latent-feature neural encoding.

### Gate 1 — Behavioral evaluation acceptance

Status: `method_gap_found`.

Before clean behavioral reruns, Codex must produce:

* `outputs/paper1_scope_reset/method_behavioral_sota_audit.md`
* `outputs/paper1_scope_reset/behavioral_metric_acceptance_table.csv`

The audit must decide:

* which metrics are primary for point-fixation maps;
* which metrics are primary for map-distribution comparisons;
* which metrics are primary for task-search outputs;
* which metrics are primary for scanpath or sequential outputs;
* whether image-level, observer-level, or clustered bootstrap uncertainty is required;
* how SALICON/CAT2000 free-viewing and COCO-Search18 task-search remain separated;
* how DeepGaze, HAT, ScanDiff, SemBA/SemBA-FAST, AdaptiveNN, center priors, task priors, attribution maps, routing maps, and token/glimpse maps are labeled as distinct behavioral objects.

A behavioral rerun can proceed only after this gate is `accepted_for_publication_rerun` or `accepted_with_limitations`.

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
- clean behavioral rerun;
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

Updated: 2026-06-14

Current implementation state is now classified by **publication readiness**, not by accumulated runs, smoke passes, or legacy Matrix V2 progress.

### Publication-readiness classification

Current implementation readiness is:

| component | current classification | publication-facing meaning |
| --- | --- | --- |
| publication contract | frozen_v0 | `configs/paper1_publication_contract.yaml` governs all future evidence |
| publication output root | created_empty | `outputs/paper1_publication_v0/` contains no empirical evidence yet |
| evidence reset | complete | every current top-level output root is classified and excluded by default |
| behavioral evaluation method | method_gap_found | clustered/hierarchical uncertainty and conditional/sequence interfaces are implemented; general map-distribution information gain still blocks the full rerun |
| latent-feature neural encoding method | accepted_with_limitations | controlled frozen-feature PCA/ridge is accepted; voxel-specific spatial-readout sensitivity is required |
| representational geometry method | accepted_for_publication_rerun | feature-space debiased linear CKA, image-resampling intervals, paired response permutations, and CKA/RSA agreement reporting are implemented |
| model adapter comparability | accepted_with_limitations | all 26 candidates have publication registry records; three external models and three built-in controls are certified, and all others have executable setup scaffolds or source-blocker records |
| efficiency/resource-allocation method | accepted_with_limitations | total per-image/task sequential/adaptive cost schema and artifact summaries are implemented; model-specific emissions remain adapter-dependent |
| cross-axis inference method | accepted_with_limitations | regime/object and minimum-panel preflight, leave-one-family sensitivity, and family-block bootstrap are implemented; no publication evidence panel exists |
| full model-role matrix | complete | 26 rows cover every required family and control without convenience-based exclusion |
| local anchor latent capability | setup_scaffold_ready | ResNet, ConvNeXt, ViT, DINOv2, and CLIP have common timm adapter entries and explicit pin/hash/smoke blockers |
| gaze/scanpath latent capability | setup_scaffold_ready | DeepGaze, HAT, and ScanDiff have typed conditioning/output contracts, runtime adapter classes, setup commands, and machine-readable blockers |
| AdaptiveNN latent capability | setup_scaffold_ready | source/environment/checkpoint/resource-hook blockers are represented by an executable runtime entry |
| SemBA / SemBA-FAST | setup_blocked_not_excluded | records explicitly identify unavailable official source/checkpoint/API as the blocker |
| legacy outputs | excluded | no existing empirical root is accepted or equivalence-certified |

### Current blockers to real paper progress
The project cannot move to publication-facing full runs until these blockers are resolved:

1. publication source pins, checkpoint hashes, environment locks, and smoke certification are missing for most local and external candidates;
2. DINOv3, DeepGaze, HAT, ScanDiff, AdaptiveNN, and related scaffold adapters still require model-specific execution hooks;
3. general free-viewing map-distribution information gain/log-likelihood remains unimplemented;
4. sequential/adaptive models must emit the new total-cost fields after their adapters are installed;
5. stream ROI subject robustness beyond subj01 is unavailable;
6. SemBA and SemBA-FAST official source/checkpoint APIs remain unresolved.

### Current implementation priority

Active priority:

> Execute Gate-Conditioned Publication Admission Panel V1: first remove the behavioral distribution-metric blocker, then admit a role-structured model panel under certification rules, then generate bounded publication-root admission evidence across behavioral, latent neural encoding, corrected geometry, efficiency/resource allocation, and cross-axis preflight.

This is not an adapter-only, metric-only, local-anchor-only, or Markdown-only milestone. The canonical artifact list lives in `Next Concrete Milestone`. Do not duplicate it here.

### Required end-of-session report

Use the publication-readiness report format in `Current Implementation Progress`. Do not use the older scientific-claim report format until publication-root evidence exists.

Do not report smoke tests, debugging fixes, old-result summaries, or legacy audits as progress unless they directly change one of the publication-contract artifacts above.

Implementation history is archived in `docs/project_status_changelog.md`.

### End-of-session report — Publication Gate-Unblocking Implementation V1

1. **Publication-contract change:** added a typed publication adapter registry and certification schema covering image-only, task-conditioned, gaze-history-conditioned, scanpath, foveated, stochastic, and active-vision models; added reusable behavioral, geometry, efficiency, and family-aware cross-axis methods without changing the frozen paper claim.
2. **Accepted artifact:** `configs/external_models/publication_registry.yaml`; `outputs/paper1_scope_reset/adapter_certification_records.jsonl`; `outputs/paper1_scope_reset/adapter_certification_summary.csv`; updated model-role, adapter-comparability, and method-gate tables under `outputs/paper1_scope_reset/`.
3. **Method gate status change:** geometry moved from `method_gap_found` to `accepted_for_publication_rerun`; adapter comparability, efficiency/resource allocation, and cross-axis inference moved to `accepted_with_limitations`; behavioral evaluation remains `method_gap_found`, with uncertainty and sequence blockers removed.
4. **Paper evidence status change:** static DeiT-S, DynamicViT, and ToMe remain `adapter_certified`; center, random, and COCO-Search18 task-prior controls have certified built-in records; all named nonlocal candidates now have `setup_scaffold_ready` or explicit `setup_blocked` records rather than `not_started`.
5. **Reviewer risk reduced:** reduced adapter incomparability, weak behavioral uncertainty, biased geometry, per-glimpse-only efficiency accounting, behavioral-regime mixing, and model-family dependence risk.
6. **Next decisive step:** execute Gate-Conditioned Publication Admission Panel V1 by resolving the behavioral log-likelihood/information-gain blocker, certifying or role-blocking enough models for a role-structured admission panel, and generating bounded `admission_panel` evidence under `outputs/paper1_publication_v0/` across behavioral, latent neural encoding, corrected geometry, efficiency/resource allocation, and cross-axis preflight.

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

Current gate state:

- behavioral evaluation: `method_gap_found`;
- latent-feature neural encoding: `accepted_with_limitations`;
- representational geometry: `accepted_for_publication_rerun`;
- model adapter comparability: `accepted_with_limitations`;
- efficiency/resource allocation: `accepted_with_limitations`;
- cross-axis inference: `accepted_with_limitations`.

Current evidence state:

- no empirical artifact is accepted as Paper 1 publication evidence;
- legacy outputs remain scaffold/provenance only;
- static DeiT-S, DynamicViT, and ToMe are adapter-certified method artifacts;
- local anchors have proven latent capability but still require publication certification.

Next milestone:

> Execute Gate-Conditioned Publication Admission Panel V1. Do not start the full final publication-root run, but do produce bounded `admission_panel` evidence under the publication root after the relevant method and adapter-admission conditions are satisfied.

## Completed Milestone — Publication Gate-Unblocking Implementation V1

Status: completed on 2026-06-14.

This milestone implemented the missing reusable infrastructure across all four required lanes. It did not run publication-root experiments or paper interpretation.

The milestone implemented the missing infrastructure that directly unblocks the publication gates.

### Required implementation lanes

Codex must work across all four lanes in one session or one coordinated implementation branch:

1. **Adapter certification and setup lane**
   - Implement the common publication adapter-certification harness.
   - Certify local anchors where possible.
   - Also create executable adapter/setup scaffolds for DINOv3, SigLIP, MambaVision, Hiera, Swin, SwinV2, DeepGaze, HAT, ScanDiff, AdaptiveNN, and SemBA/SemBA-FAST where source is available.
   - Missing environment/checkpoint/dependency is not exclusion; it must produce a concrete setup command, registry row, and machine-readable blocking reason.
   - The harness schema must support image-only, task-conditioned, gaze-history-conditioned, scanpath, foveated, stochastic, and active-vision models.

2. **Behavioral gate-unblocking lane**
   - Implement clustered image bootstrap for map metrics.
   - Implement SALICON worker-within-image hierarchical uncertainty.
   - Implement COCO-Search18 subject-within-image/task uncertainty.
   - Add sequence/task metric interface for conditional maps and generated scanpaths.
   - Preserve separate free-viewing, task-search, and scanpath outputs.

3. **Geometry gate-unblocking lane**
   - Implement debiased or corrected linear CKA.
   - Implement image-resampling intervals for CKA and RSA summaries.
   - Keep biased CKA as secondary diagnostic only.
   - Preserve response-permutation controls and CKA/RSA method-agreement reporting.

4. **Efficiency and cross-axis gate-unblocking lane**
   - Extend efficiency schema to total sequential/adaptive cost: fixation count, scanpath length, diffusion/recurrent steps, selected glimpses, stopping behavior, high-resolution sampled area, and total cost per image/task.
   - Implement leave-one-family sensitivity and family-aware uncertainty scaffolds for cross-axis analysis.
   - Do not run cross-axis interpretation; only implement the machinery needed after publication-root evidence exists.

### Required outputs

This milestone must produce executable code, tests, and machine-readable artifacts. Markdown-only updates do not count.

Required outputs include:

- adapter-certification harness code;
- adapter-certification schema tests;
- updated publication registry entries for all required model families;
- certification records or concrete setup-blocker records for all required model families;
- behavioral uncertainty implementation and tests;
- sequence/task metric interface implementation and tests;
- debiased CKA implementation and tests;
- geometry image-resampling implementation and tests;
- sequential/adaptive efficiency schema implementation and tests;
- leave-one-family / family-aware cross-axis sensitivity implementation and tests;
- updated `outputs/paper1_scope_reset/model_adapter_comparability_table.csv`;
- updated `outputs/paper1_scope_reset/model_role_matrix.csv`;
- updated method-gate tables showing which gates moved from `method_gap_found` toward `accepted_with_limitations` or `accepted_for_publication_rerun`.

### Explicit non-goals

Do not run full NSD, behavioral, geometry, efficiency, or cross-axis publication-root experiments.

Do not produce paper-facing interpretation.

Do not reduce the model universe to local anchors.

Do not perform an audit-only or planning-only session.

Do not report tests or smoke runs as progress unless they directly certify a model, unblock a method gate, or produce a reusable publication-run component.

### Acceptance rule
The milestone is complete only if executable code, tests, and machine-readable artifacts remove at least one concrete blocker from each category:

1. adapter comparability across local and nonlocal required model families;
2. behavioral uncertainty or sequence/task metrics;
3. geometry debiasing or geometry uncertainty;
4. efficiency total-cost or cross-axis family-aware inference.

A session that certifies only local anchors fails this milestone.

A session that only creates docs, audits, placeholder files, or untested schemas fails this milestone.

A session that implements only one lane fails this milestone unless it also leaves runnable, tested scaffolds for the other three lanes.

## Next Concrete Milestone

Priority: **Gate-Conditioned Publication Admission Panel V1**.

The previous run produced reusable method infrastructure and full setup scaffolds, but it did not produce publication evidence. The next milestone must move the project from infrastructure to the first bounded evidence object without bypassing method limitations or model-readiness limitations.

This is not the full final Paper 1 run. It is also not an adapter-only, metric-only, local-anchor-only, or Markdown-only session.

The milestone has two linked stages:

1. **Admission readiness:** remove the minimum blockers that make an admission panel invalid.
2. **Admission execution:** run a small publication-root admission panel only for admitted axes and admitted models.

### Stage A — Admission readiness

Codex must first complete the remaining behavioral distribution-method blocker:

* implement general probabilistic map log-likelihood;
* implement information gain against leakage-safe matched priors;
* add image-cluster intervals for these distributional metrics;
* update the behavioral method gate from `method_gap_found` to `accepted_with_limitations`, or document a tested partial implementation and explain why the gate remains blocked.

Codex must then certify or role-block enough models to form a role-structured admission panel.

The admission panel must attempt to include at minimum:

1. CNN/local hierarchy anchor:

   * ResNet-50 or ConvNeXt-Tiny;

2. ViT/DeiT anchor:

   * ViT-B/16 or static DeiT-S;

3. SSL dense-feature model:

   * DINOv2 or DINOv3;

4. VLM/semantic model:

   * CLIP or SigLIP;

5. hierarchical/hybrid model:

   * Swin, SwinV2, Hiera, or MambaVision;

6. efficient-computation model:

   * DynamicViT or ToMe;

7. gaze/scanpath/adaptive model:

   * DeepGaze III, HAT, ScanDiff, or AdaptiveNN;

8. controls:

   * center prior;
   * seeded random baseline;
   * COCO-Search18 task prior where applicable.

A model may enter the admission panel only if it has:

* source revision, package version, or official source identity;
* checkpoint or weight identity/hash;
* preprocessing contract;
* deterministic input condition;
* feature tensor export if used for neural or geometry;
* behavioral output object if used for behavior;
* efficiency/resource fields if used for efficiency;
* small deterministic certification record.

If a preferred model is blocked, Codex must substitute another model from the same role. If the entire role is blocked, Codex must emit a machine-readable role-blocker record and continue with the largest feasible role-structured panel. Do not silently shrink the model universe.

### Stage B — Admission execution

After Stage A passes for enough models and axes, Codex must generate admission-panel evidence under:

`outputs/paper1_publication_v0/`

All admission files must be labeled `admission_panel`, not `final_paper_result`.

Required outputs:

* `outputs/paper1_publication_v0/behavioral/aggregate_admission_panel.csv`
* `outputs/paper1_publication_v0/behavioral/uncertainty_admission_panel.csv`
* `outputs/paper1_publication_v0/neural_encoding/encoding_scores_admission_panel.csv`
* `outputs/paper1_publication_v0/geometry/geometry_scores_admission_panel.csv`
* `outputs/paper1_publication_v0/efficiency/efficiency_profiles_admission_panel.csv`
* `outputs/paper1_publication_v0/efficiency/resource_allocation_profiles_admission_panel.csv`
* `outputs/paper1_publication_v0/cross_axis/model_axis_scores_admission_panel.csv`
* `outputs/paper1_publication_v0/cross_axis/admission_panel_preflight.md`
* `outputs/paper1_publication_v0/audits/admission_panel_audit.csv`

### Axis admission rules

Behavioral rows may run only after the log-likelihood/information-gain blocker is resolved or explicitly marked unavailable for a specific behavioral object.

Neural rows may run only for models with certified deterministic latent-feature tensors.

Geometry rows may run only for the same admitted latent-feature tensors and must use corrected/debiased CKA plus RSA intervals.

Efficiency rows may run only for models with complete matched static or sequential/adaptive resource fields.

Cross-axis output is preflight only. It may report completeness, missingness, role coverage, family coverage, and axis availability. It must not make paper-facing convergence, dissociation, or causal claims.

### Minimum admission scope

The admission panel should include real rows for at least:

* one free-viewing behavioral dataset: SALICON or CAT2000;
* one task-search behavioral condition if a task prior or task/gaze model is admitted;
* early-retinotopic neural encoding where certified features exist;
* corrected geometry on the admitted latent features;
* efficiency/resource rows for every admitted model;
* cross-axis preflight over the admitted model/axis matrix.

If compute or setup limits prevent this minimum, Codex must produce a precise failure report naming which gate, model role, axis, or data path blocked admission execution.

### Non-goals

Do not run the full final matrix.

Do not produce paper interpretation.

Do not run only local anchors.

Do not run only adapter installation.

Do not run only the IG/log-likelihood implementation.

Do not treat `setup_scaffold_ready` as `adapter_certified`.

Do not hide method limitations or role coverage gaps.

### Acceptance rule

The milestone succeeds only if:

1. the behavioral distribution metric blocker is resolved or explicitly remains blocking after tested partial implementation;
2. at least one nonlocal model is genuinely adapter-certified, not merely setup-scaffolded;
3. at least one gaze/scanpath/adaptive candidate is either end-to-end certified or receives a concrete machine-readable role-blocker record after attempted installation/execution;
4. an admission panel produces real publication-root rows for at least behavioral, neural, geometry, and efficiency axes;
5. cross-axis preflight reports role, family, regime, and axis completeness without making paper claims.

The milestone fails if Codex only certifies a bounded image-only batch, only edits Markdown, only implements metrics, only runs local anchors, only runs one axis, or treats setup scaffolds as working models.



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

Status: audit complete; neural encoding is accepted with limitations and the
other five gates have explicit method gaps.

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

Status: model-role and capability audit complete; adapter implementation is
active. Three models are certified and all remaining candidates have explicit
setup or feature-hook requirements.

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

Status: active next milestone.

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

### Phase 3 — Clean behavioral rerun

Status: after behavioral method gate and adapter certification.

Purpose:

Regenerate behavioral evidence under the repaired publication contract.

Required outputs:

* `configs/paper1_clean_behavioral_rerun.yaml`
* `outputs/paper1_publication_v0/behavioral/per_image_metrics/`
* `outputs/paper1_publication_v0/behavioral/aggregate.csv`
* `outputs/paper1_publication_v0/audits/behavioral_rerun_audit.csv`

Acceptance rule:

SALICON, CAT2000, COCO-Search18, fixation references, task priors, gaze/scanpath models, adaptive/foveated outputs, routing maps, and controls are all run or explicitly audited as behavior-only/diagnostic/rejected.

### Phase 4 — Clean latent-feature neural and geometry rerun

Status: after neural/geometry method gates and adapter certification.

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

Status: after model adapters and efficiency method gate are certified.

Purpose:

Make resource allocation central rather than decorative.

Required outputs:

* `outputs/paper1_publication_v0/efficiency/efficiency_profiles.csv`
* `outputs/paper1_publication_v0/efficiency/resource_allocation_profiles.csv`
* `outputs/paper1_publication_v0/efficiency/alignment_per_compute.csv`

Acceptance rule:

Efficiency metrics must be connected to model role and behavioral/neural/geometry axes. Sequential and active-vision models must report total task/image cost, not only per-glimpse cost.

### Phase 7 — Cross-axis publication analysis

Status: after Phases 3–6.

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

Only publication-root evidence can enter cross-axis analysis. Quadrant labels are descriptive unless the cross-axis inference gate allows stronger language.

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

Matrix V2 external integration:

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
