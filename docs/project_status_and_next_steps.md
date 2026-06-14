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

The next task is publication method-contract freezing, scope freezing, and full model adapter certification. It is not interpretation, adaptive sweep, manuscript writing, full publication rerun, broad model execution, or legacy result consolidation.

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

- full required model-role coverage before convenience-based model reduction;
- adapter certification before evidence generation;
- literature-grounded method acceptance before full publication reruns;
- SOTA-method comparison before treating any pipeline as paper-valid;
- method-gate decisions over merely executable pipelines;
- clean publication-root reruns over legacy-result consolidation;
- latent-feature neural encoding and latent-feature geometry over output-map neural controls;
- stream/ROI structure over flat ROI averages;
- adaptive/foveated/scanpath/selective-computation mechanisms over additional post-hoc heatmap variants;
- efficiency and resource-allocation metadata as first-class evidence;
- explicit paper-evidence status for every model, artifact, and result table.


Codex should avoid:

- treating scaffold outputs as paper evidence;
- generating interpretation from the three-model adaptive pilot;
- beginning adaptive sweep or broad model execution before the full model-role matrix and adapter-certification plan exist;
- shrinking the model universe before availability, checkpoint, feature-hook, and behavioral-output audits are complete.

### Required end-of-session report

Until publication-root evidence exists, Codex must report publication-readiness changes, not claim changes.

At the end of each Codex session, update this file with:

1. `Publication-contract change`: what changed in the frozen scope, method contract, model-role matrix, adapter certification, or clean rerun plan.
2. `Accepted artifact`: exact path(s) under `outputs/paper1_scope_reset/`, `configs/`, or `outputs/paper1_publication_v0/`.
3. `Method gate status change`: which method gate moved among `not_started`, `audit_required`, `method_gap_found`, `accepted_with_limitations`, or `accepted_for_publication_rerun`.
4. `Paper evidence status change`: which model/artifact moved among `not_started`, `adapter_in_progress`, `adapter_certified`, `publication_rerun_ready`, `publication_rerun_complete`, `accepted_publication_evidence`, `diagnostic_only`, or `rejected_for_paper_evidence`.
5. `Reviewer risk reduced`: which concrete risk was reduced, such as SOTA-method mismatch, legacy-output contamination, missing latent features, behavioral-regime mixing, adapter incomparability, stream/ROI ambiguity, or weak uncertainty.
6. `Next decisive step`: the next task most likely to make publication-root evidence scientifically valid.

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
- The next accepted scientific artifact must be a frozen Paper 1 publication contract plus a full model-role and adapter-certification matrix.
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

Status: `audit_required`.

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

Status: `audit_required`.

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

Status: `audit_required`.

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

Status: `audit_required`.

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

Status: `audit_required`.

Before efficiency results enter cross-axis analysis, Codex must produce:

* `outputs/paper1_scope_reset/method_efficiency_sota_audit.md`
* `outputs/paper1_scope_reset/efficiency_acceptance_table.csv`

The audit must decide:

* which efficiency metrics are comparable across static, token-pruning, token-merging, foveated, scanpath, and active-vision models;
* how FLOPs/MACs, latency, memory, token count, retained-token fraction, selected-glimpse count, fixation count, scanpath length, stopping behavior, and foveated high-resolution area are reported;
* whether efficiency is measured under matched image resolution, batch size, hardware, and preprocessing;
* whether sequential models report total cost per image/task rather than per-glimpse cost only.

### Gate 6 — Cross-axis inference acceptance

Status: `audit_required`.

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
| behavioral benchmark code | scaffold_ready | usable for clean rerun after publication contract |
| repaired behavior map routing | scaffold_ready | must be reused in publication rerun |
| constant-map metric handling | scaffold_ready | must be reused in publication rerun |
| external artifact schema | scaffold_ready | must be extended/certified for all required models |
| static DeiT / DynamicViT / ToMe adapters | scaffold_ready_for_generic_efficiency_role | validate the adapter pattern; do not define final paper scope |
| DINOv3 / SigLIP / MambaVision / Hiera / Swin registry entries | adapter_audit_required | required candidates; must receive certification status |
| DeepGaze / HAT / ScanDiff / SemBA / SemBA-FAST | adapter_audit_required | required gaze/scanpath candidates; must be inspected for behavioral and latent-feature evidence |
| AdaptiveNN | adapter_audit_required | required human-gaze-inspired adaptive/foveated candidate |
| old behavioral aggregates | scaffold_only | excluded from publication evidence unless regenerated or equivalence-certified |
| old neural/geometry outputs | scaffold_only | excluded from publication evidence unless regenerated or equivalence-certified |
| old Matrix V2 three-model full run | method_validation_only | validates repaired pipeline; not final Paper 1 evidence |
| paper inspection packs | scaffold_only | not publication evidence |
| `docs/project_results_numbers.md` | historical_numeric_context | useful for expected ranges only; not publication evidence |
| `docs/actual_methodology_trace.md` | provenance_context | useful for avoiding repeated mistakes; not publication evidence |
| behavioral evaluation method | method_gate_audit_required | standard metrics exist, but scanpath/task-search uncertainty and metric hierarchy must be accepted before publication rerun |
| latent-feature neural encoding method | method_gate_audit_required | current PCA/ridge method is controlled-baseline grade; SOTA gap and readout-sensitivity policy must be accepted before rerun |
| representational geometry method | method_gate_audit_required | CKA/RSA exist; debiased/bias-corrected geometry and uncertainty policy must be audited before rerun |
| model adapter comparability | method_gate_audit_required | required candidates must be certified for deterministic input, latent features, behavioral outputs, and resource metadata |
| efficiency/resource-allocation method | method_gate_audit_required | static/token/adaptive/foveated/sequential cost metrics must be made comparable before cross-axis use |
| cross-axis inference method | method_gate_audit_required | no quadrant or convergence claim until model count, uncertainty, and sensitivity rules are accepted |

### Current blockers to real paper progress
The project cannot move to publication-facing full runs until these blockers are resolved:

1. no frozen Paper 1 publication contract exists;
2. no `outputs/paper1_publication_v0/` evidence root exists;
3. no full required model-role matrix exists;
4. no `paper_evidence_status` exists for each model and artifact;
5. no method-gate status exists for behavioral evaluation, neural encoding, geometry, adapter comparability, efficiency, or cross-axis inference;
6. no literature-grounded method audit compares the project pipeline against relevant SOTA methods;
7. no adapter-certification matrix exists for all required candidate models;
8. no evidence reset manifest excludes legacy outputs from paper evidence;
9. AdaptiveNN has not yet been audited as a required adaptive/foveated candidate;
10. gaze/scanpath models have not yet been audited for latent-feature extraction;
11. no clean behavioral rerun can be launched until the behavioral method gate passes;
12. no clean latent-feature neural/geometry rerun can be launched until the neural and geometry method gates pass.

### Current implementation priority

Active priority:

> Freeze Paper 1 Publication Matrix V0, certify adapters for the full required model universe, and prepare clean publication-root reruns.

The canonical artifact list lives in `Next Concrete Milestone`. Do not duplicate it here.

### Required end-of-session report

Use the publication-readiness report format in `Current Implementation Progress`. Do not use the older scientific-claim report format until publication-root evidence exists.

Do not report smoke tests, debugging fixes, old-result summaries, or legacy audits as progress unless they directly change one of the publication-contract artifacts above.

Implementation history is archived in `docs/project_status_changelog.md`.

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

## Next Concrete Milestone

Priority: **Paper 1 Publication Method Contract, Scope Reset, And Full Model Adapter Certification**.

The next task is not paper interpretation, not adaptive strength sweep, not full publication rerun, and not broad model execution. The next task is to freeze both:

1. the publication-facing scientific scope; and
2. the academic method contract that makes the future full runs paper-valid.

### Required outcome

Create:

* `outputs/paper1_scope_reset/model_role_matrix.csv`
* `outputs/paper1_scope_reset/model_availability_audit.md`
* `outputs/paper1_scope_reset/model_adapter_comparability_table.csv`
* `outputs/paper1_scope_reset/model_adapter_comparability_audit.md`
* `outputs/paper1_scope_reset/latent_feature_adapter_requirements.csv`
* `outputs/paper1_scope_reset/behavioral_output_adapter_requirements.csv`
* `outputs/paper1_scope_reset/evidence_reset_manifest.csv`
* `outputs/paper1_scope_reset/publication_output_root_plan.md`
* `outputs/paper1_scope_reset/method_sota_alignment_audit.md`
* `outputs/paper1_scope_reset/method_acceptance_gates.md`
* `outputs/paper1_scope_reset/behavioral_metric_acceptance_table.csv`
* `outputs/paper1_scope_reset/neural_encoding_acceptance_table.csv`
* `outputs/paper1_scope_reset/geometry_acceptance_table.csv`
* `outputs/paper1_scope_reset/efficiency_acceptance_table.csv`
* `outputs/paper1_scope_reset/cross_axis_inference_acceptance_table.csv`
* `outputs/paper1_scope_reset/clean_behavioral_rerun_plan.md`
* `outputs/paper1_scope_reset/clean_behavioral_rerun_matrix.csv`
* `outputs/paper1_scope_reset/latent_feature_neural_matrix_plan.md`
* `outputs/paper1_scope_reset/stream_roi_grouping_spec.md`
* `outputs/paper1_scope_reset/efficiency_resource_allocation_plan.md`
* `configs/paper1_publication_contract.yaml`
* `configs/paper1_clean_behavioral_rerun.yaml`
* `configs/paper1_latent_neural_matrix.yaml`

Publication output root:

* `outputs/paper1_publication_v0/`

### Required method SOTA alignment audit

Create `outputs/paper1_scope_reset/method_sota_alignment_audit.md`.

The audit must compare the project method against relevant academic methods for:

1. saliency/fixation-map evaluation;
2. task-search evaluation;
3. scanpath/sequential gaze prediction;
4. latent-feature neural encoding;
5. Algonauts/NSD-style fMRI prediction;
6. representational geometry with RSA/CKA/debiased CKA;
7. model-brain comparison frameworks;
8. efficiency/resource-allocation evaluation;
9. active/foveated/adaptive vision models.

For each literature reference or benchmark family, record:

* paper or benchmark name;
* task/data used;
* model output evaluated;
* method used;
* metric used;
* uncertainty/statistical control used;
* reported numbers when available;
* which part of the project method matches;
* which part of the project method differs;
* whether the project difference is acceptable, accepted-with-limitations, or blocking.

The audit must explicitly position the project method as one of:

* `controlled_benchmark_method`;
* `sota_competitive_method`;
* `diagnostic_method_only`;
* `method_gap_blocking_publication_rerun`.

The expected position is `controlled_benchmark_method` unless the audit proves otherwise.

### Required method acceptance gates

Create `outputs/paper1_scope_reset/method_acceptance_gates.md`.

The gates are:

1. behavioral evaluation;
2. latent-feature neural encoding;
3. representational geometry;
4. model adapter comparability;
5. efficiency/resource allocation;
6. cross-axis inference.

Each gate must receive one of:

* `not_started`;
* `audit_required`;
* `method_gap_found`;
* `accepted_with_limitations`;
* `accepted_for_publication_rerun`;
* `rejected_for_publication_claim`.

A publication-root full run may begin only for axes whose method gates are `accepted_for_publication_rerun` or `accepted_with_limitations`.

### Required model-role matrix

Define model roles as:

1. human gaze-prediction models;
2. human-gaze-inspired efficient or foveated models;
3. generic efficient-computation models;
4. general visual representation models;
5. self-supervised dense-feature models;
6. VLM / semantic models;
7. hierarchical / hybrid / efficient architecture models;
8. spatial-prior and task-prior controls.

Required candidate families include:

* ResNet + ConvNeXt anchors;
* ViT + DeiT anchors;
* DINOv2 and DINOv3 family;
* CLIP + SigLIP-family candidates;
* MambaVision;
* Hiera;
* Swin + SwinV2;
* DynamicViT;
* ToMe;
* DeepGaze;
* HAT;
* ScanDiff;
* SemBA + SemBA-FAST where applicable;
* AdaptiveNN;
* center, random, spatial-prior, and task-prior controls.

For each candidate model, record:

* model name;
* role;
* behavioral output available;
* latent features available;
* feature extraction path;
* neural encoding eligibility;
* representational geometry eligibility;
* efficiency/resource-allocation eligibility;
* deterministic input condition required;
* required environment/checkpoint;
* current implementation status: `ready_now`, `needs_adapter`, `needs_checkpoint`, `needs_feature_hook`, `needs_behavior_output_adapter`, `behavior_only_after_audit`, `diagnostic_only_after_audit`, or `reject_after_audit`;
* `paper_evidence_status`: `not_started`, `adapter_in_progress`, `adapter_certified`, `publication_rerun_ready`, `publication_rerun_complete`, `accepted_publication_evidence`, `diagnostic_only`, or `rejected_for_paper_evidence`.

A model can carry the central neural-alignment claim only if latent features can be extracted and passed through the same neural encoding and geometry pipeline as the other central models.

### Required adapter-certification plan

For every required model, certify whether the adapter can produce:

* one deterministic latent feature tensor per NSD image;
* candidate layer/block features;
* behavioral fixation/saliency/scanpath/task-search output where applicable;
* resource-allocation output where applicable;
* efficiency metadata;
* preprocessing metadata;
* checkpoint provenance;
* environment provenance;
* deterministic seed and input-condition record.

For gaze-history-conditioned, task-conditioned, text-conditioned, scanpath, foveated, or active-vision models, the plan must define the fixed input condition before feature extraction.

AdaptiveNN must receive a full adapter-certification row because it is a required human-gaze-inspired adaptive/foveated model. Its certification must inspect selected glimpses/fixations, stopping behavior, integrated latent states or backbone features, and compute/resource metadata.

### Required evidence reset manifest

Create `outputs/paper1_scope_reset/evidence_reset_manifest.csv`.

For every existing output root, classify it as one of:

* `scaffold_only`;
* `method_validation_only`;
* `legacy_behavior_discard`;
* `legacy_neural_discard`;
* `legacy_geometry_discard`;
* `legacy_efficiency_discard`;
* `eligible_for_equivalence_audit`;
* `publication_root_only`.

Legacy outputs must not enter publication-root summaries unless regenerated under the publication contract or certified by equivalence audit.

### Required clean rerun plans

The clean behavioral rerun plan must cover:

* SALICON free-viewing;
* CAT2000 free-viewing;
* COCO-Search18 task search;
* center and random baselines;
* DeepGaze / fixation references;
* COCO-Search18 task prior;
* attribution controls where required;
* repaired routing maps for efficient-computation models;
* gaze-prediction and scanpath outputs for HAT, ScanDiff, SemBA/SemBA-FAST, DeepGaze, and AdaptiveNN where applicable.

The clean latent-feature neural and geometry plans must specify:

* which models expose features;
* which layers or feature blocks will be candidates;
* whether the model is image-only, gaze-history-conditioned, task-conditioned, text-conditioned, foveated, active, or stochastic;
* the fixed input condition used to generate one feature tensor per NSD image;
* whether preprocessing can be matched or must be explicitly modeled;
* whether the same `flatten_pca` + validation-selected ridge protocol applies;
* whether learned spatial readout is required as sensitivity analysis;
* which ROI groups will be run;
* which models become behavior-only or diagnostic-only after audit.

Required ROI grouping:

* early visual / retinotopic: V1, V2, V3, hV4;
* dorsal/lateral/parietal spatial-selection: lateral, midlateral, midparietal, parietal;
* ventral/semantic/object-related: ventral, midventral, hV4, and fLOC category ROIs if accepted later.

The efficiency/resource-allocation plan must include:

* parameters;
* FLOPs/MACs;
* measured latency;
* peak memory;
* token counts;
* retained-token fraction;
* merge statistics;
* selected-glimpse count;
* fixation/scanpath length;
* stopping behavior;
* foveated high-resolution area or crop count where applicable.

### Acceptance criteria

This milestone is complete only when Codex can answer:

1. What is the full required model-role matrix?
2. What is each candidate model’s `paper_evidence_status`?
3. What is each method gate’s status?
4. What SOTA methods were used to judge the project method?
5. Which project methods are accepted for publication rerun, accepted with limitations, or blocking?
6. Which models have certified latent-feature extraction?
7. Which gaze-prediction, scanpath, foveated, or active-vision models can enter latent-feature neural/geometry analysis?
8. Which models are behavior-only or diagnostic-only after audit?
9. Which existing outputs are scaffold/provenance and excluded from publication evidence?
10. What exact publication-root paths will contain behavioral, neural, geometry, efficiency, and cross-axis outputs?
11. What exact clean behavioral rerun jobs are required?
12. What exact latent-feature neural/geometry configs are required?
13. What exact efficiency/resource-allocation profiles are required?
14. Which analyses are primary, which are secondary sensitivity analyses, and which are diagnostic/control only?

Do not start adaptive strength sweep, broad evidence reruns, paper-facing interpretation, or manuscript writing until this milestone is complete.


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

Status: active.

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

Status: next decisive methodological gate.

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

Status: after or alongside Phase 1.

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

## Methodology provenance audit session (2026-06-13)

Generated `docs/actual_methodology_trace.md` by tracing the current behavioral,
neural encoding, geometry, adaptive-routing, efficiency, cross-axis, and older
V1 outputs backward to their scripts, configs, inputs, and scoring functions.
Supporting provenance tables are under `outputs/methodology_trace/`.

The trace records unresolved repository-visible assumptions with explicit
`UNKNOWN_FROM_REPO` and `STATUS_CLAIM_NOT_VERIFIED` labels. It does not change
the scientific interpretation in this status document.
