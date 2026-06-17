# HMA Project Status And Next Steps

Updated: 2026-06-16

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

The current active task is **Blocked-Model Preparation And Rerun Authorization Repair**. It is not interpretation, manuscript writing, first-clean-rerun execution, unrestricted cluster execution, adapter-only installation, metric-only implementation, or model-universe reduction. The previous preflight authorized a narrowed rerun subset too early. The immediate task is to resolve every Codex-resolvable blocked model, implement remaining method/control leftovers, run actual DeepGaze tensor-export validation, and regenerate rerun-readiness artifacts before any first clean rerun is allowed.


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
- The previous first-clean-rerun authorization is suspended because it excludes multiple suitable, still-preparable model families and allows a narrowed evidence subset to define the next project step.
- The next accepted scientific artifact must be a blocked-model preparation and rerun-authorization repair pass, not a clean rerun.
- Admission-panel and preflight artifacts remain provenance until they are regenerated after blocked-model repair.
- Static DeiT-S, DynamicViT, and ToMe remain useful efficient-computation candidates, but they must not define the final model scope or paper interpretation.
- All final behavioral, latent-feature neural, latent-feature geometry, efficiency, and cross-axis evidence must be regenerated under the publication output root only after model, method, control, ROI/stream, data, environment, and expected-output readiness are revalidated.

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

Status: `accepted_with_limitations`.

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

Current limitation: free-viewing map execution is admitted, while conditional
next-fixation and generated-scanpath execution remains model-adapter gated.

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

Updated: 2026-06-16

Current implementation state is classified by **publication readiness**, not by smoke tests, local convenience runs, or legacy Matrix V2 progress.

### Publication-readiness classification

Current implementation readiness is:

| component | current classification | publication-facing meaning |
| --- | --- | --- |
| publication contract | frozen_v0 | `configs/paper1_publication_contract.yaml` remains the governing contract |
| publication output root | preflight_artifacts_only_no_final_evidence | `outputs/paper1_publication_v0/` contains admission/preflight artifacts; no clean rerun evidence is final |
| evidence reset | complete | legacy roots remain scaffold/provenance unless regenerated or equivalence-certified |
| first clean rerun authorization | blocked_after_repair_preflight | regenerated preflight keeps authorization blocked because `blocked_model_repair` remains unresolved; no clean rerun was run |
| behavioral evaluation method | accepted_with_limitations_method_leftovers_ready | map metrics, uncertainty, conditional-scanpath interfaces, and leakage-safe empirical spatial prior are implemented; non-DeepGaze scanpath/foveated breadth remains model-setup-limited |
| latent-feature neural encoding method | accepted_with_limitations_export_partially_validated | controlled frozen-feature PCA/ridge is allowed for certified tensors; DeepGaze IIE/III tensor exports are deterministic, while DeepGaze MSDB CPU export timed out after 600 seconds |
| representational geometry method | accepted_for_publication_rerun_after_tensor_validation | debiased CKA/RSA machinery is ready for certified tensors; geometry execution must wait until model tensor exports are actually validated |
| efficiency/resource-allocation method | accepted_with_limitations_evidence_backed_blockers | static/token-pruning/token-merging schemas and AdaptiveNN/HAT resource hooks are certified; ScanDiff/SemBA-family rows remain blocked by setup/source requirements |
| cross-axis inference method | accepted_with_limitations_no_interpretation | readiness/missingness reporting is allowed; convergence/dissociation interpretation remains blocked |
| model adapter comparability | partial_repair_completed_blocked_rows_evidence_backed | adapters/setup metadata were extended; remaining blocked rows now have concrete commands, logs, missing requirements, and next actions |
| easy public image encoders | attempted_public_snapshot_timeout | SigLIP and MambaVision adapters are prepared; public Hugging Face snapshot downloads timed out before complete weight snapshots were available |
| DINOv3 | adapter_certified | `dinov3_small_patch16` is certified against the manually supplied gated Hugging Face snapshot with source, environment, checkpoint, adapter, smoke, and evidence stages ready |
| empirical spatial prior | implemented_ready | leakage-safe train-split empirical spatial prior is registered, tested, and included in method readiness |
| DeepGaze IIE/III/MSDB | iie_iii_validated_msdb_timeout | IIE latent and III conditional/scanpath exports are deterministic; MSDB export was attempted and failed only after a recorded 600-second timeout |
| scanpath/foveated/adaptive candidates | partial_ready | DeepGaze III, HAT, and AdaptiveNN are certified scanpath/foveated candidates; ScanDiff and SemBA/SemBA-FAST remain setup/source blocked |
| efficient-computation models | keep_in_scope_not_main_bottleneck | DynamicViT and ToMe stay in scope, but they are not the central next-session target |
| neural ROI/stream scope | partial_ready | early retinotopic and subj01 stream manifests exist; stream/category coverage limits must remain explicit |
| cluster execution | blocked_after_repair_preflight | Codex must not launch or instruct cluster reruns while `first_clean_rerun_authorization` remains blocked |
| legacy outputs | excluded | no existing empirical root is accepted or equivalence-certified as final Paper 1 evidence |

### Current blockers to real paper progress

The regenerated preflight object is sufficient to block rerun execution truthfully. Remaining blockers are:

1. `blocked_model_repair` remains blocked for `mambavision_t`, `scandiff_freeview`, `scandiff_visual_search`, `semba`, `semba_fast`, and `siglip_base_patch16`;
2. DINOv3 `dinov3_small_patch16` is now source/checkpoint/environment/adapter/smoke certified from the manually supplied gated Hugging Face snapshot and has been removed from `user_action_checklist.csv`;
3. SigLIP and MambaVision are public suitable models with prepared adapters, but their Hugging Face snapshot downloads timed out before complete local weight snapshots were available;
4. HAT and AdaptiveNN are now source/checkpoint/environment/smoke certified and have been removed from `user_action_checklist.csv`; they are not remaining rerun blockers;
5. ScanDiff source/checkpoint/license metadata are concrete, but its pinned WSL conda/micromamba environment build timed out and checkpoint completion remains unresolved;
6. SemBA and SemBA-FAST remain user-required because no official executable source/checkpoint API is registered; setup attempts fail immediately on `UNRESOLVED_OFFICIAL_SOURCE` / `PIN_REQUIRED`;
7. DeepGaze IIE and DeepGaze III tensor exports are deterministic; DeepGaze MSDB export was attempted and failed after the recorded 600-second timeout, so MSDB-specific tensor validation remains incomplete;
8. cluster execution and the first clean rerun remain blocked until `user_action_checklist.csv` items are resolved and `rerun_readiness_table.csv` reports `first_clean_rerun_authorization=ready`;
9. no clean behavioral, latent neural encoding, geometry, efficiency, or cross-axis artifact is accepted as Paper 1 publication evidence.

<!-- Historical pre-repair blocker list retained only as provenance; do not use for current status.

The previous preflight object is no longer sufficient for rerun execution. Remaining blockers are:

1. the current first-clean-rerun plan excludes multiple suitable models that are likely Codex-preparable, including SigLIP, MambaVision, ScanDiff, SemBA/SemBA-FAST, AdaptiveNN, and the empirical spatial prior;
2. empirical spatial prior remains unimplemented even though it is a local control with no external model dependency;
3. DeepGaze latent/conditional rows are still insufficient if only manifests exist; actual tensor export, output files, shape/order/determinism checks, and hashable validation records are required;
4. SigLIP and MambaVision should not remain blocked without real Hugging Face/source download attempts, adapter execution, and certification logs;
5. DINOv3 should not remain blocked until Codex attempts official source/weight access and distinguishes a genuine gated-access failure from an unattempted setup;
6. ScanDiff, HAT, SemBA/SemBA-FAST, and AdaptiveNN should not remain blocked through vague license/checkpoint/environment language; each needs an actual clone/install/download/run attempt or an explicit user-required step;
7. “license clearance” is not a valid stopping reason unless the repo or model card explicitly restricts the planned research use; Codex must record the license and continue when use is permitted;
8. “network unavailable” is not a valid stopping reason unless a concrete command fails and the command, error, and log path are recorded;
9. cluster execution remains premature because the model universe has not been repaired and no clean full-matrix runner is registered;
10. no clean behavioral, latent neural encoding, geometry, efficiency, or cross-axis artifact is accepted as Paper 1 publication evidence.

-->

### Current implementation priority

Active priority:

> Resolve the concrete user-action/setup rows in `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`, regenerate the preflight artifacts, and keep the first clean rerun blocked until `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv` reports `first_clean_rerun_authorization=ready`.

This is not a rerun session, DynamicViT/ToMe session, cluster session, Markdown session, or smoke-test session. The canonical artifact list lives in `Next Concrete Milestone`. Do not duplicate it here.

### Required end-of-session report

Use the publication-readiness report format in `Current Implementation Progress`. Do not use the older scientific-claim report format until publication-root evidence exists.

Do not report smoke tests, debugging fixes, old-result summaries, or legacy audits as progress unless they directly change one of the publication-contract artifacts above.

Implementation history is archived in `docs/project_status_changelog.md`.

### End-of-session report - Blocked-Model Repair And Rerun Authorization

1. **Publication-contract change:** no clean rerun was launched. `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv` now blocks `first_clean_rerun_authorization` through the `blocked_model_repair` gate.
2. **Accepted artifact:** regenerated readiness artifacts include `model_setup_attempts.csv`, `model_certification_summary.csv`, `user_action_checklist.csv`, `method_rerun_readiness_table.csv`, `rerun_readiness_table.csv`, `first_clean_rerun_plan.md`, DeepGaze IIE/III deterministic validation JSONs, and the DeepGaze MSDB timeout validation JSON.
3. **Method gate status change:** empirical spatial prior is implemented and method leftovers are ready; behavioral, latent, geometry, efficiency, and cross-axis methods remain pre-rerun methods only, not final evidence.
4. **Model setup status change:** AdaptiveNN is now adapter-certified with source, checkpoint, environment, and smoke evidence. DINOv3, SigLIP, MambaVision, HAT, ScanDiff, SemBA, and SemBA-FAST remain evidence-backed setup or user-action blockers, not vague placeholders.
5. **Reviewer risk reduced:** model-universe missingness is now explicit and machine-readable; first clean rerun authorization cannot be confused with adapter smoke success or partial local setup.

### End-of-session report - Paper 1 Publication Matrix V0 Preflight
Supersession note: this report is retained as provenance only. Its rerun authorization is suspended because the generated plan excludes multiple suitable model families that remain Codex-resolvable, and because actual DeepGaze tensor export has not yet been validated. The next session must repair blocked-model readiness before any rerun.

1. **Publication-contract change:** the first clean rerun gate now uses machine-readable method/model/data/checkpoint/environment/ROI/expected-output/readiness artifacts under `outputs/paper1_publication_v0/preflight/` and `outputs/paper1_publication_v0/roi_stream/`. DeepGaze III is the certified scanpath-capable model for the initial clean rerun.
2. **Accepted artifact:** accepted preflight artifacts include `outputs/paper1_publication_v0/preflight/model_certification_summary.csv`, `outputs/paper1_publication_v0/preflight/model_setup_attempts.csv`, `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`, `outputs/paper1_publication_v0/preflight/method_rerun_readiness_table.csv`, `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv`, `outputs/paper1_publication_v0/roi_stream/stream_roi_grouping_spec.csv`, `outputs/paper1_publication_v0/roi_stream/subject_roi_availability.csv`, `outputs/paper1_publication_v0/external/scanpath_or_foveated_certified/manifest.json`, and `configs/paper1_latent_neural_stream_admission.yaml`.
3. **Method gate status change:** model adapter comparability and efficiency/resource allocation remain `accepted_with_limitations`; the prior `first_clean_rerun_authorization=authorized` state is superseded and must be regenerated after blocked-model repair.
4. **Paper evidence status change:** ConvNeXt, SwinV2, Hiera, DeepGaze III, DeepGaze MSDB, DynamicViT, and ToMe remain preflight-certified or setup-preflighted, but their status does not authorize rerun execution. DINOv3, SigLIP, MambaVision, HAT, ScanDiff, AdaptiveNN, SemBA/SemBA-FAST, and empirical spatial prior require a blocked-model repair pass. No clean rerun output is `accepted_publication_evidence`.
5. **Reviewer risk reduced:** reduced missing latent-feature risk for DeepGaze, missing scanpath-gate risk via DeepGaze III, adapter-incomparability risk through full-universe certification rows, stream/ROI ambiguity through explicit availability manifests, and stale-runbook risk through generated rerun-readiness and user-action tables.

### End-of-session report - Gate-Conditioned Publication Admission Panel V1

Historical status retained for provenance only.

1. **Publication-contract change:** no scientific claim changed. The admission configuration encoded a role-structured bounded scope, bounded SALICON/subj01 V1 execution, and explicit evidence labels.
2. **Accepted artifact:** admission-panel outputs exist under `outputs/paper1_publication_v0/`, including behavioral uncertainty, neural encoding, corrected geometry, efficiency/resource allocation, cross-axis availability, preflight, and audit artifacts.
3. **Method gate status change:** behavioral evaluation moved from `method_gap_found` to `accepted_with_limitations` after tested probabilistic log-likelihood, matched-prior information gain, and image-cluster intervals.
4. **Paper evidence status change:** ResNet-50, ViT-B/16, DINOv2, CLIP, and Swin have bounded latent neural/geometry/efficiency admission rows; DeepGaze IIE and controls have bounded free-viewing behavioral rows; all remain `admission_panel_not_final_paper_result`.
5. **Reviewer risk reduced:** reduced legacy-output contamination, distribution-metric mismatch, false latent eligibility for DeepGaze IIE, ambiguous resource units, silent role exclusion, and analytic-control overcounting in cross-axis coverage.
6. **Superseded next step:** the previous instruction to run `docs/paper1_admission_cluster_runbook.md` is no longer valid. That runbook is deleted/superseded because the setup is not yet prepared for cluster execution. The next decisive step is the Model, Method, and Stream Preflight Gate in `Next Concrete Milestone`.

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

> Complete Blocked-Model Preparation And Rerun Authorization Repair. The previous bounded first-clean-rerun authorization is suspended until suitable blocked models are either certified or converted into concrete user-action gates after real setup attempts.

## Next Concrete Milestone

Priority: **Blocked-Model Preparation And Rerun Authorization Repair**.

The previous first-clean-rerun plan is suspended. It authorized a narrowed subset before suitable blocked models were fully prepared. The next session must repair model readiness and method/control leftovers before any clean rerun.

This milestone must reduce avoidable blocked rows. It must not run the clean rerun, produce paper interpretation, treat blocked models as later work, or move forward with DeepGaze III as the only scanpath-capable model unless all other suitable scanpath/foveated/adaptive candidates have received real setup attempts.

### Stage A — Method and control leftovers

Codex must complete all method/control leftovers that are implementable inside the repository.

Required work:

1. implement the leakage-safe empirical spatial prior using training/reference data only;
2. add provenance fields showing which split and images define the empirical prior;
3. add empirical-prior sensitivity as a planned behavioral-control condition;
4. verify behavioral distribution metrics, point metrics, uncertainty, conditional-map schema, and scanpath schema against actual code;
5. verify neural encoding leakage control, PCA/ridge selection, raw/noise-normalized separation, and nonfinite ceiling handling;
6. verify whether voxel-specific or spatial-readout sensitivity is implemented; if absent, add the config and code-path target rather than treating it as prose;
7. verify geometry interval behavior for debiased CKA/RSA before scaling;
8. verify matched efficiency and total-cost schemas for static, token-pruning, token-merging, foveated, diffusion, recurrent, and scanpath models.

Required artifacts:

- `outputs/paper1_publication_v0/preflight/method_rerun_readiness_table.csv`
- `outputs/paper1_publication_v0/preflight/empirical_spatial_prior_audit.csv`
- `outputs/paper1_publication_v0/preflight/empirical_prior_sensitivity_plan.md`

### Stage B — DeepGaze actual export validation

Codex must convert DeepGaze latent/conditional certification from manifest-level preflight into actual exported evidence-readiness artifacts.

Required work:

1. inspect the actual PyTorch module graph used by DeepGaze IIE, DeepGaze III, and DeepGaze MSDB;
2. export internal tensors before final density normalization;
3. export conditional next-fixation maps for DeepGaze III;
4. export generated/iterated scanpaths where the implementation supports them;
5. validate tensor shape, image ordering, deterministic repeatability, file counts, and hashes;
6. keep final gaze-density maps as behavioral outputs;
7. keep output-map-to-fMRI rows labeled only as `output_map_neural_control`.

Required artifacts:

- `outputs/paper1_publication_v0/external/deepgaze_iie_latent/manifest.json`
- `outputs/paper1_publication_v0/external/deepgaze_iie_latent/certification.csv`
- `outputs/paper1_publication_v0/external/deepgaze_iii_conditional/manifest.json`
- `outputs/paper1_publication_v0/external/deepgaze_iii_conditional/certification.csv`
- `outputs/paper1_publication_v0/external/deepgaze_msdb_latent/manifest.json`
- `outputs/paper1_publication_v0/preflight/deepgaze_tensor_export_validation.csv`
- `outputs/paper1_publication_v0/preflight/deepgaze_hook_audit.md`

Acceptance rule:

DeepGaze certification is valid only if actual tensor files or arrays exist and pass deterministic validation. A manifest without exported tensors does not count.

### Stage C — Public image-encoder repair

Codex must resolve public image encoders that are suitable for the project and should not remain blocked through setup prose.

Required work for SigLIP:

1. install or use the supported Hugging Face/Transformers path;
2. download/cache the chosen SigLIP checkpoint;
3. implement image-tower latent export;
4. certify preprocessing, deterministic condition, tensor shapes, and candidate layers;
5. add neural/geometry/efficiency eligibility rows.

Required work for MambaVision:

1. install or clone the official source/package path;
2. download/cache the chosen MambaVision checkpoint;
3. implement stage/block latent export;
4. certify preprocessing, deterministic condition, tensor shapes, and candidate layers;
5. add neural/geometry/efficiency eligibility rows.

Required work for DINOv3:

1. implement the adapter and preprocessing contract;
2. attempt official source/checkpoint download;
3. if access is gated, record exact failed command and exact user action required;
4. prepare the adapter around the expected local checkpoint path so user-provided weights can be certified immediately.

Required artifacts:

- `outputs/paper1_publication_v0/preflight/siglip_certification.csv`
- `outputs/paper1_publication_v0/preflight/mambavision_certification.csv`
- `outputs/paper1_publication_v0/preflight/dinov3_setup_attempt.csv`

Acceptance rule:

SigLIP and MambaVision may remain blocked only after concrete failed install/download/run commands. DINOv3 may require user action only after a genuine official access/gated-weight failure is recorded.

### Stage D — Scanpath/foveated/adaptive model repair

Codex must attempt real setup for every suitable scanpath/foveated/adaptive model, rather than using license/checkpoint/environment as generic stopping language.

Required candidates:

- ScanDiff;
- HAT;
- SemBA/SemBA-FAST;
- AdaptiveNN.

For each candidate, Codex must:

1. locate and record official source/package/model-card identity;
2. record license text or license file path;
3. continue implementation unless the license explicitly forbids the planned research use;
4. clone/install dependencies where possible;
5. attempt checkpoint/model download where public;
6. implement adapter stubs around expected outputs;
7. attempt tiny inference or demo execution;
8. define behavioral output, scanpath/glimpse/fixation-history output, latent/internal-state output where available, and total-cost/resource traces;
9. produce concrete user instructions only for actions Codex cannot perform, such as gated access, manual download, credential/token setup, or user-run cluster commands.

Required artifacts:

- `outputs/paper1_publication_v0/preflight/scandiff_setup_attempt.csv`
- `outputs/paper1_publication_v0/preflight/hat_setup_attempt.csv`
- `outputs/paper1_publication_v0/preflight/semba_setup_attempt.csv`
- `outputs/paper1_publication_v0/preflight/adaptivenn_setup_attempt.csv`
- `outputs/paper1_publication_v0/preflight/scanpath_foveated_certification_summary.csv`
- `outputs/paper1_publication_v0/preflight/scanpath_foveated_user_actions.csv`

Acceptance rule:

A blocked row is valid only if it records the exact command attempted, exact error, log path, missing dependency/checkpoint/access condition, whether the next step is Codex-implementable or user-required, and the next command after that action is completed.

### Stage E — Rerun-readiness regeneration

After Stages A-D, Codex must regenerate all readiness artifacts and explicitly decide whether the first clean rerun is still blocked or newly authorized.

Required artifacts:

- `outputs/paper1_publication_v0/preflight/model_certification_summary.csv`
- `outputs/paper1_publication_v0/preflight/model_setup_attempts.csv`
- `outputs/paper1_publication_v0/preflight/model_rerun_eligibility_table.csv`
- `outputs/paper1_publication_v0/preflight/user_action_checklist.csv`
- `outputs/paper1_publication_v0/preflight/method_rerun_readiness_table.csv`
- `outputs/paper1_publication_v0/preflight/rerun_readiness_table.csv`
- `outputs/paper1_publication_v0/preflight/first_clean_rerun_plan.md`

The regenerated `first_clean_rerun_plan.md` must state whether it is authorized or blocked. If authorized, it must include only models that have actual certification records after this repair pass. If blocked, it must list the exact remaining user actions and Codex-implementable next commands.

### Non-goals

Do not run the clean rerun.

Do not launch broad cluster jobs.

Do not use the previous first-clean-rerun authorization.

Do not treat `license clearance`, `checkpoint unresolved`, `environment missing`, or `network unavailable` as a valid blocker without a concrete command, source, error, and next action.

Do not leave SigLIP, MambaVision, empirical spatial prior, or SemBA as vague blocked rows.

Do not report smoke tests as success unless they produce certification artifacts.

Do not interpret cross-axis convergence or dissociation.

### Acceptance rule

The milestone succeeds only if:

1. empirical spatial prior is implemented or fails for a concrete code-level reason;
2. DeepGaze actual tensor export and validation are completed or concretely failed;
3. SigLIP and MambaVision are certified or have concrete failed commands;
4. DINOv3 official access is attempted and any user-required action is exact;
5. ScanDiff, HAT, SemBA/SemBA-FAST, and AdaptiveNN each receive real setup attempts;
6. all blocked rows distinguish Codex-implementable next work from user-required external action;
7. rerun-readiness artifacts are regenerated after the repair pass;
8. the first clean rerun remains blocked unless the repaired model universe and method/control leftovers are actually ready.

The milestone fails if Codex runs the clean rerun, only edits Markdown, only updates tables without setup attempts, only certifies easy local anchors, only works on DynamicViT/ToMe, or leaves avoidable model blockers unresolved.

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
Status: incomplete repair required. The model-role matrix includes every required candidate, but the current certification tables still contain avoidable blocked rows. SigLIP, MambaVision, empirical spatial prior, ScanDiff, SemBA/SemBA-FAST, AdaptiveNN, HAT, DINOv3, and actual DeepGaze tensor export must be repaired or concretely failed before the first clean rerun.

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
Status: bounded local admission object generated on 2026-06-15 and retained as provenance only. The previous bounded clean-rerun authorization is suspended. Phase 2.5 now feeds into blocked-model repair and rerun-readiness regeneration, not immediate execution.

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

Status: blocked until blocked-model preparation and rerun-readiness regeneration are complete.

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
