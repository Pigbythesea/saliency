# Paper 1 Experiment Spec V1

Updated: 2026-06-05

## Paper Claim Being Tested

Human-like fixation alignment, neural encoding, and latent representational geometry are related but non-equivalent axes of visual alignment. Paper 1 tests whether these axes converge or dissociate across a method-matched model panel and an ROI-expanded `subj01` neural matrix, while keeping free-viewing fixation behavior separate from task-driven search behavior.

## Why Current Results Are Insufficient

The current accepted matrix is a diagnostic proof that the pipeline works, not a paper-grade experiment. It covers one subject (`subj01`), six models, PRF visual ROIs only (`V1`, `V2`, `V3`, `hV4`), and small model-level correlations. Its matched `flatten_pca` neural panel and full-image geometry outputs are useful baselines, but they cannot establish a top-venue claim because anatomical scope is narrow, model-level `n` remains small, and current cross-axis correlations are descriptive rather than robust evidence. The next experiment must broaden brain-region coverage before adding generic models or polishing the existing PRF-only result.

## Minimum Paper-Grade Scope

The discovery matrix uses `subj01` as the primary subject and expands the neural/geometry axis beyond PRF visual ROIs. Subject replication with `subj02`-`subj04` is a confirmatory follow-up only after the ROI-expanded `subj01` pattern exists.

Primary model panel:

- `resnet50`
- `vit_base_patch16_224`
- `vit_small_patch14_dinov2`
- `vit_base_patch16_clip_224`

Required ROI set:

- PRF visual ROIs: `V1`, `V2`, `V3`, `hV4`
- Stream ROIs: `midventral`, `midlateral`, `midparietal`, `ventral`, `lateral`, `parietal`

Behavioral evidence:

- Corrected behavioral aggregate with accepted transformer relevance integration: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior_plus_transformer_relevance.csv`
- SALICON observer controls: `outputs/observer_controls_v2/salicon_static2000_worker_json_observer_controls.csv`
- COCO-Search18 observer controls: `outputs/observer_controls_v2/coco_search18_static2000_observer_controls.csv`
- Free-viewing datasets (`SALICON`, `CAT2000`) must remain separate from task-search (`COCO-Search18`).

Encoding method:

- Full-image-count `flatten_pca`
- Validation-selected layer
- Same ridge-alpha grid, PCA component count, validation fraction, and noise-normalized selection rule as the current matched full panel
- No learned-readout rows in headline cross-model rankings

Geometry methods:

- `linear_cka_full9841`
- deterministic `subset_rsa`
- Subset sizes: `512`, `1024`, `2048`
- Subset seeds: `123`, `456`, `789`
- Correlation RDMs and Spearman RDM comparison

Attribution and behavioral method language:

- Gradients, Grad-CAM, rollout, perturbation, and dedicated fixation models must remain separate method families.
- Attention rollout may be reported only as `attention rollout attribution`, not as human or model attention.

Efficiency variables:

- Deferred from V1. Add FLOPs, latency, memory, token count, and alignment-per-compute only after the ROI-expanded matrix exists.

## Accepted Evidence Table List

The experiment is not claim-ready until these artifacts exist:

- `outputs/paper1_experiment_v1/summary/roi_expanded_encoding_model_rankings.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_encoding_roi_rankings.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_model_rankings.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_geometry_method_agreement.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_observations.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_cross_level_correlations.csv`
- `outputs/paper1_experiment_v1/summary/roi_expanded_cross_axis_decisions.csv`
- `outputs/paper1_experiment_v1/summary/behavioral_observer_control_summary.csv`
- `outputs/paper1_experiment_v1/summary/experiment_artifact_audit.csv`

## Failure Criteria

Paper 1 should be demoted to a methods/workshop paper, thesis chapter, or measurement framework if any of the following occurs:

- The ROI-expanded matrix produces no nontrivial behavior-neural, behavior-geometry, or encoding-geometry convergence/dissociation pattern.
- Any apparent pattern depends on one model, one ROI, or one geometry method.
- The expanded ROI summaries cannot separate free-viewing from task-search behavior.
- Observer-control integration changes the behavioral interpretation enough that current behavioral rows are no longer usable as an alignment axis.
- Required artifacts are incomplete or cannot be reproduced from the V1 config and recorded run sequence.

## Implementation Sequence

1. Validate current accepted diagnostic inputs and observer-control files.
2. Generate `subj01` stream-ROI manifests from `mapping_streams.npy`.
3. Generate ROI-expanded full-image `flatten_pca` configs for the four-model panel.
4. Run one smoke cell before the full ROI-expanded run.
5. Run the full `subj01` ROI-expanded matrix.
6. Compute `linear_cka_full9841` and deterministic `subset_rsa` for completed cells.
7. Summarize encoding, geometry, behavior-encoding, behavior-geometry, and encoding-geometry tables.
8. Apply failure criteria before subject replication or model expansion.

Implementation should first generalize the neural config generator to accept arbitrary ROI labels and ROI classes, then update matched-scope summary logic to read the model and ROI panel from `configs/paper1_experiment_v1.yaml` rather than relying on hardcoded PRF-only constants.

## Do-Not-Do List

- Do not add a broad model zoo before the ROI-expanded `subj01` discovery matrix exists.
- Do not treat attention rollout as evidence of human-like attention.
- Do not use learned spatial readout as a cross-model ranking row unless the same readout protocol exists for all compared models.
- Do not mix SALICON/CAT2000 free-viewing claims with COCO-Search18 task-search claims.
- Do not run subject replication before the ROI-expanded `subj01` pattern is inspected.
- Do not add fLOC category ROIs, DeepGaze MSDB, task-trained COCO-Search18 baselines, efficiency, or stronger transformer relevance in V1 unless the spec is explicitly revised.
- Do not shift this milestone into Paper 2 causal adaptive-attention intervention work.

## Verification Commands

Focused validation for this spec/config milestone:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_config_loading.py tests\test_paper1_experiment_spec.py
```

Existing neural/reporting baseline after later code changes:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_neural_roi_summary.py tests\test_paper_inspection_pack.py
```

Broader confidence after generator/summarizer changes:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_create_algonauts_manifest.py tests\test_nsd_algonauts_dataset.py tests\test_neural_roi_summary.py tests\test_neural_alignment.py tests\test_paper_inspection_pack.py
```
