# HMA Project Status Changelog

This file stores completed implementation milestones moved out of `docs/project_status_and_next_steps.md` so the primary status handoff can stay focused on current decisions and next steps.

## Completed Milestone Sequence

- Scoring policy and reporting foundation: implemented target-level raw Pearson, R2 fields, noise-ceiling metadata, valid-ceiling filtering, and noise-normalized layer/model aggregates.
- Full-image-count manifest and run configs: implemented for `subj01` `V1`, `V2`, `V3`, and `hV4` with deterministic splits and NSD-derived target-level noise ceilings.
- Feature representation upgrade: implemented train-only `flatten_pca` with metadata and batch transforms; `spatial_mean` remains useful for debug runs only.
- Cross-validated ridge baseline: implemented per-layer/ROI ridge-alpha selection from the outer training split only.
- Validation-only layer/pooling selection: implemented for `flatten_pca`; final score files contain only the selected candidate.
- Learned spatial readout: completed DINOv2 four-ROI fixed-layer learned-readout runs; this is the strongest local single-backbone method result.
- Learned-readout diagnostics: V1 learned-layer selection matched the fixed-layer result, multi-layer smoke was inconclusive, and full V1 voxel-specific low-rank readout was rejected.
- Matched small-model neural panel: completed all `24` full-image-count validation-selected `flatten_pca` cells for the six planned model families and included them in the refreshed summaries and paper inspection pack.
- Matched cross-level analysis: implemented model-level Spearman correlations and simple OLS regressions between corrected behavioral rows and the matched full-image `flatten_pca` neural panel; regenerated neural summary and paper inspection outputs.
- Scalable representational geometry V1: implemented full-image linear CKA for all `24` matched cells, regenerated matched geometry summaries and paper inspection table 10, and added geometry fields to matched cross-level reporting.
- Geometry Sensitivity And Cross-Axis Uncertainty V1: implemented method labels, subset RSA profiling, one-seed subset RSA across all `24` matched cells, geometry-method agreement, runtime summaries, leave-one-model/ROI sensitivity, model-label permutation calibration, decision labels, and paper-inspection figure/table upgrades.

## Current Historical Boundary

The current active status file should no longer carry pre-fix static2000 results, legacy leader-overlap claims, mixed-scope neural ranking details, ROI500 headline geometry claims, or repeated verification run history. Those records are historical/provenance material and should not steer future Codex sessions.
