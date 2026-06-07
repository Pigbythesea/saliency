# Multimodel Behavior-Neural Interpretation Note

This note is generated from the current static2000 behavioral summaries and multi-model ROI500 neural summaries.

## Scope

- Model/ROI winner rows: 3.
- Model ranking rows: 3.
- Matched full-image `flatten_pca` ranking rows: 3.
- Behavior-neural alignment rows: 9.
- Matched cross-level correlation rows: 6.
- Behavioral CSV: D:\Git\saliency\.tmp_pytest_neural_summary_impl\test_matched_cross_level_analy0\behavior.csv.
- Efficiency CSV: not provided.
- Interpretation boundary: descriptive only; one subject, ROI500 subset, and frozen static2000 behavioral rows.

## Neural Ranking

- Strongest mean ROI500 noise-normalized encoding model: convnext_tiny (0.9; x100=90).
- Strongest mean ROI500 raw encoding model: convnext_tiny (0.9).
- Strongest mean ROI500 RSA model: convnext_tiny (0).

## Matched Full-Image Flatten PCA Panel

- Matched panel leader by mean noise-normalized encoding: deit_small_patch16_224 (0.6; ROIs=1).
- This matched panel excludes learned-readout, ROI500, spatial-mean, and rejected voxel-specific rows.

## Efficiency-Normalized Ranking

- Efficiency CSV was not provided or did not match current models.

## Behavioral Saliency Ranking

- Behavioral leaders are computed within each static2000 dataset, metric, saliency method, and saliency family among models with matching neural outputs.
- Leader overlap counts: 0/3 match the raw encoding leader; 0/3 match the raw RSA leader.

## Bridge Interpretation

- Matched cross-level correlation tables are now the primary descriptive cross-axis evidence for the completed six-model full-image `flatten_pca` panel.
- Small-n warning: each complete correlation group is model-level and has at most six matched models from one subject.
- Legacy bridge and leader-overlap tables remain descriptive continuity diagnostics, not causal tests.
- Use `behavior_neural_alignment_summary.csv` for paper-style side-by-side behavioral and neural rows.
- Use `behavior_neural_leader_overlap.csv` for the compact leader-match check.
- Use `matched_cross_level_correlations.csv` for matched model-level correlations/regressions.

## SSL And Multimodal Candidate Prep

- Candidate inventory CSV was not present when this note was generated.
