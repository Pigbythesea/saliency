# Neural ROI Summary

This note is generated from neural ROI alignment outputs.

## Scope

- Input directories: 2.
- Encoding rows: 2.
- Encoding target rows: 0.
- RSA rows: 0.
- Geometry rows: 0 matched rows.
- Behavioral bridge CSV: D:\Git\saliency\.tmp_pytest_neural_summary_impl\test_matched_cross_level_analy1\behavior.csv.
- Efficiency CSV: not provided.
- Benchmark-style encoding scope: not available; input directories do not include per-target benchmark scores.
- Matched full-image `flatten_pca` panel models complete in summary: 2.
- Matched geometry model-ranking rows: 0.

## Best Encoding Layers

- convnext_tiny subj01 V1 correlation: layer2 score=0.4. score_type=noise_normalized.
- resnet50 subj01 V1 correlation: layer2 score=0.2. score_type=noise_normalized.

## Best RSA Layers

- No RSA best-layer rows are available.

## Behavior-Neural Bridge

- Descriptive bridge rows were generated for matching static2000 behavioral models and neural ROI outputs.
- Do not interpret bridge rows as cross-model correlations until neural outputs exist for multiple model families.

## Matched Cross-Level Analysis

- Matched cross-level observation rows: 4.
- Matched cross-level correlation/regression groups: 0/2 complete.
- These rows use only the full-image `flatten_pca` matched panel and keep COCO-Search18 separate from free-viewing datasets.

## Learned Readout Versus Flatten PCA

- No matched learned-readout and `flatten_pca` comparison rows are available.

## Matched Full-Image Flatten PCA Panel

- convnext_tiny: ROIs=1, mean_noise_normalized=0.4, rank=1.
- resnet50: ROIs=1, mean_noise_normalized=0.2, rank=2.
