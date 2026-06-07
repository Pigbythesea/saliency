# Neural ROI Summary

This note is generated from neural ROI alignment outputs.

## Scope

- Input directories: 3.
- Encoding rows: 3.
- Encoding target rows: 0.
- RSA rows: 0.
- Geometry rows: 6 matched rows.
- Behavioral bridge CSV: not provided.
- Efficiency CSV: not provided.
- Benchmark-style encoding scope: not available; input directories do not include per-target benchmark scores.
- Matched full-image `flatten_pca` panel models complete in summary: 3.
- Matched geometry model-ranking rows: 6.

## Best Encoding Layers

- convnext_tiny subj01 V1 correlation: layer1 score=0.4. score_type=noise_normalized.
- deit_small_patch16_224 subj01 V1 correlation: layer1 score=0.6. score_type=noise_normalized.
- resnet50 subj01 V1 correlation: layer1 score=0.2. score_type=noise_normalized.

## Best RSA Layers

- No RSA best-layer rows are available.

## Behavior-Neural Bridge

- No behavior-neural bridge rows were generated.
- Do not interpret bridge rows as cross-model correlations until neural outputs exist for multiple model families.

## Matched Cross-Level Analysis

- No matched cross-level correlation rows were generated.

## Learned Readout Versus Flatten PCA

- No matched learned-readout and `flatten_pca` comparison rows are available.

## Matched Full-Image Flatten PCA Panel

- deit_small_patch16_224: ROIs=1, mean_noise_normalized=0.6, rank=1.
- convnext_tiny: ROIs=1, mean_noise_normalized=0.4, rank=2.
- resnet50: ROIs=1, mean_noise_normalized=0.2, rank=3.
