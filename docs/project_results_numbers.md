# HMA Project Numeric Handoff

Updated: 2026-06-02

This is a self-contained numeric handoff of the current Human-Machine Visual Alignment project. It contains counts, metrics, rankings, model/dataset setup, and verification numbers only.

## Behavioral Benchmark Scope

The corrected behavioral benchmark contains `378` aggregate result rows.

Datasets:

| dataset | aggregate rows | images per benchmark row | fixation protocol |
| --- | ---: | ---: | --- |
| SALICON static subset | 126 | 2000 | point fixations |
| CAT2000 static subset | 126 | 2000 | point fixations |
| COCO-Search18 static subset | 126 | 2000 | task-driven point fixations |

Protocol counts:

| protocol | rows |
| --- | ---: |
| point fixations | 252 |
| task-driven point fixations | 126 |

Metric counts:

| metric | rows |
| --- | ---: |
| AUC-Borji | 54 |
| AUC-Judd | 54 |
| CC | 54 |
| KL | 54 |
| NSS | 54 |
| shuffled AUC | 54 |
| similarity | 54 |

Evaluated model rows cover `11` model/control identities:

| model/control |
| --- |
| center-bias baseline |
| ConvNeXt-T |
| DeepGaze IIE reference |
| DeiT-S/16 |
| random baseline |
| ResNet-50 |
| CLIP ResNet-50 |
| Swin-T |
| ViT-B/16 |
| CLIP ViT-B/16 |
| DINOv2 ViT-S/14 |

Saliency/explanation methods:

| method | family |
| --- | --- |
| center bias | baseline |
| random saliency | baseline |
| DeepGaze precomputed map | reference |
| Grad-CAM | class localization |
| vanilla gradient | evidence sensitivity |
| attention rollout | internal routing |

Saliency-family row counts:

| family | rows |
| --- | ---: |
| baseline | 42 |
| class localization | 63 |
| evidence sensitivity | 168 |
| reference | 21 |
| internal routing | 84 |

## Behavioral NSS Results

### CAT2000 Static Subset

| rank | model/control | method | n | NSS mean | 95% CI low | 95% CI high |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | DeepGaze IIE | DeepGaze map | 2000 | 1.838466 | 1.809760 | 1.867172 |
| 2 | center bias | center bias | 2000 | 1.618651 | 1.602291 | 1.635012 |
| 3 | ResNet-50 | Grad-CAM | 2000 | 0.881798 | 0.852829 | 0.910768 |
| 4 | DINOv2 ViT-S/14 | vanilla gradient | 2000 | 0.809503 | 0.786810 | 0.832196 |
| 5 | ConvNeXt-T | Grad-CAM | 2000 | 0.759007 | 0.726451 | 0.791564 |
| 6 | DeiT-S/16 | attention rollout | 2000 | 0.706659 | 0.677372 | 0.735945 |
| 7 | DINOv2 ViT-S/14 | attention rollout | 2000 | 0.669732 | 0.647753 | 0.691711 |
| 8 | CLIP ViT-B/16 | attention rollout | 2000 | 0.588220 | 0.560596 | 0.615844 |
| 9 | ResNet-50 | vanilla gradient | 2000 | 0.466488 | 0.448900 | 0.484076 |
| 10 | CLIP ResNet-50 | Grad-CAM | 2000 | 0.366296 | 0.338508 | 0.394083 |

### COCO-Search18 Static Subset

| rank | model/control | method | n | NSS mean | 95% CI low | 95% CI high |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | DeepGaze IIE | DeepGaze map | 2000 | 1.745170 | 1.674067 | 1.816272 |
| 2 | center bias | center bias | 2000 | 1.309644 | 1.281897 | 1.337392 |
| 3 | ResNet-50 | Grad-CAM | 2000 | 0.954973 | 0.914862 | 0.995085 |
| 4 | ConvNeXt-T | Grad-CAM | 2000 | 0.908253 | 0.863021 | 0.953484 |
| 5 | DINOv2 ViT-S/14 | vanilla gradient | 2000 | 0.712529 | 0.670568 | 0.754489 |
| 6 | CLIP ResNet-50 | Grad-CAM | 2000 | 0.640623 | 0.598453 | 0.682793 |
| 7 | DINOv2 ViT-S/14 | attention rollout | 2000 | 0.546807 | 0.503107 | 0.590506 |
| 8 | DeiT-S/16 | attention rollout | 2000 | 0.510805 | 0.481389 | 0.540221 |
| 9 | ResNet-50 | vanilla gradient | 2000 | 0.482032 | 0.445550 | 0.518514 |
| 10 | ConvNeXt-T | vanilla gradient | 2000 | 0.389143 | 0.352708 | 0.425578 |

### SALICON Static Subset

| rank | model/control | method | n | NSS mean | 95% CI low | 95% CI high |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | DeepGaze IIE | DeepGaze map | 2000 | 1.743241 | 1.710289 | 1.776193 |
| 2 | center bias | center bias | 2000 | 0.932575 | 0.913002 | 0.952147 |
| 3 | DINOv2 ViT-S/14 | vanilla gradient | 2000 | 0.735908 | 0.712098 | 0.759717 |
| 4 | ConvNeXt-T | Grad-CAM | 2000 | 0.632550 | 0.604989 | 0.660110 |
| 5 | ResNet-50 | Grad-CAM | 2000 | 0.597877 | 0.574752 | 0.621003 |
| 6 | DeiT-S/16 | attention rollout | 2000 | 0.544208 | 0.516031 | 0.572386 |
| 7 | CLIP ResNet-50 | Grad-CAM | 2000 | 0.503253 | 0.477498 | 0.529009 |
| 8 | DINOv2 ViT-S/14 | attention rollout | 2000 | 0.455435 | 0.436081 | 0.474790 |
| 9 | ViT-B/16 | attention rollout | 2000 | 0.390684 | 0.366895 | 0.414474 |
| 10 | CLIP ViT-B/16 | attention rollout | 2000 | 0.390254 | 0.365121 | 0.415386 |

## Behavioral Dataset Counts

Static subsets:

| dataset | rows/images | split | extra counts |
| --- | ---: | --- | --- |
| SALICON static subset | 2000 | validation | none recorded here |
| CAT2000 static subset | 2000 | train | none recorded here |
| COCO-Search18 static subset | 2000 | validation | 1338 target-present, 662 target-absent |

Full COCO-Search18 dataset rows:

| quantity | count |
| --- | ---: |
| total rows | 74646 |
| columns | 10 |
| train rows | 64112 |
| validation rows | 10534 |
| target-present rows | 49760 |
| target-absent rows | 24886 |

COCO-Search18 full target-category rows:

| target category | rows |
| --- | ---: |
| bottle | 3990 |
| bowl | 3390 |
| car | 2520 |
| chair | 6074 |
| clock | 2880 |
| cup | 6630 |
| fork | 5520 |
| keyboard | 4440 |
| knife | 3390 |
| laptop | 2970 |
| microwave | 3748 |
| mouse | 2636 |
| oven | 2430 |
| potted plant | 3720 |
| sink | 6720 |
| stop sign | 3030 |
| toilet | 3808 |
| tv | 6750 |

SALICON observer annotations:

| quantity | count |
| --- | ---: |
| full worker-level rows | 893854 |
| static-subset worker-level rows | 125874 |
| static-subset images | 2000 |

Observer-control results:

| dataset | rows | columns | note |
| --- | ---: | ---: | --- |
| COCO-Search18 static observer controls | 1867 | 8 | inter-observer NSS/AUC fields present |
| SALICON static observer controls | 20000 | 8 | at most 10 workers per image |

## Neural Encoding Scope

Current neural results are for `1` subject: `subj01`.

Current ROI set:

| ROI |
| --- |
| V1 |
| V2 |
| V3 |
| hV4 |

Neural summary counts:

| quantity | count |
| --- | ---: |
| input neural run directories | 48 |
| encoding summary rows | 120 |
| encoding target rows | 289740 |
| RSA rows | 92 |
| spatial-mean diagnostic rows | 92 |
| full-image matched-panel flatten-PCA rows | 24 |
| full-image learned spatial-readout rows | 4 |

Encoding target metric scopes:

| metric scope | rows |
| --- | ---: |
| benchmark-style noise-normalized | 289620 |
| benchmark-style non-noise-normalized | 120 |

hV4 zero-ceiling note:

| quantity | count |
| --- | ---: |
| hV4 zero-noise-ceiling targets | 4 |

## Matched Full-Image Neural Panel Setup

The matched neural panel has `6` models and `4` ROIs, giving `24` model-by-ROI cells.

Matched-panel cell audit:

| status | cells |
| --- | ---: |
| complete | 24 |
| missing | 0 |
| incomplete | 0 |
| explicitly skipped | 0 |

Per-run setup:

| quantity | value |
| --- | ---: |
| full image count | 9841 |
| train images | 7873 |
| test images | 1968 |
| split seed | 123 |
| PCA components | 512 |

Feature setup:

| setting | value |
| --- | --- |
| feature reduction | flatten PCA |
| PCA solver | randomized |
| PCA fit policy | train-only |
| neural target score | raw Pearson correlation and noise-normalized score |
| ridge selection | validation/inner-validation depending on run |

Targets per matched model:

| target type | count |
| --- | ---: |
| valid positive-ceiling targets | 9654 |
| zero-ceiling targets | 4 |
| invalid-ceiling targets | 0 |

## Matched Full-Image Neural Model Ranking

Ranking by mean valid-target noise-normalized encoding score:

| rank | model | mean noise-normalized score | score x100 | mean raw Pearson | valid targets | zero-ceiling targets |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | DINOv2 ViT-S/14 | 0.591079 | 59.107926 | 0.541401 | 9654 | 4 |
| 2 | CLIP ViT-B/16 | 0.580751 | 58.075140 | 0.538027 | 9654 | 4 |
| 3 | ResNet-50 | 0.580685 | 58.068504 | 0.536216 | 9654 | 4 |
| 4 | DeiT-S/16 | 0.561700 | 56.169977 | 0.527933 | 9654 | 4 |
| 5 | ViT-B/16 | 0.533908 | 53.390778 | 0.513485 | 9654 | 4 |
| 6 | ConvNeXt-T | 0.509512 | 50.951156 | 0.502095 | 9654 | 4 |

## Matched Full-Image Neural ROI Scores

| model | ROI | selected layer | raw Pearson | noise-normalized | valid targets | zero targets | selected ridge alpha |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-T | V1 | stages.0 | 0.574477 | 0.599323 | 2973 | 0 | 0.001 |
| ConvNeXt-T | V2 | stages.0 | 0.526797 | 0.527125 | 2936 | 0 | 10000.0 |
| ConvNeXt-T | V3 | stages.0 | 0.488495 | 0.473876 | 2453 | 0 | 100000.0 |
| ConvNeXt-T | hV4 | stages.2 | 0.418609 | 0.437721 | 1292 | 4 | 100000.0 |
| DeiT-S/16 | V1 | blocks.0 | 0.580590 | 0.611357 | 2973 | 0 | 10000.0 |
| DeiT-S/16 | V2 | blocks.3 | 0.556209 | 0.587888 | 2936 | 0 | 10000.0 |
| DeiT-S/16 | V3 | blocks.3 | 0.530810 | 0.560140 | 2453 | 0 | 10000.0 |
| DeiT-S/16 | hV4 | blocks.3 | 0.444122 | 0.487414 | 1292 | 4 | 100000.0 |
| ResNet-50 | V1 | layer2 | 0.585271 | 0.622254 | 2973 | 0 | 0.001 |
| ResNet-50 | V2 | layer2 | 0.561672 | 0.599519 | 2936 | 0 | 100000.0 |
| ResNet-50 | V3 | layer2 | 0.531720 | 0.561382 | 2453 | 0 | 100000.0 |
| ResNet-50 | hV4 | layer3 | 0.466201 | 0.539585 | 1292 | 4 | 1000000.0 |
| ViT-B/16 | V1 | blocks.3 | 0.563680 | 0.578286 | 2973 | 0 | 10000.0 |
| ViT-B/16 | V2 | blocks.6 | 0.529133 | 0.534086 | 2936 | 0 | 10000.0 |
| ViT-B/16 | V3 | blocks.6 | 0.514245 | 0.527996 | 2453 | 0 | 10000.0 |
| ViT-B/16 | hV4 | blocks.6 | 0.446882 | 0.495263 | 1292 | 4 | 100000.0 |
| CLIP ViT-B/16 | V1 | blocks.3 | 0.587097 | 0.626278 | 2973 | 0 | 100.0 |
| CLIP ViT-B/16 | V2 | blocks.3 | 0.565247 | 0.606858 | 2936 | 0 | 1000.0 |
| CLIP ViT-B/16 | V3 | blocks.3 | 0.539192 | 0.577204 | 2453 | 0 | 1000.0 |
| CLIP ViT-B/16 | hV4 | blocks.6 | 0.460574 | 0.512665 | 1292 | 4 | 10000.0 |
| DINOv2 ViT-S/14 | V1 | blocks.3 | 0.595029 | 0.642300 | 2973 | 0 | 10.0 |
| DINOv2 ViT-S/14 | V2 | blocks.6 | 0.568586 | 0.614450 | 2936 | 0 | 1000.0 |
| DINOv2 ViT-S/14 | V3 | blocks.6 | 0.545084 | 0.590614 | 2453 | 0 | 1000.0 |
| DINOv2 ViT-S/14 | hV4 | blocks.6 | 0.456906 | 0.516953 | 1292 | 4 | 1000.0 |

## DINOv2 Learned Spatial Readout

The learned spatial readout was run for DINOv2 ViT-S/14 on the same `subj01` PRF visual ROI set.

Setup:

| quantity | value |
| --- | ---: |
| full image count | 9841 |
| outer train images | 7873 |
| held-out test images | 1968 |
| inner train images | 6298 |
| inner validation images | 1575 |
| maximum epochs | 180 |

Scores:

| ROI | layer | mean raw Pearson | mean noise-normalized | median raw Pearson | median noise-normalized | valid targets | zero targets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | blocks.3 | 0.648 | 0.762 | 0.691 | 0.803 | 2973 | 0 |
| V2 | blocks.6 | 0.607 | 0.700 | 0.654 | 0.735 | 2936 | 0 |
| V3 | blocks.6 | 0.583 | 0.674 | 0.611 | 0.703 | 2453 | 0 |
| hV4 | blocks.6 | 0.488 | 0.586 | 0.518 | 0.588 | 1292 | 4 |

Epochs:

| ROI | best epoch | stopped epoch |
| --- | ---: | ---: |
| V1 | 127 | 142 |
| V2 | 65 | 80 |
| V3 | 63 | 78 |
| hV4 | 59 | 74 |

Learned readout minus matched flatten-PCA baseline:

| ROI | flatten-PCA raw | learned raw | raw delta | flatten-PCA noise-normalized | learned noise-normalized | noise-normalized delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.595 | 0.648 | +0.053 | 0.642 | 0.762 | +0.120 |
| V2 | 0.569 | 0.607 | +0.038 | 0.614 | 0.700 | +0.085 |
| V3 | 0.545 | 0.583 | +0.038 | 0.591 | 0.674 | +0.083 |
| hV4 | 0.457 | 0.488 | +0.031 | 0.517 | 0.586 | +0.069 |

## Representational Geometry Scope

Geometry was computed for the matched full-image panel.

Counts:

| quantity | count |
| --- | ---: |
| total geometry score rows | 96 |
| valid geometry score rows | 96 |
| full-image CKA rows | 24 |
| subset-RSA rows | 72 |
| models | 6 |
| ROIs | 4 |
| full images used by CKA | 9841 |
| subset-RSA subset sizes | 128, 256, 512 |
| subset-RSA seed | 123 |

Geometry methods:

| method | image count | comparison |
| --- | ---: | --- |
| full-image linear CKA | 9841 | CKA between model features and raw ROI responses |
| subset RSA size 128 | 128 | Spearman between correlation RDMs |
| subset RSA size 256 | 256 | Spearman between correlation RDMs |
| subset RSA size 512 | 512 | Spearman between correlation RDMs |

## Representational Geometry Rankings

### Full-Image Linear CKA

| rank | model | mean geometry score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.229035 |
| 2 | ConvNeXt-T | 0.210549 |
| 3 | ResNet-50 | 0.209893 |
| 4 | DeiT-S/16 | 0.202786 |
| 5 | CLIP ViT-B/16 | 0.187853 |
| 6 | ViT-B/16 | 0.103863 |

### Subset RSA Size 128

| rank | model | mean geometry score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.265353 |
| 2 | ConvNeXt-T | 0.236541 |
| 3 | ResNet-50 | 0.235214 |
| 4 | CLIP ViT-B/16 | 0.230689 |
| 5 | DeiT-S/16 | 0.225000 |
| 6 | ViT-B/16 | 0.127000 |

### Subset RSA Size 256

| rank | model | mean geometry score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.240000 |
| 2 | ResNet-50 | 0.233000 |
| 3 | ConvNeXt-T | 0.221000 |
| 4 | DeiT-S/16 | 0.215000 |
| 5 | CLIP ViT-B/16 | 0.202000 |
| 6 | ViT-B/16 | 0.126000 |

### Subset RSA Size 512

| rank | model | mean geometry score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.234000 |
| 2 | ResNet-50 | 0.221000 |
| 3 | ConvNeXt-T | 0.211000 |
| 4 | DeiT-S/16 | 0.205000 |
| 5 | CLIP ViT-B/16 | 0.204000 |
| 6 | ViT-B/16 | 0.123000 |

## Geometry Runtime And Agreement

Runtime:

| method | rows | valid rows | invalid rows | mean wall time sec | max wall time sec | estimated RDM bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-image linear CKA | 24 | 24 | 0 | 0.689941 | 0.933989 | 0 |
| subset RSA size 128 | 24 | 24 | 0 | 0.246518 | 0.322708 | 131072 |
| subset RSA size 256 | 24 | 24 | 0 | 0.351630 | 0.458027 | 524288 |
| subset RSA size 512 | 24 | 24 | 0 | 0.730804 | 0.934660 | 2097152 |

Agreement between full-image CKA ranking and subset-RSA ranking:

| quantity | value |
| --- | ---: |
| agreement rows | 15 |
| complete agreement rows | 15 |
| across-ROI Spearman agreement | 0.943 |
| across-ROI Kendall agreement | 0.867 |
| V1 size-128 Spearman agreement | 1.000 |
| V1 size-128 Kendall agreement | 1.000 |
| V1 size-256 Spearman agreement | 1.000 |
| V1 size-256 Kendall agreement | 1.000 |
| V1 size-512 Spearman agreement | 0.943 |
| V1 size-512 Kendall agreement | 0.867 |

## Cross-Level Correlations

Matched model-level cross-level analysis count:

| quantity | count |
| --- | ---: |
| total correlation/regression groups | 315 |
| complete groups | 210 |
| insufficient-model groups | 105 |
| groups per behavioral dataset | 105 |

Metric groups:

| metric | groups |
| --- | ---: |
| AUC-Borji | 45 |
| AUC-Judd | 45 |
| CC | 45 |
| KL | 45 |
| NSS | 45 |
| shuffled AUC | 45 |
| similarity | 45 |

NSS across-ROI complete groups:

| behavioral dataset | saliency method | matched models | Spearman: behavior vs noise-normalized encoding | OLS R2: behavior vs noise-normalized encoding | Spearman: behavior vs geometry |
| --- | --- | ---: | ---: | ---: | ---: |
| CAT2000 | attention rollout | 4 | 0.400000 | 0.588409 | 0.800000 |
| CAT2000 | vanilla gradient | 6 | 0.485714 | 0.132414 | 0.771429 |
| COCO-Search18 | attention rollout | 4 | 0.800000 | 0.281328 | 1.000000 |
| COCO-Search18 | vanilla gradient | 6 | 0.314286 | 0.061027 | 0.714286 |
| SALICON | attention rollout | 4 | 0.000000 | 0.019716 | 0.600000 |
| SALICON | vanilla gradient | 6 | 0.485714 | 0.138048 | 0.771429 |

## Cross-Axis Sensitivity And Decision Counts

Sensitivity and decision totals:

| quantity | count |
| --- | ---: |
| sensitivity rows | 4284 |
| decision rows | 945 |

Decision-label totals:

| decision label | rows |
| --- | ---: |
| stable convergence | 374 |
| unstable | 256 |
| insufficient models | 315 |

Decision labels by relationship:

| relationship | stable convergence | unstable | insufficient models | total |
| --- | ---: | ---: | ---: | ---: |
| behavior vs geometry | 163 | 47 | 105 | 315 |
| behavior vs noise-normalized encoding | 85 | 125 | 105 | 315 |
| encoding vs geometry | 126 | 84 | 105 | 315 |

## Additional Neural Subject Data Prepared

Additional neural subject data have been prepared for `subj02`, `subj03`, and `subj04`.

| subject | rows | V1 response dimension | V2 response dimension | V3 response dimension | hV4 response dimension |
| --- | ---: | ---: | ---: | ---: | ---: |
| subj02 | 39364 | 2737 | 2779 | 2615 | 1262 |
| subj03 | 36328 | 2676 | 2991 | 2418 | 887 |
| subj04 | 35116 | 2328 | 2474 | 2146 | 1190 |

Combined four-subject data:

| quantity | count |
| --- | ---: |
| combined rows for subj01-subj04 | 150172 |

Validation:

| validation item | count |
| --- | ---: |
| sampled rows per subject/ROI for bounded validation | 25 |
| subjects with bounded validation completed | 3 |
| ROIs per subject in bounded validation | 4 |

## Summary Numbers

Current headline numeric values:

| quantity | value |
| --- | ---: |
| highest behavioral NSS | 1.838 |
| highest behavioral NSS dataset | CAT2000 |
| highest behavioral NSS model/control | DeepGaze IIE |
| matched-panel neural encoding leader mean noise-normalized score | 0.591 |
| matched-panel neural encoding leader x100 score | 59.11 |
| matched-panel neural encoding leader | DINOv2 ViT-S/14 |
| matched geometry leader score | 0.229 |
| matched geometry leader | DINOv2 ViT-S/14 |
| geometry sensitivity agreement rows | 15 |
| overall behavior-to-encoding leader match rate | 0.587 |
| overall behavior-to-RSA leader match rate | 0.587 |
| learned spatial readout improved ROI comparisons | 4 |
| learned spatial readout total ROI comparisons | 4 |
| SSL/multimodal candidates dry-inspected | 8 |
| SSL/multimodal pretrained debug runs complete | 3 |
| SSL/multimodal pretrained status complete | 3 |
| SSL/multimodal pretrained status not run | 5 |

## Verification Numbers

Recorded verification results:

| verification item | result |
| --- | ---: |
| full test suite | 210 passed |
| focused neural/reporting test result | 36 passed |
| geometry-focused reporting test result | 85 passed |
| broader neural/reporting rerun with fresh temp directory | 93 passed |
| matched-panel audit complete cells | 24 |
| matched-panel audit missing cells | 0 |
| matched-panel audit incomplete cells | 0 |
| matched-panel audit explicitly skipped cells | 0 |
| neural target-scope total rows | 289740 |
| neural target-scope noise-normalized rows | 289620 |
| neural target-scope non-noise-normalized rows | 120 |
| reliability metadata audit candidate count | 0 |
| reliability metadata audit usable candidate count | 0 |
| reliability metadata audit usable metadata found | 0 |
| neural noise-ceiling smoke rows | 2973 |

## Efficiency Data Status

Efficiency fields are not populated with usable values.

| efficiency quantity | current recorded value |
| --- | --- |
| efficiency data | not provided |
| latency mean ms | not populated |
| parameter count | not populated |
| model size MB | empty |
| FLOPs | empty |
| memory footprint | not provided |
| token count | not provided |
| retained-patch statistics | not provided |
