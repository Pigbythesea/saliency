# HMA Project Results Numbers

Updated: 2026-06-12

This document contains scientific outcomes and the sample sizes needed to interpret them. It excludes implementation milestones, test results, artifact audits, runtime measurements, and validation bookkeeping.

## Headline Outcomes

| outcome | current result |
| --- | --- |
| strongest free-viewing NSS | CAT2000 DeepGaze MSDB: `1.979` |
| strongest SALICON NSS | DeepGaze MSDB: `1.760` |
| strongest task-search NSS | COCO-Search18 task prior: `2.199` |
| best scoped transformer-relevance NSS | CAT2000 DINOv2: `1.141`; SALICON DINOv2: `1.033` |
| six-model PRF encoding leader | DINOv2, noise-normalized `0.591` |
| four-model, ten-ROI discovery encoding leader | DINOv2, noise-normalized `0.556` |
| four-model, ten-ROI full-image CKA leader | DINOv2, `0.194` |
| confirmatory-subject geometry result | DINOv2 leads full-image CKA in `3/3` subjects |
| confirmatory-subject encoding result | DINOv2 leads in `2/3`; ResNet-50 leads in `subj04` |
| accepted subject-level interpretation | `geometry_replicated_encoding_ambiguous` |

## Behavioral Benchmark

The accepted aggregate contains `455` metric rows. Every benchmark cell uses `2000` images.

| dataset | rows | viewing regime | fixation protocol |
| --- | ---: | --- | --- |
| SALICON | 161 | free viewing | points |
| CAT2000 | 161 | free viewing | points |
| COCO-Search18 | 133 | task search | task points |

| saliency family | metric rows |
| --- | ---: |
| evidence sensitivity / vanilla gradient | 168 |
| internal routing / attention rollout | 84 |
| class localization / Grad-CAM | 63 |
| transformer relevance | 56 |
| baseline | 42 |
| dedicated reference | 35 |
| task-search baseline | 7 |

### Behavioral NSS Rankings

#### CAT2000

| rank | model/control | method | NSS | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | DeepGaze MSDB | fixation reference | 1.9786 | [1.9493, 2.0079] |
| 2 | DeepGaze IIE | fixation reference | 1.8385 | [1.8098, 1.8672] |
| 3 | center bias | baseline | 1.6187 | [1.6023, 1.6350] |
| 4 | DINOv2 ViT-S/14 | transformer relevance | 1.1414 | [1.1142, 1.1687] |
| 5 | CLIP ViT-B/16 | transformer relevance | 0.9404 | [0.9134, 0.9674] |
| 6 | DeiT-S/16 | transformer relevance | 0.8859 | [0.8586, 0.9133] |
| 7 | ResNet-50 | Grad-CAM | 0.8818 | [0.8528, 0.9108] |
| 8 | DINOv2 ViT-S/14 | vanilla gradient | 0.8095 | [0.7868, 0.8322] |
| 9 | ConvNeXt-T | Grad-CAM | 0.7590 | [0.7265, 0.7916] |
| 10 | ViT-B/16 | transformer relevance | 0.7334 | [0.7068, 0.7599] |

#### SALICON

| rank | model/control | method | NSS | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | DeepGaze MSDB | fixation reference | 1.7600 | [1.7247, 1.7952] |
| 2 | DeepGaze IIE | fixation reference | 1.7432 | [1.7103, 1.7762] |
| 3 | DINOv2 ViT-S/14 | transformer relevance | 1.0330 | [1.0062, 1.0598] |
| 4 | CLIP ViT-B/16 | transformer relevance | 0.9805 | [0.9536, 1.0074] |
| 5 | center bias | baseline | 0.9326 | [0.9130, 0.9521] |
| 6 | DeiT-S/16 | transformer relevance | 0.9309 | [0.9033, 0.9584] |
| 7 | ViT-B/16 | transformer relevance | 0.8513 | [0.8254, 0.8772] |
| 8 | DINOv2 ViT-S/14 | vanilla gradient | 0.7359 | [0.7121, 0.7597] |
| 9 | ConvNeXt-T | Grad-CAM | 0.6325 | [0.6050, 0.6601] |
| 10 | ResNet-50 | Grad-CAM | 0.5979 | [0.5748, 0.6210] |

#### COCO-Search18

| rank | model/control | method | NSS | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | COCO-Search18 task prior | task-search baseline | 2.1995 | [2.1460, 2.2530] |
| 2 | DeepGaze IIE | free-viewing reference, diagnostic here | 1.7452 | [1.6741, 1.8163] |
| 3 | center bias | baseline | 1.3096 | [1.2819, 1.3374] |
| 4 | ResNet-50 | Grad-CAM | 0.9550 | [0.9149, 0.9951] |
| 5 | ConvNeXt-T | Grad-CAM | 0.9083 | [0.8630, 0.9535] |
| 6 | DINOv2 ViT-S/14 | vanilla gradient | 0.7125 | [0.6706, 0.7545] |
| 7 | CLIP ResNet-50 | Grad-CAM | 0.6406 | [0.5985, 0.6828] |
| 8 | DINOv2 ViT-S/14 | attention rollout | 0.5468 | [0.5031, 0.5905] |
| 9 | DeiT-S/16 | attention rollout | 0.5108 | [0.4814, 0.5402] |
| 10 | ResNet-50 | vanilla gradient | 0.4820 | [0.4456, 0.5185] |

### Accepted Behavioral-Control Metrics

| dataset/control | AUC-Borji | AUC-Judd | CC | KL | NSS | shuffled AUC | SIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAT2000 DeepGaze MSDB | 0.8793 | 0.8798 | 0.6620 | 0.5145 | 1.9786 | 0.8495 | 0.6277 |
| CAT2000 DeepGaze IIE | 0.8588 | 0.8594 | 0.6520 | 0.5812 | 1.8385 | 0.8241 | 0.6080 |
| SALICON DeepGaze MSDB | 0.8575 | 0.8593 | 0.7475 | 0.4520 | 1.7600 | 0.8222 | 0.6521 |
| SALICON DeepGaze IIE | 0.8614 | 0.8631 | 0.8021 | 0.3555 | 1.7432 | 0.8257 | 0.6956 |
| COCO-Search18 task prior | 0.8376 | 0.8380 | 0.4482 | 1.5377 | 2.1995 | 0.6742 | 0.3377 |
| COCO-Search18 DeepGaze IIE | 0.8634 | 0.8638 | 0.2841 | 1.9216 | 1.7452 | 0.7382 | 0.2741 |

DeepGaze MSDB has higher CAT2000 NSS than DeepGaze IIE by `0.1401`. On SALICON, their NSS values are close: MSDB is higher by `0.0167`, while IIE is higher on CC, shuffled AUC, and SIM and lower on KL.

On COCO-Search18, the task-specific prior exceeds diagnostic DeepGaze IIE by `0.4543` NSS and center bias by `0.8899` NSS.

## Transformer Relevance

Transformer relevance was evaluated on SALICON and CAT2000 for four transformer models.

| model | SALICON relevance NSS | SALICON rollout NSS | SALICON gradient NSS | CAT2000 relevance NSS | CAT2000 rollout NSS | CAT2000 gradient NSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 ViT-S/14 | 1.0330 | 0.4554 | 0.7359 | 1.1414 | 0.6697 | 0.8095 |
| CLIP ViT-B/16 | 0.9805 | 0.3903 | 0.0892 | 0.9404 | 0.5882 | 0.0986 |
| DeiT-S/16 | 0.9309 | 0.5442 | 0.0448 | 0.8859 | 0.7067 | -0.1030 |
| ViT-B/16 | 0.8513 | 0.3907 | 0.0827 | 0.7334 | 0.3410 | 0.0207 |

Across the eight matched dataset-model cells:

- transformer relevance beats attention rollout on all seven metrics in `56/56` comparisons;
- transformer relevance beats vanilla gradients on all seven metrics in `56/56` comparisons;
- mean NSS gain over attention rollout is `+0.4263`, range `[+0.1793, +0.5903]`;
- mean NSS gain over vanilla gradients is `+0.7148`, range `[+0.2971, +0.9890]`.

Transformer relevance remains below DeepGaze MSDB on both free-viewing datasets. It also remains below center bias on CAT2000, but DINOv2 and CLIP transformer relevance exceed center bias on SALICON.

## Human Observer Context

These are leave-one-observer-out human-context values, not model scores.

| dataset | images | observers/subjects represented | observers per image | mean NSS | median NSS | mean AUC | median AUC |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| SALICON | 2000 | 1466 workers | 10-71 | 0.529 | 0.502 | 0.873 | 0.890 |
| COCO-Search18 | 502 | 10 subjects | 2-14 | 0.674 | 0.690 | 0.900 | 0.922 |

## Six-Model PRF Neural Encoding

This matched `subj01` panel uses four PRF visual ROIs: V1, V2, V3, and hV4. Scores are averaged over `9654` valid positive-ceiling targets per model.

| rank | model | mean raw Pearson | mean noise-normalized score |
| ---: | --- | ---: | ---: |
| 1 | DINOv2 ViT-S/14 | 0.5414 | 0.5911 |
| 2 | CLIP ViT-B/16 | 0.5380 | 0.5808 |
| 3 | ResNet-50 | 0.5362 | 0.5807 |
| 4 | DeiT-S/16 | 0.5279 | 0.5617 |
| 5 | ViT-B/16 | 0.5135 | 0.5339 |
| 6 | ConvNeXt-T | 0.5021 | 0.5095 |

CLIP ViT-B/16 and ResNet-50 are effectively tied on the mean noise-normalized score: difference `0.00007`.

## DINOv2 Learned Spatial Readout

This is a single-backbone readout-sensitivity result, not a method-matched cross-model ranking.

| ROI | flatten-PCA raw | learned raw | raw gain | flatten-PCA normalized | learned normalized | normalized gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.595 | 0.648 | +0.053 | 0.642 | 0.762 | +0.120 |
| V2 | 0.569 | 0.607 | +0.038 | 0.614 | 0.700 | +0.085 |
| V3 | 0.545 | 0.583 | +0.038 | 0.591 | 0.674 | +0.083 |
| hV4 | 0.457 | 0.488 | +0.031 | 0.517 | 0.586 | +0.069 |

The learned readout improves all `4/4` ROI comparisons.

## Ten-ROI Discovery Encoding

The `subj01` Paper 1 discovery matrix uses four models and ten ROIs: V1, V2, V3, hV4, lateral, midlateral, midparietal, midventral, parietal, and ventral.

### Model Ranking

| rank | model | mean raw Pearson | mean noise-normalized score | valid targets |
| ---: | --- | ---: | ---: | ---: |
| 1 | DINOv2 ViT-S/14 | 0.4909 | 0.5559 | 37027 |
| 2 | ResNet-50 | 0.4823 | 0.5371 | 37027 |
| 3 | CLIP ViT-B/16 | 0.4767 | 0.5212 | 37027 |
| 4 | ViT-B/16 | 0.4655 | 0.5016 | 37027 |

DINOv2 leads ResNet-50 by `0.0188` in mean noise-normalized score across the ten ROIs.

### ROI Winners

| ROI | winning model | raw Pearson | noise-normalized score |
| --- | --- | ---: | ---: |
| V1 | DINOv2 | 0.5950 | 0.6423 |
| V2 | DINOv2 | 0.5686 | 0.6144 |
| V3 | DINOv2 | 0.5451 | 0.5906 |
| hV4 | ResNet-50 | 0.4662 | 0.5396 |
| lateral | DINOv2 | 0.5264 | 0.6214 |
| midlateral | DINOv2 | 0.4459 | 0.4872 |
| midparietal | ResNet-50 | 0.4539 | 0.5366 |
| midventral | ResNet-50 | 0.4546 | 0.5166 |
| parietal | DINOv2 | 0.4168 | 0.4828 |
| ventral | DINOv2 | 0.4508 | 0.5684 |

DINOv2 wins `7/10` ROIs; ResNet-50 wins hV4, midparietal, and midventral.

## Six-Model PRF Representational Geometry

### Full-Image Linear CKA

| rank | model | mean CKA |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.2290 |
| 2 | ConvNeXt-T | 0.2105 |
| 3 | ResNet-50 | 0.2099 |
| 4 | DeiT-S/16 | 0.2028 |
| 5 | CLIP ViT-B/16 | 0.1879 |
| 6 | ViT-B/16 | 0.1039 |

### Deterministic Subset RSA

| method | 1st | score | 2nd | score |
| --- | --- | ---: | --- | ---: |
| size 128, seed 123 | DINOv2 | 0.2654 | ConvNeXt-T | 0.2365 |
| size 256, seed 123 | DINOv2 | 0.2400 | ResNet-50 | 0.2326 |
| size 512, seed 123 | DINOv2 | 0.2341 | ResNet-50 | 0.2212 |

For this six-model PRF panel, DINOv2 leads full-image CKA and all three subset-RSA sizes.

## Ten-ROI Discovery Geometry

### Full-Image Linear CKA

| rank | model | mean CKA |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.1944 |
| 2 | ResNet-50 | 0.1873 |
| 3 | CLIP ViT-B/16 | 0.0993 |
| 4 | ViT-B/16 | 0.0889 |

### Subset-RSA Rankings

| subset size | seed | leader | leader score | runner-up | runner-up score |
| ---: | ---: | --- | ---: | --- | ---: |
| 512 | 123 | ResNet-50 | 0.2328 | DINOv2 | 0.2307 |
| 512 | 456 | ResNet-50 | 0.2164 | DINOv2 | 0.2156 |
| 512 | 789 | DINOv2 | 0.2280 | ResNet-50 | 0.2258 |
| 1024 | 123 | ResNet-50 | 0.2251 | DINOv2 | 0.2210 |
| 1024 | 456 | DINOv2 | 0.2247 | ResNet-50 | 0.2212 |
| 1024 | 789 | ResNet-50 | 0.2276 | DINOv2 | 0.2242 |
| 2048 | 123 | ResNet-50 | 0.2311 | DINOv2 | 0.2280 |
| 2048 | 456 | ResNet-50 | 0.2314 | DINOv2 | 0.2303 |
| 2048 | 789 | ResNet-50 | 0.2279 | DINOv2 | 0.2268 |

The ten-ROI discovery result is method-sensitive at the top rank:

- DINOv2 leads full-image CKA;
- ResNet-50 leads `7/9` subset-RSA settings;
- DINOv2 leads `2/9` subset-RSA settings;
- the DINOv2-ResNet score differences are small in every subset-RSA setting.

Across all `99` ROI/method rank-agreement comparisons, mean CKA-to-subset-RSA agreement is Spearman `0.921` and Kendall `0.869`. At the across-ROI model-ranking level, Spearman agreement ranges from `0.600` to `0.800` and Kendall from `0.333` to `0.667`.

## Cross-Axis Results

### Decision Counts

The ROI-expanded cross-axis decision table contains `2079` relationships.

| relationship | stable convergence | unstable | insufficient models | total |
| --- | ---: | ---: | ---: | ---: |
| behavior vs encoding | 198 | 33 | 462 | 693 |
| behavior vs geometry | 198 | 33 | 462 | 693 |
| encoding vs geometry | 231 | 0 | 462 | 693 |
| total | 627 | 66 | 1386 | 2079 |

Geometry-method sensitivity outcomes:

| outcome | rows |
| --- | ---: |
| stable across geometry methods | 908 |
| direction conflict | 16 |
| insufficient models | 462 |
| not applicable to behavior-vs-encoding | 693 |

Of the geometry-tested relationships with sufficient models, `908/924` are stable in direction across CKA and subset RSA and `16/924` show direction conflict.

### Attribution-Family Cross-Axis Means

These are descriptive mean Spearman correlations across the available metric/ROI groups.

| dataset | family | model n | mean behavior-encoding rho | mean behavior-geometry rho |
| --- | --- | ---: | ---: | ---: |
| CAT2000 | vanilla gradient | 6 | 0.5314 | 0.7143 |
| CAT2000 | attention rollout | 4 | 0.5943 | 0.7486 |
| CAT2000 | transformer relevance | 4 | 0.9600 | 0.8400 |
| SALICON | vanilla gradient | 6 | 0.4465 | 0.6539 |
| SALICON | attention rollout | 4 | 0.3886 | 0.6114 |
| SALICON | transformer relevance | 4 | 0.9600 | 0.8400 |
| COCO-Search18 | vanilla gradient | 6 | 0.3551 | 0.5820 |
| COCO-Search18 | attention rollout | 4 | 0.6057 | 0.7600 |

These correlations remain small-model-panel descriptions. Transformer relevance improves the behavioral attribution result, but it does not establish a causal relationship between fixation alignment and neural or geometric alignment.

## Confirmatory Subject Robustness

The confirmatory panel uses four models over V1, V2, V3, and hV4 for `subj02`, `subj03`, and `subj04`.

### Encoding Rankings

Encoding values are mean raw Pearson correlations because usable subject-specific noise ceilings are not available for these confirmatory rows.

| subject | rank 1 | score | rank 2 | score | rank 3 | score | rank 4 | score |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| subj02 | DINOv2 | 0.5314 | ResNet-50 | 0.5304 | CLIP ViT | 0.5263 | ViT-B | 0.5025 |
| subj03 | DINOv2 | 0.4207 | ResNet-50 | 0.4198 | CLIP ViT | 0.4169 | ViT-B | 0.4000 |
| subj04 | ResNet-50 | 0.4258 | DINOv2 | 0.4220 | CLIP ViT | 0.4200 | ViT-B | 0.4057 |

### Full-Image CKA Rankings

| subject | rank 1 | score | rank 2 | score | rank 3 | score | rank 4 | score |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| subj02 | DINOv2 | 0.2539 | ResNet-50 | 0.2396 | CLIP ViT | 0.2104 | ViT-B | 0.1145 |
| subj03 | DINOv2 | 0.1607 | ResNet-50 | 0.1438 | CLIP ViT | 0.1085 | ViT-B | 0.0690 |
| subj04 | DINOv2 | 0.2043 | ResNet-50 | 0.1881 | CLIP ViT | 0.1704 | ViT-B | 0.0914 |

### DINOv2 Minus ResNet-50 Encoding Margins

| subject | paired targets | mean margin | median margin | positive-target fraction | bootstrap 95% CI | supported model |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| subj02 | 9393 | +0.00237 | +0.00213 | 0.551 | [0.00205, 0.00268] | DINOv2 |
| subj03 | 8972 | +0.00307 | +0.00314 | 0.598 | [0.00274, 0.00340] | DINOv2 |
| subj04 | 8138 | -0.00172 | -0.00010 | 0.497 | [-0.00207, -0.00141] | ResNet-50 |

ROI-level encoding margins:

| subject | V1 | V2 | V3 | hV4 |
| --- | ---: | ---: | ---: | ---: |
| subj02 | -0.00425 | +0.00393 | +0.01290 | -0.00855 |
| subj03 | +0.00011, ambiguous | +0.00695 | +0.00620 | -0.00962 |
| subj04 | -0.00102 | +0.00624 | -0.00301 | -0.01735 |

Across all confirmatory targets, the pooled DINOv2-minus-ResNet encoding margin is `+0.00135`, 95% CI `[0.00117, 0.00153]`. This pooled positive value does not remove the subject reversal: `subj04` significantly favors ResNet-50. The accepted encoding interpretation is therefore subject-sensitive and ambiguous, not a universal DINOv2 win.

### Geometry Margins

| subject | DINOv2-minus-ResNet mean CKA margin | subset-RSA settings supporting DINOv2 |
| --- | ---: | ---: |
| subj02 | +0.01429 | 9/9 |
| subj03 | +0.01688 | 9/9 |
| subj04 | +0.01615 | 9/9 |
| aggregate | +0.01577 | 9/9 |

The confirmatory geometry result is consistent across subjects, full-image CKA, and the nine deterministic subset-RSA settings. This is the strongest current replication result.

## Current Numeric Interpretation

The actual outcome is a geometry-first dissociation:

1. DINOv2 is the strongest model in the six-model PRF encoding panel and in the four-model ten-ROI discovery encoding mean.
2. DINOv2 leads full-image CKA in the ten-ROI discovery matrix, but ResNet-50 leads most discovery subset-RSA settings, so the discovery geometry top rank is method-sensitive.
3. In all three confirmatory subjects, DINOv2 leads full-image CKA and has positive aggregate geometry margins across all nine subset-RSA settings.
4. Confirmatory encoding is not equally stable: DINOv2 leads `subj02` and `subj03`, while ResNet-50 leads `subj04`.
5. Transformer relevance materially improves fixation alignment over attention rollout and vanilla gradients, but dedicated fixation references remain stronger and the improvement does not establish causal neural alignment.

The current accepted summary label is:

`geometry_replicated_encoding_ambiguous`
