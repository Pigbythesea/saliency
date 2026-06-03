# HMA Project Publication Critique Handoff

**Purpose:** This file summarizes a critical evaluation of the current Human–Machine Visual Alignment project, focusing on the achieved numerical outcomes, the current paper plan, novelty against existing literature, and what must change for top-venue publication.

**Scope:** This critique treats engineering progress as secondary. The central evaluation is based on reported numbers, figures, scientific claims, and publication readiness.

---

## Verdict

The project is scientifically pointed in the right direction, while the current evidence remains far below top-venue grade.

The proposal asks a publishable question: whether behavioral fixation alignment, model attribution/internal routing, neural encoding, representational geometry, and efficiency are actually coupled or dissociable. That is a legitimate NeuroAI question. The proposal explicitly defines the project as multi-axis: behavioral saliency, fMRI/neural alignment, representational similarity, Brain-Score-style evaluation, and compute efficiency. It also correctly warns that a model may look human-like in fixation maps while failing in cortical representations, or predict visual cortex while producing non-human-like saliency maps.

The current results only partially instantiate that plan. The project now has behavioral fixation benchmarking, a six-model one-subject neural encoding panel, and one matched geometry axis. That means the project has finally moved beyond a pure saliency-map pilot. However, the actual evidence remains too narrow, too underpowered, and too control-light for a top conference claim. The current matched full-image geometry is one subject, one metric family, and still lacks enough uncertainty and robustness checks.

---

## What the achieved numbers actually say

### 1. Behavioral results: sane, useful, and unoriginal

The behavioral benchmark has 378 aggregate rows across SALICON, CAT2000, and COCO-Search18, with 126 rows per dataset and no blank/unknown/density-fallback protocol rows. That is good because the previous protocol failure appears fixed.

Headline NSS values:

| Dataset | Best reference/control | Center bias | Best classifier-derived attribution rows |
| --- | ---: | ---: | --- |
| SALICON | DeepGaze 1.743 | 0.933 | DINOv2 gradient 0.736; ConvNeXt Grad-CAM 0.633; ResNet-50 Grad-CAM 0.598 |
| CAT2000 | DeepGaze 1.838 | 1.619 | ResNet-50 Grad-CAM 0.882; DINOv2 gradient 0.810; ConvNeXt Grad-CAM 0.759 |
| COCO-Search18 | DeepGaze 1.745 | 1.310 | ResNet-50 Grad-CAM 0.955; ConvNeXt Grad-CAM 0.908; DINOv2 gradient 0.713 |

Mean-reviewer interpretation: this is a sanity check, not a discovery. DeepGaze beating center bias means the scoring is no longer obviously broken. Classifier-derived Grad-CAM/gradient/rollout maps being below DeepGaze is exactly what the literature would expect because DeepGaze-family models are dedicated fixation predictors, while Grad-CAM and gradients are explanation maps for classification.

The behavioral layer currently proves:

> The pipeline is now sane.

It does not prove:

> Modern classifiers have human-like attention.

DeepGaze MSDB and current saliency benchmark results set a much higher bar than the current DeepGaze IIE/classifier-attribution comparison. The behavioral layer should be treated as one axis of a broader alignment matrix, not as the main publication result.

---

### 2. Neural encoding: promising local signal, weak publication evidence

The matched neural panel has six models x four ROIs = 24 complete cells for `subj01`, using full image count, train-only PCA, validation-selected layers, and noise-normalized scores.

Ranking by mean valid-target noise-normalized encoding score:

| Rank | Model | Score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.591 |
| 2 | CLIP ViT-B/16 | 0.581 |
| 3 | ResNet-50 | 0.581 |
| 4 | DeiT-S/16 | 0.562 |
| 5 | ViT-B/16 | 0.534 |
| 6 | ConvNeXt-T | 0.510 |

This is the most scientifically interesting part of the current outcome because it begins to align with modern brain-predictivity literature: self-supervised and multimodal encoders are serious baselines, and plain supervised ViT-B is weaker. The result is plausible.

Mean-reviewer attack: one subject, four early/intermediate PRF ROIs, six models, one feature policy, and no official Algonauts equivalence. That is an immediate rejection point if the paper tries to claim SOTA neural alignment.

The neural result is credible as a local diagnostic baseline. It is weak as a standalone publication claim.

---

### 3. Learned DINOv2 readout: genuinely useful, easy to overclaim

The DINOv2 learned spatial readout improves over DINOv2 flatten-PCA in all four ROIs:

| ROI | Flatten-PCA | Learned spatial readout | Delta |
| --- | ---: | ---: | ---: |
| V1 | 0.642 | 0.762 | +0.120 |
| V2 | 0.614 | 0.700 | +0.086 |
| V3 | 0.591 | 0.674 | +0.083 |
| hV4 | 0.517 | 0.586 | +0.069 |

This is one of the strongest numbers in the project, yet it is method-provenance, not cross-model evidence. It should not be used as the primary cross-model ranking row because it is not method-matched to the other backbones.

Mean-reviewer attack:

> You improved DINOv2 with a better readout. That may say more about your readout than about DINOv2's brain-likeness.

Correct use: motivate Paper 2 or a controlled readout/intervention study, not inflate the Paper 1 model ranking.

---

### 4. Geometry: now present, still fragile

The matched geometry ranking by full-image linear CKA is:

| Rank | Model | Mean CKA score |
| ---: | --- | ---: |
| 1 | DINOv2 ViT-S/14 | 0.229 |
| 2 | ConvNeXt-T | 0.211 |
| 3 | ResNet-50 | 0.210 |
| 4 | DeiT-S/16 | 0.203 |
| 5 | CLIP ViT-B/16 | 0.188 |
| 6 | ViT-B/16 | 0.104 |

This is important because it creates the first real cross-axis tension:

- DINOv2 leads both encoding and CKA.
- CLIP ViT is near the top in encoding and weaker in CKA.
- ConvNeXt is weak in encoding but strong in CKA.
- Behavioral winners differ by dataset and attribution family.
- Encoding–geometry rank agreement appears weak, with reported Spearman around 0.257.

This is the first possible seed of a paper:

> Encoding and geometry may partially dissociate even under matched model/ROI/image controls.

With `n=6` and one subject, this is still a hypothesis.

Linear CKA alone is too brittle. CKA can be sensitive to irrelevant dimensions, so subset RSA or another representational metric should be used as a check. Geometry should not become a single-metric story.

---

## Does the current project adhere to the plan?

Partially.

It adheres to the revised Paper 1 plan more than the original proposal. The original proposal included adaptive/selective computation, Brain-Score-style evaluation, model scale, efficiency, and video. Those are mostly absent or deferred.

The current project has become a narrower observational cross-axis study:

- fixation alignment;
- neural encoding;
- representational geometry;
- controlled model panel.

That narrowing is reasonable because the original proposal was too broad for a first paper.

It does not yet satisfy the revised Paper 1 acceptance gate. Paper 1 is ready only if:

1. behavioral scoring is corrected;
2. matched neural panel is method-matched;
3. full-image geometry exists;
4. cross-axis analyses include uncertainty/sensitivity;
5. claims are descriptive rather than causal;
6. at least one nontrivial dissociation/convergence survives leave-one-model-out inspection;
7. the paper motivates Paper 2.

Current status satisfies the first three. It does not yet satisfy the uncertainty, robustness, subject replication, stronger controls, or robust dissociation requirements.

Honest status:

> Current project = promising diagnostic scaffold with early signals.

> Publication-grade Paper 1 = still unproven.

> Top-venue Paper 1 = currently no.

---

## Where the novelty actually is

The novelty is not in any individual component.

Grad-CAM, gradients, attention rollout, NSS, AUC, CKA, RSA, ridge encoding, center-bias baselines, DeepGaze baselines, and NSD/Algonauts-style encoding are all established. The paper must acknowledge that saliency/fixation metrics, DeepGaze-class baselines, Algonauts/NSD, Brain-Score-style evaluation, transformer attribution, and representational geometry already exist.

The novelty is in the joint question:

> Do models that look more human-like behaviorally also become more brain-like neurally and geometrically, or do these axes separate across architecture, training objective, attribution family, ROI, and task regime?

The strongest novelty claim should become:

> Human-like fixation alignment, neural encoding, and representational geometry are related but non-equivalent axes of visual alignment; using saliency-map similarity as a proxy for brain-likeness is empirically unsafe.

That story is the best likely Paper 1 contribution.

---

## What a top-venue reviewer would reject right now

### 1. "This is just a benchmark of known pieces."

The current behavioral results say DeepGaze beats center bias and classifier attribution maps are weaker. That is expected. The neural results say DINOv2/CLIP/ResNet are strong local encoders. Also expected. The CKA result is more interesting, yet it is one metric, one subject, six models.

### 2. "n=6 model-level correlations are not evidence."

Behavior–encoding and behavior–geometry correlations are descriptive. With `n=6`, a single model can flip the sign or dominate the slope. Use bootstrap/permutation uncertainty, leave-one-model-out sensitivity, and rank stability rather than treating p-values as serious.

### 3. "One subject is too limited."

The current neural evidence is `subj01` only. For a top conference, that is a severe limitation unless framed as a methods/demo paper. Subject replication is not optional if the paper makes claims about human visual cortex.

### 4. "Attribution is being confused with attention."

Human fixations, post-hoc explanation maps, and internal transformer routing are related objects, but they are not the same object. Grad-CAM localization, perturbation sensitivity, attention rollout attribution, and transformer relevance methods must be distinguished.

Chefer-style transformer attribution or AttnLRP should be added before making any transformer-attention claim.

### 5. "The behavioral baselines are below current saliency SOTA."

DeepGaze IIE is useful, yet DeepGaze MSDB is a stronger current reference. Human/interobserver ceilings are still missing or underused.

### 6. "The project still lacks the causal/adaptive-attention component that motivated the original idea."

The original proposal's most exciting part was selective computation: token pruning, foveation, recurrent glimpses, adaptive resolution, and efficiency. Current Paper 1 has almost none of that as actual outcomes. That is acceptable for Paper 1 only if Paper 1 is explicitly observational and motivates Paper 2.

---

## What should change now

### Immediate Paper 1 target

Do not frame Paper 1 as:

> We built a benchmark.

Do not frame it as:

> DINOv2 is most human-like.

Frame it as:

> A controlled cross-axis diagnostic showing that fixation alignment, neural encoding, and representational geometry only partially agree across modern vision models, so saliency alignment cannot be treated as a proxy for brain alignment.

The paper lives or dies on whether it can show a robust dissociation.

The current candidate dissociation/convergence pattern:

- DINOv2 leads encoding and CKA.
- CLIP ViT is strong in encoding but weaker in CKA.
- ConvNeXt is weak in encoding but high in CKA.
- Behavioral winners differ by dataset and attribution family.
- Encoding–geometry rank agreement appears weak, with reported Spearman around 0.257.

This is interesting. It is not yet robust enough.

---

## Required next analyses

### 1. Run uncertainty before adding many new models

Add:

- image bootstrap for behavioral metrics;
- target bootstrap for encoding;
- subset/bootstrap uncertainty for CKA/RSA;
- leave-one-model-out sensitivity for every cross-axis correlation.

Without this, the cross-axis story is statistically ornamental.

### 2. Add subject robustness

At minimum, run a reduced panel on `subj02`–`subj04`:

- DINOv2;
- CLIP ViT;
- ResNet-50;
- ConvNeXt or DeiT.

Use additional subjects as robustness checks rather than uncontrolled discovery fishing.

### 3. Add stronger behavioral controls

Prioritize:

- DeepGaze MSDB;
- human/interobserver ceiling;
- COCO-Search18 task-specific baseline.

These are more important than adding more generic classifier attribution rows.

### 4. Add transformer attribution beyond rollout

Add:

- Chefer-style transformer attribution; or
- AttnLRP.

Current attention rollout rows should be described as rollout attribution/internal-routing proxy, not human attention.

### 5. Finish geometry sensitivity

Linear CKA alone is too brittle.

Add:

- subset RSA at feasible sizes;
- repeat subset seeds;
- method agreement reporting;
- CKA-vs-RSA disagreement analysis.

### 6. Add ROI/anatomical expansion before huge model-zoo expansion

More brain regions are more scientifically valuable than adding ten more `timm` models. Anatomical axes are more likely to reveal meaningful convergence/dissociation than another flat model ranking.

### 7. Add efficiency only after the core matrix is stable

Efficiency was central in the original proposal, but a shallow FLOP table will look decorative.

Add only after behavior–encoding–geometry is stable:

- parameter count;
- FLOPs/MACs;
- latency;
- token count;
- alignment-per-compute.

---

## Revised project shape for top publication

### Paper 1: observational dissociation paper

Core claim:

> Fixation alignment, neural encoding, and representational geometry form separable axes of human–machine visual alignment.

Necessary evidence:

- Same model panel across all axes.
- Separate free-viewing and task-driven datasets.
- Dedicated saliency baselines and human ceilings.
- Multi-subject neural robustness.
- CKA plus RSA or another geometry check.
- Exact model-level `n`, leave-one-out plots, bootstrap confidence intervals.
- Specific dissociation examples, not just global correlations.

Realistic venue status:

| Target | Current plausibility |
| --- | --- |
| Workshop / thesis chapter | plausible soon |
| NeurIPS / ICLR / CVPR main | possible only if dissociation survives sensitivity and subject checks |
| Nature / Nature Machine Intelligence / PNAS | premature unless the neural claim becomes substantially stronger |

---

### Paper 2: causal adaptive-attention intervention

Core claim:

> Human gaze or adaptive selective computation changes alignment, efficiency, or neural predictivity in a controlled way.

Candidate interventions:

- gaze-guided token masking;
- human-saliency-guided token pruning;
- foveated input policy;
- saliency regularization loss;
- adaptive readout/sampling inspired by the DINOv2 learned-readout gain.

This is where the original proposal's strongest idea belongs.

---

## Final harsh assessment

The current project is not publication-grade as a top venue paper.

It has enough evidence to justify continuing, and it has moved past toy status, but the present outcomes are still mainly a controlled internal diagnostic.

The behavioral numbers are sane. The neural numbers are plausible. The geometry numbers are the first truly useful cross-axis addition. The novelty is still mostly latent, not yet demonstrated.

The project becomes publishable only if it proves a robust cross-axis dissociation or convergence pattern under:

- uncertainty;
- subject checks;
- stronger baselines;
- attribution controls;
- geometry-method sensitivity.

The planning has improved substantially by narrowing Paper 1 into a cross-axis dissociation study. The next improvement is to make the plan less like a completeness checklist and more like a falsifiable claim:

> Can fixation alignment predict neural encoding and representational geometry across models, ROIs, and task regimes?

If the answer is:

> No, and here is the robust pattern,

then there is a paper.

If the answer is:

> We implemented many metrics,

then there is infrastructure.

---

## Codex-facing implementation priorities

Use this as the immediate implementation order:

1. Freeze accepted behavioral aggregate and matched neural/geometry panel.
2. Implement uncertainty and sensitivity analysis:
   - image bootstrap;
   - target bootstrap;
   - subset geometry bootstrap;
   - leave-one-model-out;
   - permutation over model labels.
3. Add DeepGaze MSDB or current fixation SOTA control if feasible.
4. Add human/interobserver ceiling reporting for SALICON/CAT2000/COCO-Search18 where data allow.
5. Add Chefer-style or AttnLRP transformer attribution.
6. Add reduced subject robustness panel for `subj02`–`subj04`.
7. Expand ROI/anatomical scope before broad model-zoo expansion.
8. Add efficiency profile only after the central matrix and uncertainty reports are stable.
9. Produce final cross-axis decision tables that classify each finding as:
   - robust convergence;
   - robust dissociation;
   - unstable/descriptive only;
   - insufficient models.

---

## Claim hygiene rules

Do not claim:

- "DINOv2 is brain-like" from one-subject encoding.
- "Transformer attention is human attention" from attention rollout.
- "Fixation alignment causes neural alignment" from correlations.
- "The project is SOTA" from local internal splits.
- "DeepGaze/classifier attribution comparison is novel" by itself.
- "Encoding score proves representational equivalence" without geometry sensitivity.

Safer claims:

- The benchmark now produces sane behavioral fixation-alignment diagnostics.
- Dedicated fixation models outperform generic classifier explanation maps, as expected.
- The matched neural panel provides a local, controlled encoding baseline.
- DINOv2 and CLIP-like encoders are plausible strong neural baselines in the local panel.
- Learned spatial readout materially improves DINOv2 encoding but is not method-matched across models.
- Initial geometry results suggest possible dissociation between encoding and representational geometry.
- Strong publication claims require uncertainty, subject robustness, stronger baselines, and transformer attribution controls.
