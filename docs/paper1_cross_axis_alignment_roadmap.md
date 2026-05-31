# Paper 1 Roadmap — Cross-Axis Diagnostic of Human–Machine Visual Alignment

**Working title:** *Dissociating Fixation Alignment, Neural Encoding, and Representational Geometry in Modern Vision Models*  
**Role in program:** observational foundation / diagnostic map  
**Intended venue tier:** NeurIPS / ICLR / ICML workshop as first realistic target; top-tier main conference only if subject replication, uncertainty, geometry, and a crisp dissociation result are strong.  
**Status basis:** builds directly from the current `project_status_and_next_steps.md` state dated 2026-05-27.

---

## 0. Executive judgment

The current project has become scientifically legitimate, but it is still underpowered as a top-venue main paper if framed as a benchmark. Its best path is to become a **cross-axis dissociation study** rather than a saliency leaderboard or local Algonauts imitation.

The paper should ask:

> Do models that better match human fixation behavior also better align with human visual cortex in encoding and representational geometry, or do these alignment axes dissociate across architecture, attribution family, ROI, and viewing regime?

The current repository already supports the first half of this paper:

- corrected behavioral fixation benchmarking on SALICON, CAT2000, and COCO-Search18;
- DeepGaze and center-bias sanity controls;
- a complete six-model matched full-image-count neural encoding panel for `subj01` PRF ROIs `V1`, `V2`, `V3`, and `hV4`;
- target-level noise-normalized encoding scores;
- validation-only layer/pooling selection;
- first behavior-versus-encoding cross-level tables.

The missing pieces are the ones that decide whether the result is publishable:

1. matched full-image representational geometry;
2. uncertainty estimates;
3. stronger statistical framing for small model-level `n`;
4. subject expansion or at minimum robustness checks;
5. behavioral SOTA controls and human ceilings;
6. an explicit dissociation narrative.

This paper should **not** claim that human-like attention causes better neural alignment. That belongs to Paper 2.

---

## 1. Core thesis

### Strong version

Modern vision models do not lie on a single “human-likeness” axis. Fixation alignment, visual-cortex encoding, and representational geometry can converge in some model families and dissociate in others. Therefore, saliency-like behavior, neural predictivity, and latent-space brain-likeness must be evaluated jointly and under matched controls.

### Reviewer-safe version

We introduce a controlled diagnostic framework for comparing fixation alignment, neural encoding, and representational geometry across modern vision models. Using matched model panels, corrected fixation protocols, and reliability-aware fMRI encoding scores, we find evidence that behavioral fixation similarity and neural alignment are partially dissociable.

### Claims to avoid

- “This model is brain-like.”
- “Transformer attention is human attention.”
- “Attribution/fixation similarity proves cognitive plausibility.”
- “Our fMRI scores are Algonauts-leaderboard-equivalent.”
- “Human-like saliency causes better neural alignment.”
- “COCO-Search18 and free-viewing datasets measure the same behavior.”

---

## 2. Literature positioning

### 2.1 What the field already has

The paper must openly acknowledge that its ingredients are established:

- Saliency/fixation benchmarks already use NSS, AUC variants, CC, SIM, KL, center bias, and DeepGaze-class baselines.
- DeepGaze III models free-viewing scanpaths by conditioning next fixation prediction on image information and previous fixation history.
- DeepGaze MSDB highlights dataset bias and reports state-of-the-art saliency performance across MIT/Tuebingen benchmark datasets.
- Algonauts 2023 / NSD established large-scale fMRI response prediction on natural images.
- Brain-Score-style platforms already evaluate neural and behavioral alignment across model benchmarks.
- AttnLRP and related attribution work show that transformer attribution should not be reduced to raw attention maps or vanilla gradients.
- Neural encoding with visual attention has already shown that gaze-aware or learned attention modules can improve fMRI prediction.

### 2.2 Paper 1’s novelty

Paper 1 is novel only if it contributes a **joint dissociation analysis** under matched controls:

1. Same model panel across fixation, encoding, and geometry.
2. Explicit separation of free-viewing fixation and task-driven search.
3. Explicit separation of explanation maps from operational attention.
4. Matched full-image neural encoding and full-image representational geometry.
5. A structured answer to whether fixation alignment predicts neural alignment.
6. A transition from observational findings to causal intervention in Paper 2.

### 2.3 What this paper is not

It is not:

- a new saliency model;
- a new Algonauts leaderboard method;
- a general claim about all brain areas;
- a claim about video or visual memory;
- a causal intervention study.

---

## 3. Current assets from the repository

### 3.1 Behavioral layer

Already usable:

- `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- `378` corrected behavioral rows.
- `126` rows each for SALICON, CAT2000, and COCO-Search18.
- `points` / `task_points` fixation protocols; no blank or density-fallback rows.
- Corrected DeepGaze > center-bias sanity check across all datasets.

Current behavioral headlines:

- SALICON NSS: DeepGaze `1.743`, center bias `0.933`.
- CAT2000 NSS: DeepGaze `1.838`, center bias `1.619`.
- COCO-Search18 NSS: DeepGaze `1.745`, center bias `1.310`.

Use these as sanity checks, not central novelty.

### 3.2 Neural encoding layer

Already strong enough for local diagnostic analysis:

- `outputs/neural_roi_summary/`
- `120` encoding rows.
- `289740` target-level encoding rows.
- Complete matched six-model full-image-count `flatten_pca` panel:
  - `resnet50`
  - `convnext_tiny`
  - `deit_small_patch16_224`
  - `vit_base_patch16_224`
  - `vit_small_patch14_dinov2`
  - `vit_base_patch16_clip_224`
- Four PRF ROIs:
  - `V1`
  - `V2`
  - `V3`
  - `hV4`
- Matched panel audit:
  - `24` complete cells
  - `0` missing cells
  - `0` skipped cells

Accepted matched-panel ranking:

1. `vit_small_patch14_dinov2`
2. `vit_base_patch16_clip_224`
3. `resnet50`
4. `deit_small_patch16_224`
5. `vit_base_patch16_224`
6. `convnext_tiny`

### 3.3 Learned-readout provenance

DINOv2 learned spatial readout materially improves over DINOv2 `flatten_pca` across all four ROIs.

Use this as:

- evidence that spatial readout design matters;
- a method-provenance result;
- a justification for Paper 2’s adaptive sampling/readout focus.

Do not use it as the primary cross-model ranking row because it is not method-matched to other backbones.

### 3.4 Cross-level tables

Already implemented:

- `matched_cross_level_observations.csv`
- `matched_cross_level_correlations.csv`

Current issue:

- model-level correlations are based on `n=6` models or `n=4` transformer-only rows.
- They are descriptive, not inferential.

---

## 4. Main weaknesses a mean reviewer would attack

### Weakness 1 — “This is a benchmark made of known components.”

Response strategy:

- Do not market it as a benchmark.
- Market it as a dissociation analysis across alignment axes.
- Emphasize controlled matched-panel design and negative/partial-convergence findings.

### Weakness 2 — “Small model-level n makes correlations unreliable.”

Response strategy:

- Add bootstrap/permutation uncertainty.
- Treat model-level correlations as descriptive.
- Use rank stability, leave-one-model-out sensitivity, and sign consistency across ROIs/datasets.
- Avoid overinterpreting `n=4` attention-rollout correlations.

### Weakness 3 — “One subject is too limited.”

Response strategy:

- Add at least one additional NSD/Algonauts subject if feasible.
- If full subject expansion is too expensive, run a reduced matched panel for two additional subjects on fewer models:
  - DINOv2
  - CLIP ViT
  - ResNet-50
  - one weaker transformer baseline
- Report subject-level robustness as a confirmatory supplement.

### Weakness 4 — “Attribution is not attention.”

Response strategy:

- Rename all rows carefully:
  - “gradient attribution”
  - “Grad-CAM localization”
  - “attention rollout attribution”
  - “perturbation sensitivity”
- Add AttnLRP or Chefer-style transformer attribution before making transformer-specific claims.
- Explicitly say that these are explanation-map-to-fixation comparisons, not direct attention mechanisms.

### Weakness 5 — “Encoding scores do not prove representational alignment.”

Response strategy:

- Add full-image subset RSA and linear CKA.
- Keep encoding and geometry separate.
- Report cases where encoding and CKA disagree.

### Weakness 6 — “DeepGaze and center bias are not enough.”

Response strategy:

- Add DeepGaze MSDB if feasible.
- Add human inter-observer ceiling / leave-one-subject-out fixation ceiling for SALICON/CAT2000 where data allow.
- Add task-specific COCO-Search18 baseline if available.

---

## 5. Required final experiments for Paper 1

### Experiment 1 — Behavioral fixation alignment sanity and family comparison

Goal:

- Establish corrected fixation alignment axis.
- Separate free-viewing and task-driven viewing.
- Avoid treating attribution methods as saliency-prediction models.

Required outputs:

- Dataset × method × model metric table.
- Separate NSS/AUC and map-distribution metrics.
- Free-viewing figure:
  - SALICON
  - CAT2000
- Task-driven figure:
  - COCO-Search18
- DeepGaze/center-bias sanity figure.
- Human ceiling if available.

Acceptance criteria:

- DeepGaze beats center bias in point-based metrics.
- No density-fallback rows in headline results.
- No pooling of COCO-Search18 with free-viewing datasets.

Engineering tasks:

1. Freeze corrected aggregate.
2. Add `analysis_scope` column:
   - `free_viewing`
   - `task_search`
3. Add `method_object_type`:
   - `dedicated_fixation_model`
   - `center_prior`
   - `posthoc_attribution`
   - `perturbation_sensitivity`
   - `random_control`
4. Add human ceiling / interobserver ceiling module if data support it.
5. Add DeepGaze MSDB precomputed maps if feasible.

---

### Experiment 2 — Matched neural encoding panel

Goal:

- Establish reliability-aware local neural alignment across models and ROIs.

Required outputs:

- Matched-panel neural ranking.
- ROI-specific rankings.
- Layer-selection audit.
- Noise-ceiling distribution and valid-target coverage.
- Learned-readout provenance table.

Acceptance criteria:

- Headline uses only method-matched `flatten_pca` rows.
- Learned DINOv2 readout is marked as provenance/upper-bound.
- Zero-ceiling hV4 targets are excluded from noise-normalized aggregates.
- No mixed-scope ranking in headline.

Engineering tasks:

1. Freeze `matched_full_panel_model_rankings.csv`.
2. Add confidence intervals across targets for ROI scores.
3. Add leave-one-ROI-out ranking stability.
4. Add subject expansion if feasible.
5. Keep validation-only selection artifacts in supplement.

---

### Experiment 3 — Matched full-image representational geometry

Goal:

- Add the missing latent-space axis.

Candidate methods:

1. Linear CKA between model features and neural responses.
2. Subset RSA with deterministic image subsets.
3. Procrustes / crossnobis-style representational comparison if feasible.

Recommended implementation:

- Implement linear CKA first because it is scalable and avoids full `9841 × 9841` RDM allocation.
- Implement subset RSA second as a complementary check.
- Use deterministic image subsets:
  - fixed seed;
  - sample sizes: `512`, `1024`, `2048`;
  - repeat subsets for uncertainty.

Engineering tasks:

1. Add `src/hma/neural/geometry.py`.
2. Reuse selected layers and feature-reduction metadata from matched panel.
3. Save `geometry_scores.csv` beside `encoding_scores.csv`.
4. Columns:
   - `model_name`
   - `roi`
   - `subject`
   - `num_images_total`
   - `num_images_used`
   - `geometry_method`
   - `subset_seed`
   - `subset_size`
   - `model_feature_source`
   - `neural_response_source`
   - `score`
   - `valid`
5. Extend `summarize_neural_roi_results.py`:
   - `matched_geometry_scores.csv`
   - `matched_geometry_model_rankings.csv`
   - `matched_geometry_roi_rankings.csv`
6. Extend paper inspection pack:
   - geometry ranking table;
   - behavior–geometry cross-level table;
   - encoding–geometry table.

Acceptance criteria:

- All `24` matched model × ROI cells have geometry scores.
- At least two subset sizes show qualitatively stable rankings.
- CKA and subset RSA either agree or reveal interpretable disagreement.

---

### Experiment 4 — Cross-axis dissociation analysis

Goal:

- Make the paper’s core claim.

Analyses:

1. Behavior vs encoding.
2. Behavior vs geometry.
3. Encoding vs geometry.
4. Behavior vs encoding controlling for model family.
5. Free-viewing vs task-search comparison.
6. ROI-specific cross-axis patterns.
7. Attribution-family-specific patterns.

Required statistics:

- Spearman correlation.
- Kendall tau as robustness.
- Leave-one-model-out sensitivity.
- Bootstrap over images for behavioral scores.
- Bootstrap over neural targets for encoding scores.
- Bootstrap/subset uncertainty for geometry.
- Permutation test over model labels.
- Report exact `n`.

Do not overclaim p-values. With `n=6`, the most honest result is pattern-level evidence with sensitivity analysis.

Acceptance criteria:

- Paper can identify at least one robust dissociation:
  - e.g. a model high in neural encoding but modest in fixation alignment;
  - or a method high in fixation similarity but unrelated to geometry;
  - or ROI-specific reversal.
- If no robust dissociation appears, the paper becomes weaker and should be reframed as a measurement framework plus null/negative result.

---

### Experiment 5 — Efficiency axis

Goal:

- Recover original proposal axis without overloading the paper.

Minimal metrics:

- parameter count;
- FLOPs or MACs;
- latency on a fixed GPU/CPU;
- number of visual tokens;
- retained tokens for attribution/masking variants if applicable;
- memory footprint.

Engineering tasks:

1. Add `scripts/profile_model_efficiency.py`.
2. Use fixed image resolution and batch size.
3. Save `outputs/efficiency_profiles/model_efficiency.csv`.
4. Merge into cross-axis tables.
5. Add alignment-per-compute exploratory figure.

Acceptance criteria:

- Efficiency is exploratory unless it produces a clean result.
- Do not let efficiency delay geometry and uncertainty.

---

## 6. Paper 1 structure

### Abstract

- Problem: fixation, neural encoding, and representational geometry are often treated as interchangeable signs of human-like vision.
- Method: matched cross-axis diagnostic across modern vision models.
- Result: behavioral and neural axes partially dissociate.
- Implication: observational benchmarking is insufficient; causal adaptive-attention interventions are needed.

### Introduction

1. Modern vision models are increasingly compared to human perception and brain activity.
2. Fixation prediction, fMRI encoding, and representational geometry are usually studied separately.
3. It is unclear whether human-like saliency/fixation alignment predicts neural alignment.
4. We provide a matched diagnostic across behavioral, neural, and latent-space axes.
5. The study motivates causal adaptive-attention tests.

### Related work

- Saliency prediction and DeepGaze.
- Scanpath modeling.
- Dataset bias in saliency.
- Brain-Score and multi-benchmark NeuroAI.
- NSD / Algonauts.
- Neural encoding with attention.
- Transformer attribution and AttnLRP.
- Representational similarity / CKA.

### Methods

- Datasets.
- Models.
- Fixation metrics.
- Attribution methods.
- Neural encoding.
- Noise ceilings.
- Representational geometry.
- Cross-axis statistics.
- Uncertainty.

### Results

1. Behavioral fixation sanity checks.
2. Matched neural encoding panel.
3. Matched representational geometry.
4. Cross-axis convergence/dissociation.
5. Efficiency exploratory analysis.
6. Failure modes and negative results.

### Discussion

- Fixation similarity is not a sufficient proxy for neural alignment.
- Encoding and geometry measure different aspects of brain-likeness.
- Attribution maps must not be equated with operational attention.
- Results motivate adaptive-attention interventions.

### Limitations

- One/few subjects.
- PRF visual ROIs only.
- Static images only.
- Model-level `n` small.
- Attribution methods imperfect.
- No causal manipulation.

---

## 7. Figure plan

### Figure 1 — Conceptual design

Axes:

- fixation alignment;
- neural encoding;
- representational geometry;
- efficiency.

Message:

- A model can be high on one axis and low on another.

### Figure 2 — Behavioral sanity and method families

- DeepGaze vs center bias vs attribution families.
- Separate SALICON/CAT2000 from COCO-Search18.

### Figure 3 — Matched neural encoding panel

- Six models × four ROIs.
- Noise-normalized scores.
- Learned DINOv2 readout shown as provenance/upper-bound.

### Figure 4 — Matched geometry panel

- CKA/subset RSA across same six models and ROIs.

### Figure 5 — Cross-axis matrix

- Behavior vs encoding.
- Behavior vs geometry.
- Encoding vs geometry.
- Confidence intervals and leave-one-model-out sensitivity.

### Figure 6 — Dissociation examples

- Specific models/methods that break the assumption that fixation alignment equals neural alignment.

### Supplementary figures

- Layer selection.
- Noise ceiling distributions.
- Bootstrap stability.
- Metric correlations.
- Efficiency profiles.
- Dataset-specific behavior.

---

## 8. Engineering implementation order

### Phase 1 — Freeze current assets

1. Tag current repo state.
2. Freeze corrected behavioral aggregate.
3. Freeze matched neural panel.
4. Add `paper1_config.yaml` listing exactly which files are accepted.

### Phase 2 — Geometry

1. Implement scalable CKA.
2. Implement deterministic subset RSA.
3. Add tests.
4. Run all `24` model × ROI cells.
5. Regenerate neural summary and paper pack.

### Phase 3 — Uncertainty

1. Behavioral bootstrap over images.
2. Neural bootstrap over targets.
3. Geometry bootstrap over image subsets.
4. Leave-one-model-out sensitivity.
5. Update cross-level tables.

### Phase 4 — Controls

1. Add DeepGaze MSDB if feasible.
2. Add human interobserver ceiling if data allow.
3. Add AttnLRP or Chefer-style attribution for transformers.
4. Add efficiency profiles.

### Phase 5 — Subject robustness

1. Choose minimal subject-expansion plan.
2. Run reduced matched panel if full expansion is too expensive.
3. Report subject-level sign/rank stability.

### Phase 6 — Paper pack

1. Generate final tables.
2. Generate publication figures.
3. Write manuscript skeleton.
4. Create reviewer-attack checklist.
5. Archive config, seeds, and exact outputs.

---

## 9. Acceptance gate for Paper 1

Paper 1 is ready only if all are true:

- Behavioral pipeline is corrected and sanity-checked.
- Matched neural panel is complete and method-matched.
- Full-image geometry exists for all headline model × ROI cells.
- Cross-axis analyses include uncertainty and sensitivity.
- Claims are framed as descriptive/dissociative, not causal.
- At least one nontrivial dissociation or convergence pattern is robust enough to survive leave-one-model-out inspection.
- The paper explicitly motivates Paper 2.

If this gate fails, submit as a workshop / thesis chapter / methods note, then shift effort to Paper 2.

---

## 10. Decision rules

### Continue Paper 1 to top venue only if:

- geometry adds a genuinely new pattern;
- subject or bootstrap robustness is credible;
- the dissociation story is clear;
- the paper can say more than “we correlated known metrics.”

### Stop expanding Paper 1 if:

- new additions only improve completeness without changing the story;
- the paper starts becoming a leaderboard;
- implementation work delays Paper 2’s intervention.

### Best likely Paper 1 contribution

> Human-like fixation alignment, neural encoding, and latent representational geometry are related but non-equivalent axes of visual alignment; treating saliency-map similarity as a proxy for neural brain-likeness is empirically unsafe.

That is the story to prove.
