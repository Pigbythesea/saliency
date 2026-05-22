# HMA Project Status And Next Steps

Updated: 2026-05-20

## Purpose

This document is a handoff for a fresh Codex session. It should answer:

- what the project currently does,
- which outputs are trustworthy,
- which outputs are diagnostic or superseded,
- where the relevant code lives,
- what the next concrete implementation milestone is.

It is intentionally not a full experiment diary. Older pilot results and round-by-round implementation notes were removed because they made the next action harder to see.

## Current Snapshot

The repository implements a human-machine visual alignment benchmark with three active layers:

- Behavioral saliency / fixation benchmarking on SALICON, CAT2000, and COCO-Search18.
- Neural ROI500 encoding and RSA diagnostics on local Algonauts / NSD `subj01` visual ROIs.
- Paper-style inspection tables and figures that bridge behavioral and neural summaries.

The main package is `src/hma/`. The main scripts are under `scripts/`. Current result packs are under `outputs/`.

Important current outputs:

- Core behavioral aggregate: `outputs/real_matrix_v2/aggregated/results.csv`.
- Behavioral aggregate merged with partial SSL/VLM rows: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- SSL/VLM behavioral aggregate: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`.
- Neural ROI summary: `outputs/neural_roi_summary/`.
- Inspection pack: `outputs/paper_inspection_v1/README.md`.

## Scientific Boundary

The behavioral pipeline has been fixed so NSS and AUC-style metrics can use raw fixation points instead of thresholded fixation-density maps. The relevant implementation is in:

- `src/hma/metrics/saliency_metrics.py`
- `src/hma/experiments/saliency_benchmark.py`
- `src/hma/experiments/aggregate_results.py`
- `src/hma/saliency/postprocess.py`
- `src/hma/saliency/precomputed.py`

However, much of the current static2000 aggregate was generated before that protocol fix. Treat those rows as superseded for scientific NSS/AUC claims unless `fixation_protocol` is explicitly `points` or `task_points`.

Current behavioral status:

- `outputs/real_matrix_v2/aggregated/results.csv` has blank / pre-fix `fixation_protocol` metadata and should not be used for academic NSS/AUC claims.
- `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv` contains a mix of old rows plus newer SSL/VLM rows. Some newer rows have `points` or `task_points`, but many still have `unknown`.
- The inspection pack is useful for checking pipeline shape and bridge reporting, but not for final claims about model fixation prediction.
- The table where DINOv2 attention rollout appears to beat local DeepGaze on CAT2000 is a red flag about mixed / stale protocols, not a publishable result.

Current neural status:

- ROI500 encoding and RSA have been run for `resnet50`, `convnext_tiny`, `deit_small_patch16_224`, `vit_base_patch16_224`, and `vit_small_patch14_dinov2`.
- These are one-subject, ROI500, raw-correlation diagnostics. They are not Algonauts leaderboard-equivalent noise-normalized scores.
- Current ranking in `outputs/neural_roi_summary/neural_model_rankings.csv`: DeiT-S/16 leads mean encoding; ViT-B/16 leads mean RSA; DINOv2 ViT-S/14 ranks second on both raw summaries.

## Methods Audit Findings

The current score mismatch is more consistent with benchmark-protocol / pipeline mismatch than with ordinary dataset variance.

Behavioral saliency:

- The old static2000 rows likely used density-derived positives for NSS/AUC in at least some runs. If a blurred fixation-density map is thresholded into positives, many pixels become "fixations", which compresses NSS toward the normalized map average and can make strong models look weak.
- This explains the observed pattern where local SALICON DeepGaze has plausible density alignment (`CC` about `0.802`) but implausibly low NSS (`0.435`), compared with published DeepGaze IIE SALICON NSS about `1.996` and CC about `0.872`.
- CAT2000 is the strongest red flag: MIT/Tuebingen reports CAT2000 center-bias NSS about `2.087` and DeepGaze IIE NSS about `2.112`, while local stale rows report center bias `0.519` and DeepGaze `0.275`.
- COCO-Search18 is task-driven search, not free viewing. Published task-trained search saliency reports NSS `4.64`, AUC-Judd `0.95`, sAUC `0.84`, CC `0.72`, and SIM `0.54`. Local COCO-Search18 rows around NSS `0.55` to `0.93` should remain diagnostic until task-specific protocol and baselines are validated.
- Academic saliency SOTA separates fixation-point metrics from density metrics: NSS/AUC/sAUC use fixation locations, while CC/SIM/KL use empirical density maps. Probabilistic models such as DeepGaze also use metric-appropriate prediction maps; for example, sAUC may divide predicted density by an average other-image density.
- The current repo now has the right implementation direction for point-based NSS/AUC, but existing stale outputs do not prove the corrected path works until references are rerun.

Behavioral method strength:

- The benchmark scaffold is useful, but it is not yet a SOTA-equivalent saliency benchmark.
- To make paper-level fixation-prediction claims, the pipeline should recover known reference behavior first: center bias and DeepGaze must land in a sensible range and ordering before interpreting model saliency rows.
- XAI maps such as Grad-CAM, vanilla gradients, and attention rollout are not SOTA fixation-prediction models. They are valid probes for "do explanation maps resemble fixations?", but they should not be framed as direct competitors to DeepGaze-class saliency predictors.
- A stronger SOTA-comparable behavioral layer should eventually use official or near-official evaluation code, preferably `pysaliency` / MIT-Tuebingen-style semantics, information gain, learned/KDE center bias, metric-specific DeepGaze maps, and human/inter-observer controls.

Neural alignment:

- The current ROI500 neural path is scientifically recognizable as a small linear encoding/RSA scaffold, but it is not benchmark-equivalent to Algonauts or strong NSD papers.
- Algonauts 2023 evaluates held-out test predictions by correlating predicted and measured fMRI per vertex, squaring the correlations, normalizing by each vertex noise ceiling, and averaging across selected vertices from all subjects. The local ROI500 rows instead report raw mean Pearson correlations for one subject, selected ROIs, and an internal 400/100 split.
- Current raw correlations such as DeiT mean encoding `0.261` are plausible internal diagnostics, but they should not be compared numerically with Algonauts leaderboard percentages, organizer baseline scores, or published variance-explained values.
- Stronger NSD / brain-alignment papers typically use cross-validated regularization, feature reduction or PCA, noise ceilings, held-out splits, subject-level aggregation, ROI or vertex reliability filtering, and often multiple layers/features/ensembles. Recent large-scale work also warns that flexible linear mappings can make many architectures look similarly predictive, so claims should emphasize controlled comparisons and dissociations rather than simple architecture winners.

Neural method strength:

- The current neural method is strong enough to verify data loading, activation extraction, ROI response alignment, and rough model ranking.
- It is not yet strong enough for final scientific claims about model-brain SOTA, architecture superiority, or Algonauts-level prediction quality.
- The next neural upgrade should add benchmark-equivalent metrics and validation controls before expanding to many more models.

## Dataset And Manifest State

Behavioral manifests:

- SALICON full: `data/manifests/salicon_manifest.csv`
- CAT2000 full: `data/manifests/cat2000_manifest.csv`
- COCO-Search18 full: `data/manifests/coco_search18_manifest.csv`
- Static2000 manifests: `data/manifests/v2/*_static2000_manifest.csv`
- Pilot500 manifests: `data/manifests/pilot/*_pilot500_manifest.csv`

SALICON and CAT2000 manifests include `fixation_points_path` when raw fixation files are available. COCO-Search18 uses task / scanpath fixation points.

Neural data:

- Local Algonauts / NSD subject data: `data/raw/nsd_algonauts/subj01/`
- ROI500 configs: `configs/experiments/neural_roi500/`
- DINOv2 ROI500 configs: `configs/experiments/neural_roi500_ssl/`

## What Is Already Built

Behavioral infrastructure:

- Manifest loaders for SALICON, CAT2000, COCO-Search18, and NSD / Algonauts-style data.
- Fixation parsers for SALICON and CAT2000 `.mat` files.
- Static metrics: NSS, AUC-Judd, AUC-Borji, shuffled AUC, CC, SIM, KL, EMD, MAE, Pearson.
- Saliency methods: center bias, random saliency, gradient, integrated gradients, Grad-CAM, attention rollout, occlusion, and precomputed DeepGaze-style maps.
- Matrix execution, aggregation, summaries, plots, and paper inspection pack generation.

Neural infrastructure:

- `timm` wrappers with named-layer activation extraction.
- Ridge encoding and RSA over ROI response vectors.
- ROI500 summaries, model rankings, ROI winners, efficiency-normalized summaries, and behavior-neural bridge tables.

SSL / multimodal additions:

- Candidate inspection for DINOv2, DINOv3, CLIP, SigLIP, and EVA-CLIP model names.
- Pretrained debug runs completed for `vit_small_patch14_dinov2`, `vit_base_patch16_clip_224`, and `resnet50_clip`.
- Full ROI500 completed for `vit_small_patch14_dinov2`.
- Partial behavioral SSL/VLM rows completed, especially DINOv2 static2000 gradient and attention-rollout rows.

## Next Concrete Milestone

Priority: **Corrected Behavioral Reference Rerun V1**.

Do this before adding more model families, CLIP rows, selective-computation methods, Brain-Score, CKA, or video. The current blocker is not more coverage; it is that the behavioral reference scale must be made trustworthy under the corrected fixation-point protocol.

Goal:

- Regenerate center-bias and DeepGaze reference rows under the corrected protocol.
- Confirm static2000 NSS/AUC rows report `fixation_protocol=points` for SALICON/CAT2000 and `task_points` for COCO-Search18.
- Verify DeepGaze and center-bias ranges no longer look obviously inconsistent with literature sanity checks.
- Only then rerun the model saliency rows and regenerate bridge / inspection outputs.

### Step 1: Re-export DeepGaze Maps Where Needed

CAT2000 and COCO-Search18 need collision-safe `{map_key}.npy` filenames because static manifests can reuse `image_id` values. These commands skip existing maps unless `--overwrite` is added.

```cmd
.\.venv\Scripts\python.exe scripts\export_deepgaze_maps.py --manifest data\manifests\v2\cat2000_static2000_manifest.csv --image-root data\raw\CAT2000 --output-dir data\precomputed\deepgaze\cat2000_static2000 --filename-template "{map_key}.npy"
.\.venv\Scripts\python.exe scripts\export_deepgaze_maps.py --manifest data\manifests\v2\coco_search18_static2000_manifest.csv --image-root data\raw\COCO-Search18 --output-dir data\precomputed\deepgaze\coco_search18_static2000 --filename-template "{map_key}.npy"
```

SALICON reference configs still use `{image_id}.npy`.

### Step 2: Rerun Static2000 References

Run the center-bias baselines and DeepGaze references first.

```cmd
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2\salicon_static2000__center_bias_baseline_center_bias.yaml --progress-interval 50
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2\cat2000_static2000__center_bias_baseline_center_bias.yaml --progress-interval 50
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2\coco_search18_static2000__center_bias_baseline_center_bias.yaml --progress-interval 50
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\salicon_static2000__deepgaze_reference_deepgaze_precomputed.yaml --progress-interval 50
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\cat2000_static2000__deepgaze_reference_deepgaze_precomputed.yaml --progress-interval 50
.\.venv\Scripts\python.exe scripts\run_saliency_benchmark.py --config configs\experiments\real_matrix_v2_references\coco_search18_static2000__deepgaze_reference_deepgaze_precomputed.yaml --progress-interval 50
```

### Step 3: Re-aggregate And Inspect Reference Rows

```cmd
.\.venv\Scripts\python.exe scripts\aggregate_results.py outputs\real_matrix_v2 --output outputs\real_matrix_v2\aggregated\results.csv --plots-dir outputs\real_matrix_v2\aggregated\results_plots --efficiency-csv outputs\real_matrix_v2\efficiency\model_efficiency.csv
.\.venv\Scripts\python.exe scripts\create_paper_inspection_pack.py --behavioral-csv outputs\real_matrix_v2\aggregated\results.csv
```

Check:

- `outputs/paper_inspection_v1/tables/table6_benchmark_sanity_ranges.md`
- `outputs/paper_inspection_v1/tables/table1_behavior_static2000_nss_top.md`
- `outputs/real_matrix_v2/aggregated/results.csv`

At this point, inspect only the rerun reference rows. The aggregate will still contain stale model rows until those specific output directories are regenerated.

Acceptance criteria for this milestone:

- Static2000 center-bias and DeepGaze rows have nonblank `fixation_protocol`.
- SALICON and CAT2000 reference rows use `points`.
- COCO-Search18 reference rows use `task_points`.
- DeepGaze / center-bias ordering and NSS scale are no longer obvious protocol failures.
- If the reference sanity check fails, fix protocol or map-loading code before running more model rows.

If reference sanity still fails, inspect in this order:

1. Confirm `per_image_metrics.csv` rows have `fixation_protocol=points` or `task_points`; if not, debug manifest `fixation_points_path`, dataset loaders, and `_fixation_coords_and_protocol_for_item`.
2. Confirm SALICON/CAT2000 fixation parsers return true `(x, y)` points and that image resizing scales points with the same width/height convention as the resized image.
3. Confirm DeepGaze precomputed maps are loaded from the intended file, especially CAT2000 / COCO-Search18 `{map_key}.npy` paths.
4. Confirm DeepGaze exported maps are probabilities when used for NSS/CC/SIM/KL, not log densities unless the metric explicitly expects log densities.
5. Confirm reference maps and target fixation maps are resized to the same target shape with no unintended min-max normalization before metric-specific normalization.
6. Compare a tiny hand-inspected subset against `pysaliency` or MIT/Tuebingen-style metric code before rerunning the full matrix.

### Step 4: Rerun Model Rows Only After References Pass

After reference sanity is acceptable, rerun the core static2000 model rows in `configs/experiments/real_matrix_v2/`, then re-merge SSL rows only if they were also produced under corrected protocol.

Useful script:

```cmd
.\.venv\Scripts\python.exe scripts\run_v2_matrix.py --config-dir configs\experiments\real_matrix_v2 --output-root outputs\real_matrix_v2 --phase static2000 --resume --progress-interval 50
```

Do not treat `--resume` as sufficient if an existing output was generated before the protocol fix. For corrected reruns, delete or overwrite only the specific stale output directories that need regeneration, and do not remove unrelated user work.

Behavioral paper-readiness checklist after model reruns:

- Report NSS/AUC only from rows with `points` or `task_points`.
- Report CC/SIM/KL separately as density-map metrics.
- Keep center bias, random, DeepGaze, and inter-observer or leave-one-observer-out controls visible in every headline table.
- Separate "fixation prediction" claims from "explanation-map similarity to fixations" claims. Grad-CAM, gradients, and rollout belong to the latter unless trained/evaluated as saliency predictors.
- For COCO-Search18, avoid comparing free-viewing saliency models directly with task-trained search models unless the task/category protocol is aligned.

## Later Milestones

After corrected behavioral rows are available:

1. Regenerate `results_with_ssl_behavior.csv`, neural bridge summaries, and `outputs/paper_inspection_v1/`.
2. Complete remaining CLIP / VLM behavioral static2000 rows if the bridge needs broader multimodal coverage.
3. Upgrade the neural pipeline to benchmark-equivalent metrics before making model-brain SOTA claims.
4. Run full ROI500 for CLIP or SigLIP candidates only after their behavioral rows are useful and the neural metric upgrade is specified.
5. Add Brain-Score / external neural comparisons.
6. Add CKA or deeper representational geometry summaries.
7. Add selective-computation models such as token pruning, foveation, adaptive patch selection, or glimpse-style models.
8. Consider video only after static-image behavioral and neural claims are stable.

## Neural Pipeline Upgrade Directions

Priority: **Neural Benchmark-Equivalent Evaluation V1**, after behavioral reference sanity is fixed.

Goal:

- Keep the current ROI500 raw-correlation path as an internal diagnostic.
- Add a separate benchmark-style neural report that is explicit about metric, split, subject coverage, ROI/vertex coverage, and noise-ceiling normalization.

Recommended next steps:

1. Add a neural metric report that converts raw vertex correlations into squared correlations and, where available, noise-normalized scores.
2. Add noise-ceiling or reliability metadata for vertices/ROIs, or explicitly mark runs as non-noise-normalized when those values are unavailable.
3. Replace fixed `ridge_alpha=1.0` with cross-validated ridge regularization inside the training split.
4. Add feature dimensionality control, such as PCA on training features only, to avoid unstable high-dimensional fits.
5. Preserve or test spatial information for early visual ROIs. The current `spatial_mean` reduction is convenient, but it discards retinotopic layout and may be too weak for V1/V2 claims.
6. Use held-out test-like splits that mimic Algonauts where possible, and clearly separate internal validation splits from official challenge-equivalent evaluation.
7. Expand beyond `subj01` before making subject-general claims.
8. Report ROI-specific and model-level confidence intervals via bootstrap over images, vertices, or subjects as appropriate.
9. Add controls: random/untrained model features, low-level image features, center/edge statistics, and at least one known strong CLIP/DINO/vision-language baseline.
10. Only compare to Algonauts leaderboard or NSD SOTA papers after the metric is squared, noise-normalized, and matched in subject/vertex/test-split scope.

Neural paper-readiness boundary:

- Current ROI500 tables can support statements like "the local diagnostic pipeline ranks these models this way under raw correlation."
- They cannot support statements like "DeiT is more brain-aligned than DINOv2/CLIP/SOTA" or "the model reaches Algonauts-level performance."
- For scientific claims, frame the core question as convergence/dissociation among behavioral saliency, encoding, RSA, and efficiency, with controls and uncertainty, not as a single leaderboard winner.

## Code Pointers

Dataset loading and fixation parsing:

- `src/hma/datasets/salicon.py`
- `src/hma/datasets/cat2000.py`
- `src/hma/datasets/coco_search18.py`
- `src/hma/datasets/fixation_parsers.py`
- `src/hma/datasets/fixation_utils.py`

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
- `src/hma/saliency/occlusion.py`
- `src/hma/saliency/precomputed.py`
- `src/hma/saliency/postprocess.py`

Neural alignment:

- `src/hma/neural/activations.py`
- `src/hma/neural/encoding.py`
- `src/hma/neural/rsa.py`
- `src/hma/experiments/neural_alignment.py`
- `src/hma/experiments/summarize_neural_roi_results.py`
- `scripts/run_neural_alignment.py`
- `scripts/summarize_neural_roi_results.py`

Reporting:

- `scripts/create_paper_inspection_pack.py`
- `scripts/merge_behavioral_aggregates.py`
- `scripts/merge_efficiency_profiles.py`

## Verification Baseline

Recent full-suite runs before this document cleanup reported all tests passing, with known non-blocking warnings from PyTorch Grad-CAM hooks and Windows `.pytest_cache` permissions. For code changes, run targeted tests first and then the full suite when feasible:

```cmd
.\.venv\Scripts\python.exe -m pytest tests\test_metrics.py tests\test_saliency_benchmark.py tests\test_aggregate_results.py tests\test_export_deepgaze_maps.py tests\test_paper_inspection_pack.py
.\.venv\Scripts\python.exe -m pytest
```

This edit is documentation-only; no tests are required just to consume this file.

## Artifact Cleanup State

Updated: 2026-05-20

Generated artifacts with stale or mixed behavioral protocols were cleared so the next benchmark pass cannot accidentally consume pre-fix NSS/AUC rows.

Removed generated outputs:

- `outputs/real_matrix_v1/`
- `outputs/real_matrix_v2/`
- `outputs/real_matrix_v2_ssl_behavior/`
- `outputs/paper_inspection_v1/`
- `outputs/neural_roi_summary/`
- transient smoke/debug outputs: `outputs/default/`, `outputs/saliency_static_debug/`, `outputs/salicon_resnet50_debug/`, `outputs/neural_smoke_dummy/`, `outputs/neural_roi500_debug/`, `outputs/neural_nsd_algonauts_*_smoke/`

Kept generated outputs:

- `outputs/neural_roi500/`
- `outputs/neural_roi500_ssl/`
- `outputs/neural_roi500_ssl_pretrained_debug/`
- `outputs/neural_nsd_algonauts_*_500/`

Rationale:

- The behavioral matrices and paper inspection pack are no longer scientifically usable because they were generated before, or mixed across, the corrected fixation-point protocol.
- `outputs/neural_roi_summary/` was removed because it included behavior-neural bridge tables derived from stale behavioral aggregates. It should be regenerated after corrected behavioral references and model rows exist.
- Current neural ROI500 run directories were kept because they remain valid raw-correlation diagnostics and are needed to regenerate neural-only summaries.
- No source files, configs, manifests, raw data, or tests were removed. The tests remain usable; they validate current code paths, including the corrected fixation-protocol metadata.

## Corrected Behavioral Reference Rerun V1 Status

Updated: 2026-05-20

The six static2000 reference rows were rerun after artifact cleanup:

- SALICON center bias
- SALICON DeepGaze precomputed
- CAT2000 center bias
- CAT2000 DeepGaze precomputed
- COCO-Search18 center bias
- COCO-Search18 DeepGaze precomputed

Acceptance gate status: **passed**.

Protocol metadata:

- SALICON reference rows: `fixation_protocol=points`
- CAT2000 reference rows: `fixation_protocol=points`
- COCO-Search18 reference rows: `fixation_protocol=task_points`

Corrected NSS sanity values:

- SALICON center bias: `0.933`
- SALICON DeepGaze IIE: `1.743`
- CAT2000 center bias: `1.619`
- CAT2000 DeepGaze IIE: `1.838`
- COCO-Search18 center bias: `1.310`
- COCO-Search18 DeepGaze IIE: `1.745`

Interpretation:

- The old protocol failure pattern is resolved for the reference rows. DeepGaze now beats center bias on NSS for SALICON, CAT2000, and COCO-Search18 under the corrected point/task-point protocols.
- SALICON and CAT2000 values are still not official benchmark scores and should not be numerically equated with MIT/Tuebingen or SALICON leaderboard results, but the scale and ordering are no longer obvious failures.
- COCO-Search18 remains task-driven search, so it should be interpreted separately from free-viewing SALICON/CAT2000.

Next action:

- Rerun the static2000 model rows in `configs/experiments/real_matrix_v2/`.
- Then aggregate and inspect only rows with `fixation_protocol=points` or `task_points`.
- After the core model rows pass, rerun SSL/VLM behavioral rows, regenerate merged behavioral aggregates, regenerate neural bridge summaries, and recreate `outputs/paper_inspection_v1/`.

## Corrected Core Static2000 Matrix Status

Updated: 2026-05-21

The core `real_matrix_v2` static2000 matrix was rerun and re-aggregated after the corrected reference gate passed.

Current clean aggregate:

- Path: `outputs/real_matrix_v2/aggregated/results.csv`
- Static datasets: `salicon_static2000`, `cat2000_static2000`, `coco_search18_static2000`
- Per-image output directories: `36`
- Aggregate rows: `252`
- Protocol rows: `168` metric rows with `points`, `84` metric rows with `task_points`
- Blank / `unknown` / `density_fallback` aggregate rows: none

Note:

- `scripts/run_v2_matrix.py --phase static2000` also created three SALICON pilot reliability-check outputs. Those pilot artifacts were removed before the final aggregate so `results.csv` is static2000-only.
- The optional efficiency join was skipped during re-aggregation because `outputs/real_matrix_v2/efficiency/model_efficiency.csv` is not present after cleanup. This does not affect behavioral metric validity.

Corrected static2000 NSS headline:

- SALICON: DeepGaze `1.743`, center bias `0.933`, strongest model saliency row ConvNeXt-T Grad-CAM `0.633`.
- CAT2000: DeepGaze `1.838`, center bias `1.619`, strongest model saliency row ResNet-50 Grad-CAM `0.882`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, strongest model saliency row ResNet-50 Grad-CAM `0.955`.

Interpretation:

- The core behavioral matrix is now usable for diagnostic paper-style analysis under the corrected fixation protocol.
- Dedicated DeepGaze reference rows are clearly stronger than generic XAI saliency maps, which is the expected sanity pattern.
- The current model saliency rows should still be described as explanation-map-to-fixation similarity, not SOTA fixation prediction.

Next action:

- Rerun the SSL/VLM behavioral static2000 matrix under `configs/experiments/real_matrix_v2_ssl_behavior/`.
- Re-aggregate SSL/VLM results, merge them with the corrected core aggregate, regenerate neural bridge summaries, and recreate the paper inspection pack.

## Corrected SSL/VLM Static2000 Matrix Status

Updated: 2026-05-22

The SSL/VLM behavioral static2000 matrix was rerun and merged with the corrected core aggregate.

SSL/VLM aggregate:

- Path: `outputs/real_matrix_v2_ssl_behavior/aggregated/results.csv`
- Static datasets: `salicon_static2000`, `cat2000_static2000`, `coco_search18_static2000`
- Completed configs: `18`
- Aggregate rows: `126`
- Protocol rows: `84` metric rows with `points`, `42` metric rows with `task_points`
- Blank / `unknown` / `density_fallback` aggregate rows: none

Merged behavioral aggregate:

- Path: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Rows: `378`
- Dataset rows: `126` each for SALICON, CAT2000, and COCO-Search18
- Protocol rows: `252` metric rows with `points`, `126` metric rows with `task_points`
- Blank / `unknown` / `density_fallback` aggregate rows: none

Corrected merged NSS headline:

- SALICON: DeepGaze `1.743`, center bias `0.933`, DINOv2 ViT-S/14 gradient `0.736`, ConvNeXt-T Grad-CAM `0.633`, ResNet-50 Grad-CAM `0.598`.
- CAT2000: DeepGaze `1.838`, center bias `1.619`, ResNet-50 Grad-CAM `0.882`, DINOv2 ViT-S/14 gradient `0.810`, ConvNeXt-T Grad-CAM `0.759`.
- COCO-Search18: DeepGaze `1.745`, center bias `1.310`, ResNet-50 Grad-CAM `0.955`, ConvNeXt-T Grad-CAM `0.908`, DINOv2 ViT-S/14 gradient `0.713`.

Interpretation:

- The corrected behavioral layer is now clean enough for paper-style diagnostic tables and figures.
- DeepGaze remains the strongest reference row across all datasets, and center bias remains a strong baseline.
- DINOv2 gradient is now a strong explanation-map row, especially on SALICON and CAT2000, but it should be framed as an attribution/fixation-similarity result rather than as a dedicated fixation-prediction model.

Next action:

- Regenerate `outputs/neural_roi_summary/` using `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`.
- Recreate `outputs/paper_inspection_v1/` from the corrected merged behavioral aggregate and regenerated neural summary.

## Corrected Inspection Pack Status

Updated: 2026-05-22

The neural bridge summaries and paper inspection pack were regenerated from the corrected merged behavioral aggregate.

Inputs:

- Behavioral CSV: `outputs/real_matrix_v2/aggregated/results_with_ssl_behavior.csv`
- Neural ROI outputs: `outputs/neural_roi500/` and `outputs/neural_roi500_ssl/`
- SSL/multimodal candidate inventory: `outputs/neural_roi_summary/ssl_multimodal_candidate_inventory.csv`

Regenerated outputs:

- `outputs/neural_roi_summary/`
- `outputs/paper_inspection_v1/`

Inspection pack headline:

- Top displayed behavioral NSS row: CAT2000 / DeepGaze IIE / DeepGaze, NSS `1.838`
- Raw neural encoding leader: DeiT-S/16, mean encoding `0.261`
- Raw neural RSA leader: ViT-B/16, mean RSA `0.088`
- Overall behavior-to-encoding leader match rate: `0.079`
- Overall behavior-to-RSA leader match rate: `0.000`
- SSL/multimodal candidates dry-inspected: `8`; pretrained debug runs complete: `3`

Current interpretation boundary:

- The corrected behavioral table is now suitable for diagnostic paper-style discussion.
- The behavior-neural bridge remains descriptive only because the neural side is still one-subject ROI500 raw-correlation output.
- The next scientific upgrade should be neural benchmark-equivalent evaluation rather than more static behavioral cleanup.
