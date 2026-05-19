# V2 Static2000 Results Note

Date: 2026-05-19

## Scope

This note freezes `outputs/real_matrix_v2/aggregated/results.csv` as the current behavioral-saliency baseline. Pilot rows remain useful for reliability checks, but the main static-image claims should use the three `static2000` subsets.

Static outputs:

- Aggregate table: `outputs/real_matrix_v2/aggregated/results.csv`
- Static ledger: `outputs/real_matrix_v2/run_ledgers/static2000_run_ledger.csv`
- Summary tables: `outputs/real_matrix_v2/aggregated/results_summary/`
- Plots: `outputs/real_matrix_v2/aggregated/results_plots/`

The static ledger contains 33 successful core `static2000` runs. DeepGaze reference rows were added after the core matrix through precomputed-map configs.

## Main Pattern

Center bias remains the strongest NSS row on all three static datasets:

- SALICON: `center_bias_baseline + center_bias`, NSS 0.5087300828847219.
- CAT2000: `center_bias_baseline + center_bias`, NSS 0.5193630471415818.
- COCO-Search18: `center_bias_baseline + center_bias`, NSS 0.933417318686843.

The strongest model-generated non-baseline NSS row remains consistent across datasets:

- SALICON: `resnet50 + gradcam`, NSS 0.3421211974733906.
- CAT2000: `resnet50 + gradcam`, NSS 0.39305544706738876.
- COCO-Search18: `resnet50 + gradcam`, NSS 0.6341453774338297.

After adding DeepGaze IIE precomputed reference maps:

- SALICON: `deepgaze_reference + deepgaze_precomputed`, NSS 0.43481094856746494, CC 0.8018529208600521, KL 0.3562953563779592.
- CAT2000: `deepgaze_reference + deepgaze_precomputed`, NSS 0.2749275257792324, CC 0.3711839377101278, KL 1.2703175959587096.
- COCO-Search18: `deepgaze_reference + deepgaze_precomputed`, NSS 0.5179860137457727, CC 0.2832240072847344, KL 1.9249819242060184.

This supports the current interpretation: model saliency produces meaningful fixation alignment, especially class-localization maps, but center bias remains a necessary control and currently outperforms model saliency on the main static NSS/CC/SIM-style measures.

DeepGaze changes the reference picture: it is the strongest non-baseline static row on SALICON, especially for density-alignment metrics, but it does not beat `resnet50 + gradcam` on CAT2000 or COCO-Search18 NSS.

## Metric Caveats

Shuffled AUC gives a less center-bias-dominated view. On COCO-Search18 static2000, `deit_small_patch16_224 + attention_rollout` leads shuffled AUC, followed closely by CNN Grad-CAM rows. SALICON and CAT2000 still favor center bias on shuffled AUC.

KL remains lower-is-better and favors center bias across all three static datasets.

EMD is intentionally pilot-only in the current V2 config because it is slower. Treat static EMD as a possible later targeted pass, not a missing failure.

Pilot occlusion was added as the first perturbation saliency family. It is valid and has stronger shuffled AUC than the other ResNet saliency methods in the three pilot datasets, but it does not beat Grad-CAM on NSS/CC and has weak KL. Keep it pilot-only for now.

## Next Use

Use `key_comparisons.csv` for center-bias, Grad-CAM, gradient, attention/evidence, reference, and perturbation contrasts. Use `pilot_static_stability.csv` to identify patterns that survive scaling from pilot500 to static2000.

The next implementation priority is no longer more behavioral saliency scaling. Move to the proposal's neural layer: build a real NSD / Algonauts manifest, run one neural encoding experiment, and add RSA / representational-geometry reporting over the same image set.
