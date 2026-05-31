# Literature Review: Cross-Axis Diagnostic of Human–Machine Visual Alignment

## Overview

The roadmap for **Paper 1** aims to evaluate how closely modern vision models align with human visual processing along three **axes** — **fixation alignment**, **neural encoding** and **representational geometry** — and to examine whether improvements on one axis predict improvements on the others.  The existing literature already provides strong baselines for each component and highlights several open challenges: dataset bias in saliency prediction, individual variability in scanpaths, fMRI encoding and representation comparison methods, and interpretability techniques for complex architectures.  This review synthesises research published up to **2026‑05‑31** with a focus on what is established, recent methodological advances, and gaps relevant to the proposed dissociation study.

## Saliency Prediction and Scanpath Modelling

### Established Saliency Benchmarks and Baselines

Early saliency models relied on **low‑level features** and information‑theoretic mechanisms (e.g., Itti et al., 1998; Treisman & Gelade, 1980).  Deep learning shifted the field towards high‑level features.  **DeepGaze III** demonstrates that free‑viewing scanpaths can be modelled by conditioning the next fixation on both image content and prior fixations; the model uses recurrent attention to generate scanpaths and outperforms earlier approaches on the MIT300 benchmark【713079479663537†L149-L169】.  The DeepGaze III paper notes that scene content contributes more to predictions than previous fixations and that the architecture enables investigations of factors influencing fixation selection【713079479663537†L149-L169】.

The **MIT/Tübingen saliency benchmark** uses metrics such as normalized scanpath saliency (NSS), area‑under‑curve variants, correlation (CC), similarity (SIM), Kullback–Leibler divergence and center‑bias baselines.  These metrics, as referenced in the roadmap, form the baseline evaluation for Paper 1.

### Dataset Bias and Generalization

Despite high performance on benchmarks, saliency models generalize poorly across datasets.  **Kümmerer et al. (2025)** show that when models trained on one dataset are applied to another, performance drops by ~40 %; more than half of this drop arises from dataset‑specific biases【458740451669501†L52-L66】.  To address this, the authors propose extending a mostly dataset‑agnostic encoder–decoder with fewer than 20 dataset‑specific parameters that control multi‑scale structure, center bias and fixation spread.  Adapting only these parameters closes **75 % of the generalization gap**, and as few as **50 samples** suffice for effective adaptation【458740451669501†L52-L68】.  The resulting model sets state‑of‑the‑art scores on the **MIT300**, **CAT2000** and **COCO‑Freeview** datasets and provides interpretable insights into how absolute and relative object sizes drive saliency【458740451669501†L52-L68】.

### Personalized Scanpath Prediction

Scanpath behaviour varies across individuals due to culture, memory and experience.  **Few‑Shot Personalized Scanpath Prediction (FS‑PSP)** by **Xue et al. (CVPR 2025)** formulates a **few‑shot** task where a base model must adapt to a new subject with ≤ 10 support scanpaths【393755380628548†L24-L28】.  The key innovation is the **Subject‑Embedding Network (SE‑Net)** which learns individualized embeddings; a separate scanpath prediction module conditions on these embeddings【393755380628548†L24-L30】.  SE‑Net is trained with classification and contrastive losses to maximize inter‑subject discrimination while minimizing intra‑subject variance【393755380628548†L118-L130】.  Conditioning the scanpath model on these embeddings yields accurate predictions without test‑time fine‑tuning and performs well on multiple datasets【393755380628548†L24-L30】.  FS‑PSP underscores the need for personalization and suggests that limited per‑subject data suffices when combined with transfer learning.

### Top‑Down vs Bottom‑Up Attention in Visual Search

Most models separately predict **top‑down (goal‑directed)** and **bottom‑up (free‑viewing)** attention.  **Human Attention Transformer (HAT)** (2024) unifies both within a single transformer architecture.  The model uses a simplified foveated retina and sequential dense predictions to avoid discretizing fixations.  HAT achieves state‑of‑the‑art performance on **target‑present** and **target‑absent search** as well as **“task‑less” free‑viewing**, and its sequential predictions improve interpretability【923863930084123†L19-L33】.  This demonstrates that spatio‑temporal awareness and foveation can bridge the gap between bottom‑up and top‑down tasks.

**Semantic‑Foveal Bayesian Attention (SemBA‑FAST)** (Luzio et al., 2025) focuses on goal‑directed search.  It combines **deep object detection** with a **probabilistic semantic fusion** mechanism and artificial foveation; semantic knowledge is updated sequentially to generate attention maps【948656366266731†L21-L39】.  Tested on **COCO‑Search18**, SemBA‑FAST produces fixation sequences that closely match human scanpaths and outperforms other top‑down models【948656366266731†L31-L37】.

### Diffusion‑Based and Transformer‑Based Scanpath Models

To better capture the inherent variability of human gaze, **ScanDiff** (Cartella et al., 2025) combines **diffusion models** with Vision Transformers.  The stochastic diffusion process explicitly models scanpath variability, generating diverse gaze trajectories.  Textual conditioning enables task‑driven scanpath generation, and the model surpasses state‑of‑the‑art methods in both free‑viewing and task‑driven settings【547497620819956†L14-L35】.  This suggests that generative diffusion frameworks can provide richer scanpath distributions than deterministic models.

During captioning tasks, language modulates attention.  **NevaClip** (Zanca et al., 2024) introduces **CapMIT1003**, a dataset pairing images with captions and click‑contingent explorations.  The proposed zero‑shot method combines **CLIP** embeddings with a **Neural Visual Attention (NeVA)** algorithm.  By optimizing fixations to align the foveated image representation with the caption representation, NevaClip generates scanpaths that outperform existing models for captioning and free‑viewing tasks【418081758193870†L13-L20】.  The algorithm relies on a gradient‑based alignment loss that guides fixations to maximize similarity between visual and textual embeddings【418081758193870†L107-L134】.

These developments show rapid progress in modelling scanpaths, with increasing emphasis on personalization, top‑down control, multi‑task generality, and distributional modelling.

## Neural Encoding and fMRI Datasets

### Natural Scenes Dataset (NSD) and Synthetic Extensions

The **Natural Scenes Dataset (NSD)** is a cornerstone resource for linking vision models to human brain activity.  NSD provides high‑resolution (1.8 mm isotropic, 1.6 s sampling) 7 T fMRI measurements from **eight subjects** viewing thousands of natural scenes【805066001099321†L24-L34】.  Participants perform a continuous recognition task during scanning, and the dataset supports large‑scale computational modelling【805066001099321†L24-L34】.  NSD was used in the **Algonauts 2023 Challenge**, which enabled community efforts to predict fMRI responses and spurred competitions comparing deep models【503441943515290†L53-L66】.  The roadmap’s encoding panel uses similar procedures (ridge regression, cross‑validation, noise ceilings) to compute reliability‑normalized scores.

To probe **out‑of‑distribution (OOD) generalization**, **NSD‑Synthetic** extends NSD by collecting fMRI responses to **284 synthetic images** from the same eight subjects.  The authors show that NSD‑synthetic responses reliably encode stimulus information and are OOD relative to the original NSD stimuli.  OOD generalization tests using NSD‑synthetic reveal differences among models that are not detected with natural images; the degree of stimulus OODness predicts model failure magnitude【133767529002560†L50-L67】.  This dataset provides a benchmark for testing whether models generalize beyond natural image statistics.

**NSD‑Imagery** (Kneeland et al., 2025) complements NSD by pairing fMRI activity with **mental imagery** rather than seen images.  The study collects mental images of NSD stimuli and evaluates existing decoding models on this new benchmark.  The authors find that performance on mental imagery reconstruction is largely decoupled from performance on seen image reconstruction and that simpler linear decoders generalize better to mental imagery, whereas complex models overfit visual training data【69083459642660†L32-L49】.  NSD‑Imagery thus reveals a novel dissociation between overt vision and mental imagery and argues for the importance of fMRI datasets targeted at internal cognitive states.

### Gaze‑Aware Neural Encoding

Recent works incorporate eye‑tracking into fMRI encoding.  An arXiv preprint (2026) on **gaze‑aware neural encoding** reports that sampling CNN feature maps according to fixations allows encoding models to match performance of conventional models while using **112× fewer parameters**; the improvement is especially pronounced for subjects with dynamic eye movements【862266140476645†L48-L68】.  Such results indicate that aligning model sampling with measured gaze can improve fMRI prediction efficiency and may reduce the need for heavy regressors.

### Beyond Prediction Accuracy: New Evaluation Frameworks

Standard encoding analyses correlate model features with brain responses; however, prediction accuracy alone may mask which **response dimensions** are captured.  **Target‑Space Recovery Profiles** (Nakamura et al., 2026) introduce a framework that identifies reproducible dimensions in the target brain responses using repeated trials and then measures how strongly each dimension is recovered by models【269501302558289†L45-L66】.  Applying this method to a subset of NSD reveals that early and intermediate visual cortex responses are low‑dimensional, and some pretrained or randomly initialized models can achieve similar prediction accuracy while recovering different dimensions【269501302558289†L45-L67】.  This suggests that scalar scores (e.g., correlation) may hide mismatches that recovery profiles expose and provides a diagnostic tool for Paper 1.

### Neural Predictivity and Non‑Linear Mapping

Brain‑Score has popularised evaluating models on neural and behavioural benchmarks.  However, the correlation between classification performance and neural predictivity is limited.  A 2021 study proposes using **non‑linear response mapping** instead of linear regression; a small neural network mapping from model activations to neural responses increases predictivity and shows that higher ImageNet accuracy does not guarantee better neural alignment【716304796866005†L248-L265】.  This highlights the importance of choosing appropriate mapping functions when comparing models to brain data.

### Efficiency and Sample Complexity in Encoding Models

Fractional ridge regression and other regularization techniques improve encoding efficiency.  NSD‑Synthetic uses standardized train/test splits and noise ceilings to reduce overfitting【133767529002560†L50-L67】.  FS‑PSP and gaze‑aware models (see earlier) also emphasise parameter efficiency.  Paper 1’s plan to profile model efficiency (parameters, FLOPs, token counts) fits within this broader trend.

## Representational Geometry and Similarity Metrics

### Representational Similarity Analysis (RSA) and Centered Kernel Alignment (CKA)

RSA compares dissimilarity matrices between representations without requiring one‑to‑one correspondences; it originally applied to neuroimaging and model representations【504503694756230†L161-L187】.  **Centered Kernel Alignment (CKA)** is a kernel‑based similarity measure widely used in deep learning.  A 2024 PMLR paper demonstrates that **RSA and linear CKA are mathematically equivalent** when representations are mean‑centered【561781663834307†L19-L28】.  This result unifies two literatures and advises that differences often arise from preprocessing rather than fundamental disparities.

However, a recent OpenReview commentary cautions that CKA can be manipulated and may not reliably capture similarity; irrelevant dimensions can inflate CKA scores without affecting model function【220685663952883†L24-L36】.  Consequently, Paper 1 will include both **CKA** and **subset RSA** to cross‑validate geometry results.

### Finite‑Sample Effects and Spectral Corrections

With limited recorded neurons, similarity measures underestimate true alignment.  **Kang et al. (NeurIPS 2025)** apply random matrix theory to derive an analytical framework relating CKA/CCA scores to the spectral properties of underlying representations【175331206733528†L16-L29】.  They show that eigenvector delocalization causes underestimation of similarity and introduce a **denoising method** to infer population‑level similarity even with few neurons【175331206733528†L20-L29】.  This work underscores the necessity of correcting for sampling bias when comparing model and neural representations.

## Interpretability and Attribution for Vision Models

### Gradient‑Based and Perturbation Methods

**Grad‑CAM** uses gradients of class scores with respect to convolutional features to produce coarse localization maps highlighting important regions【748807432025790†L52-L68】.  **RISE** probes a black‑box model by multiplying the input image with random masks; aggregating outputs yields a probability distribution of relevance and can match or exceed white‑box methods【101184945601009†L48-L61】.  These techniques provide saliency maps but treat attention implicitly.

### Transformer‑Specific Attribution

Standard attention maps are not explanations because they mix information from multiple heads and layers.  **Chefer et al. (CVPR 2021)** propose a relevance propagation approach based on **Deep Taylor Decomposition**.  The method assigns local relevancy scores, propagates them through transformer layers and skip connections, and outperforms attention rollout in explaining vision transformers【590327394421360†L15-L30】.  **AttnLRP** (ICML 2024) generalises **Layer‑wise Relevance Propagation** to transformer attention; it produces input and latent relevance maps with efficiency equivalent to a single backward pass and outperforms previous attribution methods【856452363927525†L18-L32】【856452363927525†L55-L63】.  These methods provide the necessary tools for Paper 1 to compare attribution‑based “explanation maps” with human fixations.

## Summary of Key Papers and Their Contributions

Below is a concise mapping of influential papers relevant to Paper 1.  Each entry lists the main questions addressed, methodologies and findings.

| Paper (year) | Question & Method | Key Findings |
| --- | --- | --- |
| **Kümmerer et al. (ICCV 2025): Modeling Saliency Dataset Bias**【458740451669501†L52-L68】 | How much does dataset bias affect cross‑dataset saliency prediction, and can a model generalize across datasets?  The authors extend a dataset‑agnostic encoder–decoder with a handful of dataset‑specific parameters controlling multi‑scale structure, center bias and fixation spread. | Performance drops ~40 % when training/testing datasets differ; 60 % of this drop arises from dataset‑specific biases.  Adapting < 20 parameters resolves 75 % of the gap and sets state‑of‑the‑art on MIT300, CAT2000 and COCO‑Freeview【458740451669501†L52-L68】. |
| **DeepGaze III (2022)**【713079479663537†L149-L169】 | Can recurrent models predict scanpaths?  DeepGaze III conditions next fixations on image content and previous fixations. | Scene content is more important than fixation history; the model achieves SOTA on MIT300 and enables analysis of fixation selection factors【713079479663537†L149-L169】. |
| **Xue et al. (CVPR 2025): Few‑Shot Personalized Scanpath Prediction**【393755380628548†L24-L30】【393755380628548†L118-L130】 | How to personalize scanpath prediction with few examples?  Introduces SE‑Net to learn subject embeddings and conditions a scanpath predictor on these embeddings. | Model adapts to new subjects with ≤ 10 support scanpaths; the subject embeddings are learned separately and robustly【393755380628548†L24-L30】【393755380628548†L118-L130】. |
| **Luzio et al. (2025): SemBA‑FAST**【948656366266731†L21-L39】 | Can a semantic‑based probabilistic framework model target‑present visual search? | Integrates deep object detection, semantic fusion and foveation to produce sequential attention maps; matches human scanpaths and outperforms baselines on COCO‑Search18【948656366266731†L21-L39】. |
| **Yang et al. (2024): Human Attention Transformer (HAT)**【923863930084123†L19-L33】 | Can one model predict both top‑down and bottom‑up scanpaths? | A transformer with sequential dense predictions and a simplified foveated retina unifies free‑viewing and search; sets SOTA across tasks and improves interpretability【923863930084123†L19-L33】. |
| **Cartella et al. (2025): ScanDiff**【547497620819956†L14-L35】 | How to generate diverse, realistic scanpaths?  Combines diffusion models with Vision Transformers and introduces textual conditioning. | Produces diverse gaze trajectories that outperform previous methods in both free‑viewing and task‑driven scenarios; highlights the benefits of diffusion modelling【547497620819956†L14-L35】. |
| **Zanca et al. (HCV 2024): NevaClip**【418081758193870†L13-L20】【418081758193870†L107-L134】 | How does language influence visual exploration?  CapMIT1003 dataset pairs images, captions and foveated clicks.  The NevaClip algorithm optimizes fixations to align CLIP‑based image and text embeddings. | Foveated vision guided by caption embeddings yields realistic scanpaths that beat existing models; introduces a new captioning eye‑tracking dataset and a zero‑shot scanning method【418081758193870†L13-L20】【418081758193870†L107-L134】. |
| **Allen et al. (2021): Natural Scenes Dataset (NSD)**【805066001099321†L24-L34】 | Large‑scale high‑resolution fMRI dataset linking natural scene viewing with brain responses. | Provides 7 T fMRI responses for eight subjects viewing thousands of images; supports high‑capacity model testing and underpins the Algonauts challenges【805066001099321†L24-L34】. |
| **Gifford et al. (2026): NSD‑Synthetic**【133767529002560†L50-L67】 | Can OOD fMRI data reveal differences between brain models? | Collects fMRI responses to 284 synthetic images; reveals that OOD generalization tests expose model failures not detected with natural images and that OODness predicts failure magnitude【133767529002560†L50-L67】. |
| **Kneeland et al. (CVPR 2025): NSD‑Imagery**【69083459642660†L32-L49】 | How well do fMRI decoders trained on vision generalize to mental imagery? | Mental imagery decoding performance is largely independent of seen‑image decoding.  Simple linear decoders generalize better than complex ones; the dataset enables benchmarking of mental imagery reconstruction【69083459642660†L32-L49】. |
| **Nakamura et al. (2026): Target‑Space Recovery Profiles**【269501302558289†L45-L67】 | Does prediction accuracy capture which brain response dimensions are recovered? | Introduces a framework identifying reproducible fMRI response dimensions and measures which dimensions models recover.  Finds that some models achieve similar accuracy but recover different dimensions, indicating hidden misalignments【269501302558289†L45-L67】. |
| **Kang et al. (NeurIPS 2025): Spectral Analysis of Representational Similarity**【175331206733528†L16-L29】 | How do finite‑neuron samples affect similarity measures (CKA/CCA)? | Uses random matrix theory to show that finite sampling underestimates similarity due to eigenvector delocalization and introduces a denoising method to infer population‑level similarity【175331206733528†L20-L29】. |
| **Chefer et al. (CVPR 2021)**【590327394421360†L15-L30】 | How to explain vision transformers beyond attention maps? | Proposes a relevance propagation method based on Deep Taylor Decomposition, propagating relevancy through attention and skip connections; improves interpretability relative to attention rollout【590327394421360†L15-L30】. |
| **AttnLRP (PMLR 2024)**【856452363927525†L18-L32】【856452363927525†L55-L63】 | How to attribute importance in transformers with efficiency? | Extends Layer‑wise Relevance Propagation to attention layers, producing input and latent relevance maps with the cost of one backward pass; outperforms other attribution methods and enables concept‑based explanations【856452363927525†L18-L32】【856452363927525†L55-L63】. |
| **Quantifying Brain Predictivity with Non‑Linear Mapping (Frontiers 2021)**【716304796866005†L248-L265】 | Can non‑linear mappings improve brain predictivity? | Replacing linear regression with a small neural network increases predictivity; classification accuracy does not correlate with neural alignment; sparser representations improve prediction【716304796866005†L248-L265】. |

## Implications for Paper 1

1. **Dataset bias and generalization**: Kümmerer et al. (ICCV 2025) show that saliency models trained on one dataset often fail on others due to dataset‑specific biases【458740451669501†L52-L68】.  Paper 1 should therefore avoid over‑interpreting metrics on a single dataset and include **multi‑dataset evaluation** and dataset‑agnostic baseline models such as **DeepGaze MSDB**.

2. **Personalization and inter‑subject variability**: FS‑PSP and SE‑Net highlight the importance of individual differences in scanpaths【393755380628548†L24-L30】.  When analyzing behavioural data, Paper 1 should consider **inter‑observer ceilings** and avoid pooling across tasks or individuals without accounting for variability.

3. **Top‑down vs bottom‑up fixation alignment**: HAT and SemBA‑FAST demonstrate that models can simultaneously handle free‑viewing and search【923863930084123†L19-L33】【948656366266731†L21-L39】.  Paper 1’s dissociation should explicitly separate **free‑viewing** from **task‑driven** viewing, as recommended in the roadmap.

4. **Diversity of scanpaths**: ScanDiff and NevaClip show that modelling **scanpath variability** and **language‑guided exploration** yields more realistic behaviours【547497620819956†L14-L35】【418081758193870†L13-L20】.  Paper 1 might explore whether diversity measures (e.g., fixation entropy) correlate with neural alignment.

5. **Neural encoding beyond accuracy**: NSD‑Synthetic and NSD‑Imagery illustrate that **generalization to synthetic and imagined stimuli** reveals different model properties【133767529002560†L50-L67】【69083459642660†L32-L49】.  Paper 1’s encoding panel should use **reliability‑normalized scores**, include **uncertainty estimates**, and possibly incorporate OOD stimuli if time permits.

6. **Representational geometry**: The equivalence between RSA and CKA (with centering) suggests that both can be applied; however, finite‑sample corrections are important【175331206733528†L16-L29】.  Paper 1 should implement **linear CKA** and **subset RSA**, cross‑validate them, and report when they disagree.

7. **Interpretability tools**: Attention maps alone are insufficient; transformer attribution methods such as **Chefer et al.** and **AttnLRP** provide more faithful explanations【590327394421360†L15-L30】【856452363927525†L18-L32】.  Paper 1 must distinguish between **post‑hoc attribution maps** and operational attention and treat them as explanation‑to‑fixation comparisons, not proxies for actual attention.

8. **New evaluation frameworks**: Target‑Space Recovery Profiles caution that high encoding accuracy can mask differences in which brain dimensions are recovered【269501302558289†L45-L67】.  Paper 1 might incorporate this framework or, at minimum, discuss its implications when interpreting correlations across axes.

9. **Efficient encoding**: Both gaze‑aware models and FS‑PSP show that parameter efficiency can be achieved without sacrificing performance【393755380628548†L24-L30】【862266140476645†L48-L68】.  If resources permit, Paper 1’s **efficiency axis** could benchmark parameter counts and FLOPs relative to alignment scores.

## Conclusion

The literature demonstrates rapid advances in saliency and scanpath modelling, neural encoding, representational geometry, and interpretability.  Importantly, **dataset bias**, **task specificity**, **individual variability** and **dimensional recovery** all pose challenges to simple correlations between fixation alignment and neural alignment.  Paper 1’s novelty lies in providing a **joint dissociation analysis** under matched controls; the review above identifies the baseline models, methods and recent innovations that will inform and benchmark such an analysis.
