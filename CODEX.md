# CODEX Instructions for This Repository

This repository implements a human-machine visual alignment benchmark.

The project compares modern computer vision architectures against:
1. human fixation and saliency maps,
2. task-driven gaze and scanpaths,
3. fMRI/neural prediction datasets,
4. representational similarity / CKA analyses,
5. Brain-Score-style external metrics,
6. computational efficiency metrics such as parameters, FLOPs, latency, and retained tokens.

## Core datasets

Initial priority:
- SALICON
- CAT2000
- COCO-Search18
- NSD / Algonauts 2023
- Brain-Score Vision comparison layer
- DHF1K as optional video extension

Secondary datasets:
- MIT1003 / MIT300
- OSIE
- BOLD5000
- THINGS-fMRI / THINGS-data
- VQA-HAT
- Hollywood-2 / UCF Sports gaze
- Ego4D gaze subset

## Coding rules

Use Python 3.10+.

Prefer:
- PyTorch
- torchvision
- timm
- numpy
- pandas
- scipy
- scikit-learn
- opencv-python
- pillow
- matplotlib
- pyyaml
- tqdm
- pytest

Do not download large datasets automatically inside tests.

Do not hard-code absolute paths.

All dataset loaders must support:
- root directory
- split
- transform
- optional max_items for debugging
- deterministic ordering
- manifest creation or manifest loading

All benchmark scripts must support:
- YAML config path
- output directory
- random seed
- CPU/GPU device selection
- debug mode with very small subsets

All metrics must be unit-tested on small synthetic arrays.

All saliency maps must be normalized through a single shared postprocessing function.

All experiment outputs should be saved as:
- CSV metrics
- JSON metadata
- optional PNG visualizations

Never build a huge monolithic script.
Use small modules and tests.

When implementing a feature:
1. inspect the existing code,
2. make the smallest coherent change,
3. add or update tests,
4. run the relevant tests,
5. summarize files changed and commands run.