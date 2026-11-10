# Compressible Representations: Functional Spines in Deep Neural Networks

Code to reproduce the analyses and figures in **"Compressible Representations: Functional Spines in Deep Neural Networks"** (Rajesh & Khosla, NeurIPS 2026).

We identify, at each layer, the subspace of a representation that is functionally sensitive to the task, using a matrix analogous to the Fisher information defined over hidden representations (the *task-sensitivity matrix*). We compare compression onto this subspace against PCA, local-neighborhood, geotopology, and random projections, and use the subspace for adversarial robustness analysis, representational comparison, shape/texture dissociation, and continual learning.

<!-- TODO: add paper link (arXiv / OpenReview / proceedings) and a one-line teaser figure from figures/ -->

## Contents

- [Setup](#setup)
- [Data](#data)
- [Repository structure](#repository-structure)
- [Reproducing the results](#reproducing-the-results)
- [Compute](#compute)
- [Citation](#citation)
- [License](#license)

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` and `uv.lock`).

## Repository structure

```
.
compressible-representation
├── utilz/                            # helper modules
├── figure_code/                      # code to generate figures from the results of our empirical expts
├── *_results/                        # when the below scripts are run, they will generate their own results directories 
├── vit_build_projectors.py           # builds the projectors for all compression criteria
├── vit_*_compression.py              # evaluates layerwise and global classification accuracy for all compressions (sec 3.1, 3.2)
├── vit_compressed_adv.py  # evaluates adv robustness for all compression types (sec 3.3)
├── task_null_alignment.py            # task-sensitive representational comparisons (sec 3.4)
├── shape_texture_alignment.py        # task-conditioned (shape vs texture) representational comparisons (sec 3.4)
├── shape_texture_ablation.py         # identificiation of functionally distinct shape and texture subspaces (sec 3.5)
├── continual_learning.py             # continual learning experiments with GPM and FGPM (sec 3.6)
├── vit_compressed_self_alignment.py  # CKA of projected vs base reps of same compression type (sec A.8)
├── vit_compressed_cross_alignment.py # CKA of projected reps across compression types (sec A.8)
├── pyproject.toml / uv.lock
```

<!-- TODO: fill in the full file names (several are truncated in my view) and fix any descriptions that are wrong. -->
## Reproducing the results
 
All code to reproduce the results in the main and supplementary text of the paper is present in this repository. Here is an example of code to generate the results for performance evaluations on compressed models.
 
First, we build and cache the projectors:
```bash
# Layer-wise, fixed allocation
uv run python vit_build_projectors.py \
  --train-dir $IMAGENET_TRAIN --val-dir $IMAGENET_VAL \
  --compression-type task_sensitive \
  --allocation-mode fixed --layerwise \
  --model-name vit_b_16 \
  --fraction-grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --calibration-samples 5000 --eval-samples 10000 \
  --output-dir vit_results/ $COMMON

# Global, utility-curve allocation
uv run python vit_build_projectors.py \
  --train-dir $IMAGENET_TRAIN --val-dir $IMAGENET_VAL \
  --compression-type task_sensitive \
  --allocation-mode utility_curve --no-layerwise \
  --model-name vit_b_16 \
  --fraction-grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --calibration-samples 5000 --eval-samples 10000 \
  --output-dir vit_results/ $COMMON
done
```
Then, we run the appropriate compression eval file to generate single-layer and global performance results
```bash
for frac in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  uv run python vit_task_sensitive_compression.py \
    --train-dir $IMAGENET_TRAIN --val-dir $IMAGENET_VAL \
    --model-name vit_b_16 --layer-subset all \
    --retain-fraction $frac \
    --fraction-grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --allocation-mode utility_curve \
    --utility-condition task_raw --utility-metric top1 \
    --calibration-samples 512 --eval-samples 2000 \
    --batch-size 64 --num-workers 2 --pin-memory \
    --utility-cache-dir vit_results/task_sensitive_cache \
    --output-dir vit_results/vit_task_sensitive_${frac}
done
 
```
## Reproducing the figures

Once the relevant results have been generated (see above), the code below reproduces each figure and table in the paper.

| Paper item | Description | Code |
|---|---|---|
| Figure 1 | Task-sensitive compression performance | [`figure_code/A_clean_accs.ipynb`](figure_code/A_clean_accs.ipynb) |
| Figure 2 | Layer-wise compression across methods | [`figure_code/A_clean_accs.ipynb`](figure_code/A_clean_accs.ipynb) |
| Figure 3 | Global compression across methods | [`figure_code/A_clean_accs.ipynb`](figure_code/A_clean_accs.ipynb) |
| Figure 4 | Adversarial robustness across methods | [`figure_code/B_adv_accs.ipynb`](figure_code/B_adv_accs.ipynb) |
| Figure 5 | Task-conditioned representational comparisons | [`task_null_alignment.py`](task_null_alignment.py),  [`shape_texture_alignment.py`](shape_texture_alignment.py) |
| Figure 6 | Functionally specific subspaces (shape vs. texture) | [`shape_texture_ablation.py`](shape_texture_alignment.py) |
| Table 1 | Continual learning results | [`continual_learning.py`](continual_learning.py) |
| Figure A.7 | Task-sensitive compression across multiple ViTs | [`figure_code/A_clean_accs.ipynb`](figure_code/A_clean_accs.ipynb) |
| Figure A.8 | Representational comparisons | [`figure_code/A_clean_accs.ipynb`](figure_code/A_clean_accs.ipynb) |

## Citation

```bibtex
@inproceedings{rajesh2026compressible,
  title     = {Compressible Representations: Functional Spines in Deep Neural Networks},
  author    = {Rajesh, Niranjan and Khosla, Meenakshi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
