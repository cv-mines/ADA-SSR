# ADA-SSR

Official implementation of **ADA-SSR: Adversarial Domain Adaptation and Self-supervised Refinement for RGB-to-Event Transfer**.

## Overview

ADA-SSR transfers an image classifier from a labeled RGB source domain to an unlabeled event target domain. It combines adversarial domain alignment with confidence-based target self-refinement using weak and strong augmentations.

<p align="center">
  <img src="assets/ada_ssr_framework.jpg" alt="Overview of the ADA-SSR framework" width="100%">
</p>

The training objective combines supervised classification on source images, adversarial domain alignment, target pseudo-label learning, and supervised classification on strongly augmented source images.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/cv-mines/ADA-SSR.git
cd ADA-SSR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a compatible PyTorch/torchvision pair for your hardware using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python main_ada_ssr.py --help
```

The current requirements pin NumPy to `1.23.5`. A Python 3.10 environment is a practical starting point; the repository does not yet provide a validated, fully pinned environment. A CUDA-capable GPU is recommended for training. The script falls back to CPU when CUDA is unavailable. Pretrained backbone weights may require an internet connection on the first run.

## Data preparation

Convert event streams into image representations before running this code: the loader reads images, not raw event files. Prepare source and target images with a shared class mapping.

The dataset root must contain these three list files:

```text
dataset_root/
├── source_images/
├── target_images/
└── image_list/
    ├── source_train.txt
    ├── target_train.txt
    └── target_test.txt
```

Image folders may have different names. Each list entry contains an image path relative to `dataset_root`, followed by an integer class label:

```text
source_images/class_0/example_001.jpg 0
source_images/class_1/example_002.jpg 1
```

Use the same contiguous labels `0, ..., K-1` in both domains. The current loader expects labels in all three lists; target training labels are used for monitoring pseudo-label accuracy, not as supervision in the target training loss.


## Training

Example for an RGB-to-event dataset stored at `data/caltech101DVS`:

```bash
python main_ada_ssr.py data/caltech101DVS \
  --arch resnet50 \
  --epochs 20 \
  --seed 1 \
  --threshold 0.7 \
  --log logs/caltech101DVS/ada_ssr_resnet50
```

Replace the dataset root with your own prepared dataset. This is a usage example, not a verified paper-reproduction configuration. Use a distinct log directory for each experiment to preserve checkpoints.

### Main options

| Option | Default | Purpose |
| --- | --- | --- |
| `--arch` | `resnet18` | Backbone; ResNet-18/34/50/101/152 are available |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `32` | Batch size per domain |
| `--iters-per-epoch` | `1000` | Training iterations per epoch |
| `--lr` | `0.01` | Optimizer learning-rate argument |
| `--threshold` | `0.7` | Confidence threshold for target pseudo-labels |
| `--trade-off` | `1.0` | Domain-alignment loss weight |
| `--trade-off-self-training` | `1.0` | Target self-training loss weight |
| `--trade-off-cls-strong` | `0.1` | Strong-source classification loss weight |
| `--seed` | unset | Training seed |
| `--workers` | `2` | Data-loader workers |
| `--phase` | `train` | `train`, `test`, or `analysis` |

See `python main_ada_ssr.py --help` for all options.

## Evaluation

Load the best classifier checkpoint from the training log directory:

```bash
python main_ada_ssr.py data/caltech101DVS \
  --arch resnet50 \
  --phase test \
  --log logs/caltech101DVS/ada_ssr_resnet50
```

Use the same architecture and model settings as during training. The dataset root must still contain all three list files, since the script initializes all datasets before evaluation.

## Feature analysis

Generate source/target feature projections with t-SNE, UMAP, Isomap, and LLE:

```bash
python main_ada_ssr.py data/caltech101DVS \
  --arch resnet50 \
  --phase analysis \
  --log logs/caltech101DVS/ada_ssr_resnet50
```

Analysis can require substantial memory and runtime for large datasets.

## Outputs

The directory passed to `--log` contains:

- `checkpoints/latest.pth`: classifier state from the latest epoch.
- `checkpoints/best.pth`: classifier state selected by validation accuracy.
- Timestamped console logs.
- `visualize/`: feature-projection PDFs produced during analysis.

Checkpoints contain classifier weights; the current script does not implement full training resume with optimizer and scheduler state.

## Code organization

| Path | Contents |
| --- | --- |
| `main_ada_ssr.py` | Training, evaluation, and analysis entry point |
| `models/` | ResNet backbones, classifier, discriminator, and gradient reversal |
| `losses/` | Adversarial alignment and confidence-based self-training |
| `utils/` | Dataset loading, augmentation, metrics, logging, and feature analysis |
| `assets/` | Framework illustration |
| `requirements.txt` | Python dependencies |

## Citation

Publication details and the BibTeX citation will be added when the final bibliographic record is available.

## Acknowledgments and licensing

Parts of this implementation are adapted from the [Transfer Learning Library (TLlib)](https://github.com/thuml/Transfer-Learning-Library). Original source notices are retained in the relevant files. See `LICENSE_TLIB.txt` for the bundled license text.

The license and copyright attribution for the complete ADA-SSR project should be confirmed before adding a project-wide license.
