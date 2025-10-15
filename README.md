# WormVT: Vision Transformer for C. elegans Developmental Biology

A research-grade implementation of Volumetric-Temporal Former (VT-Former) for processing 3D+time lightsheet microscopy data of C. elegans embryo development.

## Overview

This project implements the core vision encoder for the WormVLA (Vision-Language-Action) system described in the comprehensive architecture plan. The VT-Former processes 4D data (3D space + time) natively using factorized space-time attention, enabling efficient processing of developmental biology time-series.

### Key Features

- **3D+Time Vision Encoder**: Native 4D processing with factorized attention
- **Self-Supervised Pretraining**: VideoMAE-3D on unlabeled lightsheet data
- **Stage Classification**: Fine-tuned on VLM-labeled developmental stages
- **Modular Architecture**: Clean, extensible design for research
- **PyTorch Lightning**: Production-ready training infrastructure

## Architecture

The VT-Former uses divided space-time attention (inspired by TimeSformer):

```
Input: [B, T, C, H, W, D]  # 3D+time volumes
  ↓
3D Patch Embedding (16×16×8 voxels)
  ↓
Positional Encoding (spatial + temporal)
  ↓
Divided Space-Time Attention Blocks
  ├── Spatial Attention (within timepoints)
  └── Temporal Attention (across time)
  ↓
Output: [B, N_tokens, D_model]
```

**Efficiency**: O(THW + THWD) vs joint attention O((THWD)²)

## Project Structure

```
worm_VT/
├── src/
│   ├── models/           # Model architectures
│   │   ├── vt_former.py      # Main VT-Former encoder
│   │   ├── patch_embed.py    # 3D patch embedding
│   │   ├── attention.py      # Space-time attention blocks
│   │   └── heads.py          # Task-specific heads
│   ├── data/            # Data loaders and preprocessing
│   │   ├── nih_ls_dataset.py # NIH-LS dataset loader
│   │   ├── transforms.py     # 3D augmentations
│   │   └── utils.py          # Data utilities
│   ├── training/        # Training logic
│   │   ├── pretrain.py       # Self-supervised pretraining
│   │   ├── finetune.py       # Supervised fine-tuning
│   │   └── callbacks.py      # Custom callbacks
│   └── utils/           # General utilities
│       ├── visualization.py
│       └── metrics.py
├── configs/             # Hydra configuration files
│   ├── config.yaml          # Main config
│   ├── model/              # Model configs
│   ├── data/               # Data configs
│   └── training/           # Training configs
├── scripts/             # Bash scripts
│   ├── pretrain.sh
│   └── finetune.sh
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
└── requirements.txt    # Dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Data

The project expects the NIH-LS dataset from the sibling directory:

```bash
# Assuming data is in ../vlm_worm_test/nih-ls/
# Data structure:
# nih-ls/
# └── nih_diSPIM_deconv_1/
#     ├── tracks/
#     │   └── tracks.txt
#     └── frames/
#         └── *.tif
```

### 2. Self-Supervised Pretraining

```bash
python src/training/pretrain.py \
    model=vt_former_small \
    data=nih_ls \
    training=videomae_3d
```

### 3. Stage Classification Fine-tuning

```bash
python src/training/finetune.py \
    model=vt_former_small \
    data=nih_ls_classified \
    training=stage_classification \
    checkpoint=outputs/pretrain/best.ckpt
```

## Configuration

Configuration is managed with Hydra. See `configs/` for all options.

### Model Configurations

**Small Model** (for testing):
- Depth: 6 layers
- Width: 512
- Heads: 8
- Parameters: ~50M

**Base Model** (recommended):
- Depth: 12 layers
- Width: 768
- Heads: 12
- Parameters: ~300M

**Large Model** (for offline analysis):
- Depth: 12 layers
- Width: 1024
- Heads: 16
- Parameters: ~500M

## Training Details

### VideoMAE-3D Pretraining

- Mask ratio: 75%
- Batch size: 4 (with gradient accumulation)
- Learning rate: 1.5e-4 with cosine decay
- Optimizer: AdamW (weight decay 0.05)
- Duration: ~2 weeks on 4× A100

### Stage Classification Fine-tuning

- Training data: 400 VLM-labeled frames
- Batch size: 16
- Learning rate: 5e-5
- Epochs: 50 with early stopping
- Target accuracy: >90%

## Dataset

This project uses the NIH-LS dataset:

> Moyle, M.W., Barnes, K.M., Kuchroo, M. et al. Structural and developmental principles of neuropil assembly in C. elegans. Nature 591, 99–104 (2021).

**Dataset**: [Zenodo Record 6460375](https://zenodo.org/records/6460375) (27 GB)
- 3 C. elegans embryos
- 3D time-lapse lightsheet microscopy
- Nucleus tracking data
- ~400 frames per embryo at 1 min/frame

## Evaluation

The model is evaluated on multiple tasks:

1. **Stage Classification**: Accuracy on developmental stages
2. **Temporal Consistency**: No impossible transitions
3. **Feature Quality**: t-SNE visualization of learned embeddings
4. **Reconstruction**: Visual quality of masked patch prediction

## Future Work

This is the foundation (MVP) for the full WormVLA system:

- [ ] Multi-modal fusion (EM, connectomics, tracking)
- [ ] Language decoder integration (Qwen-2.5-3B)
- [ ] Action tokens for microscope control
- [ ] Real-time model distillation
- [ ] Hypothesis generation with RAG

See `../vlm_worm_test/docs/WORMVLA_COMPREHENSIVE_PLAN.md` for the full roadmap.

## License

MIT

## Acknowledgments

- Architecture inspired by TimeSformer, ViViT, VideoMAE
- Dataset from Moyle et al., Nature 2021
- Built with PyTorch, PyTorch Lightning, MONAI
