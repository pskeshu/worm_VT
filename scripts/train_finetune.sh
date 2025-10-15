#!/bin/bash
# Fine-tuning script for stage classification

# Usage:
#   bash scripts/train_finetune.sh

# Or with custom checkpoint:
#   bash scripts/train_finetune.sh training.checkpoint=outputs/pretrain/best.ckpt

python train.py \
    training=finetune \
    model=vt_former_small \
    data=nih_ls_classified \
    experiment_name=vt_former_finetune \
    "$@"
