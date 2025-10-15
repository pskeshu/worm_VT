#!/bin/bash
# Pretraining script for VT-Former with VideoMAE-3D

# Usage:
#   bash scripts/train_pretrain.sh

# Or with custom config:
#   bash scripts/train_pretrain.sh model=vt_former_base data.batch_size=4

python train.py \
    training=pretrain \
    model=vt_former_small \
    data=nih_ls \
    experiment_name=vt_former_pretrain \
    "$@"
