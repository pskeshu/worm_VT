#!/bin/bash
# Monitor training progress (Linux/Mac)

# Usage:
#   bash scripts/monitor_training.sh outputs/vt_former_pretrain
#   bash scripts/monitor_training.sh outputs/vt_former_pretrain 5

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/monitor_training.sh output_dir [refresh_interval]"
    echo "Example: bash scripts/monitor_training.sh outputs/vt_former_pretrain 5"
    exit 1
fi

if [ $# -eq 1 ]; then
    python scripts/monitor_training.py "$1"
else
    python scripts/monitor_training.py "$1" --refresh "$2"
fi
