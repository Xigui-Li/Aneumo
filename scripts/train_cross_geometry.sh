#!/bin/bash
# Cross-geometry generalization: geometry split (hold out deforms).
# Supports multi-GPU via torchrun.
#
# Usage:
#   bash scripts/train_cross_geometry.sh [MODEL]
#   NGPU=4 bash scripts/train_cross_geometry.sh fno

MODEL=${1:-fno}

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}
echo "Using $NGPU GPU(s) | Model: $MODEL | Split: geometry"

if [ "$NGPU" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=$NGPU"
else
    LAUNCH="python"
fi

COMMON_ARGS="
    --split_mode geometry
    --h5_dir ./h5_multi_cross
    --boundary_cut 0.1
    --epochs 10000
    --lr 3e-4
    --save_interval 500
    --vis_interval 100
"

echo "========================================="
echo "Cross-geometry $MODEL (geometry split)"
echo "Start: $(date)"
echo "========================================="

$LAUNCH -m baselines.train_cross \
    --model $MODEL \
    $COMMON_ARGS

echo "========================================="
echo "Done: $(date)"
echo "========================================="
