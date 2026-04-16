#!/bin/bash
# Train all baseline models sequentially on a single GPU.
#
# Usage:
#   bash scripts/train_baselines.sh [GPU_ID]
#
# Models: FNO, U-Net, MeshGraphNet, PointNet++, Transolver

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "================================================"
echo "Training all baselines on GPU $GPU_ID"
echo "Start: $(date)"
echo "================================================"

COMMON_ARGS="
    --h5_path data/case.h5
    --boundary_cut 0.1
    --epochs 10000
    --lr 3e-4
    --batch_size 4
    --save_interval 100
"

for MODEL in fno unet mgn pointnet2 transolver; do
    echo ""
    echo ">>> Training: $MODEL"
    python -m baselines.train \
        --model $MODEL \
        --checkpoint_dir checkpoint_baselines/$MODEL \
        $COMMON_ARGS
    echo ">>> $MODEL done"
done

echo ""
echo "================================================"
echo "All baselines done: $(date)"
echo "================================================"
