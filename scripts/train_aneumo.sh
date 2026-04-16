#!/bin/bash
# Train Aneumo (Temporal DeepONet V2) for transient WSS prediction.
#
# Usage:
#   bash scripts/train_aneumo.sh [GPU_ID] [with_swin|no_swin]
#
# Examples:
#   bash scripts/train_aneumo.sh 0 with_swin   # With Swin geometry encoder
#   bash scripts/train_aneumo.sh 0 no_swin     # Without geometry (ablation)

GPU_ID=${1:-0}
MODE=${2:-with_swin}

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "Aneumo Temporal DeepONet V2"
echo "GPU: $GPU_ID | Mode: $MODE"
echo "Start: $(date)"
echo "========================================="

COMMON_ARGS="
    --epochs 10000
    --output_vars wss
    --num_wall_samples 8000
    --boundary_cut 0.1
    --cache_mode ram
    --save_interval 100
    --vis_interval 100
    --num_vis_samples 3
"

if [ "$MODE" = "with_swin" ]; then
    python -m aneumo.train \
        --history_encoder transformer \
        --checkpoint_dir checkpoint/v2_with_swin \
        $COMMON_ARGS
elif [ "$MODE" = "no_swin" ]; then
    python -m aneumo.train \
        --history_encoder transformer \
        --no_geometry \
        --checkpoint_dir checkpoint/v2_no_swin \
        $COMMON_ARGS
else
    echo "Unknown mode: $MODE. Use 'with_swin' or 'no_swin'."
    exit 1
fi

echo "========================================="
echo "Done: $(date)"
echo "========================================="
