#!/bin/bash
# Stage 3: Multimodal merge (projector + LoRA jointly)
# Requires Stage 1 and Stage 2 checkpoints
set -e

STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-output/stage1}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-output/stage2}

export STAGE1_CHECKPOINT STAGE2_CHECKPOINT

echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Stage 2 checkpoint: $STAGE2_CHECKPOINT"

swift sft --config configs/swift/stage3_multimodal.yaml "$@"
