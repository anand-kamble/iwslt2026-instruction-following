#!/bin/bash
# =============================================================================
# Local smoke-run of the full training pipeline (Stage 1 → 2 → 3)
# Uses 20 samples, 1 epoch, tiny batches — runs on CPU/MPS, no GPU required.
#
# Usage:
#   bash scripts/train_local.sh            # run all stages
#   START_STAGE=2 END_STAGE=2 bash scripts/train_local.sh  # only stage 2
# =============================================================================

set -euo pipefail

START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-3}"

STAGE1_DIR="outputs/local/stage1_asr"
STAGE2_DIR="outputs/local/stage2_text_lora"
STAGE3_DIR="outputs/local/stage3_multitask"

export WANDB_MODE=disabled

# Ensure flash-attn is installed if running on CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    if ! python -c "import flash_attn" 2>/dev/null; then
        echo "Installing flash-attn..."
        pip install wheel && pip install flash-attn --no-build-isolation
    fi
fi

echo "============================================================"
echo "IWSLT 2026 Local Training Pipeline (20 samples)"
echo "============================================================"
echo "Stages:  ${START_STAGE} → ${END_STAGE}"
echo "Date:    $(date)"
echo "============================================================"

# ── Stage 1: Projector alignment on ASR (NAVER Stage A) ─────────────────────
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 1: Projector alignment (ASR, LLM frozen, 20 samples)"
    echo "============================================================"

    python src/train.py --config configs/local_stage1.yaml
    echo "Stage 1 complete: $STAGE1_DIR"
fi

# ── Stage 2: Text-only LoRA pre-training ─────────────────────────────────────
if [ "$START_STAGE" -le 2 ] && [ "$END_STAGE" -ge 2 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 2: Text-only LoRA (MT, 20 samples)"
    echo "============================================================"

    python src/train.py \
        --config configs/local_stage2.yaml

    echo "Stage 2 complete: $STAGE2_DIR"
fi

# ── Stage 3: Multimodal merge ────────────────────────────────────────────────
if [ "$START_STAGE" -le 3 ] && [ "$END_STAGE" -ge 3 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 3: Multimodal merge (20 samples)"
    echo "============================================================"

    STAGE3_ARGS=(
        --config configs/local_stage3.yaml
    )

    if [ -d "$STAGE1_DIR" ]; then
        STAGE3_ARGS+=(--stage1_checkpoint "$STAGE1_DIR")
        echo "Loading projector from Stage 1"
    else
        echo "WARNING: No Stage 1 checkpoint found at $STAGE1_DIR"
    fi

    if [ -d "$STAGE2_DIR" ]; then
        STAGE3_ARGS+=(--stage2_checkpoint "$STAGE2_DIR")
        echo "Loading LoRA from Stage 2"
    else
        echo "WARNING: No Stage 2 checkpoint found at $STAGE2_DIR"
    fi

    python src/train.py "${STAGE3_ARGS[@]}"
    echo "Stage 3 complete: $STAGE3_DIR"
fi

echo ""
echo "============================================================"
echo "LOCAL PIPELINE COMPLETE"
echo "============================================================"
echo "Date: $(date)"
