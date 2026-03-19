#!/bin/bash
# =============================================================================
# Local smoke-run of the full training pipeline (Stage 0 → 1 → 2 → 3)
# Uses 20 samples, 1 epoch, tiny batches — runs on CPU/MPS, no GPU required.
#
# Usage:
#   bash scripts/train_local.sh            # run all stages
#   START_STAGE=1 bash scripts/train_local.sh   # skip stage 0
#   START_STAGE=2 END_STAGE=2 bash scripts/train_local.sh  # only stage 2
# =============================================================================

set -euo pipefail

START_STAGE="${START_STAGE:-0}"
END_STAGE="${END_STAGE:-3}"
SKIP_STAGE0="${SKIP_STAGE0:-0}"

STAGE0_DIR="outputs/local_stage0"
STAGE1_DIR="outputs/local/stage1_asr"
STAGE2_DIR="outputs/local/stage2_text_lora"
STAGE3_DIR="outputs/local/stage3_multitask"
ALIGNMENT_DATA="data/alignments/local_alignment_data.pt"

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

# ── Stage 0: Precompute alignments + MSE pre-alignment ──────────────────────
if [ "$START_STAGE" -le 0 ] && [ "$END_STAGE" -ge 0 ] && [ "$SKIP_STAGE0" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 0: Precompute alignments + MSE pre-alignment (20 samples)"
    echo "============================================================"

    if [ ! -f "$ALIGNMENT_DATA" ]; then
        echo "Precomputing speech features + text embeddings..."
        python scripts/precompute_alignments.py \
            --output_dir data/alignments \
            --max_samples 20
        # Rename to local-specific name
        mv data/alignments/alignment_data.pt "$ALIGNMENT_DATA"
    else
        echo "Alignment data already exists at $ALIGNMENT_DATA, skipping precompute."
    fi

    echo "Training Stage 0 projector (MSE pre-alignment)..."
    python src/train_stage0.py \
        --config configs/local_stage0.yaml \
        --data "$ALIGNMENT_DATA" \
        --output_dir "$STAGE0_DIR"

    echo "Stage 0 complete: $STAGE0_DIR/projector.pt"
fi

# ── Stage 1: Projector alignment on ASR ──────────────────────────────────────
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 1: Projector alignment (ASR, LLM frozen, 20 samples)"
    echo "============================================================"

    STAGE1_ARGS=(
        --config configs/local_stage1.yaml
    )

    if [ -f "$STAGE0_DIR/projector.pt" ]; then
        STAGE1_ARGS+=(--stage1_checkpoint "$STAGE0_DIR")
        echo "Warm-starting from Stage 0 projector"
    fi

    python src/train.py "${STAGE1_ARGS[@]}"
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
