#!/bin/bash
#SBATCH -A atathe2-ic
#SBATCH -p IllinoisComputes-GPU
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J iwslt2026
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# IWSLT 2026 — Full training pipeline (Stage 0 → 1 → 2 → 3) + HF upload
#
# Submit:
#   sbatch scripts/slurm_train.sh
#
# Skip Stage 0 / resume from a later stage:
#   sbatch --export=SKIP_STAGE0=1 scripts/slurm_train.sh
#   sbatch --export=START_STAGE=3 scripts/slurm_train.sh
#
# Interactive GPU shell (for debugging):
#   srun -A atathe2-ic -p IllinoisComputes-GPU --gres=gpu:H200:1 \
#        --cpus-per-task=8 --mem=64G -t 01:30:00 --pty bash
#
# Outputs:
#   outputs/stage0_prealign/   — Stage 0 projector
#   outputs/stage1_asr/        — Stage 1 projector
#   outputs/stage2_text_lora/  — Stage 2 LoRA adapters
#   outputs/stage3_multitask/  — Stage 3 final checkpoint (uploaded to HF)
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
HF_REPO="anand-kamble/iwslt2026-stage3"
START_STAGE="${START_STAGE:-0}"
SKIP_STAGE0="${SKIP_STAGE0:-0}"

STAGE0_DIR="outputs/stage0_prealign"
STAGE1_DIR="outputs/stage1_asr"
STAGE2_DIR="outputs/stage2_text_lora"
STAGE3_DIR="outputs/stage3_multitask"
ALIGNMENT_DATA="data/alignments/alignment_data.pt"

# ── Wandb ────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-iwslt2026}"
# Store run data on scratch to avoid NFS quota issues
_SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
export WANDB_DIR="${WANDB_DIR:-${_SCRATCH}/wandb_${SLURM_JOB_ID:-local}}"
mkdir -p "$WANDB_DIR"
# Set WANDB_MODE=offline to train without internet, sync later with `wandb sync`
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11.11 cuda/12.8

echo "============================================================"
echo "IWSLT 2026 Training Pipeline"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start stage: ${START_STAGE}"
echo "Date:        $(date)"
echo "============================================================"

mkdir -p logs
source .venv/bin/activate

# ── Stage 0: Precompute alignments + MSE pre-alignment ──────────────────────
if [ "$START_STAGE" -le 0 ] && [ "$SKIP_STAGE0" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 0: Precompute alignments + MSE pre-alignment"
    echo "============================================================"

    if [ ! -f "$ALIGNMENT_DATA" ]; then
        echo "Precomputing speech features + text embeddings..."
        python scripts/precompute_alignments.py \
            --output_dir data/alignments \
            --max_samples 30000
    else
        echo "Alignment data already exists at $ALIGNMENT_DATA, skipping precompute."
    fi

    echo "Training Stage 0 projector (MSE pre-alignment)..."
    python src/train_stage0.py \
        --config configs/stage0_prealign.yaml \
        --data "$ALIGNMENT_DATA" \
        --output_dir "$STAGE0_DIR"

    echo "Stage 0 complete: $STAGE0_DIR/projector.pt"
fi

# ── Stage 1: Projector alignment on ASR ──────────────────────────────────────
if [ "$START_STAGE" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 1: Projector alignment (ASR, LLM frozen)"
    echo "============================================================"

    STAGE1_ARGS=(
        --config configs/stage1_alignment.yaml
    )

    # Load Stage 0 projector if available
    if [ -f "$STAGE0_DIR/projector.pt" ]; then
        STAGE1_ARGS+=(--stage1_checkpoint "$STAGE0_DIR")
        echo "Warm-starting from Stage 0 projector"
    fi

    python src/train.py "${STAGE1_ARGS[@]}"
    echo "Stage 1 complete: $STAGE1_DIR"
fi

# ── Stage 2: Text-only LoRA pre-training ─────────────────────────────────────
if [ "$START_STAGE" -le 2 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 2: Text-only LoRA (MT + SQA, no audio)"
    echo "============================================================"

    python src/train.py \
        --config configs/stage2_text_lora.yaml

    echo "Stage 2 complete: $STAGE2_DIR"
fi

# ── Stage 3: Multimodal merge ────────────────────────────────────────────────
if [ "$START_STAGE" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 3: Multimodal merge (projector + LoRA)"
    echo "============================================================"

    STAGE3_ARGS=(
        --config configs/stage3_multimodal.yaml
    )

    # Load Stage 1 projector
    if [ -d "$STAGE1_DIR" ]; then
        STAGE3_ARGS+=(--stage1_checkpoint "$STAGE1_DIR")
        echo "Loading projector from Stage 1"
    else
        echo "WARNING: No Stage 1 checkpoint found at $STAGE1_DIR"
    fi

    # Load Stage 2 LoRA
    if [ -d "$STAGE2_DIR" ]; then
        STAGE3_ARGS+=(--stage2_checkpoint "$STAGE2_DIR")
        echo "Loading LoRA from Stage 2"
    else
        echo "WARNING: No Stage 2 checkpoint found at $STAGE2_DIR"
    fi

    python src/train.py "${STAGE3_ARGS[@]}"
    echo "Stage 3 complete: $STAGE3_DIR"
fi

# ── Upload to HuggingFace ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Uploading checkpoint to HuggingFace"
echo "============================================================"

if [ -d "$STAGE3_DIR" ]; then
    echo "Uploading $STAGE3_DIR → hf.co/$HF_REPO"
    huggingface-cli upload "$HF_REPO" "$STAGE3_DIR" --repo-type model
    echo "Upload complete: https://huggingface.co/$HF_REPO"
else
    echo "ERROR: Stage 3 output not found at $STAGE3_DIR"
    exit 1
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Date: $(date)"
