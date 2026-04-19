#!/bin/bash
#SBATCH -A atathe2-ic
#SBATCH -p IllinoisComputes-GPU
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J stage3_v1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# Stage 3 v1: Multimodal merge using Stage 2 v1 LoRA adapter
#   (original hyperparams: rank 8, alpha 16, specific target modules, lr 3e-4)
# Isolated run — outputs to outputs/stage3_multimodal/
# Does NOT touch stage3_multimodal_v2/ or any other artifacts.
#
# Speech encoder features are computed ON THE FLY per batch (no pre-extraction).
# The collator drops any sample whose audio fails to load; a single corrupt
# file cannot crash the run.
#
# Inputs:
#   outputs/stage1_alignment/projector.pt         (pre-extracted)
#   outputs/stage2_text_lora/lora_adapters/       (Stage 2 v1 LoRA)
#
# Output:
#   outputs/stage3_multimodal/                    (final multimodal checkpoint)
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/atathe2/hf_cache

# ── Config ───────────────────────────────────────────────────────────────────
STAGE1_DIR="outputs/stage1_alignment"
STAGE2_DIR="outputs/stage2_text_lora"
STAGE3_CONFIG="configs/stage3_multimodal.yaml"

# ── Wandb ────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-iwslt2026}"
_SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
export WANDB_DIR="${WANDB_DIR:-${_SCRATCH}/wandb_${SLURM_JOB_ID:-local}}"
mkdir -p "$WANDB_DIR"
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11.11 cuda/12.8

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found."
    exit 1
fi
source .venv/bin/activate
mkdir -p logs

# ── Sanity checks ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 3 v1: Multimodal merge (using Stage 2 v1 LoRA)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:       $(date)"
echo "Stage 1:    $STAGE1_DIR/projector.pt"
echo "Stage 2 v1: $STAGE2_DIR/lora_adapters/"
echo "Config:     $STAGE3_CONFIG"
echo "Output:     outputs/stage3_multimodal/"
echo "============================================================"

# Fail fast if required inputs are missing
if [ ! -f "$STAGE1_DIR/projector.pt" ]; then
    echo "ERROR: missing $STAGE1_DIR/projector.pt"
    exit 1
fi
if [ ! -d "$STAGE2_DIR/lora_adapters" ]; then
    echo "ERROR: missing $STAGE2_DIR/lora_adapters/"
    exit 1
fi
if [ ! -f "$STAGE3_CONFIG" ]; then
    echo "ERROR: missing $STAGE3_CONFIG"
    exit 1
fi

# ── Stage 3: Multimodal training (on-the-fly feature extraction) ──────────
echo ""
echo "============================================================"
echo "STAGE 3: Multimodal merge (projector + LoRA), features on-the-fly"
echo "============================================================"
python src/train.py \
    --config "$STAGE3_CONFIG" \
    --stage1_checkpoint "$STAGE1_DIR" \
    --stage2_checkpoint "$STAGE2_DIR"

echo "============================================================"
echo "Stage 3 v1 COMPLETE"
echo "Output: outputs/stage3_multimodal/"
echo "Date: $(date)"
echo "============================================================"
