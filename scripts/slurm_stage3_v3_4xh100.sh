#!/bin/bash
#SBATCH -p spiderverse-overflow
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J stage3_v3_4xh100
#SBATCH --exclude=bully-maguire
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# Stage 3 v3 on 4x H100 (spiderverse-overflow partition, 4h max).
#   Uses Stage 2 v3 LoRA adapter (rank 32, alpha 64, dropout 0.05, all-linear, lr 2e-4).
#
# Output: outputs/stage3_multimodal_v3_4xh100/   (fully isolated)
# Does NOT touch stage3_multimodal/, stage3_multimodal_v2/, v1, or any other run.
#
# Uses ON-THE-FLY speech features only (no precompute step).
#
# Inputs:
#   outputs/stage1_alignment/projector.pt
#   outputs/stage2_text_lora_v3/lora_adapters/
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pizza_time/hf_cache

# ── Config ───────────────────────────────────────────────────────────────────
STAGE1_DIR="${STAGE1_DIR:-outputs/stage1_alignment}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_text_lora_v3}"
STAGE3_CONFIG="${STAGE3_CONFIG:-configs/stage3_multimodal_v3_4xh100.yaml}"

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

if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$(( (${SLURM_CPUS_PER_TASK:-32} + 3) / 4 ))
fi
export TOKENIZERS_PARALLELISM=false

# NCCL hardening: surface rank divergence promptly and survive slow data loads.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

if ! python -c "import flash_attn" 2>/dev/null; then
    echo "WARNING: flash-attn not installed. Falling back to default attention kernels."
fi

# ── Sanity checks ────────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 3 v3: 4xH100 on spiderverse-overflow, DDP torchrun, on-the-fly features"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:       $(date)"
echo "Stage 1:    $STAGE1_DIR/projector.pt"
echo "Stage 2 v3: $STAGE2_DIR/lora_adapters/"
echo "Config:     $STAGE3_CONFIG"
echo "Output:     outputs/stage3_multimodal_v3_4xh100/"
echo "============================================================"

if [ ! -f "$STAGE1_DIR/projector.pt" ]; then
    echo "ERROR: missing $STAGE1_DIR/projector.pt"; exit 1
fi
if [ ! -d "$STAGE2_DIR/lora_adapters" ]; then
    echo "ERROR: missing $STAGE2_DIR/lora_adapters/"; exit 1
fi
if [ ! -f "$STAGE3_CONFIG" ]; then
    echo "ERROR: missing $STAGE3_CONFIG"; exit 1
fi

# ── Stage 3 (no precompute) ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STAGE 3: Multimodal merge (on-the-fly speech features)"
echo "============================================================"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    src/train.py \
    --config "$STAGE3_CONFIG" \
    --stage1_checkpoint "$STAGE1_DIR" \
    --stage2_checkpoint "$STAGE2_DIR"

echo "============================================================"
echo "Stage 3 v3 4xH100 COMPLETE"
echo "Output: outputs/stage3_multimodal_v3_4xh100/"
echo "Date: $(date)"
echo "============================================================"
