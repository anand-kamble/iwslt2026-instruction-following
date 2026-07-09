#!/bin/bash
#SBATCH -A spidey-pointing-research
#SBATCH -p web-slinger-gpu
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH -J stage3_4xh100
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# Stage 3 submission run on a single 4xH100 node (DDP via torchrun).
#
# Uses ON-THE-FLY speech features only (no precompute/extract_features step).
# Requires:
#   - Stage 1 projector checkpoint: outputs/stage1_alignment/projector.pt
#   - Stage 2 LoRA adapters: outputs/stage2_text_lora/lora_adapters
#
# Submit:
#   sbatch scripts/slurm_stage3_4xh100.sh
#
# Override paths/config:
#   sbatch --export=ALL,STAGE1_DIR=...,STAGE2_DIR=...,STAGE3_CONFIG=... \
#       scripts/slurm_stage3_4xh100.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pizza_time/hf_cache

# ── Config ───────────────────────────────────────────────────────────────────
STAGE1_DIR="${STAGE1_DIR:-outputs/stage1_alignment}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_text_lora}"
STAGE3_CONFIG="${STAGE3_CONFIG:-configs/stage3_multimodal_4xh100.yaml}"

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

# Keep CPU threads balanced across the 4 torchrun workers.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$(( (${SLURM_CPUS_PER_TASK:-32} + 3) / 4 ))
fi
export TOKENIZERS_PARALLELISM=false

if ! python -c "import flash_attn" 2>/dev/null; then
    echo "WARNING: flash-attn not installed. Falling back to default attention kernels."
fi

# ── Sanity checks ────────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 3 submission: 4xH100, DDP torchrun, on-the-fly features"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:       $(date)"
echo "Stage 1:    $STAGE1_DIR/projector.pt"
echo "Stage 2:    $STAGE2_DIR/lora_adapters/"
echo "Config:     $STAGE3_CONFIG"
echo "============================================================"

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
echo "Stage 3 4xH100 COMPLETE"
echo "Date: $(date)"
echo "============================================================"
