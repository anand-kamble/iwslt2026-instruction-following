#!/bin/bash
#SBATCH -A atathe2-ic
#SBATCH -p IllinoisComputes-GPU
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J stage2_v3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# Stage 2 v3: Text-only LoRA — higher-capacity variant
#   LoRA rank 32, alpha 64, lr 2e-4, 1 epoch
# Isolated run — outputs to outputs/stage2_text_lora_v3/
# Does NOT touch outputs/stage2_text_lora/ (v1) or outputs/stage2_text_lora_v2/
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/atathe2/hf_cache

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

# ── Sanity prints ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 2 v3: Text-only LoRA (rank 32, alpha 64, lr 2e-4)"
echo "============================================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo "Output:  outputs/stage2_text_lora_v3/"
echo "============================================================"

# ── Generate translated SQA (if missing) ────────────────────────────────────
SQA_DIR="data/sqa_translated"
if [ ! -f "$SQA_DIR/librisqa_de.jsonl" ] || \
   [ ! -f "$SQA_DIR/librisqa_it.jsonl" ] || \
   [ ! -f "$SQA_DIR/librisqa_zh.jsonl" ]; then
    echo "Generating translated SQA (de, it, zh)..."
    python -m src.data.translate_sqa \
        --output_dir "$SQA_DIR" \
        --languages de it zh
    echo "Translated SQA complete: $SQA_DIR"
else
    echo "Translated SQA already exists at $SQA_DIR, skipping"
fi

# ── Stage 2 v3 training ────────────────────────────────────────────────────
python src/train.py \
    --config configs/stage2_text_lora_v3.yaml

echo "============================================================"
echo "Stage 2 v3 COMPLETE"
echo "Output: outputs/stage2_text_lora_v3/"
echo "Date: $(date)"
echo "============================================================"
