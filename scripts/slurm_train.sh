#!/bin/bash
#SBATCH -A atathe2-ic
#SBATCH -p IllinoisComputes-GPU
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J iwslt2026
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# IWSLT 2026 — Full training pipeline (Stage 1 → 2 → 3) + HF upload
# Following NAVER IWSLT 2025 (arxiv 2506.01808) 3-stage approach.
#
# Submit:
#   sbatch scripts/slurm_train.sh
#
# Resume from a later stage:
#   sbatch --export=ALL,START_STAGE=3 scripts/slurm_train.sh
#
# Run only one stage:
#   sbatch --export=ALL,START_STAGE=2,END_STAGE=2 scripts/slurm_train.sh
#
# Skip HF upload:
#   sbatch --export=ALL,SKIP_UPLOAD=1 scripts/slurm_train.sh
#
# Interactive GPU shell (for debugging):
#   srun -A atathe2-ic -p IllinoisComputes-GPU --gres=gpu:H200:1 \
#        --cpus-per-task=8 --mem=64G -t 01:30:00 --pty bash
#
# Outputs:
#   outputs/stage1_alignment/  — Stage 1 projector (NAVER Stage A)
#   outputs/stage2_text_lora/  — Stage 2 LoRA adapters (NAVER Stage B)
#   data/precomputed/          — Pre-extracted speech features (Stage 3a)
#   outputs/stage3_multimodal/ — Stage 3 final checkpoint (NAVER Stage C, uploaded to HF)
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # ensure we're in project root
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/atathe2/hf_cache

# ── Config ───────────────────────────────────────────────────────────────────
HF_REPO="anand-kamble/iwslt2026-stage3"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-99}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

STAGE1_DIR="outputs/stage1_alignment"
STAGE2_DIR="outputs/stage2_text_lora"
STAGE3_DIR="outputs/stage3_multimodal"
PRECOMPUTED_DIR="data/precomputed"

# ── Wandb ────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-iwslt2026}"
_SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
export WANDB_DIR="${WANDB_DIR:-${_SCRATCH}/wandb_${SLURM_JOB_ID:-local}}"
mkdir -p "$WANDB_DIR"
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11.11 cuda/12.8

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[cuda]'"
    exit 1
fi
source .venv/bin/activate
mkdir -p logs

# flash-attn must be pre-installed (compiles for 10-30 min, unsafe during jobs):
#   pip install wheel && pip install flash-attn --no-build-isolation
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "ERROR: flash-attn not installed. Run the pip command above in an interactive session first."
    exit 1
fi

# ── Failure trap ─────────────────────────────────────────────────────────────
_CURRENT_STAGE="setup"
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "============================================================"
        echo "PIPELINE FAILED at stage: ${_CURRENT_STAGE}"
        echo "Exit code: ${exit_code}"
        echo "Check logs: logs/iwslt2026-${SLURM_JOB_ID:-local}.err"
        echo "============================================================"
    fi
    # Finish any active wandb run
    python -c "
try:
    import wandb
    if wandb.run is not None:
        wandb.finish(exit_code=$exit_code)
except Exception:
    pass
" 2>/dev/null || true
}
trap cleanup EXIT

echo "============================================================"
echo "IWSLT 2026 Training Pipeline"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Stages:      ${START_STAGE} → ${END_STAGE}"
echo "Date:        $(date)"
echo "============================================================"

# ── Stage 1: Projector alignment on ASR (NAVER Stage A) ──────────────────────
if [ "$START_STAGE" -le 1 ] && [ "$END_STAGE" -ge 1 ]; then
    _CURRENT_STAGE="stage1"
    echo ""
    echo "============================================================"
    echo "STAGE 1: Projector alignment — NAVER Stage A (ASR, LLM frozen)"
    echo "============================================================"

    python src/train.py --config configs/stage1_alignment.yaml
    echo "Stage 1 complete: $STAGE1_DIR"
fi

# ── Stage 2: Text-only LoRA pre-training ─────────────────────────────────────
if [ "$START_STAGE" -le 2 ] && [ "$END_STAGE" -ge 2 ]; then
    _CURRENT_STAGE="stage2"
    echo ""
    echo "============================================================"
    echo "STAGE 2: Text-only LoRA (MT + SQA, no audio)"
    echo "============================================================"

    python src/train.py \
        --config configs/stage2_text_lora.yaml

    echo "Stage 2 complete: $STAGE2_DIR"
fi

# ── Stage 3: Pre-extract features + Multimodal merge ─────────────────────────
if [ "$START_STAGE" -le 3 ] && [ "$END_STAGE" -ge 3 ]; then
    _CURRENT_STAGE="stage3_extract"
    echo ""
    echo "============================================================"
    echo "STAGE 3a: Pre-extract speech encoder features"
    echo "============================================================"

    # Extract once, skip if already done (extract_features.py has resume support)
    STAGE3_CONFIG="${STAGE3_CONFIG:-configs/stage3_multimodal.yaml}"
    python src/extract_features.py \
        --config "$STAGE3_CONFIG" \
        --output_dir "$PRECOMPUTED_DIR" \
        --batch_size 32

    _CURRENT_STAGE="stage3"
    echo ""
    echo "============================================================"
    echo "STAGE 3b: Multimodal merge (projector + LoRA)"
    echo "============================================================"

    STAGE3_ARGS=(
        --config "$STAGE3_CONFIG"
        --precomputed_dir "$PRECOMPUTED_DIR"
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

# ── Upload to HuggingFace ────────────────────────────────────────────────────
if [ "$SKIP_UPLOAD" -eq 0 ] && [ "$START_STAGE" -le 3 ] && [ "$END_STAGE" -ge 3 ]; then
    _CURRENT_STAGE="upload"
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
fi

_CURRENT_STAGE="done"
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Date: $(date)"
