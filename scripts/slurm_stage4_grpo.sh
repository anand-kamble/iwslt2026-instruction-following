#!/bin/bash
#SBATCH -A spidey-pointing-research
#SBATCH -p web-slinger-gpu
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 16:00:00
#SBATCH -J stage4_grpo
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# =============================================================================
# Stage 4: text-only GRPO on Qwen3-4B + co-deployed Prometheus-7B judge.
#
# GPU layout (defaults; override via env vars):
#   TRAIN_GPU=0     — policy + vLLM colocate + null_ref_context reference
#   JUDGE_GPU=1     — Prometheus-7B-v2.0 inference server (swift deploy)
#
# Prerequisites:
#   - data/grpo/train.jsonl  — built by scripts/prepare_grpo_data.py
#   - STAGE2_CHECKPOINT env var — path to Stage 2 / Stage 3 LoRA adapter dir
#   - ms-swift + vllm + comet + openai installed in the active venv/conda
#
# Launch:
#   STAGE2_CHECKPOINT=outputs/stage2_text_lora_v2 sbatch scripts/slurm_stage4_grpo.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONUNBUFFERED=1

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load python/3.11.11 cuda/12.8

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Create it with: python -m venv .venv && pip install -e .[cuda,eval,dev]"
    exit 1
fi
source .venv/bin/activate
mkdir -p logs data/grpo outputs

# ── Caches off NFS ───────────────────────────────────────────────────────────
export HF_HOME=/scratch/pizza_time/hf_cache
export HF_DATASETS_CACHE=/scratch/pizza_time/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/pizza_time/hf_cache/transformers
export TORCH_HOME=/scratch/pizza_time/torch
export TRITON_CACHE_DIR=/scratch/pizza_time/triton_cache
export VLLM_CACHE_DIR=/scratch/pizza_time/vllm_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
         "$TORCH_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export WANDB_PROJECT="${WANDB_PROJECT:-iwslt2026}"
_SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}}"
export WANDB_DIR="${WANDB_DIR:-${_SCRATCH}/wandb_${SLURM_JOB_ID:-local}}"
mkdir -p "$WANDB_DIR"

# ── Required inputs ──────────────────────────────────────────────────────────
: "${STAGE2_CHECKPOINT:?set STAGE2_CHECKPOINT=<path to stage 2/3 LoRA dir>}"
DATA_JSONL="${DATA_JSONL:-data/grpo/train.jsonl}"
CONFIG="${CONFIG:-configs/swift/stage4_grpo.yaml}"

if [ ! -f "$DATA_JSONL" ]; then
    echo "ERROR: missing $DATA_JSONL. Run:"
    echo "    python scripts/prepare_grpo_data.py \\"
    echo "        --mix configs/swift/stage4_mix.yaml \\"
    echo "        --config configs/stage2_text_lora_v2.yaml \\"
    echo "        --out $DATA_JSONL"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then echo "ERROR: missing $CONFIG"; exit 1; fi
if [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "ERROR: missing Stage 2 checkpoint dir: $STAGE2_CHECKPOINT"; exit 1
fi

# ── GPU assignment ───────────────────────────────────────────────────────────
TRAIN_GPU="${TRAIN_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
PROMETHEUS_MODEL="${PROMETHEUS_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-8000}"
export PROMETHEUS_URL="http://127.0.0.1:${PROMETHEUS_PORT}/v1"
export PROMETHEUS_API_KEY="${PROMETHEUS_API_KEY:-EMPTY}"

JUDGE_LOG="logs/${SLURM_JOB_NAME:-stage4_grpo}-${SLURM_JOB_ID:-local}.judge.log"

# ── Sanity prints ────────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage 4 GRPO — text-only Qwen3-4B + Prometheus judge"
echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Node:            $(hostname)"
echo "Train GPU:       $TRAIN_GPU"
echo "Judge GPU:       $JUDGE_GPU"
echo "Prometheus URL:  $PROMETHEUS_URL"
echo "Dataset:         $DATA_JSONL ($(wc -l < "$DATA_JSONL") rows)"
echo "Stage 2 ckpt:    $STAGE2_CHECKPOINT"
echo "swift:           $(which swift)"
nvidia-smi -L || true
echo "============================================================"

# ── Launch Prometheus judge in background ────────────────────────────────────
echo "[judge] starting Prometheus server on GPU $JUDGE_GPU ..."
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" swift deploy \
    --model "$PROMETHEUS_MODEL" \
    --infer_backend vllm \
    --port "$PROMETHEUS_PORT" \
    --torch_dtype bfloat16 \
    --max_new_tokens 1000 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 8192 \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!
trap 'echo "[judge] shutting down Prometheus (pid $JUDGE_PID)"; kill "$JUDGE_PID" 2>/dev/null || true' EXIT

echo "[judge] PID=$JUDGE_PID — log: $JUDGE_LOG"
echo "[judge] waiting for readiness at $PROMETHEUS_URL ..."
for attempt in $(seq 1 120); do
    if kill -0 "$JUDGE_PID" 2>/dev/null; then :; else
        echo "[judge] process died during startup. Tail of log:"
        tail -40 "$JUDGE_LOG"; exit 1
    fi
    if curl -fs --max-time 2 "http://127.0.0.1:${PROMETHEUS_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[judge] ready after ${attempt}0s."
        break
    fi
    sleep 10
    if [ "$attempt" -eq 120 ]; then
        echo "[judge] timeout waiting for server. Tail of log:"
        tail -40 "$JUDGE_LOG"; exit 1
    fi
done

# ── GRPO training ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "GRPO training starting on GPU $TRAIN_GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" swift rlhf "$CONFIG" \
    --resume_from_checkpoint "$STAGE2_CHECKPOINT" \
    --resume_only_model true \
    --ignore_data_skip true

echo "============================================================"
echo "Stage 4 GRPO COMPLETE — $(date)"
echo "Output: outputs/stage4_grpo/"
echo "Judge log: $JUDGE_LOG"
echo "============================================================"
