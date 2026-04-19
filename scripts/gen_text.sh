#!/usr/bin/env bash
# Generate synthetic data for T1-T7 (text tasks) with Gemma-4-31B.
#
# Prerequisite: .venv_gen activated (generation-only env with torch 2.10 + vllm 0.19).
#   cd /work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following
#   source .venv_gen/bin/activate
#   module load cuda/12.8
#
# Launches 2 DP replicas, one per GPU. By default uses GPU 0 and 1.
# Override via: GPUS="2,3" bash scripts/gen_text.sh
#
# Each replica processes half of the task-pair list. Outputs are written
# under data/synthetic/<task>/gemma-4-31B/part-<replica>.jsonl
#
# Usage:
#   bash scripts/gen_text.sh                           # full run (5k per task × 7 = 35k) on covost2
#   SMOKE=1 bash scripts/gen_text.sh                   # 500-sample smoke per replica
#   TASKS="T1,T2" bash scripts/gen_text.sh             # only specific tasks
#   SOURCE=ytseg TASKS="T3,T6,T7" bash scripts/gen_text.sh  # gist tasks on long YTSeg chapters

set -euo pipefail

PROJECT_ROOT="/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following"
cd "${PROJECT_ROOT}"

TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-${PROJECT_ROOT}/models/gemma-4-31B}"
GPUS="${GPUS:-0,1}"
TASKS="${TASKS:-T1,T2,T3,T4,T5,T6,T7}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-5000}"
SMOKE="${SMOKE:-0}"
SOURCE="${SOURCE:-covost2}"
# Per-replica GPU memory share. Lower this to pack multiple replicas on the
# same GPU — e.g. GPUS="0,0" GPU_MEM_UTIL=0.45 for two replicas on one H200.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

if [[ "${SMOKE}" == "1" ]]; then
    SAMPLES_PER_TASK=500
    echo ">>> SMOKE MODE: ${SAMPLES_PER_TASK} samples per task"
fi

OUTPUT_DIR="${PROJECT_ROOT}/data/synthetic"
LOG_DIR="${PROJECT_ROOT}/logs/gen_text"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_REPLICAS=${#GPU_LIST[@]}

echo ">>> Text generation"
echo "    model   : ${TEXT_MODEL_PATH}"
echo "    tasks   : ${TASKS}"
echo "    source  : ${SOURCE}"
echo "    samples : ${SAMPLES_PER_TASK} per task × ${NUM_REPLICAS} replicas (split)"
echo "    gpus    : ${GPUS} (${NUM_REPLICAS} DP replicas)"
echo "    output  : ${OUTPUT_DIR}"
echo ""

PIDS=()
for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    log="${LOG_DIR}/replica-${i}-gpu${gpu}.log"
    echo ">>> Launching replica ${i} on GPU ${gpu} (log: ${log})"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HOME="${HF_HOME:-/work/nvme/bggk/atathe/hf_cache}" \
    FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/work/nvme/bggk/atathe/flashinfer_cache}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/work/nvme/bggk/atathe/xdg_cache}" \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/nvme/bggk/atathe/triton_cache}" \
    "${PROJECT_ROOT}/.venv_gen/bin/python" -m scripts.generate_synthetic \
        --model "${TEXT_MODEL_PATH}" \
        --tasks "${TASKS}" \
        --modality text \
        --samples-per-task "${SAMPLES_PER_TASK}" \
        --replica-id "${i}" \
        --num-replicas "${NUM_REPLICAS}" \
        --source-model-tag "gemma-4-31B" \
        --output-dir "${OUTPUT_DIR}" \
        --prompts-dir "${PROJECT_ROOT}/prompts/synthetic" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len 4096 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --source "${SOURCE}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

echo ">>> Replica PIDs: ${PIDS[*]}"
echo ">>> Tail logs with:  tail -f ${LOG_DIR}/replica-*.log"
echo ">>> Waiting for all replicas to finish..."

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo ">>> Replica with PID ${pid} FAILED"
        FAIL=1
    fi
done

if [[ "${FAIL}" == "1" ]]; then
    echo ">>> Text generation had failures — check logs in ${LOG_DIR}"
    exit 1
fi

echo ">>> Text generation complete"
echo ">>> Outputs under ${OUTPUT_DIR}"
