#!/usr/bin/env bash
# Generate synthetic data for T8-T10 (audio-aware tasks) with Gemma-4-E4B-it.
#
# Prerequisite: .venv_gen activated (generation-only env with torch 2.10 + vllm 0.19).
#   cd /work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following
#   source .venv_gen/bin/activate
#   module load cuda/12.8
#
# Launches 2 DP replicas, one per GPU. By default uses GPU 2 and 3
# (leaves 0,1 for the text-generator script).
# Override via: GPUS="0,1" bash scripts/gen_audio.sh
#
# Outputs: data/synthetic/<task>/gemma-4-E4B-it/part-<replica>.jsonl
#
# Usage:
#   bash scripts/gen_audio.sh                # full run (5k per task × 3 = 15k)
#   SMOKE=1 bash scripts/gen_audio.sh        # 500-sample smoke per replica
#   TASKS="T8" bash scripts/gen_audio.sh     # only specific tasks

set -euo pipefail

PROJECT_ROOT="/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following"
cd "${PROJECT_ROOT}"

AUDIO_MODEL_PATH="${AUDIO_MODEL_PATH:-${PROJECT_ROOT}/models/gemma-4-E4B-it}"
GPUS="${GPUS:-2,3}"
TASKS="${TASKS:-T8,T9,T10}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-5000}"
SMOKE="${SMOKE:-0}"
# Per-replica GPU memory share. E4B is tiny (~16GB), so 3–4 replicas fit
# comfortably on one 80GB A100 or 141GB H200.
#   GPUS="0,0,0,0" GPU_MEM_UTIL=0.22  -> 4 replicas on one A100
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

if [[ "${SMOKE}" == "1" ]]; then
    SAMPLES_PER_TASK=500
    echo ">>> SMOKE MODE: ${SAMPLES_PER_TASK} samples per task"
fi

OUTPUT_DIR="${PROJECT_ROOT}/data/synthetic"
LOG_DIR="${PROJECT_ROOT}/logs/gen_audio"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_REPLICAS=${#GPU_LIST[@]}

echo ">>> Audio generation"
echo "    model   : ${AUDIO_MODEL_PATH}"
echo "    tasks   : ${TASKS}"
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
        --model "${AUDIO_MODEL_PATH}" \
        --tasks "${TASKS}" \
        --modality audio \
        --samples-per-task "${SAMPLES_PER_TASK}" \
        --replica-id "${i}" \
        --num-replicas "${NUM_REPLICAS}" \
        --source-model-tag "gemma-4-E4B-it" \
        --output-dir "${OUTPUT_DIR}" \
        --prompts-dir "${PROJECT_ROOT}/prompts/synthetic" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len 8192 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
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
    echo ">>> Audio generation had failures — check logs in ${LOG_DIR}"
    exit 1
fi

echo ">>> Audio generation complete"
echo ">>> Outputs under ${OUTPUT_DIR}"
