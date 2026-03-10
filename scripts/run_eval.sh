#!/usr/bin/env bash
set -euo pipefail

# Run evaluation for a task.
# Usage: ./scripts/run_eval.sh <task_config> <checkpoint>
#
# Example:
#   ./scripts/run_eval.sh configs/asr.yaml your-org/iwslt2026-asr-v1

CONFIG="${1:?Usage: run_eval.sh <task_config> <checkpoint>}"
CHECKPOINT="${2:?Usage: run_eval.sh <task_config> <checkpoint>}"

python src/evaluate.py --config "$CONFIG" --checkpoint "$CHECKPOINT"
