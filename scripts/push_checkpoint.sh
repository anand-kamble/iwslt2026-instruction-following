#!/usr/bin/env bash
set -euo pipefail

# Push a checkpoint to HuggingFace Hub.
# Usage: ./scripts/push_checkpoint.sh <checkpoint_dir> <repo_id>
#
# Example:
#   ./scripts/push_checkpoint.sh ./outputs/asr-v1 your-org/iwslt2026-asr-v1

CHECKPOINT_DIR="${1:?Usage: push_checkpoint.sh <checkpoint_dir> <repo_id>}"
REPO_ID="${2:?Usage: push_checkpoint.sh <checkpoint_dir> <repo_id>}"

echo "Pushing $CHECKPOINT_DIR to hf.co/$REPO_ID"

huggingface-cli upload "$REPO_ID" "$CHECKPOINT_DIR" --repo-type model

echo "Done: https://huggingface.co/$REPO_ID"
