#!/bin/bash
# Stage 1: Projector alignment on ASR
# Trains only the projector (speech encoder and LLM frozen)
set -e

swift sft --config configs/swift/stage1_projector.yaml "$@"
