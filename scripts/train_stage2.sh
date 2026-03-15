#!/bin/bash
# Stage 2: Text-only LoRA pre-training (native Qwen3, no custom model)
set -e

swift sft --config configs/swift/stage2_text_lora.yaml "$@"
