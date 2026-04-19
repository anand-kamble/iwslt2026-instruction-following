#!/usr/bin/env bash
# Download synthetic-data generator models to local NVMe.
#
# Models:
#   google/gemma-4-31B       — text generator for T1-T7 (≈62 GB bf16)
#   google/gemma-4-E4B-it    — audio-aware generator for T8-T10 (≈16 GB bf16)
#
# Both are Gemma-licensed; you must have accepted the license at
#   https://huggingface.co/google/gemma-4-31B
#   https://huggingface.co/google/gemma-4-E4B-it
# before this script will succeed. `huggingface-cli login` with a HF token
# that has accepted the license.
#
# Usage:
#   bash scripts/download_generator_models.sh            # both models
#   bash scripts/download_generator_models.sh text       # only gemma-4-31B
#   bash scripts/download_generator_models.sh audio      # only gemma-4-E4B-it

set -euo pipefail

PROJECT_ROOT="/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following"
MODELS_DIR="${PROJECT_ROOT}/models"

TEXT_MODEL_REPO="google/gemma-4-31B"
AUDIO_MODEL_REPO="google/gemma-4-E4B-it"

TEXT_MODEL_DIR="${MODELS_DIR}/gemma-4-31B"
AUDIO_MODEL_DIR="${MODELS_DIR}/gemma-4-E4B-it"

WHICH="${1:-both}"

mkdir -p "${MODELS_DIR}"

# Fast downloads: hf_transfer is ~3-5x faster than the default backend.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Verify hf_transfer is installed (standalone package, not an extra).
if ! python -c "import hf_transfer" 2>/dev/null; then
    echo ">>> Installing hf_transfer for faster downloads..."
    pip install -q hf_transfer
fi

# Verify the user is logged in (needed for gated Gemma models).
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "ERROR: not logged in to HuggingFace. Run:"
    echo "    huggingface-cli login"
    echo "and paste a token that has accepted the Gemma license."
    exit 1
fi

download_model() {
    local repo="$1"
    local dest="$2"
    local label="$3"

    if [[ -f "${dest}/config.json" ]]; then
        echo ">>> ${label}: already present at ${dest} (skipping)"
        return 0
    fi

    echo ">>> ${label}: downloading ${repo} → ${dest}"
    mkdir -p "${dest}"

    # --exclude original consolidation files to avoid duplicate weights
    # (HF repos sometimes ship both safetensors and pytorch_model.bin).
    huggingface-cli download "${repo}" \
        --local-dir "${dest}" \
        --exclude "*.bin" "*.pth" "consolidated.*" "original/*" \
        --max-workers 8

    # Sanity check: ensure config + at least one safetensors shard landed.
    if [[ ! -f "${dest}/config.json" ]]; then
        echo "ERROR: ${label} download failed — config.json missing"
        exit 1
    fi

    if ! ls "${dest}"/*.safetensors >/dev/null 2>&1; then
        echo "ERROR: ${label} download failed — no safetensors files"
        exit 1
    fi

    local size
    size=$(du -sh "${dest}" | cut -f1)
    echo ">>> ${label}: done (${size})"
}

case "${WHICH}" in
    both)
        download_model "${TEXT_MODEL_REPO}"  "${TEXT_MODEL_DIR}"  "text generator (Gemma-4-31B)"
        download_model "${AUDIO_MODEL_REPO}" "${AUDIO_MODEL_DIR}" "audio generator (Gemma-4-E4B-it)"
        ;;
    text)
        download_model "${TEXT_MODEL_REPO}"  "${TEXT_MODEL_DIR}"  "text generator (Gemma-4-31B)"
        ;;
    audio)
        download_model "${AUDIO_MODEL_REPO}" "${AUDIO_MODEL_DIR}" "audio generator (Gemma-4-E4B-it)"
        ;;
    *)
        echo "Usage: $0 [both|text|audio]"
        exit 1
        ;;
esac

echo ""
echo "==============================================="
echo "Generator models ready at:"
echo "  text  : ${TEXT_MODEL_DIR}"
echo "  audio : ${AUDIO_MODEL_DIR}"
echo ""
echo "Set these in your gen scripts:"
echo "  export TEXT_MODEL_PATH=${TEXT_MODEL_DIR}"
echo "  export AUDIO_MODEL_PATH=${AUDIO_MODEL_DIR}"
echo "==============================================="
