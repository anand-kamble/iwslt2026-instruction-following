#!/usr/bin/env bash
set -euo pipefail

# Download permitted training datasets for the constrained setting.
# Each dataset goes into data/<dataset_name>/

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading datasets ==="
echo "Target directory: $DATA_DIR"
echo ""

# EuroParlST v1.1
# Download from: https://www.mllp.upv.es/europarl-st/
echo "=== EuroParlST ==="
if [ -d "$DATA_DIR/europarlst" ]; then
    echo "Already exists at $DATA_DIR/europarlst"
else
    echo "Download v1.1 from https://www.mllp.upv.es/europarl-st/"
    echo "Extract to: $DATA_DIR/europarlst/"
    echo "  wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz"
    echo "  tar xzf v1.1.tar.gz -C $DATA_DIR/europarlst"
    mkdir -p "$DATA_DIR/europarlst"
fi
echo ""

# LibriSpeech (audio for LibriSQA)
echo "=== LibriSpeech (for LibriSQA audio) ==="
if [ -d "$DATA_DIR/LibriSpeech" ]; then
    echo "Already exists at $DATA_DIR/LibriSpeech"
else
    echo "Downloading LibriSpeech train-clean-360 (~23GB)..."
    mkdir -p "$DATA_DIR/LibriSpeech"
    wget -q --show-progress -P "$DATA_DIR" \
        https://www.openslr.org/resources/12/train-clean-360.tar.gz || \
        echo "FAILED: Download manually from https://www.openslr.org/12/"
    if [ -f "$DATA_DIR/train-clean-360.tar.gz" ]; then
        echo "Extracting..."
        tar xzf "$DATA_DIR/train-clean-360.tar.gz" -C "$DATA_DIR"
        rm "$DATA_DIR/train-clean-360.tar.gz"
    fi
fi
echo ""

# GigaST (translations for GigaSpeech audio)
echo "=== GigaST ==="
echo "Audio: speechcolab/gigaspeech on HuggingFace (requires access request)"
echo "Translations: https://st-benchmark.github.io/resources/GigaST.html"
echo "  Place in: $DATA_DIR/gigast/{de,zh}/train.tsv"
echo "  TSV format: utterance_id<TAB>translation"
mkdir -p "$DATA_DIR/gigast/de" "$DATA_DIR/gigast/zh"
echo ""

# LibriSQA (metadata from HuggingFace, audio from LibriSpeech above)
echo "=== LibriSQA ==="
echo "Metadata auto-downloads from HuggingFace (ZihanZhao/LibriSQA)"
echo "Audio files come from LibriSpeech above"
echo ""

# NUTSHELL (summarization)
echo "=== NUTSHELL ==="
echo "Auto-downloads from HuggingFace (maikezu/abstract-gen-acl-17-22)"
echo ""

# YTSeg (audio chaptering)
echo "=== YTSeg ==="
echo "Auto-downloads from HuggingFace (retkowski/ytseg)"
echo ""

# MCIF (validation)
echo "=== MCIF ==="
echo "Auto-downloads from HuggingFace (FBK-MT/MCIF)"
echo ""

# Synthetic multilingual SQA (generated after other data is ready)
echo "=== Synthetic SQA (generated later) ==="
echo "Run after downloading: uv run python -m src.data.translate_sqa --output_dir $DATA_DIR/sqa_translated"
mkdir -p "$DATA_DIR/sqa_translated"
echo ""

echo "=== Done ==="
echo "Next steps:"
echo "  1. Download EuroParlST v1.1 manually from https://www.mllp.upv.es/europarl-st/"
echo "  2. Download GigaST translations from https://st-benchmark.github.io/resources/GigaST.html"
echo "  3. Run: uv run python -m src.data.translate_sqa (after LibriSpeech is ready)"
