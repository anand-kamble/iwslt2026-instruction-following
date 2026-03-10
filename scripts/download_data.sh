#!/usr/bin/env bash
set -euo pipefail

# Download permitted training datasets for the constrained setting.
# Each dataset goes into data/<dataset_name>/

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading datasets ==="
echo "Target directory: $DATA_DIR"
echo ""

# TODO: Add download commands for each dataset
# EuroParlST
echo "TODO: Download EuroParlST"

# CoVoST2
echo "TODO: Download CoVoST2"

# GigaST
echo "TODO: Download GigaST"

# LibriSQA
echo "TODO: Download LibriSQA"

# NUTSHELL
echo "TODO: Download NUTSHELL"

# YTSeg
echo "TODO: Download YTSeg"

# MCIF (validation)
echo "TODO: Download MCIF validation set"

echo ""
echo "=== Done ==="
