#!/bin/bash
# Wait for precompute to finish, then run Stage 0

DATA_FILE="data/alignments/alignment_data.pt"

while [ ! -f "$DATA_FILE" ]; do
    sleep 30
    echo "Waiting for precompute to finish..."
done

echo "Starting Stage 0!"
uv run python src/train_stage0.py \
    --config configs/stage0_prealign.yaml \
    --data "$DATA_FILE"
