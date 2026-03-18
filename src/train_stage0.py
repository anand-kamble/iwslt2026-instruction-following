"""Stage 0: MSE pre-alignment training.

Trains the projector to map speech features → LLM text embeddings using MSE loss.
No LLM in the training loop — only the projector (~39M params) is loaded.
Uses precomputed data from scripts/precompute_alignments.py.

Memory: ~2-3 GB (projector only)
Speed: ~50-100x faster than Stage 1

Usage:
    uv run python src/train_stage0.py --config configs/stage0_prealign.yaml
    uv run python src/train_stage0.py --config configs/stage0_prealign.yaml --data data/alignments/alignment_data.pt
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from src.model.build_model import build_projector, get_device


class AlignmentDataset(Dataset):
    """Dataset of (speech_features, text_embeddings) pairs for MSE alignment."""

    def __init__(self, data_path: str):
        data = torch.load(data_path, weights_only=False)
        self.speech_features = data["speech_features"]  # list of (T_i, 1024)
        self.text_embeddings = data["text_embeddings"]   # list of (S_i, llm_dim)
        self.llm_dim = data["llm_dim"]
        print(f"Loaded {len(self.speech_features)} alignment pairs (LLM dim: {self.llm_dim})")

    def __len__(self):
        return len(self.speech_features)

    def __getitem__(self, idx):
        return {
            "speech_features": self.speech_features[idx],   # (T, 1024)
            "text_embeddings": self.text_embeddings[idx],    # (S, llm_dim)
        }


def collate_alignment(batch):
    """Collate variable-length speech/text pairs.

    Pads speech features and text embeddings to the max length in the batch.
    Returns padded tensors + length info for the loss function.
    """
    speech = [item["speech_features"] for item in batch]
    text = [item["text_embeddings"] for item in batch]

    # Pad speech
    max_speech_len = max(s.shape[0] for s in speech)
    speech_dim = speech[0].shape[1]
    speech_padded = torch.zeros(len(batch), max_speech_len, speech_dim)
    speech_lengths = []
    for i, s in enumerate(speech):
        speech_padded[i, :s.shape[0]] = s
        speech_lengths.append(s.shape[0])

    # Pad text
    max_text_len = max(t.shape[0] for t in text)
    text_dim = text[0].shape[1]
    text_padded = torch.zeros(len(batch), max_text_len, text_dim)
    text_lengths = []
    for i, t in enumerate(text):
        text_padded[i, :t.shape[0]] = t
        text_lengths.append(t.shape[0])

    return {
        "speech_features": speech_padded,
        "text_embeddings": text_padded,
        "speech_lengths": torch.tensor(speech_lengths),
        "text_lengths": torch.tensor(text_lengths),
    }


def compute_alignment_loss(projected, text_embeddings, speech_lengths, text_lengths):
    """Compute MSE loss between projected speech and text embeddings.

    Since speech and text have different lengths, we use mean pooling
    over time for both, then compute MSE on the pooled representations.
    This teaches the projector to produce embeddings in the right space
    without requiring exact length matching.
    """
    batch_size = projected.shape[0]
    loss = 0.0

    for i in range(batch_size):
        # Mean pool over valid timesteps
        speech_pooled = projected[i, :speech_lengths[i]].mean(dim=0)  # (llm_dim,)
        text_pooled = text_embeddings[i, :text_lengths[i]].mean(dim=0)  # (llm_dim,)

        loss += nn.functional.mse_loss(speech_pooled, text_pooled)

    return loss / batch_size


def main():
    parser = argparse.ArgumentParser(description="Stage 0: MSE pre-alignment")
    parser.add_argument("--config", default="configs/stage0_prealign.yaml")
    parser.add_argument("--data", default="data/alignments/alignment_data.pt")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    training_config = config.get("training", {})
    output_dir = args.output_dir or training_config.get("output_dir", "outputs/stage0_prealign")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load precomputed data
    dataset = AlignmentDataset(args.data)

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 64),
        shuffle=True,
        collate_fn=collate_alignment,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # Build projector only — no LLM needed
    projector = build_projector(
        llm_dim=dataset.llm_dim,
        downsample_factor=1,  # already downsampled in precompute
        projector_type=training_config.get("projector_type", "transformer"),
        device=device,
    )

    # Load from checkpoint if resuming
    projector_path = os.path.join(output_dir, "projector.pt")
    if os.path.exists(projector_path) and training_config.get("resume", False):
        print(f"Resuming from {projector_path}")
        projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))

    total_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"Projector trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )

    num_epochs = training_config.get("num_epochs", 10)
    log_steps = training_config.get("logging_steps", 10)
    save_steps = training_config.get("save_steps", 100)

    # Enable tf32 on CUDA for faster matmuls
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # Training loop
    global_step = 0
    projector.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            speech = batch["speech_features"].to(device)
            text_emb = batch["text_embeddings"].to(device)
            speech_len = batch["speech_lengths"]
            text_len = batch["text_lengths"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward: projector maps speech features → LLM embedding space
                projected = projector(speech)  # (B, T, llm_dim)

                # MSE loss: projected speech should match text embeddings
                loss = compute_alignment_loss(projected, text_emb, speech_len, text_len)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_steps == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / num_batches
                print(f"Step {global_step} | Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                      f"Time: {elapsed:.0f}s")

            if global_step % save_steps == 0:
                torch.save(projector.state_dict(), projector_path)
                print(f"Saved checkpoint at step {global_step}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} done | Avg loss: {avg_epoch_loss:.4f}")

    # Final save
    torch.save(projector.state_dict(), projector_path)
    elapsed = time.time() - start_time
    print(f"\nStage 0 complete! ({elapsed:.0f}s)")
    print(f"Projector saved to {projector_path}")
    print(f"Use this as starting point for Stage 1:")
    print(f"  uv run python src/train.py --config configs/stage1_alignment.yaml "
          f"--stage1_checkpoint {output_dir}")


if __name__ == "__main__":
    main()
