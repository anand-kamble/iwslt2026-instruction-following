"""Precompute speech encoder features for faster training.

Since the speech encoder is frozen, we can extract features once and reuse them.

Usage:
    python scripts/precompute_features.py --dataset europarlst --output_dir data/features/europarlst
    python scripts/precompute_features.py --dataset librisqa --output_dir data/features/librisqa
    python scripts/precompute_features.py --dataset gigast --output_dir data/features/gigast
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.build_model import load_speech_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["europarlst", "librisqa", "gigast"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SeamlessM4T v2 encoder...")
    encoder, processor = load_speech_encoder(device=args.device)

    print(f"Loading {args.dataset}...")
    if args.dataset == "europarlst":
        ds = load_dataset("facebook/europarl_st", "en", split=args.split, trust_remote_code=True)
    elif args.dataset == "librisqa":
        ds = load_dataset("westbrook/librisqa", split=args.split, trust_remote_code=True)
    elif args.dataset == "gigast":
        ds = load_dataset(
            "speechcolab/gigaspeech", "xl", split=args.split,
            trust_remote_code=True, streaming=True,
        )

    print(f"Precomputing features...")
    for i, example in enumerate(tqdm(ds)):
        output_path = os.path.join(args.output_dir, f"{i:08d}.pt")
        if os.path.exists(output_path):
            continue

        audio = example.get("audio", {})
        if isinstance(audio, dict):
            waveform = audio["array"]
            sr = audio.get("sampling_rate", 16000)
        else:
            continue

        inputs = processor(
            audios=waveform, sampling_rate=sr,
            return_tensors="pt", return_attention_mask=True,
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = encoder(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            features = output.last_hidden_state.cpu()

        torch.save(features.squeeze(0), output_path)

    print(f"Done! Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
