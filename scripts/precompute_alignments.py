"""Precompute speech features + LLM text embeddings for Stage 0 MSE pre-alignment.

Loads models sequentially to minimize memory:
  1. Load speech encoder → extract all speech features → free encoder
  2. Load LLM embed_tokens → extract all text embeddings → free

Usage:
    uv run python scripts/precompute_alignments.py --output_dir data/alignments --max_samples 500
    uv run python scripts/precompute_alignments.py --output_dir data/alignments  # full dataset
"""

import argparse
import gc
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.build_model import get_device


def _resample_one(item):
    """Resample a single audio to 16kHz. Must be top-level for pickling."""
    import torch
    import torchaudio
    audio_array, sr = item
    if sr != 16000:
        waveform = torch.tensor(audio_array).unsqueeze(0)
        audio_array = torchaudio.transforms.Resample(sr, 16000)(waveform).squeeze(0).numpy()
    return audio_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/alignments")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--llm_name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--downsample_factor", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ===== Phase 1: Extract speech features =====
    print("\n=== Phase 1: Extracting speech features ===")
    from src.model.build_model import load_speech_encoder
    speech_encoder, processor = load_speech_encoder(device=device)

    # Load dataset
    from src.data.asr import load_europarlst_asr
    print("Loading EuroParlST...")
    ds = load_europarlst_asr("train", max_samples=args.max_samples)
    print(f"Samples: {len(ds)}")

    import torchaudio

    resampler = torchaudio.transforms.Resample(48000, 16000)

    speech_features_list = []
    transcriptions = []

    for i in tqdm(range(len(ds)), desc="Extracting speech features"):
        sample = ds[i]

        # Extract audio from HF's native audio dict
        audio_data = sample["audio"]
        if isinstance(audio_data, dict):
            audio_array = audio_data["array"]
            sr = audio_data.get("sampling_rate", 16000)
        else:
            audio_array = audio_data
            sr = 16000

        # Resample to 16kHz
        if sr != 16000:
            waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
            audio_array = resampler(waveform).squeeze(0).numpy()

        # Encode
        audio_input = processor(
            audios=audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        audio_input = {k: v.to(device) for k, v in audio_input.items()}

        with torch.no_grad():
            encoder_out = speech_encoder(
                input_features=audio_input["input_features"],
                attention_mask=audio_input["attention_mask"],
                return_dict=True,
            )
            speech_feats = encoder_out.last_hidden_state.squeeze(0).cpu()

        # Downsample via frame averaging
        T = speech_feats.shape[0]
        T_down = T // args.downsample_factor
        if T_down == 0:
            continue

        speech_feats_down = speech_feats[:T_down * args.downsample_factor].reshape(
            T_down, args.downsample_factor, -1
        ).mean(dim=1)

        speech_features_list.append(speech_feats_down)
        transcriptions.append(sample["messages"][-1]["content"])

    print(f"Extracted {len(speech_features_list)} speech feature tensors")

    # Free speech encoder
    del speech_encoder, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Freed speech encoder memory")

    # ===== Phase 2: Extract text embeddings =====
    print("\n=== Phase 2: Extracting text embeddings ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)

    # Load model in float16 on CPU, extract embed_tokens, delete model
    print("Loading LLM to extract embed_tokens...")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name, trust_remote_code=True, torch_dtype=torch.float16,
    )
    # Copy embed_tokens to device, delete everything else
    embed_tokens = llm.model.embed_tokens.to(device)
    embed_tokens.eval()
    llm_dim = embed_tokens.embedding_dim
    del llm
    gc.collect()
    print(f"LLM embedding dim: {llm_dim}")

    text_embeddings_list = []

    for i in tqdm(range(len(transcriptions)), desc="Text embeddings"):
        text_ids = tokenizer(transcriptions[i], return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            text_embeds = embed_tokens(text_ids).squeeze(0).cpu()  # (S, llm_dim)

        text_embeddings_list.append(text_embeds)

    print(f"Extracted {len(text_embeddings_list)} text embedding tensors")

    # Free embed_tokens
    del embed_tokens, tokenizer
    gc.collect()

    # ===== Save =====
    save_path = os.path.join(args.output_dir, "alignment_data.pt")
    torch.save({
        "speech_features": speech_features_list,
        "text_embeddings": text_embeddings_list,
        "llm_dim": llm_dim,
        "downsample_factor": args.downsample_factor,
        "num_samples": len(speech_features_list),
    }, save_path)

    print(f"\nSaved to {save_path}")
    print(f"  {len(speech_features_list)} pairs")
    print(f"  Speech: (T_i, 1024), Text: (S_i, {llm_dim})")


if __name__ == "__main__":
    main()
