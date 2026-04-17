"""Pre-extract speech encoder features for faster Stage 3 training.

The speech encoder is frozen and deterministic — running it on every
micro-batch during training is pure waste. This script extracts features
once and saves them to disk. Stage 3 then loads pre-extracted features
and skips the encoder entirely, cutting per-step time dramatically.

Usage:
    python src/extract_features.py --config configs/stage3_multimodal.yaml
    python src/extract_features.py --config configs/stage3_multimodal.yaml --output_dir data/precomputed --batch_size 32

Then train with:
    python src/train.py --config configs/stage3_multimodal.yaml --precomputed_dir data/precomputed ...
"""

import argparse
import os
import re

import torch
from tqdm import tqdm

from src.config import load_config
from src.model.build_model import load_speech_encoder, get_device


def load_audio_from_sample(sample, resamplers):
    """Extract raw audio array from a dataset sample (mirrors collator logic)."""
    import torchaudio

    sr = 16000

    if "audio_path" in sample and sample["audio_path"]:
        waveform, sr = torchaudio.load(sample["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze(0).numpy()
    elif "audio_array" in sample:
        audio = sample["audio_array"]
        sr = sample.get("sampling_rate", 16000)
    elif isinstance(sample.get("audio"), dict):
        audio = sample["audio"]["array"]
        sr = sample["audio"].get("sampling_rate", 16000)
    else:
        audio = sample.get("audio")

    if audio is None:
        return None

    # Resample to 16kHz if needed (cache resamplers like the collator does)
    if sr != 16000:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        waveform = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio = resamplers[sr](waveform).squeeze(0).numpy()

    return audio


def compute_output_lengths(encoder_config, input_lengths):
    """Compute speech encoder output lengths (same formula as SpeechLLM)."""
    kernel_size = getattr(encoder_config, "adaptor_kernel_size", 3)
    stride = getattr(encoder_config, "adaptor_stride", 2)
    num_layers = getattr(encoder_config, "num_adapter_layers", 1)
    pad = kernel_size // 2

    lengths = input_lengths.float()
    for _ in range(num_layers):
        lengths = torch.floor((lengths + 2 * pad - kernel_size) / stride + 1)
    return lengths.long()


_PT_INDEX_RE = re.compile(r"^(\d+)\.pt$")


def _parse_existing_indices(task_dir):
    """Parse numeric .pt filenames, ignoring non-numeric files safely."""
    indices = set()
    for f in os.listdir(task_dir):
        m = _PT_INDEX_RE.match(f)
        if m:
            indices.add(int(m.group(1)))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Pre-extract speech encoder features")
    parser.add_argument("--config", type=str, required=True, help="Task config (e.g. stage3)")
    parser.add_argument("--output_dir", type=str, default="data/precomputed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_audio_seconds", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"Loading speech encoder on {device}...")
    speech_encoder, processor = load_speech_encoder(
        config.get("model", {}).get("sfm", "facebook/seamless-m4t-v2-large")
    )

    print("Loading datasets...")
    from src.data.multitask import load_all_task_datasets
    task_datasets = load_all_task_datasets(config)

    max_audio_samples = args.max_audio_seconds * 16000
    resamplers = {}  # cache resamplers across samples

    for task_name, dataset in task_datasets.items():
        task_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Resume support: skip already-extracted samples
        existing = _parse_existing_indices(task_dir)
        remaining = [i for i in range(len(dataset)) if i not in existing]

        if not remaining:
            print(f"  {task_name}: all {len(dataset)} already extracted, skipping")
            continue

        print(f"  {task_name}: extracting {len(remaining)}/{len(dataset)} samples...")
        skipped = 0

        for batch_start in tqdm(
            range(0, len(remaining), args.batch_size), desc=task_name
        ):
            batch_indices = remaining[batch_start : batch_start + args.batch_size]

            # Load audio per-sample, skipping any sample whose lazy decode or
            # waveform load raises (e.g. corrupt file, libsndfile format errors).
            # A single bad sample must not crash the whole extraction.
            audios = []
            valid_indices = []
            for idx in batch_indices:
                try:
                    sample = dataset[idx]
                    audio = load_audio_from_sample(sample, resamplers)
                except Exception as e:
                    print(f"  {task_name}[{idx}]: skipping ({type(e).__name__}: {e!r})", flush=True)
                    skipped += 1
                    continue
                if audio is None:
                    skipped += 1
                    continue
                if len(audio) > max_audio_samples:
                    audio = audio[:max_audio_samples]
                audios.append(audio)
                valid_indices.append(idx)

            if not audios:
                continue

            # Process through SeamlessM4T processor
            audio_inputs = processor(
                audios=audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            # Run frozen encoder
            with torch.no_grad():
                input_features = audio_inputs["input_features"].to(device)
                attention_mask = audio_inputs["attention_mask"].to(device)

                output = speech_encoder(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                all_features = output.last_hidden_state

            # Compute valid output lengths
            input_lengths = attention_mask.sum(dim=1)
            valid_lengths = compute_output_lengths(
                speech_encoder.config, input_lengths
            ).clamp(max=all_features.shape[1])

            # Save per-sample (only valid frames, no padding)
            for j, idx in enumerate(valid_indices):
                valid_len = valid_lengths[j].item()
                features = all_features[j, :valid_len].cpu()
                torch.save(features, os.path.join(task_dir, f"{idx}.pt"))

        # Completeness check
        extracted = len(_parse_existing_indices(task_dir))
        expected = len(dataset)
        size_mb = sum(
            os.path.getsize(os.path.join(task_dir, f))
            for f in os.listdir(task_dir)
            if f.endswith(".pt")
        ) / 1e6
        print(f"  {task_name}: {extracted}/{expected} features, {size_mb:.0f} MB")
        if skipped:
            print(f"  WARNING: {skipped} samples had null/corrupt audio and were skipped")
        # Only raise if more are missing than we accounted for with skips —
        # legitimate per-sample failures (corrupt audio, bad format) are expected.
        missing = expected - extracted
        unaccounted = missing - skipped
        if unaccounted > 0:
            raise RuntimeError(
                f"{task_name}: {unaccounted} samples are unaccounted for in {task_dir} "
                f"(expected={expected}, extracted={extracted}, skipped={skipped}). "
                "Refusing to continue with an incomplete precompute cache."
            )

    print(f"\nDone. Use: --precomputed_dir {args.output_dir}")


if __name__ == "__main__":
    main()
