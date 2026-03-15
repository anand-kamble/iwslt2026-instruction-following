"""Prepare training data as JSONL + wav files for SWIFT.

Usage:
    # Download and prepare all datasets
    uv run python swift_plugin/prepare_data.py --all

    # Prepare specific datasets
    uv run python swift_plugin/prepare_data.py --speech-asr --max-samples 3000
    uv run python swift_plugin/prepare_data.py --text-mt
"""

import argparse
import io
import json
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.utils import TASK_PROMPTS

DATA_DIR = Path("data")
EUROPARLST_DIR = DATA_DIR / "europarlst"
AUDIO_DIR = EUROPARLST_DIR / "audio"
MT_TEXT_DIR = DATA_DIR / "mt_text"


def _extract_transcription(text: str) -> str:
    """Extract transcription from CLAPv2 format: 'A person is saying: \"actual text\"'"""
    if '"' in text:
        return text.split('"', 1)[1].rsplit('"', 1)[0]
    return text


def prepare_speech_asr(max_samples: int = None):
    """Download EuroParlST English speech, resample to 16kHz, save as wav + JSONL.

    Loads directly from CLAPv2/Europarl-st parquet files to bypass broken
    datasets metadata ('List' feature type not recognized in newer datasets lib).
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = EUROPARLST_DIR / "asr_train.jsonl"

    # Count existing samples
    existing_count = 0
    if output_jsonl.exists():
        with open(output_jsonl) as f:
            existing_count = sum(1 for _ in f)
        if max_samples and existing_count >= max_samples:
            print(f"ASR JSONL already has {existing_count} samples, skipping.")
            return
        print(f"ASR JSONL has {existing_count} samples, continuing...")

    # List train parquet files from HuggingFace
    print("Listing CLAPv2/Europarl-st parquet files...")
    all_files = list_repo_files("CLAPv2/Europarl-st", repo_type="dataset")
    train_files = sorted([f for f in all_files if f.startswith("data/train/")])
    print(f"Found {len(train_files)} train parquet files")

    resampler = torchaudio.transforms.Resample(48000, 16000)
    instruction = TASK_PROMPTS["asr"]["en"]
    count = existing_count
    target = max_samples or float("inf")

    with open(output_jsonl, "a") as jsonl_f:
        for pq_file in tqdm(train_files, desc="Processing parquet files"):
            if count >= target:
                break

            # Download parquet file
            local_path = hf_hub_download(
                "CLAPv2/Europarl-st", pq_file, repo_type="dataset",
            )
            df = pd.read_parquet(local_path)

            for _, row in df.iterrows():
                if count >= target:
                    break

                text = row["text"]
                if not text.startswith("A person is saying:"):
                    continue

                # Decode audio from bytes
                audio_bytes = row["audio"]["bytes"]
                if audio_bytes is None:
                    continue

                try:
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                except Exception:
                    continue

                # Convert to tensor and resample to 16kHz
                waveform = torch.tensor(audio_data, dtype=torch.float32)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 2 and waveform.shape[1] <= 2:
                    # (samples, channels) -> (channels, samples)
                    waveform = waveform.T
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                if sr != 16000:
                    r = torchaudio.transforms.Resample(sr, 16000)
                    waveform = r(waveform)

                # Save as wav
                audio_path = str(AUDIO_DIR / f"{count:06d}.wav")
                torchaudio.save(audio_path, waveform, 16000)

                # Write JSONL
                transcription = _extract_transcription(text)
                record = {
                    "messages": [
                        {"role": "user", "content": f"<audio>\n{instruction}"},
                        {"role": "assistant", "content": transcription},
                    ],
                    "audios": [audio_path],
                }
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                count += 1

    print(f"ASR: {count} total samples saved to {output_jsonl}")


def prepare_speech_s2tt(max_samples: int = None):
    """Generate S2TT JSONL files reusing the same audio files from ASR prep.

    Requires prepare_speech_asr to have run first (audio files must exist).
    Uses tj-solergibert/Europarl-ST for translations paired with CLAPv2 audio.
    """
    # S2TT uses the same audio as ASR — we need a mapping from audio index
    # to translation. Since CLAPv2 and tj-solergibert may not align by index,
    # we generate S2TT manifests from the same streaming source.
    for lang in ["de", "it"]:
        output_jsonl = EUROPARLST_DIR / f"s2tt_{lang}_train.jsonl"
        if output_jsonl.exists():
            print(f"S2TT {lang} JSONL exists, skipping.")
            continue
        print(f"S2TT {lang}: generating from tj-solergibert/Europarl-ST (text-only, no audio needed for S2TT training data)")
        # Note: For S2TT with audio, we'd need parallel alignment between
        # CLAPv2 and tj-solergibert datasets. For now, generate text-only S2TT
        # that can be used with audio during multitask Stage 3.
        print(f"  Skipping S2TT speech manifest for {lang} — use text-only MT for Stage 2")


def prepare_text_mt():
    """Generate text-only MT JSONL for Stage 2 from tj-solergibert/Europarl-ST."""
    MT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tj-solergibert/Europarl-ST for text MT pairs...")
    ds = load_dataset("tj-solergibert/Europarl-ST", split="train")
    num_proc = os.cpu_count() or 1

    for lang in ["de", "it"]:
        output_jsonl = MT_TEXT_DIR / f"europarlst_{lang}.jsonl"
        if output_jsonl.exists():
            with open(output_jsonl) as f:
                existing = sum(1 for _ in f)
            print(f"MT {lang} JSONL already has {existing} samples, skipping.")
            continue

        instruction = TASK_PROMPTS["s2tt"][lang]

        # Filter for samples with both EN and target language
        def has_both(example):
            t = example.get("transcriptions", {})
            if not isinstance(t, dict):
                return False
            return bool(t.get("en", "").strip() and t.get(lang, "").strip())

        filtered = ds.filter(has_both, num_proc=num_proc)

        count = 0
        with open(output_jsonl, "w") as f:
            for i in tqdm(range(len(filtered)), desc=f"MT {lang}"):
                example = filtered[i]
                en_text = example["transcriptions"]["en"]
                tgt_text = example["transcriptions"][lang]

                record = {
                    "messages": [
                        {"role": "user", "content": f"{instruction}\n\n{en_text}"},
                        {"role": "assistant", "content": tgt_text},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"MT {lang}: {count} samples saved to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SWIFT training data")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--speech-asr", action="store_true", help="Prepare EuroParlST ASR")
    parser.add_argument("--text-mt", action="store_true", help="Prepare text MT pairs")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for speech datasets")
    args = parser.parse_args()

    if not any([args.all, args.speech_asr, args.text_mt]):
        parser.print_help()
        return

    if args.all or args.speech_asr:
        prepare_speech_asr(max_samples=args.max_samples)

    if args.all or args.text_mt:
        prepare_text_mt()


if __name__ == "__main__":
    main()
