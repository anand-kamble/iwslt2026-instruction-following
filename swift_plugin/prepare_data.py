"""Prepare training data as JSONL + wav files for SWIFT.

Usage:
    # Download and prepare all datasets
    uv run python swift_plugin/prepare_data.py --all

    # Prepare specific datasets
    uv run python swift_plugin/prepare_data.py --speech-asr --max-samples 3000
    uv run python swift_plugin/prepare_data.py --text-mt
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
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
    """Download EuroParlST English speech, resample to 16kHz, save as wav + JSONL."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = EUROPARLST_DIR / "asr_train.jsonl"

    if output_jsonl.exists():
        with open(output_jsonl) as f:
            existing = sum(1 for _ in f)
        if max_samples and existing >= max_samples:
            print(f"ASR JSONL already has {existing} samples, skipping.")
            return
        print(f"ASR JSONL has {existing} samples, continuing...")

    # Check which audio files already exist
    existing_ids = set()
    if output_jsonl.exists():
        with open(output_jsonl) as f:
            for line in f:
                record = json.loads(line)
                existing_ids.add(record["audios"][0])

    resampler = torchaudio.transforms.Resample(48000, 16000)
    ds_stream = load_dataset("CLAPv2/Europarl-st", split="train", streaming=True)

    instruction = TASK_PROMPTS["asr"]["en"]
    count = len(existing_ids)
    target = max_samples or float("inf")

    print(f"Streaming EuroParlST — collecting up to {max_samples or 'all'} English samples...")

    with open(output_jsonl, "a") as f:
        for sample in tqdm(ds_stream, desc="Collecting speech samples"):
            if count >= target:
                break

            if not sample["text"].startswith("A person is saying:"):
                continue

            # Generate filename
            audio_path = str(AUDIO_DIR / f"{count:06d}.wav")

            if audio_path in existing_ids:
                count += 1
                continue

            # Extract and resample audio
            audio_decoder = sample["audio"]
            samples = audio_decoder.get_all_samples()
            waveform = samples.data.squeeze(0)  # (num_samples,)
            sr = int(samples.sample_rate)

            if sr != 16000:
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

            # Save as 16kHz wav
            torchaudio.save(audio_path, waveform.unsqueeze(0), 16000)

            # Write JSONL entry
            transcription = _extract_transcription(sample["text"])
            record = {
                "messages": [
                    {"role": "user", "content": f"<audio>\n{instruction}"},
                    {"role": "assistant", "content": transcription},
                ],
                "audios": [audio_path],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"ASR: {count} samples saved to {output_jsonl}")


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
