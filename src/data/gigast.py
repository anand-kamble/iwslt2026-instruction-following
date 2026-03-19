"""GigaST dataset loading and preprocessing.

GigaST provides machine-translated text for GigaSpeech audio.
Audio: GigaSpeech (speechcolab/gigaspeech on HuggingFace)
Translations: downloaded from https://st-benchmark.github.io/resources/GigaST.html

Language pairs: en-de, en-zh
Format: TSV files mapping utterance IDs to translations.

Setup:
    1. Request GigaSpeech access on HuggingFace (speechcolab/gigaspeech)
    2. Download GigaST translations from st-benchmark.github.io
    3. Place translations in data/gigast/{de,zh}/train.tsv
       TSV columns: utt_id<TAB>translation
"""

import csv
from pathlib import Path

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages, get_data_dir


def _load_translations(target_lang: str, split: str = "train") -> dict[str, str]:
    """Load GigaST translation TSV file into a dict {utt_id: translation}."""
    tsv_path = get_data_dir() / "gigast" / target_lang / f"{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"GigaST translations not found at {tsv_path}. "
            f"Download from https://st-benchmark.github.io/resources/GigaST.html"
        )
    translations = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                translations[parts[0]] = parts[1]
    return translations


def load_gigast_asr(split: str = "train"):
    """Load GigaSpeech audio with English transcriptions for ASR."""
    ds = load_dataset(
        "speechcolab/gigaspeech", "xl", split=split, trust_remote_code=True
    )

    def _format(example):
        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(instruction, example["text"])
        return example

    return ds.map(_format)


def load_gigast_s2tt(target_lang: str, split: str = "train"):
    """Load GigaSpeech audio with GigaST translations for S2TT.

    Requires GigaST translation files to be downloaded locally.
    """
    if target_lang not in ("de", "zh"):
        raise ValueError(f"GigaST only supports EN→de and EN→zh, got: {target_lang}")

    translations = _load_translations(target_lang, split)

    ds = load_dataset(
        "speechcolab/gigaspeech", "xl", split=split, trust_remote_code=True
    )

    # Filter to only utterances that have translations
    ds = ds.filter(lambda ex: ex["segment_id"] in translations)

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        translation = translations[example["segment_id"]]
        example["messages"] = format_chat_messages(instruction, translation)
        return example

    return ds.map(_format)
