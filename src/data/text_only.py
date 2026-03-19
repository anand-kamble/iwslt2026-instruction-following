"""Text-only data loaders for Stage 2 LoRA pre-training.

Loads MT parallel text and SQA question-answer pairs WITHOUT audio.
Used to train LoRA adapters on text reasoning before multimodal merge.
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, UNANSWERABLE, format_text_only_messages, get_data_dir


def load_mt_europarlst(target_lang: str, split: str = "train"):
    """Load EuroParlST text-only MT pairs (EN transcript -> target translation).

    Uses tj-solergibert/Europarl-ST which has transcriptions in all 9 languages.
    No audio loaded — text only.
    """
    lang_map = {"de": "de", "it": "it"}
    if target_lang not in lang_map:
        raise ValueError(f"EuroParlST text doesn't support EN->{target_lang}")

    ds = load_dataset("tj-solergibert/Europarl-ST", split=split)

    # Filter for samples that have both English and target language transcriptions
    def has_both_langs(example):
        transcriptions = example.get("transcriptions", {})
        if not isinstance(transcriptions, dict):
            return False
        en_text = transcriptions.get("en", "")
        tgt_text = transcriptions.get(target_lang, "")
        return bool(en_text and en_text.strip() and tgt_text and tgt_text.strip())

    import os
    num_proc = os.cpu_count() or 1
    ds = ds.filter(has_both_langs, num_proc=num_proc)

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        en_text = example["transcriptions"]["en"]
        target_text = example["transcriptions"][target_lang]
        # Include English source so the model learns translation, not just generation
        full_instruction = f"{instruction}\n\n{en_text}"
        example["messages"] = format_text_only_messages(full_instruction, target_text)
        return example

    ds = ds.map(_format, num_proc=num_proc)
    return ds


def load_mt_gigast(target_lang: str, split: str = "train"):
    """Load GigaST text-only MT pairs from local translation TSV files.

    Does NOT load audio — reads TSV translations directly.
    Loads English transcriptions from GigaST en/ TSV for source text.
    """
    from src.data.gigast import _load_translations

    if target_lang not in ("de", "zh"):
        raise ValueError(f"GigaST text only supports EN->de and EN->zh, got: {target_lang}")

    translations = _load_translations(target_lang, split)

    # Try to load English transcriptions for source text
    try:
        en_transcriptions = _load_translations("en", split)
    except FileNotFoundError:
        en_transcriptions = {}

    # Format each translation pair as text-only chat
    formatted = []
    instruction = TASK_PROMPTS["s2tt"][target_lang]
    for utt_id, translation in translations.items():
        en_text = en_transcriptions.get(utt_id, "")
        full_instruction = f"{instruction}\n\n{en_text}" if en_text else instruction
        formatted.append({
            "messages": format_text_only_messages(full_instruction, translation),
        })

    from datasets import Dataset
    return Dataset.from_list(formatted)


def load_sqa_text_only(split: str = "train", language: str = "en"):
    """Load SQA as text-only QA pairs (no audio).

    For English: loads from original LibriSQA.
    For other languages: loads from translated JSONL files.
    """
    if language != "en":
        return _load_translated_sqa_text_only(language)

    split_map = {"train": "partI.train", "test": "partI.test"}
    hf_split = split_map.get(split, split)
    ds = load_dataset("ZihanZhao/LibriSQA", split=hf_split)

    def _format(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt_template = TASK_PROMPTS["sqa"][language]
        instruction = prompt_template.replace("{QUESTION}", question)

        if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
            answer = UNANSWERABLE[language]

        example["messages"] = format_text_only_messages(instruction, answer)
        return example

    return ds.map(_format)


def _load_translated_sqa_text_only(language: str):
    """Load translated SQA from local JSONL files (text-only, no audio)."""
    jsonl_path = get_data_dir() / "sqa_translated" / f"librisqa_{language}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Translated SQA data not found at {jsonl_path}. "
            f"Run: python -m src.data.translate_sqa --languages {language}"
        )

    records = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            prompt_template = TASK_PROMPTS["sqa"][language]
            instruction = prompt_template.replace("{QUESTION}", record["question"])
            answer = record["answer"]
            if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
                answer = UNANSWERABLE[language]
            records.append({
                "messages": format_text_only_messages(instruction, answer),
            })

    from datasets import Dataset
    return Dataset.from_list(records)


def load_text_only_datasets(config: dict):
    """Load all text-only datasets for Stage 2.

    Returns a list of datasets.  Respects data.max_samples if set.
    """
    datasets_list = []
    data_names = config.get("data", {}).get("train", [])
    max_samples = config.get("data", {}).get("max_samples")

    for name in data_names:
        if name == "mt_europarlst":
            for lang in ["de", "it"]:
                try:
                    ds = load_mt_europarlst(lang)
                    if max_samples and len(ds) > max_samples:
                        ds = ds.select(range(max_samples))
                    datasets_list.append(ds)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT EuroParlST {lang}: {e}")
        elif name == "mt_gigast":
            for lang in ["de", "zh"]:
                try:
                    ds = load_mt_gigast(lang)
                    if max_samples and len(ds) > max_samples:
                        ds = ds.select(range(max_samples))
                    datasets_list.append(ds)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT GigaST {lang}: {e}")
        elif name == "sqa_translated":
            for lang in ["de", "it", "zh"]:
                try:
                    ds = load_sqa_text_only("train", lang)
                    if max_samples and len(ds) > max_samples:
                        ds = ds.select(range(max_samples))
                    datasets_list.append(ds)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping SQA translated {lang}: {e}")
        elif name == "sqa_english":
            ds = load_sqa_text_only("train", "en")
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            datasets_list.append(ds)
        else:
            print(f"Unknown text-only dataset: {name}")

    return datasets_list
