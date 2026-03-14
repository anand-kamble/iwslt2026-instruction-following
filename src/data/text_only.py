"""Text-only data loaders for Stage 2 LoRA pre-training.

Loads MT parallel text and SQA question-answer pairs WITHOUT audio.
Used to train LoRA adapters on text reasoning before multimodal merge.
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, UNANSWERABLE, format_text_only_messages, get_data_dir


def load_mt_europarlst(target_lang: str, split: str = "train"):
    """Load EuroParlST text-only MT pairs (EN transcript -> target translation)."""
    lang_map = {"de": "en-de", "it": "en-it"}
    if target_lang not in lang_map:
        raise ValueError(f"EuroParlST text doesn't support EN->{target_lang}")

    ds = load_dataset(
        "facebook/europarl_st", lang_map[target_lang],
        split=split, trust_remote_code=True,
    )

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        # EuroParlST stores translations in the target language field
        target_text = example.get("translation", example.get("target_text", example.get("text", "")))
        example["messages"] = format_text_only_messages(instruction, target_text)
        return example

    return ds.map(_format)


def load_mt_gigast(target_lang: str, split: str = "train"):
    """Load GigaST text-only MT pairs from local translation TSV files.

    Does NOT load audio — reads TSV translations directly.
    """
    from src.data.gigast import _load_translations

    if target_lang not in ("de", "zh"):
        raise ValueError(f"GigaST text only supports EN->de and EN->zh, got: {target_lang}")

    translations = _load_translations(target_lang, split)

    # Format each translation pair as text-only chat
    formatted = []
    instruction = TASK_PROMPTS["s2tt"][target_lang]
    for utt_id, translation in translations.items():
        formatted.append({
            "messages": format_text_only_messages(instruction, translation),
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

    ds = load_dataset("westbrook/librisqa", split=split, trust_remote_code=True)

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

    Returns a list of datasets.
    """
    datasets_list = []
    data_names = config.get("data", {}).get("train", [])

    for name in data_names:
        if name == "mt_europarlst":
            for lang in ["de", "it"]:
                try:
                    datasets_list.append(load_mt_europarlst(lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT EuroParlST {lang}: {e}")
        elif name == "mt_gigast":
            for lang in ["de", "zh"]:
                try:
                    datasets_list.append(load_mt_gigast(lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT GigaST {lang}: {e}")
        elif name == "sqa_translated":
            for lang in ["de", "it", "zh"]:
                try:
                    datasets_list.append(load_sqa_text_only("train", lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping SQA translated {lang}: {e}")
        elif name == "sqa_english":
            datasets_list.append(load_sqa_text_only("train", "en"))
        else:
            print(f"Unknown text-only dataset: {name}")

    return datasets_list
