"""Spoken Question Answering dataset loading and preprocessing.

Permitted dataset: LibriSQA
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, UNANSWERABLE, format_chat_messages


def load_sqa_dataset(split: str = "train", language: str = "en"):
    """Load the LibriSQA dataset for spoken question answering."""
    ds = load_dataset("westbrook/librisqa", split=split, trust_remote_code=True)

    def _format(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt_template = TASK_PROMPTS["sqa"][language]
        instruction = prompt_template.replace("{QUESTION}", question)

        # Mark unanswerable
        if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
            answer = UNANSWERABLE[language]

        example["messages"] = format_chat_messages(instruction, answer)
        return example

    return ds.map(_format)
