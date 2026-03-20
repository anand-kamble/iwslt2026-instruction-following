"""Speech-to-Text Summarization dataset loading and preprocessing.

Permitted dataset: NUTSHELL
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_s2tsum_dataset(split: str = "train", language: str = "en"):
    """Load the NUTSHELL summarization dataset."""
    ds = load_dataset("maikezu/nutshell", split=split, trust_remote_code=True)

    def _format(example):
        instruction = TASK_PROMPTS["s2tsum"][language]
        summary = example.get("summary", example.get("abstract", ""))
        example["messages"] = format_chat_messages(instruction, summary)
        return example

    return ds.map(_format)
