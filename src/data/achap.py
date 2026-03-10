"""Audio Chaptering dataset loading and preprocessing.

Permitted dataset: YTSeg
Output format: [CSTART] Title [CEND] section text...
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_achap_dataset(split: str = "train"):
    """Load the YTSeg audio chaptering dataset."""
    ds = load_dataset("retkowski/ytseg", split=split, trust_remote_code=True)

    def _format(example):
        instruction = TASK_PROMPTS["achap"]["en"]

        # Format chapters as [CSTART] Title [CEND] pattern
        chapters = example.get("chapters", [])
        if isinstance(chapters, list) and chapters:
            formatted_parts = []
            for chapter in chapters:
                title = chapter.get("title", "")
                text = chapter.get("text", "")
                formatted_parts.append(f"[CSTART] {title} [CEND] {text}")
            response = "\n".join(formatted_parts)
        else:
            response = example.get("text", "")

        example["messages"] = format_chat_messages(instruction, response)
        return example

    return ds.map(_format)
