"""Audio Chaptering dataset loading and preprocessing.

Permitted dataset: YTSeg (retkowski/ytseg)
Output format: [CSTART] Title [CEND] section text...

YTSeg configs:
  - "text": transcripts only (no audio blob), ~1.86 GB
  - "audio": embedded 16kHz mono audio, ~68 GB
  - "titles": per-chapter title generation, ~726 MB

Key fields:
  - target_text_ref: structured "[CSTART] Title [CEND] text..." output
  - chapter_titles: list of chapter title strings
  - chapter_timestamps: list of start times (seconds)
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_achap_dataset(split: str = "train", language: str = "en", use_audio: bool = False):
    """Load the YTSeg audio chaptering dataset.

    Args:
        split: dataset split ("train", "validation", "test")
        language: target language for the prompt
        use_audio: if True, load "audio" config (68GB); otherwise "text" config
    """
    config_name = "audio" if use_audio else "text"
    ds = load_dataset("retkowski/ytseg", config_name, split=split)

    def _format(example):
        instruction = TASK_PROMPTS["achap"][language]

        # Use the pre-formatted structured text output from YTSeg
        # Format: [CSTART] Title [CEND] section text...
        response = example.get("target_text_ref", "")

        if not response:
            return None

        example["messages"] = format_chat_messages(instruction, response)
        return example

    ds = ds.map(_format)
    # Filter out any samples where target_text_ref was empty
    ds = ds.filter(lambda x: x.get("messages") is not None)
    return ds
