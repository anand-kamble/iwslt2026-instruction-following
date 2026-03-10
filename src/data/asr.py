"""ASR dataset loading and preprocessing.

Permitted datasets: EuroParlST, CoVoST2, GigaST
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_europarlst_asr(split: str = "train"):
    """Load EuroParlST for ASR (English transcription)."""
    ds = load_dataset("facebook/europarl_st", "en", split=split, trust_remote_code=True)

    def _format(example):
        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(instruction, example["text"])
        return example

    return ds.map(_format)


def load_covost2_asr(split: str = "train"):
    """Load CoVoST2 for ASR (English transcription)."""
    ds = load_dataset("facebook/covost2", "en", split=split, trust_remote_code=True)

    def _format(example):
        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(instruction, example["sentence"])
        return example

    return ds.map(_format)


def load_asr_dataset(dataset_name: str, split: str = "train"):
    """Load an ASR dataset by name."""
    loaders = {
        "europarlst": load_europarlst_asr,
        "covost2": load_covost2_asr,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown ASR dataset: {dataset_name}. Available: {list(loaders.keys())}")
    return loaders[dataset_name](split)
