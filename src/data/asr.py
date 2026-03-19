"""ASR dataset loading and preprocessing.

Permitted datasets: EuroParlST, GigaST

EuroParlST: loaded from CLAPv2/Europarl-st on HuggingFace (has embedded audio).
English samples identified by text prefix "A person is saying:".
Audio is AudioDecoder format — extracted to numpy arrays.
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def _extract_transcription(text: str) -> str:
    """Extract transcription from CLAPv2 format: 'A person is saying: "actual text"'"""
    if '"' in text:
        return text.split('"', 1)[1].rsplit('"', 1)[0]
    return text


def load_europarlst_asr(split: str = "train", max_samples: int = None):
    """Load EuroParlST for ASR (English transcription).

    Uses CLAPv2/Europarl-st which has embedded audio for all languages.
    Filters for English samples only. Downloads the full dataset (~100GB,
    cached after first run) and keeps HF's native audio column to avoid
    PyArrow int32 overflow on large audio arrays.

    Args:
        max_samples: Limit to first N English samples. None = full dataset.
    """
    import os
    num_proc = os.cpu_count() or 1

    # Full download — HF manages the Arrow table so no int32 overflow on audio
    print("Downloading EuroParlST (cached after first run)...")
    ds = load_dataset("CLAPv2/Europarl-st", split=split)

    # Text filter is lightweight — safe to parallelize
    print(f"Filtering for English samples using {num_proc} cores...")
    ds = ds.filter(
        lambda ex: ex["text"].startswith("A person is saying:"),
        num_proc=num_proc,
    )
    print(f"Found {len(ds)} English samples")

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        print(f"Using first {max_samples} samples")

    def _format(example):
        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(
            instruction, _extract_transcription(example["text"])
        )
        return example

    # Only format text — keep the original "audio" column untouched.
    # The collator and precompute script handle audio extraction at read time.
    ds = ds.map(_format, num_proc=num_proc)
    return ds


def load_asr_dataset(dataset_name: str, split: str = "train", **kwargs):
    """Load an ASR dataset by name."""
    if dataset_name == "gigast":
        from src.data.gigast import load_gigast_asr
        return load_gigast_asr(split)

    loaders = {
        "europarlst": load_europarlst_asr,
    }
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown ASR dataset: {dataset_name}. Available: {list(loaders.keys()) + ['gigast']}"
        )
    return loaders[dataset_name](split, **kwargs)
