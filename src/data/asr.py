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
    Filters for English samples only.

    When max_samples is set, uses streaming to avoid downloading the full
    ~100GB dataset — only fetches what's needed.

    Args:
        max_samples: Max English samples to collect. None = full dataset
                     (requires full download, ~100GB cached).
    """
    from datasets import Dataset as HFDataset

    if max_samples is not None:
        # Streaming mode: collect N English samples without full download
        print(f"Streaming EuroParlST — collecting {max_samples} English samples...")
        ds_stream = load_dataset("CLAPv2/Europarl-st", split=split, streaming=True)

        collected = []
        for sample in ds_stream:
            if not sample["text"].startswith("A person is saying:"):
                continue

            audio_decoder = sample["audio"]
            samples = audio_decoder.get_all_samples()

            collected.append({
                "audio_array": samples.data.squeeze(0).numpy(),
                "sampling_rate": int(samples.sample_rate),
                "text": sample["text"],
                "messages": format_chat_messages(
                    TASK_PROMPTS["asr"]["en"],
                    _extract_transcription(sample["text"]),
                ),
            })

            if len(collected) >= max_samples:
                break

        print(f"Collected {len(collected)} English samples via streaming")
        return HFDataset.from_list(collected)

    # Full download mode
    import os
    num_proc = os.cpu_count() or 1

    ds = load_dataset("CLAPv2/Europarl-st", split=split)

    # Text filter is lightweight — safe to parallelize
    print(f"Filtering for English samples using {num_proc} cores...")
    ds = ds.filter(
        lambda ex: ex["text"].startswith("A person is saying:"),
        num_proc=num_proc,
    )
    print(f"Found {len(ds)} English samples")

    def _format(example):
        audio_decoder = example["audio"]
        samples = audio_decoder.get_all_samples()
        example["audio_array"] = samples.data.squeeze(0).numpy()
        example["sampling_rate"] = int(samples.sample_rate)

        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(
            instruction, _extract_transcription(example["text"])
        )
        return example

    # Audio extraction is memory-heavy — single process to avoid OOM
    print(f"Extracting audio from {len(ds)} samples (single process to save memory)...")
    ds = ds.map(_format)
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
