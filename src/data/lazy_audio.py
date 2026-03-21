"""Lazy-loading wrapper for large audio datasets.

Avoids PyArrow int32 offset overflow and OOM by:
  1. Loading the HF dataset normally (memory-mapped Arrow, audio on disk).
  2. Never calling .map()/.filter() on the audio-heavy dataset.
  3. Formatting text on-the-fly in __getitem__.
  4. Audio is decoded only when accessed per-sample by the collator.

This is the recommended pattern for datasets with long-form audio
(NUTSHELL, YTSeg) where .map()/.filter() trigger Arrow overflow.
"""

from __future__ import annotations

from collections.abc import Callable
from torch.utils.data import Dataset as TorchDataset


class LazyAudioDataset(TorchDataset):
    """Wraps an HF Dataset to format text lazily, keeping audio on disk.

    The underlying HF dataset is memory-mapped from Arrow files on disk.
    Audio columns use HF's Audio() feature which decodes lazily — the
    actual audio bytes are only read and decoded when __getitem__ accesses
    the sample. Text formatting happens on-the-fly, so no .map() call is
    ever made on the audio-heavy dataset.

    Args:
        hf_dataset: a HuggingFace Dataset (map-style, memory-mapped).
        indices: list of integer indices into hf_dataset to use.
            Allows filtering without calling hf_dataset.filter().
        format_fn: callable(sample) -> dict with "messages" key.
            Called per-sample in __getitem__.
        audio_key: name of the audio column (default "audio").
    """

    def __init__(
        self,
        hf_dataset,
        indices: list[int],
        format_fn: Callable[[dict], dict],
        audio_key: str = "audio",
    ):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.format_fn = format_fn
        self.audio_key = audio_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # HF decodes audio lazily here — only this sample is read from disk
        sample = self.hf_dataset[real_idx]
        formatted = self.format_fn(sample)
        # Carry over audio (just decoded for this single sample)
        if self.audio_key in sample:
            formatted[self.audio_key] = sample[self.audio_key]
        return formatted
