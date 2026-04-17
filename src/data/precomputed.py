"""Wrapper dataset that loads pre-extracted speech encoder features.

Replaces raw audio with precomputed features so the frozen speech encoder
is skipped entirely during training.
"""

import os
import re

import torch
from torch.utils.data import Dataset


_PT_INDEX_RE = re.compile(r"^(\d+)\.pt$")


class PrecomputedWrapper(Dataset):
    """Wraps a dataset to load pre-extracted speech features from disk.

    Each sample's raw audio is replaced with the corresponding .pt file
    containing the speech encoder output tensor.

    Samples whose features were skipped during extraction (corrupt audio,
    libsndfile errors, etc.) are permanently filtered out here so training
    never hits a missing .pt file.
    """

    def __init__(self, dataset: Dataset, features_dir: str):
        self.dataset = dataset
        self.features_dir = features_dir

        available = set()
        for f in os.listdir(features_dir):
            m = _PT_INDEX_RE.match(f)
            if m:
                available.add(int(m.group(1)))
        self._valid_indices = sorted(i for i in range(len(dataset)) if i in available)

        missing = len(dataset) - len(self._valid_indices)
        if missing:
            print(
                f"PrecomputedWrapper[{features_dir}]: {missing} samples "
                f"skipped (no feature file); {len(self._valid_indices)} usable."
            )

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        real_idx = self._valid_indices[idx]
        item = self.dataset[real_idx]
        feature_path = os.path.join(self.features_dir, f"{real_idx}.pt")
        item["precomputed_features"] = torch.load(feature_path, weights_only=True)
        # Drop raw audio — no longer needed
        for key in ("audio", "audio_path", "audio_array"):
            item.pop(key, None)
        return item
