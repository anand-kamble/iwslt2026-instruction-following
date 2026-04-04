"""Wrapper dataset that loads pre-extracted speech encoder features.

Replaces raw audio with precomputed features so the frozen speech encoder
is skipped entirely during training.
"""

import os

import torch
from torch.utils.data import Dataset


class PrecomputedWrapper(Dataset):
    """Wraps a dataset to load pre-extracted speech features from disk.

    Each sample's raw audio is replaced with the corresponding .pt file
    containing the speech encoder output tensor.
    """

    def __init__(self, dataset: Dataset, features_dir: str):
        self.dataset = dataset
        self.features_dir = features_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        feature_path = os.path.join(self.features_dir, f"{idx}.pt") 
        item["precomputed_features"] = torch.load(feature_path, weights_only=True)
        # Drop raw audio — no longer needed
        for key in ("audio", "audio_path", "audio_array"):
            item.pop(key, None)
        return item
