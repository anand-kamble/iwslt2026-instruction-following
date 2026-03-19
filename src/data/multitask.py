"""Multi-task dataset with temperature-based sampling.

Combines datasets from all tasks for unified instruction-following training.
Uses temperature-based sampling to balance tasks with different dataset sizes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class MultiTaskDataset(Dataset):
    """Wraps multiple task datasets into a single dataset.

    Each sample is tagged with its source task for logging/debugging.
    """

    def __init__(self, task_datasets: dict[str, Dataset]):
        """
        Args:
            task_datasets: mapping from task name to HuggingFace dataset
        """
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.task_sizes = {name: len(ds) for name, ds in task_datasets.items()}

        # Build flat index → (task_name, local_idx) mapping
        self._index_map: list[tuple[str, int]] = []
        for name, ds in task_datasets.items():
            for i in range(len(ds)):
                self._index_map.append((name, i))

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx):
        task_name, local_idx = self._index_map[idx]
        item = self.task_datasets[task_name][local_idx]
        item["_task"] = task_name
        return item


class TemperatureSampler(Sampler[int]):
    """Sampler that uses temperature-based task sampling to balance training.

    With temperature T:
    - T=1.0: sample proportional to dataset size (no rebalancing)
    - T→0: sample uniformly across tasks (full rebalancing)
    - T=0.5: moderate rebalancing (good default)

    Each epoch samples `total_samples` indices. Smaller datasets are
    oversampled and larger datasets are undersampled based on temperature.
    """

    def __init__(
        self,
        dataset: MultiTaskDataset,
        temperature: float = 0.5,
        total_samples: int | None = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.temperature = temperature
        self.seed = seed

        # Compute sampling weights per task
        sizes = dataset.task_sizes
        total_original = sum(sizes.values())

        # Temperature-scaled probabilities: p_i ∝ n_i^T
        raw_probs = {
            name: math.pow(size, temperature) for name, size in sizes.items()
        }
        prob_sum = sum(raw_probs.values())
        self.task_probs = {name: p / prob_sum for name, p in raw_probs.items()}

        # Total samples per epoch (default: sum of all datasets)
        self.total_samples = total_samples or total_original

        # Build per-task index ranges for fast lookup
        self._task_indices: dict[str, list[int]] = {}
        for global_idx, (task_name, _) in enumerate(dataset._index_map):
            self._task_indices.setdefault(task_name, []).append(global_idx)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        self.seed += 1  # different order each epoch

        task_names = list(self.task_probs.keys())
        task_weights = [self.task_probs[name] for name in task_names]

        indices = []
        for _ in range(self.total_samples):
            # Pick a task based on temperature-scaled probabilities
            task = rng.choices(task_names, weights=task_weights, k=1)[0]
            # Pick a random sample from that task
            idx = rng.choice(self._task_indices[task])
            indices.append(idx)

        return iter(indices)

    def __len__(self):
        return self.total_samples


def load_all_task_datasets(config: dict) -> dict[str, Dataset]:
    """Load datasets for all tasks specified in a multitask config.

    Args:
        config: multitask config dict with 'tasks' section

    Returns:
        dict mapping task identifiers to datasets
    """
    from datasets import concatenate_datasets

    tasks_config = config["tasks"]
    all_datasets: dict[str, Dataset] = {}

    max_samples = config.get("data", {}).get("max_samples")

    # ASR
    if "asr" in tasks_config:
        from src.data.asr import load_asr_dataset
        asr_datasets = []
        for ds_name in tasks_config["asr"].get("train", []):
            try:
                asr_datasets.append(load_asr_dataset(ds_name, "train", max_samples=max_samples))
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping ASR/{ds_name}: {e}")
        if asr_datasets:
            all_datasets["asr"] = concatenate_datasets(asr_datasets)

    # S2TT
    if "s2tt" in tasks_config:
        from src.data.s2tt import load_s2tt_dataset
        s2tt_config = tasks_config["s2tt"]
        for lang in s2tt_config.get("target_languages", ["de", "it", "zh"]):
            s2tt_datasets = []
            for ds_name in s2tt_config.get("train", []):
                try:
                    ds = load_s2tt_dataset(ds_name, lang, "train")
                    if max_samples and len(ds) > max_samples:
                        ds = ds.select(range(max_samples))
                    s2tt_datasets.append(ds)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping S2TT/{ds_name}/{lang}: {e}")
            if s2tt_datasets:
                all_datasets[f"s2tt_{lang}"] = concatenate_datasets(s2tt_datasets)

    # SQA
    if "sqa" in tasks_config:
        from src.data.sqa import load_sqa_dataset
        sqa_config = tasks_config["sqa"]
        for lang in sqa_config.get("languages", ["en"]):
            try:
                ds = load_sqa_dataset("train", lang)
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                all_datasets[f"sqa_{lang}"] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping SQA/{lang}: {e}")

    # S2TSUM
    if "s2tsum" in tasks_config:
        from src.data.s2tsum import load_s2tsum_dataset
        s2tsum_config = tasks_config["s2tsum"]
        for lang in s2tsum_config.get("languages", ["en"]):
            try:
                ds = load_s2tsum_dataset("train", lang)
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                all_datasets[f"s2tsum_{lang}"] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping S2TSUM/{lang}: {e}")

    # ACHAP
    if "achap" in tasks_config:
        from src.data.achap import load_achap_dataset
        achap_config = tasks_config["achap"]
        for lang in achap_config.get("languages", ["en"]):
            try:
                ds = load_achap_dataset("train", lang)
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                all_datasets[f"achap_{lang}"] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping ACHAP/{lang}: {e}")

    if not all_datasets:
        raise RuntimeError("No datasets were loaded. Check your config and data paths.")

    print(f"Loaded {len(all_datasets)} task-language datasets:")
    for name, ds in all_datasets.items():
        print(f"  {name}: {len(ds)} samples")

    return all_datasets
