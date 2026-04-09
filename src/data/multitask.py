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

    Supports paired text interleaving (NAVER IWSLT 2025 Stage C):
    when a speech task with a text_pair is sampled, a batch from the
    paired text task immediately follows, ensuring speech→text pairing.
    """

    def __init__(
        self,
        dataset: MultiTaskDataset,
        temperature: float = 0.5,
        ratios: dict[str, float] | None = None,
        text_pairs: dict[str, str] | None = None,
        total_samples: int | None = None,
        seed: int = 42,
        batch_size: int = 1,
    ):
        self.dataset = dataset
        self.temperature = temperature
        self.seed = seed
        self.batch_size = batch_size
        self.text_pairs = text_pairs or {}

        sizes = dataset.task_sizes
        # Epoch size based on speech tasks only (NAVER: text tasks piggyback)
        text_pair_targets = set(self.text_pairs.values())
        speech_total = sum(s for n, s in sizes.items() if n not in text_pair_targets)
        total_fallback = speech_total if text_pair_targets else sum(sizes.values())

        if ratios is not None:
            # Use explicit ratios — normalize to sum to 1.0
            present = {k: v for k, v in ratios.items() if k in sizes}
            ratio_sum = sum(present.values())
            self.task_probs = {name: p / ratio_sum for name, p in present.items()}
            # Tasks in dataset but not in ratios get zero weight
            # (text-pair targets are sampled via pairing, not independently)
            for name in sizes:
                if name not in self.task_probs:
                    self.task_probs[name] = 0.0
        else:
            # Temperature-scaled probabilities: p_i ∝ n_i^T
            # Exclude text-pair targets from independent sampling
            active_sizes = {n: s for n, s in sizes.items() if n not in text_pair_targets}
            raw_probs = {name: math.pow(size, temperature) for name, size in active_sizes.items()}
            prob_sum = sum(raw_probs.values())
            self.task_probs = {name: p / prob_sum for name, p in raw_probs.items()}
            for name in sizes:
                if name not in self.task_probs:
                    self.task_probs[name] = 0.0

        # Total samples per epoch (default: speech tasks only)
        self.total_samples = total_samples or total_fallback

        # Build per-task index ranges for fast lookup
        self._task_indices: dict[str, list[int]] = {}
        for global_idx, (task_name, _) in enumerate(dataset._index_map):
            self._task_indices.setdefault(task_name, []).append(global_idx)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        self.seed += 1  # different order each epoch

        # Only independently sample tasks with positive weight
        task_names = [n for n in self.task_probs if self.task_probs[n] > 0]
        task_weights = [self.task_probs[n] for n in task_names]

        indices = []
        if self.batch_size > 1:
            # Yield in batch-sized groups from the same task.
            # This ensures homogeneous batches (all speech or all text-only),
            # required for interleaved text training (NAVER IWSLT 2025 Stage C).
            while len(indices) < self.total_samples:
                task = rng.choices(task_names, weights=task_weights, k=1)[0]
                for _ in range(self.batch_size):
                    idx = rng.choice(self._task_indices[task])
                    indices.append(idx)
                # Paired text interleaving (NAVER Stage C): speech batch is
                # immediately followed by a batch from the text equivalent.
                text_pair = self.text_pairs.get(task)
                if text_pair and text_pair in self._task_indices:
                    for _ in range(self.batch_size):
                        idx = rng.choice(self._task_indices[text_pair])
                        indices.append(idx)
        else:
            for _ in range(self.total_samples):
                task = rng.choices(task_names, weights=task_weights, k=1)[0]
                idx = rng.choice(self._task_indices[task])
                indices.append(idx)

        return iter(indices)

    def __len__(self):
        return self.total_samples


def _align_audio_features(datasets: list) -> list:
    """Cast audio columns to 16kHz so datasets from different sources can be concatenated.

    CoVoST2 declares Audio(sampling_rate=16000) while EuroParlST uses
    Audio(sampling_rate=None). concatenate_datasets rejects the schema
    mismatch. Casting to a uniform 16kHz resolves this.
    """
    from datasets import Audio as AudioFeature

    aligned = []
    for ds in datasets:
        if "audio" in ds.features and hasattr(ds.features["audio"], "sampling_rate"):
            if ds.features["audio"].sampling_rate != 16000:
                ds = ds.cast_column("audio", AudioFeature(sampling_rate=16000))
        aligned.append(ds)
    return aligned


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
            all_datasets["asr"] = concatenate_datasets(_align_audio_features(asr_datasets))

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
                all_datasets[f"s2tt_{lang}"] = concatenate_datasets(
                    _align_audio_features(s2tt_datasets)
                )

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

    # S2TSUM (lazy audio — no max_samples needed for memory)
    if "s2tsum" in tasks_config:
        from src.data.s2tsum import load_s2tsum_dataset

        s2tsum_config = tasks_config["s2tsum"]
        s2tsum_max = s2tsum_config.get("max_samples", max_samples)
        for lang in s2tsum_config.get("languages", ["en"]):
            try:
                ds = load_s2tsum_dataset("train", lang, max_samples=s2tsum_max)
                all_datasets[f"s2tsum_{lang}"] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping S2TSUM/{lang}: {e}")

    # ACHAP (lazy audio — no max_samples needed for memory)
    if "achap" in tasks_config:
        from src.data.achap import load_achap_dataset

        achap_config = tasks_config["achap"]
        achap_max = achap_config.get("max_samples", max_samples)
        for lang in achap_config.get("languages", ["en"]):
            try:
                ds = load_achap_dataset(
                    "train",
                    lang,
                    use_audio=achap_config.get("use_audio", True),
                    max_samples=achap_max,
                )
                all_datasets[f"achap_{lang}"] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping ACHAP/{lang}: {e}")

    # Load text-only datasets for interleaving (NAVER IWSLT 2025 Stage C).
    # Each speech task with a textual equivalent (ST↔MT, SQA↔QA) gets a
    # matching text-only dataset. The sampler yields homogeneous batches so
    # the collator can auto-detect text-only vs speech batches.
    text_interleave = config.get("text_interleave", {})
    if text_interleave:
        from src.data.text_only import load_mt_for_lang, load_sqa_text_only

        for speech_task, loader_spec in text_interleave.items():
            try:
                loader_type, lang = loader_spec.split("_", 1)
                if loader_type == "mt":
                    ds = load_mt_for_lang(lang)
                elif loader_type == "qa":
                    ds = load_sqa_text_only("train", lang)
                else:
                    print(f"Unknown text interleave loader: {loader_spec}")
                    continue
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                all_datasets[loader_spec] = ds
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping text interleave {speech_task}→{loader_spec}: {e}")

    if not all_datasets:
        raise RuntimeError("No datasets were loaded. Check your config and data paths.")

    print(f"Loaded {len(all_datasets)} task-language datasets:")
    for name, ds in all_datasets.items():
        print(f"  {name}: {len(ds)} samples")

    return all_datasets
