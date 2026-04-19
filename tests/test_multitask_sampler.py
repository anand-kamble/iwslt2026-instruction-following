from torch.utils.data import BatchSampler, Dataset

from src.data.multitask import MultiTaskDataset, TemperatureSampler


class _DummyDataset(Dataset):
    def __init__(self, n: int, task: str):
        self._n = n
        self._task = task

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"_task": self._task, "idx": idx}


def _task_name(dataset: MultiTaskDataset, global_idx: int) -> str:
    return dataset._index_map[global_idx][0]


def test_temperature_sampler_world_aligned_blocks():
    batch_size = 2
    world_size = 4
    dataset = MultiTaskDataset(
        {
            "asr": _DummyDataset(200, "asr"),
            "s2tt_de": _DummyDataset(200, "s2tt_de"),
            "mt_de": _DummyDataset(200, "mt_de"),
        }
    )
    sampler = TemperatureSampler(
        dataset,
        ratios={"asr": 0.5, "s2tt_de": 0.5},
        text_pairs={"s2tt_de": "mt_de"},
        seed=7,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=0,
        total_samples=96,
    )

    indices = list(iter(sampler))
    global_batches = list(BatchSampler(indices, batch_size=batch_size, drop_last=False))

    # Every per-device batch is homogeneous.
    for batch in global_batches:
        tasks = {_task_name(dataset, idx) for idx in batch}
        assert len(tasks) == 1

    # Every local step sees a single modality across ranks after accelerate-style
    # batch sharding (rank r takes batch ids step * world_size + r).
    for step in range(len(global_batches) // world_size):
        step_tasks = {
            _task_name(dataset, global_batches[step * world_size + rank][0])
            for rank in range(world_size)
        }
        assert len(step_tasks) == 1


def test_temperature_sampler_len_matches_iteration_and_rank_is_global():
    dataset = MultiTaskDataset(
        {
            "asr": _DummyDataset(100, "asr"),
            "s2tt_de": _DummyDataset(100, "s2tt_de"),
            "mt_de": _DummyDataset(100, "mt_de"),
        }
    )

    s_rank0 = TemperatureSampler(
        dataset,
        ratios={"asr": 0.4, "s2tt_de": 0.6},
        text_pairs={"s2tt_de": "mt_de"},
        seed=11,
        batch_size=2,
        num_replicas=4,
        rank=0,
        total_samples=100,
    )
    s_rank1 = TemperatureSampler(
        dataset,
        ratios={"asr": 0.4, "s2tt_de": 0.6},
        text_pairs={"s2tt_de": "mt_de"},
        seed=11,
        batch_size=2,
        num_replicas=4,
        rank=1,
        total_samples=100,
    )

    s_rank0.set_epoch(3)
    expected_len = len(s_rank0)
    s_rank0.set_epoch(3)
    emitted = list(iter(s_rank0))
    assert expected_len == len(emitted)

    s_rank1.set_epoch(3)
    assert emitted == list(iter(s_rank1))
