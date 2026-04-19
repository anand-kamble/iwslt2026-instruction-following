"""Synthetic T1–T10 dataset loading for Stage 3 surprise-aware training.

Reads JSONL shards produced by ``scripts/generate_synthetic.py`` at
``data/synthetic/<task>/<model_tag>/part-*.jsonl`` and emits HuggingFace
Datasets in the same ChatML shape as CoVoST2 / LibriSQA loaders:

    example["messages"]     = ChatML user/assistant turns
    example["audio_path"]   = (audio tasks) HF reference — decoded lazily
    example["_task"]        = task identifier

Text tasks (T1–T7) consume the transcript via ``<text>...</text>`` so no
audio is attached. Audio tasks (T8–T10) re-open the underlying CoVoST2
rows by index and cast the audio column to 16 kHz; decoding is lazy.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Audio, Dataset, load_dataset

from src.data.utils import (
    SPEECH_PLACEHOLDER,
    SYNTHETIC_TASK_PROMPTS,
    format_chat_messages,
    format_text_only_messages,
    get_data_dir,
)

TEXT_TASKS = frozenset(
    {"T1_keywords", "T2_ner", "T3_gist_en", "T4_topic",
     "T5_numeric_qa", "T6_gist_de", "T7_gist_zh"}
)
AUDIO_TASKS = frozenset({"T8_tone_en", "T9_tone_de", "T10_tone_zh"})
ALL_SYNTHETIC_TASKS = TEXT_TASKS | AUDIO_TASKS

_DEFAULT_TEXT_MODEL = "gemma-4-31B"
_DEFAULT_AUDIO_MODEL = "gemma-4-E4B-it"
_COVOST2_REPO = "fixie-ai/covost2"
_COVOST2_CONFIG = "en_de"


def _synthetic_root(data_root: Path | None = None) -> Path:
    return (data_root or get_data_dir()) / "synthetic"


def _iter_jsonl_rows(shard_dir: Path):
    """Yield non-empty, non-rejected records from every part-*.jsonl shard."""
    if not shard_dir.exists():
        raise FileNotFoundError(
            f"Synthetic shard dir missing: {shard_dir}. "
            f"Run scripts/gen_text.sh or scripts/gen_audio.sh first."
        )
    shards = sorted(shard_dir.glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No part-*.jsonl under {shard_dir}")

    for shard in shards:
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Hard guarantee: zero empties anywhere. The generator already
                # writes rejects to *.rejects, but we double-check here.
                if rec.get("audit", {}).get("rejected"):
                    continue
                answer = (rec.get("answer") or "").strip()
                transcript = (rec.get("transcript") or "").strip()
                if not answer or answer.upper() == "NONE":
                    continue
                if not transcript:
                    continue
                rec["answer"] = answer
                rec["transcript"] = transcript
                yield rec


def _parse_hf_index(audio_path: str) -> int:
    """Extract the CoVoST2 train-split row index from the stored URI."""
    return int(audio_path.rstrip("/").split("/")[-1])


def _load_shards(task_name: str, model_tag: str, data_root: Path | None):
    shard_dir = _synthetic_root(data_root) / task_name / model_tag
    return list(_iter_jsonl_rows(shard_dir))


def load_synthetic_text_task(
    task_name: str,
    model_tag: str = _DEFAULT_TEXT_MODEL,
    data_root: Path | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Load a synthetic text task (T1–T7) as a HF Dataset.

    Each example carries a ``messages`` field formatted with
    ``format_text_only_messages`` so the SpeechTextCollator treats it as
    text-only (no <|speech|> placeholder).
    """
    if task_name not in TEXT_TASKS:
        raise ValueError(
            f"{task_name!r} is not a text task. Expected one of {sorted(TEXT_TASKS)}."
        )

    rows = _load_shards(task_name, model_tag, data_root)
    if max_samples is not None:
        rows = rows[:max_samples]

    instruction = SYNTHETIC_TASK_PROMPTS[task_name]
    records = [
        {
            "id": r["id"],
            "task": task_name,
            "transcript": r["transcript"],
            "answer": r["answer"],
            "messages": format_text_only_messages(
                instruction=instruction,
                response=r["answer"],
                source_text=r["transcript"],
            ),
        }
        for r in rows
    ]

    ds = Dataset.from_list(records)
    print(f"Synthetic {task_name} (text): {len(ds)} samples")
    return ds


def load_synthetic_audio_task(
    task_name: str,
    model_tag: str = _DEFAULT_AUDIO_MODEL,
    data_root: Path | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Load a synthetic audio task (T8–T10) as a HF Dataset.

    Pulls the referenced CoVoST2 rows by index, swaps in the synthetic
    answer, and attaches a ``messages`` field. The ``audio`` column is
    cast to 16 kHz and decoded lazily by the collator.
    """
    if task_name not in AUDIO_TASKS:
        raise ValueError(
            f"{task_name!r} is not an audio task. Expected one of {sorted(AUDIO_TASKS)}."
        )

    rows = _load_shards(task_name, model_tag, data_root)
    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise RuntimeError(f"Synthetic {task_name}: no rows after filtering.")

    indices = [_parse_hf_index(r["audio_path"]) for r in rows]
    answers = [r["answer"] for r in rows]
    ids = [r["id"] for r in rows]

    cv = load_dataset(_COVOST2_REPO, _COVOST2_CONFIG, split="train")
    cv = cv.select(indices)
    cv = cv.cast_column("audio", Audio(sampling_rate=16000))
    cv = cv.add_column("synthetic_answer", answers)
    cv = cv.add_column("synthetic_id", ids)
    cv = cv.add_column("synthetic_task", [task_name] * len(cv))

    instruction = SYNTHETIC_TASK_PROMPTS[task_name]

    def _format(example):
        example["messages"] = format_chat_messages(
            instruction, example["synthetic_answer"]
        )
        return example

    cv = cv.map(_format)
    print(f"Synthetic {task_name} (audio): {len(cv)} samples")
    return cv


def load_synthetic_task(
    task_name: str,
    model_tag: str | None = None,
    data_root: Path | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Dispatch to the text or audio loader based on task identity."""
    if task_name in TEXT_TASKS:
        tag = model_tag or _DEFAULT_TEXT_MODEL
        return load_synthetic_text_task(task_name, tag, data_root, max_samples)
    if task_name in AUDIO_TASKS:
        tag = model_tag or _DEFAULT_AUDIO_MODEL
        return load_synthetic_audio_task(task_name, tag, data_root, max_samples)
    raise ValueError(
        f"Unknown synthetic task {task_name!r}. Expected one of {sorted(ALL_SYNTHETIC_TASKS)}."
    )


__all__ = [
    "ALL_SYNTHETIC_TASKS",
    "AUDIO_TASKS",
    "SPEECH_PLACEHOLDER",
    "TEXT_TASKS",
    "load_synthetic_audio_task",
    "load_synthetic_task",
    "load_synthetic_text_task",
]
