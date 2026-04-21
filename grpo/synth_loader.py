"""Loader for synthetic T1..T10 JSONL data → ms-swift GRPO rows.

Each synthetic row has `transcript`, `answer`, plus metadata. We wrap them in
the project's NAVER Table 4 text-only chat format using the per-task
instruction drafts below. Drop rows where `audit.rejected == True`.
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Iterable

from src.data.utils import SYSTEM_PROMPT


# Per-task instruction drafts. Tweak wording here if needed — every synthetic
# row of a given task reuses its instruction verbatim.
TASK_INSTRUCTIONS = {
    "T1_keywords":   "Extract the three most salient keywords from the transcript. Return them as a comma-separated list.",
    "T3_gist_en":    "Summarize the transcript in one concise English sentence capturing the main claim.",
    "T4_topic":      "In one or two words, state the topic of the transcript.",
    "T5_numeric_qa": "Read the transcript and answer the question with a number and its unit.",
    "T6_gist_de":    "Fasse den Transkript in einem kurzen deutschen Satz zusammen, der die zentrale Aussage wiedergibt.",
    "T7_gist_zh":    "请用一句简洁的中文概括该文本记录的主要观点。",
    "T8_tone_en":    "Describe the speaker's tone, pace, and delivery style in one or two English sentences.",
    "T9_tone_de":    "Beschreibe den Tonfall, das Sprechtempo und den Vortragsstil des Sprechers in ein bis zwei deutschen Sätzen.",
    "T10_tone_zh":   "请用一到两句中文描述说话者的语气、语速和表达风格。",
}

# T2_ner is intentionally missing: dropped from GRPO.


def _format_user_content(task: str, transcript: str, row: dict) -> str:
    """Build the user turn. Matches src/data/utils.format_text_only_messages."""
    instruction = TASK_INSTRUCTIONS[task]
    # T5 carries its question inside `answer` as "Q: ... || A: ...". Split so
    # the Q goes into the prompt and only the A becomes the solution.
    if task == "T5_numeric_qa":
        question = row.get("_split_question", "")
        if question:
            instruction = f"{instruction}\nQuestion: {question}"
    return (
        f"Content: <text>{transcript}</text>\n"
        f"Question: {instruction}\n"
        f"Your answer:"
    )


def _extract_solution(task: str, row: dict) -> str | None:
    """Return the gold answer (ms-swift's `solution` column)."""
    raw = (row.get("answer") or "").strip()
    if not raw:
        return None
    if task == "T5_numeric_qa":
        # "Q: ... || A: 1 referendum" → split and keep A as solution; side-effect
        # stores question on the row so the formatter can reuse it.
        if "||" in raw:
            q, a = raw.split("||", 1)
            q = q.strip()
            a = a.strip()
            q = q[2:].strip() if q.upper().startswith("Q:") else q
            a = a[2:].strip() if a.upper().startswith("A:") else a
            row["_split_question"] = q
            return a or None
        return raw
    return raw


def iter_synth_rows(synth_root: str, task: str) -> Iterable[dict]:
    """Yield GRPO rows for one synthetic task."""
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown or dropped synthetic task: {task}")
    pattern = os.path.join(synth_root, task, "**", "*.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("audit", {}).get("rejected"):
                    continue
                transcript = (row.get("transcript") or "").strip()
                if not transcript:
                    continue
                solution = _extract_solution(task, row)
                if not solution:
                    continue
                user_content = _format_user_content(task, transcript, row)
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    "solution": solution,
                    "task":     task,
                }


def sample_synth(
    synth_root: str,
    task: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Reservoir-sample up to `n` rows from a synthetic task's files."""
    pool = list(iter_synth_rows(synth_root, task))
    if not pool:
        return []
    if len(pool) <= n:
        rng.shuffle(pool)
        return pool
    # Uniform subsample without replacement
    return rng.sample(pool, n)
