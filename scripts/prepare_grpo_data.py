"""Build the GRPO JSONL: 50% Stage 3 text-only + 50% synthetic T1..T10.

Driven by a mix YAML (configs/swift/stage4_mix.yaml). Each row in the output
JSONL contains:
    - "messages": [system, user]   (assistant turn stripped; GRPO generates)
    - "solution": gold answer
    - "task":     task name used by the task-routed reward ORMs

Usage:
    python scripts/prepare_grpo_data.py \\
        --mix    configs/swift/stage4_mix.yaml \\
        --config configs/stage2_text_lora_v2.yaml \\
        --out    data/grpo/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def _load_project_config(path: str) -> dict:
    from src.config import load_config
    return load_config(path)


def _row_from_stage3(row: dict, task: str) -> dict | None:
    """Strip the assistant turn from a Stage 3 text-only row; move it to `solution`."""
    msgs = row.get("messages")
    if not msgs or msgs[-1].get("role") != "assistant":
        return None
    solution = (msgs[-1].get("content") or "").strip()
    if not solution:
        return None
    return {
        "messages": msgs[:-1],
        "solution": solution,
        "task":     task,
    }


def _collect_stage3(
    config_path: str,
    ratios: dict[str, float],
    total: int,
    rng: random.Random,
) -> list[dict]:
    from src.data.text_only import load_text_only_datasets

    cfg = _load_project_config(config_path)
    datasets = load_text_only_datasets(cfg)

    out: list[dict] = []
    for task, frac in ratios.items():
        if task not in datasets:
            print(f"  [stage3] {task}: dataset missing — skipping")
            continue
        target = int(round(total * frac))
        ds = datasets[task]
        pool = []
        for row in ds:
            rec = _row_from_stage3(row, task)
            if rec is not None:
                pool.append(rec)
        if not pool:
            print(f"  [stage3] {task}: 0 usable rows")
            continue
        if len(pool) <= target:
            rng.shuffle(pool)
            chosen = pool
        else:
            chosen = rng.sample(pool, target)
        out.extend(chosen)
        print(f"  [stage3] {task}: {len(chosen)} rows (target {target}, pool {len(pool)})")
    return out


def _collect_synth(
    synth_root: str,
    ratios: dict[str, float],
    total: int,
    rng: random.Random,
) -> list[dict]:
    from grpo.synth_loader import sample_synth

    out: list[dict] = []
    for task, frac in ratios.items():
        target = int(round(total * frac))
        chosen = sample_synth(synth_root, task, target, rng)
        if not chosen:
            print(f"  [synth]  {task}: 0 rows found under {synth_root}/{task}")
            continue
        out.extend(chosen)
        print(f"  [synth]  {task}: {len(chosen)} rows (target {target})")
    return out


def main(mix_path: str, config_path: str, out_path: str) -> None:
    with open(mix_path) as f:
        mix = yaml.safe_load(f)

    total = int(mix["total_samples"])
    stage3_frac = float(mix.get("stage3_fraction", 0.5))
    stage3_total = int(round(total * stage3_frac))
    synth_total  = total - stage3_total

    rng = random.Random(int(mix.get("shuffle_seed", 42)))

    print(f"Target total: {total}  |  Stage 3: {stage3_total}  |  synth: {synth_total}")
    print("Collecting Stage 3 rows...")
    stage3_rows = _collect_stage3(config_path, mix["stage3_ratios"], stage3_total, rng)

    print("Collecting synthetic rows...")
    synth_root = mix.get("synth_root", "data/synthetic")
    synth_rows = _collect_synth(synth_root, mix["synth_ratios"], synth_total, rng)

    rows = stage3_rows + synth_rows
    rng.shuffle(rows)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    task_counts = Counter(r["task"] for r in rows)
    print(f"\nWrote {len(rows)} rows → {out}")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:16s} {count:>6d}  ({count / max(1, len(rows)):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix",    required=True, help="Mix YAML (e.g. configs/swift/stage4_mix.yaml)")
    ap.add_argument("--config", required=True, help="Project YAML that lists text-only datasets")
    ap.add_argument("--out",    required=True, help="Output JSONL path")
    args = ap.parse_args()
    main(args.mix, args.config, args.out)
