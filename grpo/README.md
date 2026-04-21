# Stage 4 — text-only GRPO on Qwen3-4B (constrained track)

Minimal GRPO pipeline using the official ms-swift CLI — no custom model
registration, no rollout-server setup. Trains the LoRA adapter that is shared
between the speech and text paths, so instruction-following gains transfer
back to speech inference.

## Why text-only

The IWSLT 2026 constrained track forbids non-Qwen3-4B / non-SeamlessM4Tv2
models. Without a custom `register_model` plugin, ms-swift CLI cannot load
the composite `SpeechLLM`. The remaining constrained-track option is GRPO on
Qwen3-4B text paths, with the trained LoRA applied back to the speech model
at evaluation time.

## Files

| File                                 | Purpose                                |
| ------------------------------------ | -------------------------------------- |
| `grpo/rewards.py`                    | Task-routed reward ORMs                |
| `scripts/prepare_grpo_data.py`       | Converts text-only Stage 2/3 datasets  |
| `configs/swift/stage4_grpo.yaml`     | GRPO CLI config                        |
| `scripts/slurm_stage4_grpo.sh`       | SLURM launcher                         |

## Reward functions (registered in `grpo/rewards.py`)

| Name             | Task family              | Metric                              |
| ---------------- | ------------------------ | ----------------------------------- |
| `mt_chrf`        | `mt_*`, `s2tt_*`         | sacrebleu sentence chrF, ∈ [0, 1]   |
| `qa_token_f1`    | `qa_*`, `sqa_*`          | SQuAD-style normalized token F1     |
| `language_match` | `mt_*`, `s2tt_*`         | 1 on target-lang match, 0 on miss   |
| `repetition`     | all (built-in)           | n-gram repetition penalty           |
| `soft_overlong`  | all (built-in)           | Length penalty near max_completion  |

Each ORM returns `None` for rows outside its task family (ms-swift's
documented multi-task pattern — see `GRPO/DeveloperGuide/multi_task.md`).

## Run

```bash
# 1. Prepare JSONL from your Stage 2 text config
python scripts/prepare_grpo_data.py \
    --config configs/stage2_text_lora_v2.yaml \
    --out data/grpo/train.jsonl

# 2. Launch GRPO
STAGE2_CHECKPOINT=outputs/stage2_text_lora_v2 \
    sbatch scripts/slurm_stage4_grpo.sh
```

## Knobs worth tuning

- `num_generations` (default 8) — group size for advantage calc; drop to 4 if OOM.
- `beta` (default 0.001) — KL to reference; 0 disables ref model entirely.
- `reward_weights` — bump `mt_chrf` / `qa_token_f1` if hygiene terms dominate.
- `vllm_gpu_memory_utilization` — raise to 0.8 if the policy fits comfortably.

## Known limits

- **Text-only**: does not directly optimize ASR/WER — that's an audio metric.
  Transfer effect via shared LoRA is empirical; measure on MCIF after training.
- **No vLLM for custom models**: we rely on vLLM only because Qwen3-4B is a
  stock HuggingFace model. Any future switch back to `SpeechLLM` via
  `register.py` requires `--use_vllm false` and transformers rollout.
