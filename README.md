# IWSLT 2026 — Instruction Following (Constrained)

Fine-tuning Seamless M4T v2 + Qwen3-4B-Instruct for the [IWSLT 2026 Instruction Following](https://iwslt.org/2026/instruction-following) shared task.

## Architecture

```
Audio → Seamless M4T v2 (frozen) → Projector (trainable) → Qwen3-4B (LoRA) → Text
         1024-dim, ~62.5 tok/s      Transformer (39M)        16-rank LoRA
```

- **Speech encoder:** `facebook/seamless-m4t-v2-large` — frozen, extracts 1024-dim features
- **Projector:** 4-layer Transformer encoder (1024 → 2560) with 3x frame-averaging downsample
- **LLM:** `Qwen/Qwen3-4B-Instruct-2507` — LoRA fine-tuning (r=16, bf16)
- **Evaluation:** MCIF benchmark, XML output format

## Tasks

| Task | Description | Training Data | Eval Metric |
|------|-------------|---------------|-------------|
| ASR | Speech Recognition (EN) | EuroParlST, GigaST | WER |
| S2TT | Speech Translation (EN→DE/IT/ZH) | EuroParlST, GigaST | COMET |
| SQA | Spoken Question Answering | LibriSQA | BERTScore |
| S2TSUM | Summarization (EN) | NUTSHELL | BERTScore |
| ACHAP | Audio Chaptering (EN) | YTSeg | TBD |
| Surprise | TBD (in-context learning) | — | — |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install -e .

# CUDA + quantization support
pip install -e ".[cuda]"

# Evaluation tools
pip install -e ".[eval]"

# Development
pip install -e ".[dev]"
```

## Training (3-stage pipeline)

```bash
# Full pipeline via SLURM (H200)
sbatch scripts/slurm_train.sh

# Or run stages individually:
python src/train.py --config configs/stage1_alignment.yaml
python src/train.py --config configs/stage2_text_lora.yaml
python src/train.py --config configs/stage3_multimodal.yaml \
    --stage1_checkpoint outputs/stage1_asr \
    --stage2_checkpoint outputs/stage2_text_lora
```

## Evaluation

```bash
python src/evaluate.py --config configs/stage3_multimodal.yaml \
    --checkpoint ./outputs/stage3_multitask --languages en de it zh
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# Full pipeline smoke test (100 samples, mock audio)
python scripts/smoke_pipeline.py --samples 100 --steps 20
```

## Project Structure

```
configs/              Per-task and per-stage training configs (YAML)
src/
  data/               Dataset loaders + preprocessing per task
    collator.py       Multimodal data collator (audio + text)
    multitask.py      Multi-task dataset with temperature sampling
    utils.py          Shared utilities, prompts, constants
  model/
    adapters.py       Speech → LLM projection layers
    build_model.py    Model loading (encoder, LLM, projector)
    speech_llm.py     End-to-end SpeechLLM model
  train.py            Training entry point (all stages)
  evaluate.py         MCIF evaluation entry point
  inference.py        Submission generation
scripts/              SLURM job scripts, data download, smoke tests
tests/                Smoke tests
```

## Acknowledgments

Some training and experimentation for this project used compute resources from
the [UIUC Campus Cluster](https://campuscluster.illinois.edu/).
