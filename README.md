# IWSLT 2026 — Instruction Following (Constrained)

Fine-tuning Seamless M4T v2 + Qwen3-4B-Instruct for the [IWSLT 2026 Instruction Following](https://iwslt.org/2026/instruction-following) shared task.

## Architecture

```
Audio → Seamless M4T v2 (frozen) → Projector (trainable) → Qwen3-4B (LoRA) → Text
         1024-dim, ~62.5 tok/s      Linear+GELU+Linear       32-rank LoRA
```

- **Speech encoder:** `facebook/seamless-m4t-v2-large` — frozen, extracts 1024-dim features
- **Projector:** 2-layer MLP (1024 → 2560) — bridges speech to LLM embedding space
- **LLM:** `Qwen/Qwen3-4B-Instruct-2507` via Unsloth — QLoRA (4-bit) fine-tuning
- **Evaluation:** MCIF benchmark, XML output format

## Tasks

| Task | Description | Training Data | Eval Metric |
|------|-------------|---------------|-------------|
| ASR | Speech Recognition (EN) | EuroParlST, CoVoST2 | WER |
| S2TT | Speech Translation (EN→DE/IT/ZH) | EuroParlST, CoVoST2 | COMET |
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

# Unsloth (for training)
pip install -e ".[unsloth]"

# Evaluation tools
pip install -e ".[eval]"

# Development
pip install -e ".[dev]"
```

## Training

```bash
# Train ASR
python src/train.py --config configs/asr.yaml

# Train S2TT
python src/train.py --config configs/s2tt.yaml

# Train SQA
python src/train.py --config configs/sqa.yaml

# Custom output directory
python src/train.py --config configs/asr.yaml --output_dir ./outputs/asr-v2

# Resume from checkpoint
python src/train.py --config configs/asr.yaml --resume_from ./outputs/asr/checkpoint-1000
```

## Evaluation

```bash
# Evaluate on MCIF
python src/evaluate.py --config configs/asr.yaml --checkpoint ./outputs/asr --language en

# Evaluate S2TT for German
python src/evaluate.py --config configs/s2tt.yaml --checkpoint ./outputs/s2tt --language de
```

## Inference & Submission

```bash
# Generate submission from test audio
python src/inference.py --checkpoint ./outputs/asr --input_dir ./test_data --output_dir ./submission --task asr

# Push checkpoint to HuggingFace
bash scripts/push_checkpoint.sh ./outputs/asr your-org/iwslt2026-asr-v1
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
configs/              Per-task training configs (YAML)
src/
  data/               Dataset loaders + preprocessing per task
    collator.py       Multimodal data collator (audio + text)
    utils.py          Shared utilities, prompts, constants
  model/
    adapters.py       Speech → LLM projection layer
    build_model.py    Model loading (encoder, LLM, projector)
    speech_llm.py     End-to-end SpeechLLM model
  train.py            Training entry point
  evaluate.py         MCIF evaluation entry point
  inference.py        Submission generation
scripts/              Shell helpers
tests/                Smoke tests
```

## Collaboration

- Branch per task: `feat/asr`, `feat/sqa`, etc.
- All hyperparams in YAML configs — committed to git
- Checkpoints on HuggingFace Hub (not in git)
