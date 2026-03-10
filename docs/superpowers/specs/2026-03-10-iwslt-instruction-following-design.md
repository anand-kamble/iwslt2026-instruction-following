# IWSLT 2026 Instruction Following — Project Design

## Overview

Fine-tune a system combining a Speech Foundation Model (Seamless M4T v2) with Qwen3-4B-Instruct for the IWSLT 2026 Instruction Following shared task (constrained setting).

## Task

The system must handle 6 tasks via instruction following:
- **ASR** — Automatic Speech Recognition (English)
- **S2TT** — Speech-to-Text Translation (EN → DE, IT, ZH)
- **SQA** — Spoken Question Answering (EN, multilingual)
- **S2TSUM** — Speech-to-Text Summarization (English)
- **ACHAP** — Audio Chaptering (English)
- **Surprise** — Unknown task, solvable via in-context learning

### Tracks
- **Short track:** 15-20s auto-segmented audio
- **Long track:** ~6min audio files (all short tasks + S2TSUM + ACHAP)

## Constrained Setting

### Models
- SFM: `facebook/seamless-m4t-v2-large`
- LLM: `Qwen/Qwen3-4B-Instruct-2507`

### Permitted Training Data
| Task | Datasets |
|------|----------|
| ASR/S2TT | EuroParlST, CoVoST2, GigaST |
| SQA | LibriSQA |
| S2TSUM | NUTSHELL |
| ACHAP | YTSeg |

### Validation
- ASR/S2TT/SQA/S2TSUM: MCIF dataset (includes IWSLT25 test set)
- ACHAP: YTSeg

## Architecture Decisions

- **Framework-agnostic** — training framework TBD (likely Unsloth or similar)
- **Task-oriented configs** — one YAML per task, shared base config
- **HuggingFace Hub for checkpoints** — no large files in git
- **Branch-per-task collaboration** — `feat/asr`, `feat/sqa`, etc.

## Collaboration

- 2-3 people working on different tasks independently
- GPU setup varies (cloud-based, mixed)
- Configs committed to git for reproducibility
- Checkpoints pushed to HuggingFace Hub via scripts

## Evaluation

- Output format: MCIF standard
- Evaluation scripts from MCIF GitHub repo
- Submission via Meetween SPEECHM Evaluation Server
