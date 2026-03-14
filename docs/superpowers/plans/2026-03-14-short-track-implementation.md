# IWSLT 2026 Short Track Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 3-stage training pipeline (projector alignment → text LoRA → multimodal merge) for the IWSLT 2026 instruction-following short track using Seamless M4T v2 + Qwen3-4B.

**Architecture:** Frozen SeamlessM4T encoder → TransformerProjector (Conv1d stride-3 + 4-layer Transformer + Linear) → Qwen3-4B with LoRA r=16 in 8-bit. Three training stages: (1) projector-only on ASR, (2) text-only LoRA on MT+SQA, (3) brief multimodal merge on all tasks.

**Tech Stack:** PyTorch, HuggingFace Transformers, Unsloth, PEFT, bitsandbytes, mcif-bench, datasets

**Spec:** `docs/superpowers/specs/2026-03-14-short-track-training-strategy.md`

---

## File Structure

### New files:
| File | Responsibility |
|------|---------------|
| `src/model/adapters.py` | Add `TransformerProjector` class (keep existing `SpeechProjector`) |
| `src/data/text_only.py` | Text-only data loaders for Stage 2 (MT pairs, SQA text, no audio) |
| `src/data/translate_sqa.py` | Script to generate synthetic multilingual SQA via SeamlessM4T |
| `scripts/precompute_features.py` | Precompute frozen speech encoder outputs to `.pt` files |
| `configs/stage1_alignment.yaml` | Stage 1 config: ASR-only, no_lora, projector-only |
| `configs/stage2_text_lora.yaml` | Stage 2 config: text-only LoRA training |
| `configs/stage3_multimodal.yaml` | Stage 3 config: multimodal merge, max_steps=2000 |

### Modified files:
| File | Changes |
|------|---------|
| `src/model/build_model.py` | Add `projector_type`, `load_in_8bit`, `no_lora` support |
| `configs/base.yaml` | Update defaults: 8-bit, lora_r=16, new flags |
| `src/data/collator.py` | Support precomputed features + text-only samples |
| `src/data/sqa.py` | Load synthetic translated SQA datasets |
| `src/data/utils.py` | Add `format_text_only_messages()` for Stage 2 |
| `src/train.py` | 3-stage support: no_lora, text_only, checkpoint loading, differential LR |
| `src/evaluate.py` | Fix projector loading, 8-bit, multi-language, generation params |
| `src/inference.py` | MCIF dataset mode, 4-language output, generation params |

---

## Chunk 1: Architecture & Data Pipeline

### Task 1: TransformerProjector

**Files:**
- Modify: `src/model/adapters.py`
- Test: `tests/test_configs.py` (existing projector test), `tests/test_smoke.py`

- [ ] **Step 1: Write failing test for TransformerProjector shape**

Add to `tests/test_smoke.py`:

```python
class TestTransformerProjector:
    def test_output_shape(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=3)
        x = torch.randn(2, 150, 1024)
        out = proj(x)
        assert out.shape == (2, 50, 2560)

    def test_downsample_factor_1(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=1)
        x = torch.randn(2, 100, 1024)
        out = proj(x)
        assert out.shape == (2, 100, 2560)

    def test_param_count(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=3)
        total = sum(p.numel() for p in proj.parameters())
        assert 35_000_000 < total < 45_000_000  # ~39M expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_smoke.py::TestTransformerProjector -v`
Expected: FAIL with `ImportError: cannot import name 'TransformerProjector'`

- [ ] **Step 3: Implement TransformerProjector**

Add to `src/model/adapters.py` after `SpeechProjector`:

```python
class TransformerProjector(nn.Module):
    """Projects speech encoder outputs into the LLM's embedding space.

    Architecture: Conv1d downsample → Transformer encoder → Linear projection.
    Based on NAVER IWSLT 2025 constrained system projector.
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        llm_dim: int = 2560,
        downsample_factor: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        if downsample_factor > 1:
            self.downsample = nn.Conv1d(
                speech_dim, speech_dim,
                kernel_size=downsample_factor, stride=downsample_factor,
            )
        else:
            self.downsample = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=speech_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(speech_dim, llm_dim)

    def forward(self, speech_features: torch.Tensor) -> torch.Tensor:
        """Project speech features to LLM embedding space.

        Args:
            speech_features: (batch, seq_len, speech_dim)
        Returns:
            (batch, seq_len // downsample_factor, llm_dim)
        """
        x = speech_features

        if self.downsample is not None:
            x = x.transpose(1, 2)  # (B, C, T)
            x = self.downsample(x)
            x = x.transpose(1, 2)  # (B, T', C)

        x = self.transformer(x)
        return self.output_proj(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_smoke.py::TestTransformerProjector -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/model/adapters.py tests/test_smoke.py
git commit -m "feat: add TransformerProjector (Conv1d + 4-layer Transformer + Linear)"
```

---

### Task 2: Build model updates (8-bit, no_lora, projector_type)

**Files:**
- Modify: `src/model/build_model.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write failing test for build_projector with projector_type**

Add to `tests/test_smoke.py`:

```python
class TestBuildModel:
    def test_build_transformer_projector(self):
        from src.model.build_model import build_projector
        proj = build_projector(projector_type="transformer", downsample_factor=3, device="cpu")
        assert hasattr(proj, "transformer")
        x = torch.randn(1, 90, 1024)
        out = proj(x)
        assert out.shape == (1, 30, 2560)

    def test_build_legacy_projector(self):
        from src.model.build_model import build_projector
        proj = build_projector(projector_type="linear", downsample_factor=1, device="cpu")
        x = torch.randn(1, 100, 1024)
        out = proj(x)
        assert out.shape == (1, 100, 2560)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_smoke.py::TestBuildModel -v`
Expected: FAIL with `TypeError: build_projector() got an unexpected keyword argument 'projector_type'`

- [ ] **Step 3: Update build_projector to support projector_type**

Modify `src/model/build_model.py`:

```python
from src.model.adapters import SpeechProjector, TransformerProjector

def build_projector(
    speech_dim: int = SPEECH_DIM,
    llm_dim: int = QWEN3_4B_DIM,
    downsample_factor: int = 3,
    projector_type: str = "transformer",
    device: str = "cuda",
) -> nn.Module:
    """Build the trainable projection layer between speech encoder and LLM."""
    if projector_type == "transformer":
        projector = TransformerProjector(
            speech_dim=speech_dim,
            llm_dim=llm_dim,
            downsample_factor=downsample_factor,
        )
    else:
        projector = SpeechProjector(
            speech_dim=speech_dim,
            llm_dim=llm_dim,
            downsample_factor=downsample_factor,
        )
    return projector.to(device)
```

- [ ] **Step 4: Update load_llm_with_lora to support 8-bit and no_lora**

Modify `src/model/build_model.py` — update `load_llm_with_lora`:

```python
def load_llm_with_lora(
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    load_in_8bit: bool = True,
    no_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    if not no_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            max_seq_length=max_seq_length,
        )

    return model, tokenizer
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_smoke.py::TestBuildModel -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/model/build_model.py tests/test_smoke.py
git commit -m "feat: support projector_type, 8-bit quantization, and no_lora mode in build_model"
```

---

### Task 3: Config updates

**Files:**
- Modify: `configs/base.yaml`
- Create: `configs/stage1_alignment.yaml`, `configs/stage2_text_lora.yaml`, `configs/stage3_multimodal.yaml`
- Test: `tests/test_configs.py`

- [ ] **Step 1: Update base.yaml**

```yaml
# Base configuration — shared across all tasks
model:
  sfm: facebook/seamless-m4t-v2-large
  llm: unsloth/Qwen3-4B-Instruct-2507

hub:
  organization: anand-kamble
  push_to_hub: true

training:
  seed: 42
  bf16: true
  gradient_checkpointing: true
  load_in_4bit: false
  load_in_8bit: true
  no_lora: false
  text_only: false
  projector_type: transformer
  max_seq_length: 4096
  lora_r: 16
  lora_alpha: 16
  gradient_accumulation_steps: 4
  optim: adamw_8bit
  lr_scheduler_type: cosine
  warmup_steps: 100
  logging_steps: 50
  save_steps: 500
  eval_steps: 500
  downsample_factor: 5

paths:
  data_dir: ./data
  output_dir: ./outputs
  checkpoint_dir: ./checkpoints
```

- [ ] **Step 2: Create stage1_alignment.yaml**

```yaml
# Stage 1: Projector alignment on ASR (no LoRA)
defaults:
  - base

task: asr
language: en
stage: 1

data:
  train:
    - europarlst
    - gigast
  eval:
    - mcif

training:
  no_lora: true
  downsample_factor: 3
  num_epochs: 2
  per_device_train_batch_size: 4
  learning_rate: 1e-4
```

- [ ] **Step 3: Create stage2_text_lora.yaml**

```yaml
# Stage 2: Text-only LoRA pre-training (MT + SQA, no audio)
defaults:
  - base

task: text_lora
stage: 2
text_only: true

data:
  train:
    - mt_europarlst
    - mt_gigast
    - sqa_translated
    - sqa_english

training:
  text_only: true
  num_epochs: 2
  per_device_train_batch_size: 8
  learning_rate: 3e-4
  downsample_factor: 3
```

- [ ] **Step 4: Create stage3_multimodal.yaml**

```yaml
# Stage 3: Multimodal merge (all tasks with audio, brief training)
defaults:
  - base

task: multitask
track: short
stage: 3

tasks:
  asr:
    train:
      - europarlst
      - gigast
  s2tt:
    train:
      - europarlst
      - gigast
    target_languages:
      - de
      - it
      - zh
  sqa:
    train:
      - librisqa
    languages:
      - en
      - de
      - it
      - zh

sampling:
  temperature: 0.5

training:
  downsample_factor: 3
  max_steps: 2000
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  lr_projector: 1e-5
  lr_lora: 1e-4
  warmup_steps: 50
  save_steps: 500
  logging_steps: 10
```

- [ ] **Step 5: Write test for new configs**

Add to `tests/test_configs.py`:

```python
def test_stage_configs_parse():
    """Stage configs should parse and have required fields."""
    for filename, expected_stage in [
        ("stage1_alignment.yaml", 1),
        ("stage2_text_lora.yaml", 2),
        ("stage3_multimodal.yaml", 3),
    ]:
        with open(CONFIGS_DIR / filename) as f:
            config = yaml.safe_load(f)
        assert config["stage"] == expected_stage

def test_base_config_8bit():
    """Base config should default to 8-bit quantization."""
    with open(CONFIGS_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)
    assert config["training"]["load_in_8bit"] is True
    assert config["training"]["load_in_4bit"] is False
    assert config["training"]["lora_r"] == 16
    assert config["training"]["lora_alpha"] == 16

def test_stage1_no_lora():
    """Stage 1 must have no_lora=true."""
    with open(CONFIGS_DIR / "stage1_alignment.yaml") as f:
        config = yaml.safe_load(f)
    assert config["training"]["no_lora"] is True

def test_stage2_text_only():
    """Stage 2 must be text-only."""
    with open(CONFIGS_DIR / "stage2_text_lora.yaml") as f:
        config = yaml.safe_load(f)
    assert config["training"]["text_only"] is True
    # Or top-level
    assert config.get("text_only", config["training"].get("text_only")) is True
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_configs.py -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 7: Commit**

```bash
git add configs/base.yaml configs/stage1_alignment.yaml configs/stage2_text_lora.yaml configs/stage3_multimodal.yaml tests/test_configs.py
git commit -m "feat: add 3-stage training configs (alignment, text LoRA, multimodal merge)"
```

---

### Task 4: Text-only data loader and format helpers

**Files:**
- Modify: `src/data/utils.py`
- Create: `src/data/text_only.py`
- Test: `tests/test_configs.py` (add format test)

- [ ] **Step 1: Add format_text_only_messages to utils.py**

Add to `src/data/utils.py`:

```python
def format_text_only_messages(instruction: str, response: str | None = None) -> list[dict]:
    """Format instruction and response as ChatML messages WITHOUT speech placeholder.

    Used for Stage 2 text-only LoRA training.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that follows instructions."},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
```

- [ ] **Step 2: Write test for format_text_only_messages**

Add to `tests/test_configs.py`:

```python
def test_format_text_only_messages():
    """Text-only messages should NOT contain speech placeholder."""
    from src.data.utils import format_text_only_messages, SPEECH_PLACEHOLDER
    messages = format_text_only_messages("Translate to German.", "Hallo Welt.")
    assert len(messages) == 3
    for m in messages:
        assert SPEECH_PLACEHOLDER not in m["content"]
    assert messages[1]["content"] == "Translate to German."
    assert messages[2]["content"] == "Hallo Welt."
```

- [ ] **Step 3: Run test**

Run: `python3 -m pytest tests/test_configs.py::test_format_text_only_messages -v`
Expected: PASS

- [ ] **Step 4: Create src/data/text_only.py**

```python
"""Text-only data loaders for Stage 2 LoRA pre-training.

Loads MT parallel text and SQA question-answer pairs WITHOUT audio.
Used to train LoRA adapters on text reasoning before multimodal merge.
"""

from datasets import load_dataset, concatenate_datasets

from src.data.utils import TASK_PROMPTS, UNANSWERABLE, format_text_only_messages


def load_mt_europarlst(target_lang: str, split: str = "train"):
    """Load EuroParlST text-only MT pairs (EN transcript → target translation)."""
    lang_map = {"de": "en-de", "it": "en-it"}
    if target_lang not in lang_map:
        raise ValueError(f"EuroParlST text doesn't support EN→{target_lang}")

    ds = load_dataset(
        "facebook/europarl_st", lang_map[target_lang],
        split=split, trust_remote_code=True,
    )

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_text_only_messages(instruction, example["text"])
        return example

    return ds.map(_format)


def load_mt_gigast(target_lang: str, split: str = "train"):
    """Load GigaST text-only MT pairs (EN transcript → target translation).

    Requires GigaST translation files downloaded locally.
    """
    from src.data.gigast import _load_translations

    if target_lang not in ("de", "zh"):
        raise ValueError(f"GigaST text only supports EN→de and EN→zh, got: {target_lang}")

    translations = _load_translations(target_lang, split)

    ds = load_dataset(
        "speechcolab/gigaspeech", "xl", split=split, trust_remote_code=True,
    )
    ds = ds.filter(lambda ex: ex["segment_id"] in translations)

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        translation = translations[example["segment_id"]]
        example["messages"] = format_text_only_messages(instruction, translation)
        return example

    return ds.map(_format)


def load_sqa_text_only(split: str = "train", language: str = "en"):
    """Load LibriSQA as text-only QA pairs (no audio)."""
    ds = load_dataset("westbrook/librisqa", split=split, trust_remote_code=True)

    def _format(example):
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt_template = TASK_PROMPTS["sqa"][language]
        instruction = prompt_template.replace("{QUESTION}", question)

        if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
            answer = UNANSWERABLE[language]

        example["messages"] = format_text_only_messages(instruction, answer)
        return example

    return ds.map(_format)


def load_text_only_datasets(config: dict):
    """Load all text-only datasets for Stage 2.

    Returns a list of datasets.
    """
    datasets_list = []
    data_names = config.get("data", {}).get("train", [])

    for name in data_names:
        if name == "mt_europarlst":
            for lang in ["de", "it"]:
                try:
                    datasets_list.append(load_mt_europarlst(lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT EuroParlST {lang}: {e}")
        elif name == "mt_gigast":
            for lang in ["de", "zh"]:
                try:
                    datasets_list.append(load_mt_gigast(lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping MT GigaST {lang}: {e}")
        elif name == "sqa_translated":
            for lang in ["de", "it", "zh"]:
                try:
                    datasets_list.append(load_sqa_text_only("train", lang))
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping SQA translated {lang}: {e}")
        elif name == "sqa_english":
            datasets_list.append(load_sqa_text_only("train", "en"))
        else:
            print(f"Unknown text-only dataset: {name}")

    return datasets_list
```

- [ ] **Step 5: Commit**

```bash
git add src/data/utils.py src/data/text_only.py tests/test_configs.py
git commit -m "feat: add text-only data loaders for Stage 2 LoRA pre-training"
```

---

### Task 5: Synthetic SQA translation script

**Files:**
- Create: `src/data/translate_sqa.py`

- [ ] **Step 1: Create translate_sqa.py**

```python
"""Generate synthetic multilingual SQA data by translating LibriSQA.

Usage:
    python -m src.data.translate_sqa --output_dir data/sqa_translated --languages de it zh
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText


LANG_CODES = {
    "de": "deu",
    "it": "ita",
    "zh": "cmn",
}


def translate_batch(model, processor, texts, src_lang="eng", tgt_lang="deu", batch_size=32):
    """Translate a batch of texts using SeamlessM4T."""
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, src_lang=src_lang, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, tgt_lang=tgt_lang, max_new_tokens=256)

        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


def main():
    parser = argparse.ArgumentParser(description="Translate LibriSQA for multilingual SQA")
    parser.add_argument("--output_dir", type=str, default="data/sqa_translated")
    parser.add_argument("--languages", nargs="+", default=["de", "it", "zh"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SeamlessM4T for text translation...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
    model = model.to(args.device)
    model.eval()

    print("Loading LibriSQA...")
    ds = load_dataset("westbrook/librisqa", split="train", trust_remote_code=True)

    questions = [ex.get("question", "") for ex in ds]
    answers = [ex.get("answer", "") for ex in ds]

    for lang in args.languages:
        print(f"\n=== Translating to {lang} ===")
        tgt_code = LANG_CODES[lang]

        print(f"Translating {len(questions)} questions...")
        translated_questions = translate_batch(
            model, processor, questions, tgt_lang=tgt_code, batch_size=args.batch_size
        )

        print(f"Translating {len(answers)} answers...")
        translated_answers = translate_batch(
            model, processor, answers, tgt_lang=tgt_code, batch_size=args.batch_size
        )

        # Save as JSONL
        output_path = Path(args.output_dir) / f"librisqa_{lang}.jsonl"
        count = 0
        with open(output_path, "w") as f:
            for i, (q, a) in enumerate(zip(translated_questions, translated_answers)):
                record = {
                    "id": i,
                    "question": q,
                    "answer": a,
                    "original_question": questions[i],
                    "original_answer": answers[i],
                    "language": lang,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"Saved {count} translated pairs to {output_path}")

    print("\nDone! Run COMETKiwi filtering separately if needed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/data/translate_sqa.py
git commit -m "feat: add synthetic SQA translation script using SeamlessM4T"
```

---

### Task 6: SQA multilingual loader update

**Files:**
- Modify: `src/data/sqa.py`

- [ ] **Step 1: Update sqa.py to support translated datasets**

Add to `src/data/sqa.py`:

```python
def load_translated_sqa_dataset(split: str = "train", language: str = "de"):
    """Load synthetic translated SQA dataset from local JSONL files.

    These are generated by src/data/translate_sqa.py.
    """
    from src.data.utils import get_data_dir

    jsonl_path = get_data_dir() / "sqa_translated" / f"librisqa_{language}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Translated SQA data not found at {jsonl_path}. "
            f"Run: python -m src.data.translate_sqa --languages {language}"
        )

    # Load the original LibriSQA for audio
    original_ds = load_dataset("westbrook/librisqa", split=split, trust_remote_code=True)

    # Load translations
    import json
    translations = {}
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            translations[record["id"]] = record

    def _format(example, idx):
        if idx not in translations:
            return example

        t = translations[idx]
        question = t["question"]
        answer = t["answer"]

        prompt_template = TASK_PROMPTS["sqa"][language]
        instruction = prompt_template.replace("{QUESTION}", question)

        if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
            answer = UNANSWERABLE[language]

        example["messages"] = format_chat_messages(instruction, answer)
        return example

    return original_ds.map(_format, with_indices=True)
```

Update `load_sqa_dataset` to route non-English languages:

```python
def load_sqa_dataset(split: str = "train", language: str = "en"):
    """Load the LibriSQA dataset for spoken question answering."""
    if language != "en":
        return load_translated_sqa_dataset(split, language)

    # ... existing English-only code unchanged ...
```

- [ ] **Step 2: Commit**

```bash
git add src/data/sqa.py
git commit -m "feat: support loading translated multilingual SQA datasets"
```

---

### Task 7: Precompute speech features script

**Files:**
- Create: `scripts/precompute_features.py`

- [ ] **Step 1: Create precompute_features.py**

```python
"""Precompute speech encoder features for faster training.

Usage:
    python scripts/precompute_features.py --dataset europarlst --output_dir data/features/europarlst
    python scripts/precompute_features.py --dataset librisqa --output_dir data/features/librisqa
"""

import argparse
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["europarlst", "librisqa"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SeamlessM4T v2 encoder...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    full_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    encoder = full_model.speech_encoder.to(args.device)
    encoder.eval()
    del full_model
    torch.cuda.empty_cache()

    print(f"Loading {args.dataset}...")
    if args.dataset == "europarlst":
        ds = load_dataset("facebook/europarl_st", "en", split=args.split, trust_remote_code=True)
    elif args.dataset == "librisqa":
        ds = load_dataset("westbrook/librisqa", split=args.split, trust_remote_code=True)

    print(f"Precomputing features for {len(ds)} samples...")
    for i in tqdm(range(len(ds))):
        output_path = os.path.join(args.output_dir, f"{i:08d}.pt")
        if os.path.exists(output_path):
            continue

        example = ds[i]
        audio = example.get("audio", {})
        if isinstance(audio, dict):
            waveform = audio["array"]
            sr = audio.get("sampling_rate", 16000)
        else:
            continue

        inputs = processor(
            audios=waveform, sampling_rate=sr,
            return_tensors="pt", return_attention_mask=True,
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = encoder(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            features = output.last_hidden_state.cpu()

        torch.save(features.squeeze(0), output_path)

    print(f"Done! Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/precompute_features.py
git commit -m "feat: add speech feature precomputation script"
```

---

### Task 8: Collator updates (precomputed features + text-only)

**Files:**
- Modify: `src/data/collator.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write test for text-only collation**

Add to `tests/test_smoke.py`:

```python
class TestCollator:
    def test_text_only_collation(self):
        """Collator should handle text-only samples (no audio)."""
        from src.data.collator import SpeechTextCollator
        from src.data.utils import format_text_only_messages
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

        collator = SpeechTextCollator(
            processor=None,  # No processor needed for text-only
            tokenizer=tokenizer,
            max_seq_length=512,
            text_only=True,
        )

        features = [
            {"messages": format_text_only_messages("Translate to German.", "Hallo Welt.")},
            {"messages": format_text_only_messages("Answer the question.", "Yes.")},
        ]

        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "input_features" not in batch
        assert batch["input_ids"].shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_smoke.py::TestCollator -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'text_only'`

- [ ] **Step 3: Update SpeechTextCollator**

Modify `src/data/collator.py`:

```python
@dataclass
class SpeechTextCollator:
    """Collator that handles both audio features and tokenized text.

    Supports three modes:
    - Standard: audio + text (Stage 1 and 3)
    - Text-only: no audio processing (Stage 2)
    - Precomputed: load pre-extracted speech features from .pt files
    """

    processor: object  # SeamlessM4T processor (None for text-only)
    tokenizer: object  # Qwen tokenizer
    max_seq_length: int = 4096
    text_only: bool = False

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        if not self.text_only:
            # Process audio (raw or precomputed)
            if "precomputed_features" in features[0]:
                # Precomputed mode: pad pre-extracted features
                feats = [f["precomputed_features"] for f in features]
                max_len = max(f.shape[0] for f in feats)
                dim = feats[0].shape[1]
                padded = torch.zeros(len(feats), max_len, dim)
                mask = torch.zeros(len(feats), max_len, dtype=torch.long)
                for i, f in enumerate(feats):
                    padded[i, : f.shape[0]] = f
                    mask[i, : f.shape[0]] = 1
                batch["input_features"] = padded
                batch["input_features_attention_mask"] = mask
            else:
                # Raw audio mode
                audios = [f["audio"] for f in features]
                audio_inputs = self.processor(
                    audios=audios,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                )
                batch["input_features"] = audio_inputs["input_features"]
                batch["input_features_attention_mask"] = audio_inputs["attention_mask"]

        # Tokenize text
        texts = []
        for f in features:
            text = self.tokenizer.apply_chat_template(
                f["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        batch["input_ids"] = text_inputs["input_ids"]
        batch["attention_mask"] = text_inputs["attention_mask"]

        # Create labels: mask everything except assistant response
        labels = text_inputs["input_ids"].clone()
        for i, f in enumerate(features):
            prompt_messages = [m for m in f["messages"] if m["role"] != "assistant"]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(
                prompt_text, truncation=True, max_length=self.max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100

        labels[text_inputs["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch
```

- [ ] **Step 4: Run test**

Run: `python3 -m pytest tests/test_smoke.py::TestCollator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/collator.py tests/test_smoke.py
git commit -m "feat: collator supports text-only and precomputed feature modes"
```

---

## Chunk 2: Training Pipeline & Evaluation

### Task 9: Training pipeline — 3-stage support

**Files:**
- Modify: `src/train.py`
- Test: manual smoke run (requires GPU)

- [ ] **Step 1: Update train.py for 3-stage training**

Replace the `main()` function in `src/train.py` with support for:
- `no_lora` mode (Stage 1)
- `text_only` mode (Stage 2)
- `--stage1_checkpoint` / `--stage2_checkpoint` for checkpoint loading
- `max_steps` config option
- Differential learning rates via custom optimizer

Key changes to `main()`:

```python
def main():
    parser = argparse.ArgumentParser(description="Train IWSLT instruction-following model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Load projector weights from Stage 1")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Load LoRA weights from Stage 2")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    text_only = training_config.get("text_only", False) or config.get("text_only", False)
    no_lora = training_config.get("no_lora", False)

    output_dir = args.output_dir or config.get("paths", {}).get("output_dir", "./outputs")
    task_label = config["task"]
    stage = config.get("stage", "")
    if stage:
        task_label = f"stage{stage}_{task_label}"
    output_dir = os.path.join(output_dir, task_label)

    print(f"Task: {config['task']} | Stage: {stage} | Text-only: {text_only} | No-LoRA: {no_lora}")
    print(f"Output: {output_dir}")

    # --- Text-only Stage 2: skip speech encoder, use LLM directly ---
    if text_only:
        print("=== Stage 2: Text-only LoRA training ===")
        llm_config = config.get("model", {})
        llm, tokenizer = load_llm_with_lora(
            model_name=llm_config.get("llm", "unsloth/Qwen3-4B-Instruct-2507"),
            max_seq_length=training_config.get("max_seq_length", 4096),
            load_in_4bit=training_config.get("load_in_4bit", False),
            load_in_8bit=training_config.get("load_in_8bit", True),
            no_lora=False,  # Stage 2 always uses LoRA
            lora_r=training_config.get("lora_r", 16),
            lora_alpha=training_config.get("lora_alpha", 16),
        )

        # Load Stage 2 text datasets
        from src.data.text_only import load_text_only_datasets
        from datasets import concatenate_datasets

        train_datasets = load_text_only_datasets(config)
        train_dataset = concatenate_datasets(train_datasets)
        print(f"Text-only training samples: {len(train_dataset)}")

        collator = SpeechTextCollator(
            processor=None, tokenizer=tokenizer,
            max_seq_length=training_config.get("max_seq_length", 4096),
            text_only=True,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("num_epochs", 2),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            learning_rate=float(training_config.get("learning_rate", 3e-4)),
            warmup_steps=training_config.get("warmup_steps", 100),
            logging_steps=training_config.get("logging_steps", 50),
            save_steps=training_config.get("save_steps", 500),
            bf16=training_config.get("bf16", True),
            optim=training_config.get("optim", "adamw_8bit"),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            seed=training_config.get("seed", 42),
            report_to="none",
            save_total_limit=3,
            remove_unused_columns=False,
        )

        trainer = Trainer(model=llm, args=training_args,
                          train_dataset=train_dataset, data_collator=collator)
        trainer.train()

        # Save LoRA
        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        print(f"Stage 2 LoRA saved to {output_dir}/lora_adapters")
        return

    # --- Stage 1 or 3: Speech + LLM pipeline ---
    print("Loading speech encoder...")
    speech_encoder, processor = load_speech_encoder(
        config.get("model", {}).get("sfm", "facebook/seamless-m4t-v2-large")
    )

    print("Loading LLM...")
    llm_config = config.get("model", {})
    llm, tokenizer = load_llm_with_lora(
        model_name=llm_config.get("llm", "unsloth/Qwen3-4B-Instruct-2507"),
        max_seq_length=training_config.get("max_seq_length", 4096),
        load_in_4bit=training_config.get("load_in_4bit", False),
        load_in_8bit=training_config.get("load_in_8bit", True),
        no_lora=no_lora,
        lora_r=training_config.get("lora_r", 16),
        lora_alpha=training_config.get("lora_alpha", 16),
    )

    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
        llm.resize_token_embeddings(len(tokenizer))

    print("Building projector...")
    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 3),
        projector_type=training_config.get("projector_type", "transformer"),
    )

    # Load checkpoints from previous stages
    if args.stage1_checkpoint:
        projector_path = os.path.join(args.stage1_checkpoint, "projector.pt")
        print(f"Loading Stage 1 projector from {projector_path}")
        projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    if args.stage2_checkpoint:
        from unsloth import FastLanguageModel
        lora_path = os.path.join(args.stage2_checkpoint, "lora_adapters")
        print(f"Loading Stage 2 LoRA from {lora_path}")
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=training_config.get("max_seq_length", 4096),
            load_in_4bit=training_config.get("load_in_4bit", False),
            load_in_8bit=training_config.get("load_in_8bit", True),
        )
        if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
            tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
            llm.resize_token_embeddings(len(tokenizer))

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)

    # Load data
    print("Loading datasets...")
    sampler = None

    if config["task"] == "multitask":
        from src.data.multitask import MultiTaskDataset, TemperatureSampler, load_all_task_datasets
        task_datasets = load_all_task_datasets(config)
        train_dataset = MultiTaskDataset(task_datasets)
        sampling_config = config.get("sampling", {})
        temperature = sampling_config.get("temperature", 0.5)
        sampler = TemperatureSampler(
            train_dataset, temperature=temperature,
            seed=training_config.get("seed", 42),
        )
    else:
        train_datasets = load_task_datasets(config)
        if len(train_datasets) > 1:
            from datasets import concatenate_datasets
            train_dataset = concatenate_datasets(train_datasets)
        else:
            train_dataset = train_datasets[0]

    print(f"Training samples: {len(train_dataset)}")

    collator = SpeechTextCollator(
        processor=processor, tokenizer=tokenizer,
        max_seq_length=training_config.get("max_seq_length", 4096),
    )

    # Training args — support max_steps
    max_steps = training_config.get("max_steps", -1)
    num_epochs = training_config.get("num_epochs", 3) if max_steps <= 0 else 100

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        bf16=training_config.get("bf16", True),
        optim=training_config.get("optim", "adamw_8bit"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        seed=training_config.get("seed", 42),
        report_to="none",
        save_total_limit=3,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=collator,
    )

    if sampler is not None:
        trainer = Trainer(**trainer_kwargs)
        trainer._get_train_sampler = lambda: sampler
    else:
        trainer = Trainer(**trainer_kwargs)

    # Differential LR for Stage 3
    lr_projector = training_config.get("lr_projector")
    lr_lora = training_config.get("lr_lora")
    if lr_projector and lr_lora and not no_lora:
        original_create_optimizer = trainer.create_optimizer

        def create_optimizer_with_param_groups():
            projector_params = list(model.projector.parameters())
            projector_ids = {id(p) for p in projector_params}
            other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in projector_ids]

            from torch.optim import AdamW
            optimizer = AdamW([
                {"params": projector_params, "lr": float(lr_projector)},
                {"params": other_params, "lr": float(lr_lora)},
            ])
            trainer.optimizer = optimizer
            return optimizer

        trainer.create_optimizer = create_optimizer_with_param_groups

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save
    print(f"Saving to {output_dir}")
    torch.save(projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    if not no_lora:
        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))

    hub_config = config.get("hub", {})
    if hub_config.get("push_to_hub", False) and hub_config.get("organization", "YOUR_HF_ORG") != "YOUR_HF_ORG":
        repo_id = f"{hub_config['organization']}/iwslt2026-{task_label}"
        print(f"Pushing to {repo_id}...")
        if not no_lora:
            llm.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)

    print("Done!")
```

- [ ] **Step 2: Commit**

```bash
git add src/train.py
git commit -m "feat: 3-stage training support (no_lora, text_only, checkpoint loading, differential LR)"
```

---

### Task 10: Evaluation fixes

**Files:**
- Modify: `src/evaluate.py`

- [ ] **Step 1: Fix load_checkpoint to use config params**

Update `load_checkpoint()` in `src/evaluate.py`:

```python
def load_checkpoint(checkpoint_dir: str, config: dict = None, device: str = "cuda"):
    """Load a trained SpeechLLM from a checkpoint directory."""
    from unsloth import FastLanguageModel

    config = config or {}
    training_config = config.get("training", {})

    speech_encoder, processor = load_speech_encoder(device=device)

    lora_path = os.path.join(checkpoint_dir, "lora_adapters")
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=training_config.get("max_seq_length", 4096),
        load_in_4bit=training_config.get("load_in_4bit", False),
        load_in_8bit=training_config.get("load_in_8bit", True),
    )
    FastLanguageModel.for_inference(llm)

    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 3),
        projector_type=training_config.get("projector_type", "transformer"),
        device=device,
    )
    projector_path = os.path.join(checkpoint_dir, "projector.pt")
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    projector.eval()

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)
    return model, processor, tokenizer
```

- [ ] **Step 2: Update generation params and multi-language support**

Update `run_inference_on_mcif()`:
- Change `temperature=0.7, top_p=0.8` → `temperature=0.1, repetition_penalty=1.1`

Update `main()`:
- Accept `--languages` flag (default: `en de it zh`)
- Loop over all languages and run evaluation for each
- Pass `config` to `load_checkpoint()`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "fix: evaluation uses correct projector, 8-bit, multi-language, and generation params"
```

---

### Task 11: Inference MCIF mode

**Files:**
- Modify: `src/inference.py`

- [ ] **Step 1: Add MCIF dataset mode and multi-language output**

Update `src/inference.py` to add:
- `--mcif` flag: read prompts from FBK-MT/MCIF HuggingFace dataset
- `--languages` flag: generate outputs for multiple languages in one run
- Update generation params: `temperature=0.1, repetition_penalty=1.1`

Key new function:

```python
def run_mcif_submission(model, processor, tokenizer, output_dir, languages, track="short"):
    """Generate submission XML files for all languages from MCIF dataset."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:
        print(f"\n=== Generating {track}/{lang} ===")
        config_name = f"{track}_fixedprompt"
        ds = load_dataset("FBK-MT/MCIF", config_name, split="test")
        prompt_key = f"prompt_{lang}"

        samples = []
        for example in tqdm(ds, desc=f"{lang}"):
            instruction = example[prompt_key]
            output_text = run_inference(
                model, processor, tokenizer,
                audio_path=None,  # use example["audio"] directly
                instruction=instruction,
                audio_array=example["audio"]["array"],
                max_new_tokens=512,
            )
            samples.append(OutputSample(id=example["id"], value=output_text))

        output_path = os.path.join(output_dir, f"{track}_{lang}.xml")
        write_output(samples, track=track, language=lang,
                     output_name="iwslt2026", output=output_path)
        print(f"Written: {output_path} ({len(samples)} samples)")
```

Update `run_inference()` to accept `audio_array` directly (alternative to `audio_path`).

Update generation params: `temperature=0.1, repetition_penalty=1.1`.

- [ ] **Step 2: Commit**

```bash
git add src/inference.py
git commit -m "feat: MCIF dataset mode for multi-language submission generation"
```

---

### Task 12: Update existing tests

**Files:**
- Modify: `tests/test_configs.py`, `tests/test_smoke.py`

- [ ] **Step 1: Fix existing tests broken by base.yaml changes**

The `test_speech_projector_shapes` test in `test_configs.py` still tests `SpeechProjector` — keep it for backward compatibility. The new `TransformerProjector` tests are in `test_smoke.py`.

Update `test_base_config_has_model` if needed to check new fields.

Update `TestForwardPass` tests in `test_smoke.py` to work with the updated `build_projector` signature.

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: update tests for 3-stage training pipeline changes"
```

---

## Implementation Errata

These corrections must be applied during implementation (found during plan review):

### E1: Add COMETKiwi filtering to translate_sqa.py (Task 5)
The `translate_sqa.py` script must include COMETKiwi quality filtering, not defer it. After translating, score each pair with `Unbabel/wmt22-cometkiwi-da` (reference-free) and discard pairs below 0.85. Add `--comet_threshold 0.85` argument and filter before saving JSONL.

### E2: Fix EuroParlST field name in text_only.py (Task 4)
`load_mt_europarlst()` uses `example["text"]` which is the English transcript. The translation target is in a different field (varies by language pair in `facebook/europarl_st`). Verify the actual field name from the HuggingFace dataset card and use the correct target-language field.

### E3: Fix SQA text-only loader for non-English (Task 4)
`load_sqa_text_only(split, language)` for `sqa_translated` (de/it/zh) must load from the local JSONL files generated by `translate_sqa.py`, NOT from `westbrook/librisqa`. Read from `data/sqa_translated/librisqa_{lang}.jsonl`.

### E4: Set remove_unused_columns=False globally (Task 9)
Set `remove_unused_columns=False` in `TrainingArguments` for ALL training paths (not just the sampler path), since custom fields like `precomputed_features`, `_task`, and `messages` will be stripped otherwise.

### E5: Fix text_only config location (Task 3)
Use `text_only` only under `training:` key, not at top level. Remove the top-level `text_only: true` from `stage2_text_lora.yaml`. Read from `training_config.get("text_only", False)` only.

### E6: Fix run_inference for audio_array (Task 11)
`run_inference()` must accept an optional `audio_array` parameter as alternative to `audio_path`. When `audio_array` is provided, skip `load_audio()` and process the array directly. Update the function signature and add the conditional.

### E7: Avoid double LLM loading in Stage 3 (Task 9)
When `--stage2_checkpoint` is provided, skip the initial LLM load and directly load from the LoRA checkpoint path. Restructure the loading logic to be conditional: if stage2_checkpoint is set, load LLM from there; otherwise load fresh and optionally apply LoRA.

### E8: Use load_speech_encoder from build_model.py in precompute_features.py (Task 7)
Instead of manually loading SeamlessM4Tv2Model and extracting the encoder, import and use `load_speech_encoder()` from `src/model/build_model.py` for consistency. Add `gigast` as a `--dataset` option.

### E9: Add prompt paraphrasing (not in plan, from spec line 191)
During Stage 3 data loading, randomly vary task prompts with paraphrases (e.g., "Transcribe the English content." / "Write down what is spoken in English." / "Please transcribe the audio."). Add a `TASK_PROMPT_VARIANTS` dict in `utils.py` and randomly select during collation.

### E10: mWERSegmenter check (not in plan, from spec line 265)
In `evaluate.py`, before running TRANS-COMET evaluation, check if `mwerSegmenter` is in PATH or `MWERSEGMENTER_ROOT` is set. Print a clear error message if missing with install instructions.

---

## Execution Checklist

After all tasks complete:

1. Run full test suite: `python3 -m pytest tests/ -v`
2. Run a 10-step smoke training on CPU/Mac:
   - Stage 1: `python3 src/train.py --config configs/stage1_alignment.yaml`
   - Stage 2: `python3 src/train.py --config configs/stage2_text_lora.yaml`
3. Verify configs load correctly: `python3 -c "from src.config import load_config; print(load_config('configs/stage3_multimodal.yaml'))"`
4. Verify MCIF output format: generate a dummy XML and validate structure
