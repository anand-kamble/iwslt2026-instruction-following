# IWSLT 2026 Short Track Training Strategy

**Date:** 2026-03-14
**Setting:** Constrained (Seamless M4T v2 + Qwen3-4B-Instruct-2507)
**Track:** Short (ASR, S2TT, SQA, Surprise)
**Approach:** Three-stage training (projector alignment → text LoRA → multimodal merge)

## Context

The IWSLT 2026 Instruction Following shared task requires a single model that handles multiple speech tasks based on text instructions. The short track evaluates on 15-20 second audio segments across 4 tasks: ASR (English), S2TT (EN→DE/IT/ZH), SQA (multilingual), and a surprise task.

**Evaluation (MCIF):**
- Short track EN: ASR-WER + QA-BERTScore
- Short track DE/IT/ZH: TRANS-COMET + QA-BERTScore (note: "TRANS-COMET" is the MCIF benchmark's own metric name using `Unbabel/wmt22-comet-da`; distinct from COMETKiwi used for our synthetic SQA filtering)
- One XML output file per language direction (4 files total)
- Surprise task scored via method TBD at evaluation time

**Key constraints:**
- CoVoST2 removed from constrained setting
- No multilingual training data for SQA (must create synthetic data)
- EuroParlST: en-de, en-it only. GigaST: en-de, en-zh only.
- Surprise task unknown at submission time — model must generalize via in-context learning

**Deadlines:**
- Apr 1-15: Evaluation period
- Apr 15: System submission
- Apr 24: System paper (4-8 pages, ACL format)

**Research basis:** Design informed by NAVER LABS IWSLT 2025 constrained system (same SeamlessM4T encoder, 3-stage training), DS2ST-LM, SLAM-ASR, and the MCIF benchmark paper.

## Architecture

```
Audio → Seamless M4T v2 (frozen, 1024-dim, layer 24)
     → Conv1d (stride=3, downsample)
     → 4-layer Transformer encoder (1024-dim, 8 heads, FFN 2048)
     → Linear (1024→2560)
     → Qwen3-4B (LoRA r=16, 8-bit quantized)
```

### Speech Encoder
- Model: `facebook/seamless-m4t-v2-large`
- Frozen throughout all stages
- Output: 1024-dim features at ~62.5 tokens/sec from the 24th (last) encoder layer
- Verified: `last_hidden_state` returns layer 24 output. Config confirms 24 layers.

### Projector (upgraded from 2-layer MLP)
Motivated by NAVER IWSLT 2025 (4-layer Transformer, won constrained track). DS2ST-LM showed simple projectors win for single tasks, but multi-task requires more capacity since LoRA alone can't carry all adaptation.

Architecture:
1. **Conv1d downsample**: kernel=3, stride=3. Reduces 15-20s audio from 937-1250 → 312-416 tokens. Matches NAVER's k=3 approach ("minimal quality difference vs k=2, significantly faster training").
2. **4-layer Transformer encoder**: dim=1024, 8 heads, FFN=2048. Learns contextualized speech representations before projection. ~33M trainable params.
3. **Linear projection**: 1024→2560 (Qwen3-4B hidden_size).

Total projector params: ~39M (Conv1d ~3.1M + 4-layer Transformer ~33.6M + Linear ~2.6M; vs ~5M for old 2-layer MLP).

### LLM
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- **8-bit quantization** (not 4-bit): 4-bit quantization is understudied for speech-LLMs and may limit the LLM's ability to represent speech embeddings. 8-bit ≈ ~8GB VRAM, fits easily on A100. Fall back to 4-bit only if memory is insufficient.
- LoRA: r=16, lora_alpha=16, targets q/k/v/o/gate/up/down projections (NAVER used r=8; r=16 is a middle ground. Monitor for overfitting — drop to r=8 if needed.)
- Special token `<|speech|>` marks where speech embeddings are inserted

## Synthetic Multilingual SQA Data

**Problem:** LibriSQA is English-only. MCIF evaluates SQA in DE/IT/ZH. NAVER IWSLT 2025 found zero-shot cross-lingual SQA performs poorly and explicitly created translated SQA data.

**Solution:** Use SeamlessM4T's text translation capability to translate LibriSQA questions and answers into DE, IT, ZH.

**Pipeline:**
1. Extract all question-answer pairs from LibriSQA
2. Translate questions and answers independently using SeamlessM4T text translation
3. Score translation quality with reference-free COMETKiwi (Unbabel/wmt22-cometkiwi-da) — no reference translations available, so we use quality estimation
4. Filter: keep only pairs with COMETKiwi ≥ 0.85
5. Format with target-language SQA prompts (already defined in `TASK_PROMPTS["sqa"]`)
6. Unanswerable examples: use the exact target-language unanswerable string (already in `UNANSWERABLE` dict)

**Output:** Three new SQA datasets (LibriSQA-DE, LibriSQA-IT, LibriSQA-ZH) with same audio as original LibriSQA but translated text.

**Constrained setting compliance:** NAVER did this in the constrained track and was accepted. The organizers note "Training data details will be collected at submission time to ensure a fair comparison."

## Precompute Speech Features (Setup Phase)

Since the speech encoder is frozen across all stages, precompute and cache its outputs to avoid redundant forward passes during training. This is what NAVER did and provides significant speedup.

**Pipeline:**
1. For each audio sample in EuroParlST, GigaST, and LibriSQA:
   - Run through SeamlessM4T v2 encoder
   - Save `last_hidden_state` (1024-dim features) to disk as `.pt` files
2. Modify collator to load precomputed features instead of raw audio
3. Skip speech encoder during training entirely

**Storage estimate:** ~62.5 tokens/sec × 1024 dims × 4 bytes ≈ 250KB/sec of audio. EuroParlST (~1.6K hours) + LibriSQA (~960 hours) ≈ ~640GB. If also precomputing GigaST XL (~10K hours), total is ~9TB. Start with EuroParlST + LibriSQA only; stream GigaST if storage is limited.

**Fallback:** If storage is prohibitive, skip precomputation and run the encoder online. Training will be slower but still feasible.

## Stage 1: Projector Alignment

**Goal:** Learn the speech→LLM embedding mapping via ASR transcription. Projector only — no LoRA.

| Parameter | Value |
|-----------|-------|
| Data | EuroParlST (en) + GigaST (en) for ASR (precomputed speech features) |
| Trainable | Projector only (Conv1d + Transformer + Linear, ~39M params) |
| Frozen | Speech encoder (skipped — using precomputed features), LLM (no LoRA attached) |
| Learning rate | 1e-4 |
| Epochs | 1-2 (best checkpoint may come early — NAVER's best was at 18h) |
| Batch size | 4-8 (A100) |
| max_seq_length | 4096 |
| Quantization | 8-bit |

**No LoRA in Stage 1:** `train.py` must support a `no_lora: true` config option that skips the `FastLanguageModel.get_peft_model()` call entirely. The LLM is loaded in 8-bit but not wrapped with LoRA adapters.

**Validation gate:** Run `mcif_eval -t short -l en` manually after training. ASR-WER must be < 30% (target: 15-20%) before proceeding. If it fails, debug projector alignment before continuing.

**Rationale:** Projector-only alignment is the standard approach (NAVER Stage A, SLAM-ASR). No LoRA in this stage means the LLM's multilingual and instruction-following capabilities are fully preserved.

## Stage 2: Text LoRA Pre-training

**Goal:** Teach the LLM translation and multilingual QA reasoning via text-only data (no audio). Based on NAVER IWSLT 2025 Stage B.

| Parameter | Value |
|-----------|-------|
| Data | Text-only (no audio): MT pairs from EuroParlST/GigaST + synthetic multilingual SQA |
| Trainable | LoRA only (r=16, lora_alpha=16) |
| Frozen | Speech encoder, projector |
| Learning rate | 3e-4 (matches NAVER's text LoRA LR) |
| Epochs | 1-2 |
| Batch size | 8-16 (text-only is fast, no audio processing) |
| max_seq_length | 4096 |
| Quantization | 8-bit |

**Stage 2 bypasses SpeechLLM entirely.** Since there is no audio, Stage 2 trains the LLM directly using standard HuggingFace Trainer with text chat data. No `SpeechLLM` wrapper, no speech encoder, no projector in the forward pass. This simplifies Stage 2 significantly — it's just text LoRA fine-tuning.

**Data sources (text-only, no `<|speech|>` placeholder):**
- **MT pairs:** Source EN transcript → target DE/IT/ZH translation from EuroParlST and GigaST text corpora (text side only, no audio). Formatted as chat using `TASK_PROMPTS["s2tt"]` but without the `<|speech|>` prefix — user message is just the instruction text.
- **SQA multilingual:** Translated LibriSQA (from synthetic data pipeline). Formatted as chat using `TASK_PROMPTS["sqa"]` in target language, response is translated answer.
- **SQA English:** Original LibriSQA text pairs (question→answer, no audio).

**Rationale:** NAVER found that training LoRA on text-only data first improves reasoning quality. The LLM learns translation and multilingual QA patterns without the complexity of speech embeddings. Stage 3 then merges speech modality with these text capabilities.

## Stage 3: Multimodal Merge

**Goal:** Combine speech understanding (Stage 1 projector) with text reasoning (Stage 2 LoRA) into a unified instruction-following model.

| Parameter | Value |
|-----------|-------|
| Initialize from | Stage 1 projector + Stage 2 LoRA weights |
| Data | All tasks with audio: ASR + S2TT + SQA (all languages), precomputed speech features |
| Trainable | Projector + LoRA (both fine-tuned jointly) |
| Frozen | Speech encoder (skipped — using precomputed features) |
| Learning rate | 1e-5 (projector), 1e-4 (LoRA) — via parameter groups in `Trainer.create_optimizer()` |
| Duration | **1K-2K steps** (not epochs — NAVER's Stage C was only 1K steps / ~6 hours. Brief merging is sufficient.) |
| Batch size | 4 per device, gradient_accumulation_steps=8 → effective batch=32 |
| Sampling | Temperature=0.5 + monitor per-task metrics for starvation |
| max_seq_length | 4096 |
| Quantization | 8-bit |

**Data (all with audio + `<|speech|>` placeholder):**
- ASR: EuroParlST (en) + GigaST (en)
- S2TT: EuroParlST (en-de, en-it) + GigaST (en-de, en-zh)
- SQA EN: LibriSQA with English prompts
- SQA DE/IT/ZH: LibriSQA with translated prompts and answers (same audio, multilingual text)

**Task sampling:** Temperature=0.5 upweights smaller datasets. Additionally, monitor per-task metrics during training — if ASR dominates (IT-IST IWSLT 2025 lesson), adjust task ratio caps. NAVER used explicit ratios: ASR 0.2, ST 0.4, SQA 0.4.

**S2TT data coverage:**
- EN→DE: EuroParlST + GigaST (good)
- EN→IT: EuroParlST only (less data)
- EN→ZH: GigaST only (machine-translated refs)

**Differential learning rates:** Override `Trainer.create_optimizer()` with parameter groups. Projector gets 1e-5 (preserve alignment from Stage 1), LoRA gets 1e-4 (continue adapting from Stage 2 but lower than Stage 2's 3e-4, since this is a short merge phase). Note: Unsloth patches the Trainer/optimizer internally (especially with `optim: adamw_8bit`). If the custom optimizer conflicts with Unsloth, fallback to a single LR of 1e-5 for all parameters, or use standard PEFT without Unsloth for Stage 3.

**Stage 3 is intentionally short.** NAVER found that Stages 1 and 2 do the heavy lifting. Stage 3 just fuses speech understanding with text reasoning. If results are good after 1K steps, stop. Over-training risks overwriting the alignment from Stage 1 and the text capabilities from Stage 2.

## Inference & Submission

**Per-sample flow:**
1. Load audio from MCIF test set
2. Read instruction prompt from MCIF dataset
3. Encode audio → projector (Conv1d + Transformer + Linear) → speech embeddings
4. Merge with tokenized instruction → generate response
5. Collect as OutputSample → write XML via `mcif.io.write_output()`

**Generation parameters:**
- `max_new_tokens=512`
- `temperature=0.1` or greedy decoding (deterministic, accurate)
- `repetition_penalty=1.1`

**Output:** 4 XML files: `short_en.xml`, `short_de.xml`, `short_it.xml`, `short_zh.xml`

**Surprise task:** Same pipeline — model receives a novel instruction and must follow it via in-context learning. The 3-stage approach helps preserve Qwen3's instruction-following ability: Stage 1 has no LoRA (LLM untouched), Stage 2 only improves text reasoning, Stage 3 is brief (1K-2K steps) to avoid disrupting base capabilities. To further improve generalization, vary ASR/S2TT/SQA prompts with paraphrases during Stage 3 training (e.g., "Transcribe this audio" vs "Write down what is said" — different wordings for the same task). This is free and within constrained rules.

**Chinese SQA note:** NAVER found Llama-3.1-8B struggled with coherent Chinese SQA. Qwen3-4B has native Chinese support so this is lower risk, but monitor Chinese QA-BERTScore specifically after Stage 3.

**Unanswerable SQA:** Must output exact strings per language:
- EN: "Not answerable."
- DE: "Nicht zu beantworten."
- IT: "Non è possibile rispondere."
- ZH: "无法回答。"

## Validation & Iteration

**Timeline (Mar 14 → Apr 1 = 17 days):**

| Days | Phase | Milestone |
|------|-------|-----------|
| 1-2 | Setup | Cluster setup, precompute speech features, synthetic SQA generation, 100-step smoke run |
| 3-5 | Stage 1 | Projector alignment on ASR. Gate: WER < 30% on MCIF dev |
| 6-7 | Stage 2 | Text LoRA on MT + multilingual SQA. Spot-check translation quality |
| 8-9 | Stage 3 | Multimodal merge (1K-2K steps, ~6-12 hours). Validate all 4 language directions |
| 10-14 | Iterate | Fix issues, tune hyperparams (LoRA rank, task ratios, Stage 3 duration), re-train |
| 15-17 | Submit | Generate final outputs, verify XML format, submit |

**Validation checkpoints:**
- After Stage 1: ASR-WER < 30% on MCIF dev (gate). Target 15-20%.
- After Stage 2: Spot-check text-only translation quality (no formal metric — this stage is fast).
- After Stage 3: Full `mcif_eval` for all 4 languages. Check:
  - ASR-WER shouldn't degrade much from Stage 1
  - TRANS-COMET > 0 (positive = better than baseline)
  - QA-BERTScore across all languages — compare EN vs DE/IT/ZH gap
- If cross-lingual SQA still poor after Stage 3: increase SQA task weight in temperature sampling or add more translated SQA data.

**Quantization fallback:** If ASR-WER plateaus > 25% and doesn't improve with more training, try 8-bit→16-bit or increase projector capacity. On A100 80GB, 16-bit Qwen3-4B (~8GB) + encoder (~5GB) + projector (~0.15GB) fits with room for batch_size=2.

## Codebase Changes Required

### New files:
- `scripts/precompute_features.py` — Precompute speech encoder outputs for all training audio and save as `.pt` files
- `src/data/text_only.py` — Text-only data loaders for Stage 2 (MT pairs, SQA text, no audio). Uses `TASK_PROMPTS` but formats chat messages without `<|speech|>` placeholder.
- `src/data/translate_sqa.py` — Script to generate synthetic multilingual SQA via SeamlessM4T text translation + COMETKiwi filtering
- `configs/stage1_alignment.yaml` — Projector-only ASR alignment, `no_lora: true`, downsample_factor=3
- `configs/stage2_text_lora.yaml` — Text-only LoRA pre-training, `text_only: true`, no audio
- `configs/stage3_multimodal.yaml` — Multimodal merge, max_steps=2000, gradient_accumulation_steps=8, downsample_factor=3

### Modified files:

**`src/model/adapters.py`:**
- Keep `SpeechProjector` (for backward compatibility with tests)
- Add `TransformerProjector`: Conv1d (stride=3) + 4-layer Transformer encoder (1024-dim, 8 heads, FFN 2048) + Linear (1024→2560)

**`src/model/build_model.py`:**
- Update `build_projector()` to accept `projector_type` param — construct `TransformerProjector` or `SpeechProjector`
- Support `load_in_8bit` via `BitsAndBytesConfig(load_in_8bit=True)` passed as `quantization_config` to `FastLanguageModel.from_pretrained()`. Verify Unsloth supports this; if not, use standard `AutoModelForCausalLM` with bitsandbytes directly.
- Support `no_lora: true` mode that skips `FastLanguageModel.get_peft_model()` — returns the base model without LoRA adapters

**`src/train.py`:**
- `--stage1_checkpoint` argument to load projector weights from Stage 1
- `--stage2_checkpoint` argument to load LoRA weights from Stage 2
- Support `no_lora: true` config — skip LoRA attachment for Stage 1
- Support `text_only: true` config — Stage 2 uses LLM directly (no `SpeechLLM` wrapper, no speech encoder loaded), standard HuggingFace Trainer with text chat data
- Override `Trainer.create_optimizer()` for parameter groups (projector LR vs LoRA LR) in Stage 3. Fallback: if Unsloth's internal optimizer patching conflicts, use single LR or bypass Unsloth for Stage 3.
- Support `max_steps` config (Stage 3 uses steps, not epochs)

**`src/data/collator.py`:**
- Support loading precomputed speech features (`.pt` files) instead of raw audio
- Handle text-only samples (no audio field) for Stage 2

**`src/data/sqa.py`:**
- Support loading from synthetic translated SQA datasets (LibriSQA-DE/IT/ZH) for Stage 3. Same audio as original LibriSQA but translated question/answer text.

**`src/evaluate.py`:**
- Fix `load_checkpoint`: pass `downsample_factor` and `projector_type` to `build_projector` (bug: currently defaults to factor=1 and old projector class). Also fix hardcoded `load_in_4bit=True` → respect config.
- Run all 4 language directions in one invocation
- Update generation params: temperature=0.1, repetition_penalty=1.1 (currently 0.7/0.8)
- mWERSegmenter installation check
- Parse `mcif_eval` JSON output

**`src/inference.py`:**
- MCIF dataset mode (read prompts from HuggingFace dataset)
- Generate all 4 XML files in one run
- Update generation params: temperature=0.1, repetition_penalty=1.1

**`configs/base.yaml`:**
- Keep downsample_factor=5 (default for long track). Stage configs override to 3.
- Change `load_in_4bit: false`, add `load_in_8bit: true`
- Change `lora_r: 16`, `lora_alpha: 16` (was 32)
- Add `no_lora: false` (default)
- Add `text_only: false` (default)
- Add `projector_type: transformer` (default for new architecture)
- Stage configs inherit from base via existing `load_config()` merge system

### No changes needed:
- `SpeechLLM` model architecture (embedding merge logic unchanged — used in Stage 1 and 3 only)
- Data loaders (asr.py, s2tt.py, gigast.py) — audio loading unchanged
- `MultiTaskDataset` + `TemperatureSampler` — used in Stage 3
- `mcif.io` integration

## Out of Scope

- Ensemble or model averaging
- Contrastive pre-training (KIT approach)
- Data augmentation beyond SQA translation
- Hyperparameter search
- Long track (stretch goal only after short track submission)
