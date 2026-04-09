# Stage 4: GRPO + TARS Alignment Pipeline (Revised)

**Date:** 2026-04-03
**Status:** CONSENSUS APPROVED (Architect APPROVE + Critic conditions resolved)
**Revision:** R3 (all review conditions incorporated)
**Deadline:** April 15 submission, April 24 paper

---

## RALPLAN-DR Summary (Revised)

### Principles (5)

1. **Honest complexity accounting** -- Choose the approach with lower *actual* implementation + debugging cost, not lower theoretical elegance.
2. **Minimal SpeechLLM modifications** -- Allow small, backward-compatible additions to SpeechLLM (`.config` property, remove `@inference_mode`), but no architectural changes.
3. **HF ecosystem alignment** -- Stay within Transformers + TRL + PEFT. No custom CUDA kernels or external RL libraries.
4. **Incremental testability** -- Each component (reward functions, TARS alignment, training script) must be independently testable before integration.
5. **Memory-first design** -- Every design choice must fit on H200 80GB with 4-bit Qwen3-4B + frozen encoder + projector + generation buffers.

### Decision Drivers (Top 3)

1. **Crash surface area** -- GRPOTrainer `__init__` has 7+ crash points when `model` is `nn.Module` (not `PreTrainedModel`). Each requires a workaround. The cumulative debugging cost is the primary risk of Option B.
2. **Override realism** -- The real override point is `_generate_and_score_completions` (300+ lines), not `_generate_single_turn`. Option B requires overriding this plus `_get_per_token_logps_and_entropies` plus patching `__init__`. Total override: ~600-800 lines.
3. **Time pressure** -- 12 days to submission. The approach must minimize debugging surface, not minimize lines of new code.

### Revised Option Analysis

#### Option A: Manual GRPO Training Loop (CHOSEN)

Implement a focused GRPO training script that:
- Uses SpeechLLM directly for generation and forward passes (no compatibility shims)
- Implements the core GRPO algorithm: generate N completions, compute rewards, calculate group-relative advantages, apply clipped surrogate loss
- Reuses HF Trainer for optimization infrastructure (optimizer, scheduler, gradient accumulation, checkpointing, logging)

**Realistic line count:**

| Component | Lines |
|-----------|-------|
| Speech-aware generation (encode + project + merge + generate, no `@inference_mode`) | ~80 |
| Per-token logprob computation (forward with `inputs_embeds`, extract logps) | ~60 |
| GRPO loss (advantage normalization, clipped surrogate, KL penalty) | ~80 |
| Training loop glue (batch handling, gradient accumulation, logging) | ~120 |
| Dataset adapter (wrap existing loaders for GRPO format) | ~60 |
| `main()` entry point | ~80 |
| **Total `src/train_grpo.py`** | **~480** |

**Pros:**
- Full control over speech/text dual-path. No API compatibility issues.
- SpeechLLM's `forward()` and `_merge_speech_and_text()` work as-is for logprob computation.
- Generation uses `encode_speech` + `projector` + `_merge_speech_and_text` + `llm.generate()` directly (avoids `@inference_mode` decorator on `SpeechLLM.generate()`).
- No need to work around 7+ crash points in GRPOTrainer `__init__`.
- Speech embedding caching is straightforward (compute once, reuse in logprob step).
- Checkpoint saving is explicit (save projector.pt + LoRA adapters, not entire model state).
- Debugging surface is our own code, not TRL internals.

**Cons:**
- Must implement advantage normalization and clipped surrogate loss (~80 lines, but the algorithm is well-documented).
- Must implement own wandb logging for GRPO-specific metrics.
- No benefit from future TRL improvements.

**Risk:** Medium -- the GRPO algorithm is simple (the paper is clear), and the speech integration is straightforward because we control the loop.

#### Option B: Subclass GRPOTrainer (REJECTED)

**Rejection rationale (updated with evidence from source inspection):**

The original plan estimated ~200 lines of overrides. Verified crash points in GRPOTrainer `__init__`:

1. **Line 265**: `get_config_model_id(model.config)` -- SpeechLLM has no `.config`
2. **Line 286**: `inspect.signature(model.forward)` captures SpeechLLM params, loses `logits_to_keep` optimization
3. **Line 294**: Auto-loads `processing_class` via `model.config` -- must pass explicitly
4. **Line 319**: `is_peft_model(model)` returns False -- with `beta != 0.0`, triggers `create_model_from_path(get_config_model_id(self.model.config))` at line 576 -- crash
5. **Line 725**: `model.add_model_tags(...)` -- `nn.Module` has no such method
6. **Lines 935/959/970** (Trainer.__init__): Writes to `self.model.config.eos_token_id` etc.
7. **Line 4314-4329** (Trainer._save): Falls through to `state_dict()` path, saves entire model including frozen encoder

To fix all of these requires: adding `.config` property to SpeechLLM, adding `.add_model_tags()` stub, adding `.generation_config` property, overriding `__init__` to skip/patch parent behavior, overriding `_generate_and_score_completions` (~300 lines), overriding `_get_per_token_logps_and_entropies` (~70 lines), overriding `save_model` for proper checkpoint saving. Realistic total: **600-800 lines of override/shim code**, plus we still depend on TRL internals that could shift.

At 600-800 lines of overrides vs ~480 lines of purpose-built code, Option B offers higher risk for comparable effort.

#### Option C: Use `rollout_func` Parameter (REJECTED -- unchanged)

Experimental API, only solves generation half.

---

## ADR: Architectural Decision Record (Revised)

**Decision:** Manual GRPO training loop (Option A) using HF Trainer for optimization infrastructure.

**Drivers:**
- GRPOTrainer has 7+ crash points with `nn.Module` models, requiring extensive shims
- Real override surface is 600-800 lines (not 200), eliminating the "reuse" advantage
- Manual loop gives full control over speech embedding flow, generation, and checkpoint saving
- GRPO algorithm is simple enough to implement correctly in ~80 lines of loss code

**Alternatives considered:**
- Option B (subclass GRPOTrainer): 600-800 lines of overrides + shims, high crash risk, coupled to TRL internals
- Option C (rollout_func): Experimental API, only solves half the problem

**Why chosen:** Option A has lower total risk. The GRPO algorithm is well-understood (clipped surrogate + group-relative advantages). The speech-specific integration is straightforward because we control the entire generation and forward pass flow. Total new code (~480 lines) is less than Option B's override surface (~600-800 lines), and every line is debuggable without TRL source diving.

**Consequences:**
- Positive: Full control, no TRL internal coupling, straightforward speech embedding caching, explicit checkpoint saving
- Negative: Must implement advantage normalization, clipped loss, and wandb logging ourselves. No automatic benefit from TRL improvements.
- Mitigation: The loss computation is <80 lines and directly follows the GRPO paper equations. wandb logging is ~20 lines.

**Follow-ups:**
- After submission, evaluate whether to upstream multimodal GRPO support to TRL
- If TRL adds first-class multimodal support, consider migrating

---

## Context

### Existing Architecture (unchanged)
- `SpeechLLM` (in `src/model/speech_llm.py`): Wraps frozen `speech_encoder`, trainable `projector`, and LoRA-adapted `llm`. Custom `forward()` takes `input_features` + `input_ids`, encodes speech, projects, merges embeddings via `_merge_speech_and_text`, then runs LLM. Custom `generate()` does the same but calls `llm.generate()` -- decorated with `@torch.inference_mode()`.
- `build_model.py`: Factory functions `load_speech_encoder()`, `load_llm_with_lora()`, `load_llm_from_checkpoint()`, `build_projector()`.
- `src/train.py`: HF Trainer-based training with multitask support, differential LR, per-group scheduling.
- Data pipeline: `SpeechTextCollator` handles audio processing + tokenization. Per-task loaders produce `{"audio": ..., "messages": [...]}` dicts.

### Key SpeechLLM Methods Used by GRPO

| Method | Used for | Note |
|--------|----------|------|
| `encode_speech()` | Extract speech features from audio | `@torch.no_grad()`, safe to call in any context |
| `projector()` | Project speech features to LLM space | Trainable, needs gradients for loss backward |
| `_get_embed_tokens()` | Get text embedding layer | Works with PEFT and vanilla models |
| `_merge_speech_and_text()` | Replace `<\|speech\|>` tokens with speech embeddings | Core method, handles variable-length sequences |
| `llm.generate()` | Generate completions | Use directly (NOT `SpeechLLM.generate()` which has `@inference_mode`) |
| `llm(inputs_embeds=...)` | Forward pass for logprobs | Standard HF CausalLM interface |

---

## Work Objectives

Build a Stage 4 RL post-training pipeline that:
1. Takes a Stage 3 checkpoint (projector + LoRA weights)
2. Runs GRPO training with speech-aware generation and logprob computation
3. Computes task-specific rewards (WER, COMET, BERTScore) + optional TARS alignment rewards
4. Updates projector + LoRA weights via GRPO loss
5. Fits on H200 80GB with 4-bit Qwen3-4B

---

## Guardrails

### Must Have
- Accept any Stage 3 checkpoint (projector.pt + lora_adapters/)
- All reward functions return valid floats (no silent NaN)
- Smoke test passes: load -> generate -> reward -> gradient step -> no crash
- Config-driven (YAML), consistent with existing config system
- Explicit checkpoint saving: projector.pt + lora_adapters/ (NOT full model state_dict)

### Must NOT Have
- No custom CUDA kernels
- No dependency on TRL experimental APIs
- No separate model copy for TARS text path (reuse same LLM)
- No architecture redesign of SpeechLLM (small backward-compatible additions OK)

### Relaxed Guardrails (from original plan)
- **SpeechLLM modifications allowed:** Adding `.config` property (delegating to `self.llm.config`) and removing `@torch.inference_mode()` from `generate()` (replacing with `@torch.no_grad()`) are acceptable small changes. These are backward-compatible.
- **TARS behavior alignment defaults to OFF.** It is extremely slow (generates text for every sample). Representation alignment only as default.

---

## Task Flow

```
Step 1: Small SpeechLLM Additions (src/model/speech_llm.py)
    |  (backward-compatible .config property, fix @inference_mode)
    v
Step 2: Reward Functions (src/rewards/)
    |
    v
Step 3: TARS Alignment (src/rewards/tars_alignment.py)
    |
    v
Step 4: GRPO Training Script (src/train_grpo.py)
    |  (depends on Steps 1-3)
    v
Step 5: Config + Smoke Test
    |  (depends on Step 4)
    v
Step 6: Integration Validation + Go/No-Go
```

---

## Detailed TODOs

### Step 1: SpeechLLM Backward-Compatible Additions
**Files:** `src/model/speech_llm.py`
**Complexity:** S
**Dependencies:** None

**What to change:**

1. Add a `.config` property that delegates to `self.llm.config`:
```python
@property
def config(self):
    """Delegate config to the LLM for Trainer/TRL compatibility."""
    return self.llm.config
```

2. Add `add_model_tags` stub:
```python
def add_model_tags(self, tags):
    """Stub for Trainer compatibility."""
    if hasattr(self.llm, 'add_model_tags'):
        self.llm.add_model_tags(tags)
```

3. Change `@torch.inference_mode()` to `@torch.no_grad()` on `generate()`:
   - `@torch.inference_mode()` permanently disables gradient tracking and cannot be overridden. GRPO needs to call the generation sub-steps (encode_speech, projector, merge) where projector gradients may be needed later.
   - `@torch.no_grad()` is sufficient for inference and does not break the computational graph for later backward passes.

4. Add a `generate_for_grpo()` method that does NOT use `@torch.no_grad()` on the projector path:
```python
def generate_for_grpo(self, input_features, input_features_attention_mask,
                      input_ids, attention_mask, **generate_kwargs):
    """Generate without @inference_mode. Returns (outputs, speech_embeds, speech_lengths).
    
    Speech embeddings are returned for caching (reuse in logprob computation).
    """
    with torch.no_grad():
        speech_features, speech_lengths = self.encode_speech(
            input_features, input_features_attention_mask
        )
    # Projector forward WITH gradient tracking (needed for GRPO backward)
    speech_embeds, speech_lengths = self.projector(speech_features, speech_lengths)
    speech_embeds = speech_embeds.to(self.dtype)
    
    text_embeds = self._get_embed_tokens()(input_ids)
    inputs_embeds, new_attention_mask, _ = self._merge_speech_and_text(
        speech_embeds, text_embeds, input_ids, attention_mask, labels=None,
        speech_lengths=speech_lengths,
    )
    
    with torch.no_grad():  # Generation itself doesn't need gradients
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            **generate_kwargs,
        )
    # CRITICAL: speech_embeds retains gradient graph (projector ran outside no_grad).
    # Do NOT call .detach() on speech_embeds -- projector gradients flow through
    # the logprob computation path via these cached tensors.
    return outputs, speech_embeds, speech_lengths
```

**Acceptance criteria:**
- [ ] `SpeechLLM.config` returns the LLM's config object
- [ ] `SpeechLLM.add_model_tags()` does not crash
- [ ] `SpeechLLM.generate()` still works for inference (now with `@torch.no_grad()`)
- [ ] `SpeechLLM.generate_for_grpo()` returns outputs + cached speech embeddings
- [ ] Existing `tests/test_smoke.py` still passes (backward compatible)

---

### Step 2: Task Reward Functions
**Files:** `src/rewards/__init__.py`, `src/rewards/task_rewards.py`
**Complexity:** M
**Dependencies:** None

**What to build:**

Three reward functions matching TRL's callable signature: `fn(prompts, completions, completion_ids, **kwargs) -> list[float|None]`.

Each function inspects `kwargs["_tasks"]` (note: plural, matching the dataset field name in `src/train.py` line 487) to determine which samples it applies to.

```python
def wer_reward(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """WER reward for ASR samples. Returns 1 - WER (higher is better)."""
    from jiwer import wer
    rewards = []
    for i, completion in enumerate(completions):
        task = kwargs.get("_tasks", [None])[i]
        reference = kwargs.get("references", [None])[i]
        if task != "asr" or reference is None:
            rewards.append(None)
            continue
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = max(0.0, 1.0 - wer(reference, text))
        rewards.append(score)
    return rewards

def comet_reward(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """COMET reward for S2TT samples. Batch computation."""
    # Module-level cached COMET model (lazy load)
    # Collect S2TT samples, batch-compute COMET, return scores
    # Return None for non-S2TT samples

def bertscore_reward(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """BERTScore reward for SQA/S2TSUM samples."""
    # Collect applicable samples, batch-compute BERTScore F1
    # Return None for non-applicable samples
```

**Key details:**
- Field name is `_tasks` (plural) -- consistent with `src/train.py` line 487 and the multitask dataset
- COMET model loaded lazily on first call, cached at module level
- BERTScore uses default `roberta-large` for English tasks

**Acceptance criteria:**
- [ ] `wer_reward` returns float in [0, 1] for ASR samples, `None` for others
- [ ] `comet_reward` returns float for S2TT samples, `None` for others
- [ ] `bertscore_reward` returns float for SQA/S2TSUM samples, `None` for others
- [ ] All functions use `_tasks` (plural) field name consistently
- [ ] Lazy model loading (COMET model loaded on first call, cached)

---

### Step 3: TARS Alignment Rewards
**Files:** `src/rewards/tars_alignment.py`
**Complexity:** L
**Dependencies:** Step 2 (shared patterns)

**What to build:**

Two TARS reward functions. **Representation alignment is the default; behavior alignment is OFF by default** due to extreme slowness (sequential autoregressive generation for every sample).

```python
class TARSRewardFactory:
    def __init__(self, speech_llm, tokenizer):
        self.speech_llm = speech_llm
        self.tokenizer = tokenizer

    def representation_alignment(self, prompts, completions, completion_ids, **kwargs):
        """Cosine similarity of speech-path vs text-path hidden states."""
        transcripts = kwargs.get("transcripts", [])
        rewards = []
        with torch.no_grad():
            for i, transcript in enumerate(transcripts):
                if not transcript:
                    rewards.append(None)
                    continue
                # Text-path: embed transcript, forward through LLM
                text_ids = self.tokenizer(transcript, return_tensors="pt", ...)
                text_out = self.speech_llm.llm(
                    input_ids=text_ids["input_ids"].to(device),
                    attention_mask=text_ids["attention_mask"].to(device),
                    output_hidden_states=True,
                )
                text_hidden = text_out.hidden_states[-1].mean(dim=1)

                # Speech-path: use cached speech embeddings from generation step
                speech_hidden = kwargs.get("_speech_hidden_states", [None])[i]
                if speech_hidden is None:
                    rewards.append(None)
                    continue

                cos_sim = F.cosine_similarity(text_hidden, speech_hidden, dim=-1)
                rewards.append(cos_sim.item())
        return rewards

    def behavior_alignment(self, prompts, completions, completion_ids, **kwargs):
        """BERTScore between speech-path and text-path outputs.
        
        WARNING: Extremely slow. Generates text for each sample sequentially.
        Default: OFF. Enable via config `tars.enable_behavior_alignment: true`.
        """
        # Same as before but with clear warning about speed
```

**Speech embedding caching strategy (addresses review issue #4):**

During generation in the GRPO loop, after encoding speech and projecting:
1. Compute `speech_embeds` via `encode_speech()` + `projector()`
2. Run a short forward pass through the LLM with the prompt (speech + instruction) to get hidden states
3. Store the mean hidden state per sample in `_speech_hidden_states` list
4. Pass this list through to reward functions via `kwargs`

This avoids needing to re-encode speech in the reward function. The hidden states are small tensors (one (1, D) per sample) and survive batch splitting trivially.

**Acceptance criteria:**
- [ ] `representation_alignment` computes cosine similarity between text-path and speech-path hidden states
- [ ] `behavior_alignment` is disabled by default (config flag to enable)
- [ ] Both functions handle missing transcripts gracefully (return `None`)
- [ ] Both functions use `torch.no_grad()` for their internal forward passes
- [ ] Text-path bypasses speech encoder/projector (uses `speech_llm.llm` directly)

---

### Step 4: GRPO Training Script + Config
**Files:** `src/train_grpo.py`, `configs/stage4_grpo.yaml`
**Complexity:** L
**Dependencies:** Steps 1-3

**What to build:**

A self-contained GRPO training script that uses HF Trainer's optimization infrastructure but implements the GRPO-specific generation/scoring/loss manually.

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│  GRPOSpeechTrainer (subclass of HF Trainer)     │
│                                                  │
│  Overrides:                                      │
│  - compute_loss() → GRPO loss computation        │
│  - _prepare_inputs() → generation + scoring      │
│  - save_model() → projector.pt + LoRA            │
│                                                  │
│  Inherits from HF Trainer:                       │
│  - Optimizer creation + stepping                 │
│  - LR scheduling                                 │
│  - Gradient accumulation                         │
│  - Checkpointing cadence (save_steps)            │
│  - wandb integration                             │
│  - Distributed training (if needed)              │
└─────────────────────────────────────────────────┘
```

**Key insight:** We subclass HF `Trainer` (NOT TRL `GRPOTrainer`). HF Trainer is designed for `nn.Module` models. Its `_save` method (trainer.py line 4314-4329) falls through to `state_dict()` for non-PreTrainedModel, which we override to save just projector + LoRA. This avoids all 7 crash points in GRPOTrainer while still getting optimizer, scheduler, gradient accumulation, and logging for free.

**Core components in `src/train_grpo.py`:**

#### 4a. `GRPOSpeechDataset`
Wraps existing task datasets to produce samples in GRPO format:
```python
{
    "prompt": [{"role": "user", "content": "<|speech|>\nTranscribe..."}],
    "audio": <array>,
    "sampling_rate": 16000,
    "_tasks": "asr",
    "reference": "Hello world.",
    "transcript": "Hello world.",
}
```

#### 4b. `speech_grpo_generate()`
Generate N completions per prompt using SpeechLLM components directly:

```python
def speech_grpo_generate(speech_llm, processor, batch, num_generations, generation_config):
    """Generate completions and cache speech embeddings.
    
    Returns:
        prompt_ids: list[list[int]] per sample
        completion_ids: list[list[int]] per sample (N per prompt)
        speech_embeds_cache: dict mapping sample_idx -> speech_embeds tensor
    """
    # 1. Process audio through Seamless processor
    # 2. Encode speech (frozen encoder, no_grad)
    # 3. Project to LLM space (projector, WITH gradients detached for generation)
    # 4. Cache projected speech embeddings for reuse in logprob step
    #    CRITICAL: do NOT detach -- projector gradients flow through logprob path
    # 5. (If TARS rep. alignment enabled) Forward prompt through LLM with
    #    output_hidden_states=True, extract hidden_states[-1].mean(dim=1),
    #    store in speech_hidden_states_cache for TARS reward kwargs
    # 6. For each prompt, repeat N times and generate via llm.generate()
    # 7. Split outputs into prompt_ids + completion_ids
```

Key: Use `speech_llm.encode_speech()` + `speech_llm.projector()` + `speech_llm._merge_speech_and_text()` + `speech_llm.llm.generate()` directly. This bypasses the `@torch.inference_mode()` decorator on `SpeechLLM.generate()`.

#### 4c. `compute_per_token_logps()`
Compute per-token log probabilities with speech embeddings:

```python
def compute_per_token_logps(speech_llm, prompt_completion_ids, attention_mask,
                            speech_embeds_cache, logits_to_keep):
    """Forward pass with inputs_embeds for logprob computation.
    
    Uses cached speech_embeds from generation step.
    CRITICAL: speech_embeds must be the SAME tensor objects from generate_for_grpo,
    NOT detached or cloned. Projector gradients flow through this path.
    Replaces <|speech|> token embeddings with cached speech embeddings.
    """
    # 1. Get text embeddings via speech_llm._get_embed_tokens()
    # 2. Replace speech placeholder region with cached speech_embeds
    #    (reuse speech_llm._merge_speech_and_text logic)
    # 3. Forward through speech_llm.llm(inputs_embeds=..., attention_mask=...)
    # 4. Extract logits for completion tokens only (logits_to_keep optimization)
    # 5. Compute per-token log probs via log_softmax + gather
```

#### 4d. `grpo_loss()`
Core GRPO loss computation (~80 lines):

```python
def grpo_loss(per_token_logps, old_per_token_logps, advantages, completion_mask,
              epsilon=0.2, beta=0.0, ref_per_token_logps=None):
    """GRPO clipped surrogate loss with optional KL penalty.
    
    Implements equations from DeepSeekMath paper:
    L = -E[min(r*A, clip(r, 1-eps, 1+eps)*A)] + beta * KL
    where r = exp(logp - old_logp), A = group-relative advantages
    """
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    per_token_loss = -torch.min(ratio * advantages.unsqueeze(1),
                                clipped_ratio * advantages.unsqueeze(1))
    
    if beta > 0.0 and ref_per_token_logps is not None:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
                       - (ref_per_token_logps - per_token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl
    
    loss = ((per_token_loss * completion_mask).sum(-1)
            / completion_mask.sum(-1).clamp(min=1.0)).mean()
    return loss
```

#### 4e. `GRPOSpeechTrainer`
Subclass HF Trainer with GRPO-specific logic:

```python
class GRPOSpeechTrainer(Trainer):
    def __init__(self, model, reward_funcs, grpo_config, processor, ...):
        # Store GRPO-specific config (num_generations, epsilon, beta, etc.)
        # Store reward functions list
        # Store processor for audio processing
        # Call super().__init__(...) -- works fine with nn.Module
        
    def _prepare_inputs(self, inputs):
        """Override: generate completions, compute rewards, compute advantages."""
        # 1. Call speech_grpo_generate() -> completions + cached speech embeds
        # 2. Decode completions
        # 3. Call each reward function
        # 4. Compute group-relative advantages (normalize per prompt group)
        # 5. Compute old_per_token_logps (for importance sampling)
        # 6. Store everything in a dict for compute_loss
        # 7. Buffer for gradient accumulation steps
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override: compute GRPO loss instead of standard CE loss."""
        # 1. Compute per_token_logps via compute_per_token_logps()
        # 2. Call grpo_loss() with advantages from inputs
        # 3. Log metrics to wandb
        # 4. Return loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override: save projector + LoRA only."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.projector.state_dict(),
                   os.path.join(output_dir, "projector.pt"))
        self.model.llm.save_pretrained(
            os.path.join(output_dir, "lora_adapters"))
        self.model.tokenizer.save_pretrained(
            os.path.join(output_dir, "lora_adapters"))
```

#### 4f. `main()` Entry Point

```python
def main():
    # 1. Parse args (--config, --stage3_checkpoint)
    # 2. Load config
    # 3. Build model (same as src/train.py: load_speech_encoder, load_llm_from_checkpoint, build_projector, SpeechLLM)
    # 4. Build GRPOSpeechDataset (wrapping multitask loaders)
    # 5. Build reward functions list
    # 6. Build GRPOSpeechTrainer
    # 7. trainer.train()
```

#### 4g. Config YAML

```yaml
defaults:
  - base

task: multitask
track: short
stage: 4

tasks:
  asr:
    train:
      - europarlst
  s2tt:
    train:
      - europarlst
    target_languages:
      - de
      - it
  sqa:
    train:
      - librisqa
    languages:
      - en
  s2tsum:
    train:
      - nutshell         # dataset spec added
    languages:
      - en
  achap:
    train:
      - ytseg            # dataset spec added
    languages:
      - en
    use_audio: true

sampling:
  ratios:
    asr: 0.20
    s2tt_de: 0.15
    s2tt_it: 0.15
    sqa_en: 0.25
    s2tsum_en: 0.10
    achap_en: 0.15

grpo:
  num_generations: 4
  max_completion_length: 256
  temperature: 0.8
  epsilon: 0.2                # clipping parameter
  beta: 0.0                   # KL penalty OFF (no ref model needed)
                              # was 0.01, but with PEFT model this requires
                              # creating a "ref" adapter copy -- use 0.0 to avoid
  loss_type: grpo
  scale_rewards: group

  reward_weights:
    wer: 1.0
    comet: 1.0
    bertscore: 1.0
    tars_representation: 0.3

tars:
  enable_representation_alignment: true
  enable_behavior_alignment: false    # OFF by default (too slow)
  lambda_rep: 0.3
  lambda_beh: 0.3                    # only used if behavior_alignment enabled

training:
  downsample_factor: 3
  max_steps: 500
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-7
  lr_projector: 1e-6
  lr_lora: 5e-7
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  gradient_checkpointing: true
  save_steps: 100
  logging_steps: 5
  load_in_4bit: true
```

**Acceptance criteria:**
- [ ] `GRPOSpeechTrainer` generates completions using audio input
- [ ] Per-token logprobs computed correctly with cached speech embeddings
- [ ] GRPO loss computes correct clipped surrogate objective
- [ ] Advantage normalization is per-group (per-prompt, across N generations)
- [ ] `save_model()` saves projector.pt + lora_adapters/ (NOT full state_dict)
- [ ] wandb logging shows: loss, reward/mean, reward/std, per-function rewards, advantages
- [ ] `main()` loads Stage 3 checkpoint and starts training without error
- [ ] Config YAML loads via `load_config()` without error
- [ ] `beta: 0.0` -- no ref model created

---

### Step 5: Smoke Test
**Files:** `tests/test_grpo_smoke.py`
**Complexity:** M
**Dependencies:** Steps 1-4

**What to build:**

End-to-end smoke test using a tiny LLM (Qwen3-0.6B) on CPU.

```python
class TestGRPOSmoke:
    def test_speechllm_config_property(self):
        """SpeechLLM.config delegates to llm.config."""

    def test_speechllm_generate_for_grpo(self):
        """generate_for_grpo returns outputs + cached speech_embeds."""

    def test_reward_functions_signature(self):
        """Each reward function returns list[float|None] of correct length."""
        # Test with _tasks (plural) field

    def test_tars_representation_alignment(self):
        """TARS representation alignment returns cosine similarity."""

    def test_grpo_loss_computation(self):
        """GRPO loss computes correct clipped surrogate objective."""
        # Synthetic logps + advantages, verify loss is finite and correct sign

    def test_speech_grpo_generate(self):
        """speech_grpo_generate produces completions + cached embeds."""

    def test_end_to_end_step(self, real_speech_encoder, real_processor):
        """Full loop: generate -> reward -> gradient step -> no crash."""
```

**Acceptance criteria:**
- [ ] SpeechLLM property tests pass
- [ ] Reward function tests pass (with `_tasks` plural)
- [ ] GRPO loss unit test passes
- [ ] End-to-end: generate -> reward -> loss -> backward -> no crash
- [ ] No GPU required (CPU with tiny model)

---

### Step 6: Integration Validation + Go/No-Go
**Files:** None (manual validation on GPU)
**Complexity:** S
**Dependencies:** Steps 1-5

**Go/No-Go criteria (2-day spike):**

| Checkpoint | Criterion | Day |
|------------|-----------|-----|
| Day 1 AM | SpeechLLM additions pass existing tests | 1 |
| Day 1 PM | Reward functions work with synthetic data | 1 |
| Day 1 EOD | `GRPOSpeechTrainer` initializes, generates 1 batch, computes loss | 1 |
| Day 2 AM | End-to-end: 10 GRPO steps complete without OOM on H200 | 2 |
| Day 2 PM | wandb logs show reasonable reward progression | 2 |
| **GO** | All above pass | 2 |
| Day 1 PM+ | Reward functions execute on 1 batch in <10 seconds (with cached models) | 1 |
| **NO-GO** | Any of: OOM, loss divergence, reward computation >30sec/step | 2 |

**Rollback plan (if NO-GO):**
- Fall back to supervised fine-tuning only (no RL). Stage 3 checkpoint is the submission model.
- Time saved: remaining 10 days for evaluation, ablations, and paper writing.

**Validation checklist:**
- [ ] `python src/train_grpo.py --config configs/stage4_grpo.yaml --stage3_checkpoint outputs/stage3` starts without error
- [ ] wandb logs show: loss, reward/mean, per-function rewards
- [ ] Memory usage on H200 stays under 65GB (leaving headroom)
- [ ] Checkpoint saves projector.pt + lora_adapters/ in correct format
- [ ] Saved checkpoint is loadable by existing inference pipeline
- [ ] Single training step completes in <60 seconds

---

## Memory Budget Estimate (H200 80GB) -- REVISED

| Component | Estimate |
|---|---|
| Seamless M4T v2 encoder (fp16, frozen) | ~2.5 GB |
| Qwen3-4B (4-bit NF4) | ~2.5 GB |
| LoRA adapters (r=8, bf16) | ~0.05 GB |
| Projector (Transformer, fp32) | ~0.15 GB |
| Generation buffer (4 completions x 256 tokens x bf16) | ~4 GB |
| Optimizer states (projector + LoRA, AdamW: 2x param size) | ~1 GB |
| Gradient checkpointing activation memory | ~12 GB |
| Speech embedding cache (per batch, small) | ~0.1 GB |
| TARS text-path forward pass (temp, no_grad) | ~4 GB |
| Reward model caches (COMET ~2GB, BERTScore ~1.3GB) | ~3.5 GB |
| KV cache during generation | ~4 GB |
| Padding overhead (variable-length sequences) | ~3 GB |
| **Total estimated** | **~37.5 GB** |
| **Headroom** | **~42.5 GB** |

Revised upward from 24GB to ~37.5GB based on reviewer feedback. Still comfortable on H200 80GB. Could increase `num_generations` to 8 if quality demands it (adds ~4GB for generation buffers).

---

## Speech Embedding Caching Strategy (addresses review issue #4)

**Problem:** Between generation and logprob computation, the speech embeddings must be available. GRPOTrainer's `_buffered_inputs` / `split_tensor_dict` would split batches and lose the association. In our manual loop, we control this directly.

**Solution:**

1. During generation: compute speech embeddings once per prompt (not per generation).
   ```python
   speech_features, speech_lengths = speech_llm.encode_speech(input_features, mask)
   speech_embeds, speech_lengths = speech_llm.projector(speech_features, speech_lengths)
   # Cache: {prompt_idx: (speech_embeds[i], speech_lengths[i])}
   ```

2. For N generations per prompt: reuse the same speech embeddings (repeat N times).

3. For logprob computation: retrieve cached speech embeddings by prompt index.
   ```python
   # prompt_idx = sample_idx // num_generations
   cached_embeds = speech_embeds_cache[prompt_idx]
   ```

4. For TARS reward: extract hidden states from the prompt-only forward pass and pass via `_speech_hidden_states` in reward kwargs.

5. Cache lifetime: one generation batch. Cleared when new batch is generated.

This is trivial in the manual loop because we control the entire flow. In the GRPOTrainer subclass approach, this would require hooking into `_buffered_inputs` splitting logic.

---

## File Summary

| File | Action | Lines (est.) |
|---|---|---|
| `src/model/speech_llm.py` | MODIFY | +40 (config prop, add_model_tags, generate_for_grpo) |
| `src/rewards/__init__.py` | CREATE | 15 |
| `src/rewards/task_rewards.py` | CREATE | 150 |
| `src/rewards/tars_alignment.py` | CREATE | 180 |
| `src/train_grpo.py` | CREATE | 480 |
| `configs/stage4_grpo.yaml` | CREATE | 65 |
| `tests/test_grpo_smoke.py` | CREATE | 200 |
| **Total** | | **~1130 lines** |

**Comparison with original plan:** +155 lines total, but:
- No TRL internal dependency
- No crash debugging
- Explicit checkpoint saving
- Full control over speech embedding flow

---

## Open Questions

See `.omc/plans/open-questions.md` for tracked items.
