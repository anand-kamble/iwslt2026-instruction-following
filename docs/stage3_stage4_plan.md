# Stage 3 + Stage 4 Plan — IWSLT 2026 Instruction Following

**Status:** living document. Last updated 2026-04-17.

**Paper target:** IWSLT 2026 system paper, deadline Friday 2026-04-24.

**Team:** Aniket Tathe et al.

---

## 0. Goals

Extend the NAVER 2025 3-stage recipe (arXiv:2506.01808) with two additions that target the 2026 track:

1. **Stage 3 (surprise-aware SFT):** Add a synthetic curriculum of novel speech-instruction tasks + an in-context 1-shot format layer, so the model is prepared for the held-out Surprisal task and for the 2026-added S2TSUM/ACHAP tasks.
2. **Stage 4 (GRPO post-hoc):** After SFT, run a short GRPO stage targeting residual failure modes (repetition, instruction drift) and/or direct optimization of the generalization signal.

Both additions are hedges for the **Surprisal** task, which is unknown at submission time, en→de / en→zh only, and explicitly solvable via in-context / 1-shot learning (per organizer rules).

---

## 1. Base system (unchanged)

- **SFM:** `facebook/seamless-m4t-v2-large` (frozen, 1024-dim, ~62.5 tok/s).
- **LLM:** `Qwen/Qwen3-4B-Instruct-2507` (LoRA, r=8 / r=16 / r=32 depending on Stage 2 variant).
- **Projector:** 4-layer Transformer encoder, 1024→2560, 3× frame-averaging downsample.
- **Stages 1 & 2:** already trained. Checkpoints on disk:
  - `outputs/stage1_alignment/` (projector).
  - `outputs/stage2_text_lora_v3/` (LoRA r=32, α=64, lr 2e-4, 1 epoch — preferred starting point for Stage 3).

---

## 2. Generator models — committed

Both Gemma-family models, split by modality:

| Generator | Role | HF repo | Params | Modality | Used for |
|---|---|---|---|---|---|
| **Gemma-4-31B** | Text-content annotator | `google/gemma-4-31B` (verify URL before download) | ~31B (dense) | text only | T1–T7 |
| **Gemma-4-E4B-it** | Audio-aware annotator | `google/gemma-4-E4B-it` | ~8B effective | text + image + audio | T8–T10 |

Diversity within a single generator is sourced from **2 prompt variants per task** (50/50 split at generation time) + sampling (temperature 0.7 / 0.9 alternated).

**Risks to verify before committing compute:**
- `google/gemma-4-31B` URL exists and loads — confirm in browser.
- vLLM support for `Gemma4ForConditionalGeneration` — if not yet registered in vLLM, fall back to `transformers` + `accelerate` (slower; ~60–70% of vLLM throughput).
- 31B bf16 (~62 GB) on A100 80GB is tight — plan for **TP=2** across two A100s per instance, run 2 instances in parallel across 4 A100s.

---

## 3. Final synthetic task list (10 tasks)

Task IDs locked. Ratios, sample counts, generator ownership:

### Text-content tasks (generator: Gemma-4-31B, transcript input)

| ID | Task | Input format | Output format | Pool | Active in training |
|---|---|---|---|---|---|
| **T1** | Keyword extraction | EN transcript | Comma-separated list, 5–7 items, 1–3 words each | 5k | 5k |
| **T2** | Named entity extraction | EN transcript | JSON `{"PER":[…],"ORG":[…],"LOC":[…]}` | 5k | 5k |
| **T3** | One-line gist (EN) | EN transcript | Single EN sentence | 5k | 5k |
| **T4** | Topic / domain label | EN transcript | Short free-form phrase (1–3 words) | 5k | 5k |
| **T5** | Numeric / quantitative answer | EN transcript + auto-generated question | Number + unit | 5k | 5k |
| **T6** | Cross-lingual one-liner (DE) | EN transcript | Single DE sentence | 5k | 5k |
| **T7** | Cross-lingual one-liner (ZH) | EN transcript | Single ZH sentence | 5k | 5k |

### Audio-aware tasks (generator: Gemma-4-E4B-it, audio input)

| ID | Task | Input format | Output format | Pool | Active |
|---|---|---|---|---|---|
| **T8** | Speaker tone / style description (EN) | Audio (≤20 s) | 1–2 EN sentences describing how the speaker sounds (tone, emphasis, pace) | 5k | 5k |
| **T9** | Cross-lingual tone description (DE) | Audio (≤20 s) | 1–2 DE sentences describing how the speaker sounds | 5k | 5k |
| **T10** | Cross-lingual tone description (ZH) | Audio (≤20 s) | 1–2 ZH sentences | 5k | 5k |

**Dropped:**
- Register classification (T6-original): weak signal, mostly text-derivable, now replaced by audio-aware T8.
- Sentiment (Tier 2 originally): transcript-only sentiment is noisy supervision.
- Speaker count, prosody emotion, speech style: pure-acoustic, unreliable from text generators.

**Rationale for the 7-text / 3-audio split:**
- Text tasks cover extractive (T1, T2), compressive (T3, T4), grounded (T5), and cross-lingual generative (T6, T7) — 4 granularities of content understanding.
- Audio tasks give the SpeechLLM training signal for answers that *require* listening to the audio (not just reading the transcript), hedging a possibly-acoustic Surprisal task.
- T6–T7 and T9–T10 directly mirror the Surprisal language pairs (en→de, en→zh).

---

## 4. Generation pipeline

### Audio / transcript pool

Sample from training data of:
- **CoVoST2** (en audio + transcript)
- **EuroParlST** (en audio + transcript)
- **GigaST** (en audio + transcript, new in 2026 constrained)

Draw ~10k unique (audio, transcript) pairs uniformly across the three sources. Each pair is used:
- Text tasks (T1–T7): transcript sent to Gemma-4-31B — one (audio, transcript) pair supports up to 7 text-task training examples.
- Audio tasks (T8–T10): audio sent to Gemma-4-E4B-it — one audio clip supports up to 3 audio-task examples.

Total LLM calls = 7 × 5k + 3 × 5k = 50k across both generators.

### Per-record JSONL schema

```json
{
  "id": "synth_T1_000042",
  "task": "T1_keywords",
  "source_model": "gemma-4-31B",
  "prompt_variant": "A",
  "decoding": {"temperature": 0.7, "top_p": 0.9, "seed": 42},
  "audio_path": "data/covost2/common_voice_en_12345.mp3",
  "transcript": "Full English transcript used as input…",
  "instruction_en": "List 5–7 keywords…",
  "instruction_paraphrases": ["…", "…"],
  "answer": "climate change, IPCC, …",
  "generation_timestamp": "2026-04-18T02:14:03Z",
  "audit": {"rejected": false, "filter_reasons": []}
}
```

Stored under `data/synthetic/<task_id>/<source_model>/part-*.jsonl`.

### Per-task generator prompts

Two prompt variants (A/B) per task, each tagged `prompt_variant` in the JSONL. Prompts to be drafted in the next work cycle and saved under:

```
prompts/synthetic/
  T1_keywords_gemma_A.txt
  T1_keywords_gemma_B.txt
  T2_ner_gemma_A.txt
  ...
  T8_tone_gemma4e4b_A.txt
  T8_tone_gemma4e4b_B.txt
  ...
```

Each prompt contains:
- **System/context preamble** (role description, dataset context).
- **Output contract** (format spec, rejection rules).
- **2–3 few-shot demonstrations** (frozen, hand-audited).
- **Slot markers** for transcript or audio reference.

Decoding defaults (per task class):
- Short outputs (T1, T2, T4, T6): `temperature=0.7, top_p=0.9, max_new_tokens=120`, alternate 50% with `temperature=0.9`.
- Medium outputs (T3, T5, T7, T8, T9, T10): `temperature=0.7, top_p=0.9, max_new_tokens=200`.

### Post-generation filters (per task)

Hard rejects (drop the sample, keep the count):
- Contains `<think>`, `I think`, `Let me reason`, or reasoning prefix (mostly a Qwen-Claude concern but keep the filter for Gemma too).
- Output language mismatch (for T6/T7/T9/T10): run fastText `lid.176.bin` on the answer; reject if top language ≠ target.
- Empty output or refusal tokens ("I cannot", "As an AI…").
- Output shorter than 3 tokens (for any task) or longer than 3× expected length.
- Non-parseable JSON for T2.
- T5 answer missing a numeral.

Soft normalizations (keep the sample, record in `audit`):
- Strip leading/trailing whitespace.
- Collapse whitespace runs.
- Remove trailing backticks / quotes.
- Lowercase keywords unless proper-noun heuristic triggers.

### Generator orchestration — 2× H200 + 2× A100, two parallel scripts

Two independent generator processes launched simultaneously. No phase ordering.

| GPUs | Script | Model | Layout | Workload |
|---|---|---|---|---|
| H200 #1 + #2 (141 GB each) | `scripts/gen_text.sh` | Gemma-4-31B | 2 DP replicas (one per GPU) | T1–T7, 35k samples |
| A100 #1 + #2 (80 GB each) | `scripts/gen_audio.sh` | Gemma-4-E4B-it | 2 DP replicas (one per GPU) | T8–T10, 15k samples |

**DP, not TP.** Both models fit per-GPU with room to spare (31B bf16 ≈ 62 GB in 141 GB H200; 8B bf16 ≈ 16 GB in 80 GB A100), so TP would only add communication overhead. DP replication scales linearly.

Estimated per-replica throughput (vLLM, bf16, short outputs):
- Gemma-4-31B on H200: ~4–6k samples/hr (H200 is ~2–3× A100 for inference).
- Gemma-4-E4B-it on A100: ~3–5k samples/hr.

Estimated wall times (2 replicas each, parallel):
- Text pool: 35k / ~10k/hr = **~3–4 hrs**.
- Audio pool: 15k / ~8k/hr = **~2 hrs**.
- Overall wall time ≈ **~3–4 hrs**.

Audio finishes first by ~1–2 hrs. Those A100s free up for feature precompute or paraphrased-MCIF dev generation.

vLLM flags (31B on H200 has plenty of KV-cache headroom so less brittle than on A100):
- `--gpu-memory-utilization 0.9`
- `--max-model-len 4096`
- `--max-num-seqs 64` (can go higher on H200)

Edge-case fallback if smoke reveals a ceiling: switch the text side to TP=2 across both H200s (one vLLM instance, larger batch). Only needed if a single-GPU replica can't push batch past ~64, which is unlikely for 31B short-output workloads.

### Throughput smoke test (required before full gen)

Before committing 9 hrs of compute, run a 500-sample smoke on each generator instance to measure real tokens/sec and abort if more than 2× slower than estimated. Script: `scripts/benchmark_gen_throughput.py` (to be written).

---

## 5. Stage 3 training plan

### Task ratios (Layer 1 — task sampler)

Sum ≈ 1.00.

| Bucket | Ratio | Notes |
|---|---|---|
| ASR (en) | 0.15 | down from NAVER's 0.20 to free space for synthetic |
| S2TT-de | 0.08 | |
| S2TT-it | 0.08 | |
| S2TT-zh | 0.08 | |
| SQA-en | 0.08 | |
| SQA-de | 0.04 | |
| SQA-it | 0.04 | |
| SQA-zh | 0.04 | |
| S2TSUM-en | 0.08 | |
| ACHAP-en | 0.10 | |
| synth_T1 … synth_T10 | 0.023 each (10 × 0.023 = 0.23) | uniform across synthetic tasks |
| **Total** | **~1.00** | |

Synthetic aggregate share = 23% of Stage 3 sample views.

### ICL overlay (Layer 2 — collator coin flip)

For each sampled **short-audio** example (ASR / S2TT / SQA / synthetic), with probability **p = 0.25** reformat as 1-shot:

```
<audio_exemplar> → <answer_exemplar>    ← demonstration, same task
<audio_target> + instruction → <answer_target>  ← to learn
```

Exemplar constraints:
- Drawn from the same task (not cross-task).
- Different audio from target.
- Exemplar audio duration ≤ 15 s (cap; resample if longer — skip that exemplar and redraw).

S2TSUM (~6 min audio) and ACHAP (~6 min audio) are **always zero-shot** (OOM risk doubles with exemplar). Audio tasks T8–T10 use ≤20 s clips — they participate in ICL.

### Training hyperparameters

Start from `configs/stage3_multimodal.yaml` and overlay `configs/stage3_surpriseaware.yaml` (new, to be written). Key values:

- **Starting weights:**
  - Projector: `outputs/stage1_alignment/projector.pt`
  - LoRA: `outputs/stage2_text_lora_v3/lora_adapters/` (r=32, α=64)
- **Epochs:** 1 (explicit — user requirement)
  - With combined pool ~1.2M samples and batch 256 → **~4700 steps**
  - At ~5–8 s/step on 2× H200 DDP (or 2× A100 DDP), wall time ~6–10 hrs
  - Save every 500 steps; use the best checkpoint at eval
- **Learning rate:** LoRA 3e-4, projector 1e-5 (constant projector schedule, cosine LoRA)
- **Warmup:** 3% of total steps
- **Weight decay:** 0.1
- **Precision:** bf16 on H200/A100
- **Gradient checkpointing:** enabled
- **Max seq length:** 4096 (may need to rise for 1-shot ICL batches — monitor OOM)

### Text interleave (keep NAVER-style pairing)

- `s2tt_de → mt_de`, `s2tt_it → mt_it`, `s2tt_zh → mt_zh`
- `sqa_en → qa_en`, `sqa_de → qa_de`, `sqa_it → qa_it`, `sqa_zh → qa_zh`
- Synthetic tasks: **no text interleave partner** (they have no textual equivalent; they are the novel signal themselves).

### Precompute

Run `scripts/precompute_features.py` for S2TSUM (NUTSHELL) and ACHAP (YTSeg) audio before Stage 3. Cuts per-step time substantially by removing live encoder forward. Estimated 4–6 hrs on 1 GPU, parallel with generation.

---

## 6. Paraphrased-MCIF dev harness (for novelty evaluation)

To measure whether the synthetic curriculum + ICL overlay actually helps generalization:

1. Take MCIF dev set (the `IWSLT2025` branch of `FBK-MT/MCIF` — IWSLT 2025 IF test set repurposed as 2026 dev).
2. For each sample's instruction prompt, generate **4 paraphrases** using Qwen3-4B (text-only, offline). Preserve intent, vary surface form.
3. Save as `data/paraphrased_mcif_dev.jsonl` with original and paraphrased prompts side-by-side.
4. At eval, score the Stage 3 novel checkpoint and Stage 3 baseline on:
   - Standard MCIF dev (original prompts) — replicates published eval
   - Paraphrased MCIF dev (reworded prompts, same audio) — measures novelty
5. Paper table: per-task, per-language, per-prompt-type score. The *gap* (baseline drops more on paraphrased than novel does) is the novelty claim.

Script: `scripts/build_paraphrased_mcif.py` (to be written).

---

## 7. Stage 3 ablation plan (run two models in parallel)

With 4× A100 available and Stage 3 training ~6–10 hrs, run two variants side-by-side:

| Variant | Synthetic tasks | ICL overlay | Steps | Purpose |
|---|---|---|---|---|
| **V-base (NAVER-replica)** | off | off (zero-shot only) | 1000 (NAVER's optimal) | Published-baseline reference |
| **V-novel (surprise-aware)** | on (T1–T10, 23%) | on (p=0.25 short-audio) | ~4700 (1 epoch) | Novelty claim |

V-base runs faster because it skips ICL, skips synthetic samples, and uses 1K steps — completes in ~2–3 hrs on 2× A100. V-novel is the full 1-epoch run on 2× H200 or 2× A100 DDP.

Evaluation matrix (per variant):

|  | MCIF dev (standard prompts) | Paraphrased MCIF dev | Paper column |
|---|---|---|---|
| V-base | ✓ | ✓ | "baseline" |
| V-novel | ✓ | ✓ | "+ synthetic + ICL" |

---

## 8. Stage 4 — GRPO plan

### Section status: PARTIAL. To be finalized after Cameron Wolfe's *GRPO Tricks* post is digested.

### Principles (locked)

- **Start from V-novel.** Reference policy = V-novel SFT checkpoint (NOT V-base).
- **LoRA-only update.** Projector frozen. LoRA adapter trainable.
- **KL-to-reference** regularization against V-novel with β in [0.005, 0.02].
- **Group size G = 4** initially (compute-bound; raise if throughput allows).
- **Target one reward design** — don't chase two at once in this timeline.

### Candidate reward designs

Ranked by expected payoff × feasibility for the Mon–Thu window:

**R1 — Instruction-adherence judge (preferred)**
- Reward = 0.6 · judge_score + 0.3 · task_floor + 0.1 · repetition_penalty
  - **judge_score**: Qwen3-4B frozen judge scoring "does the output respond to the instruction?" on [0, 1]. Judge sees prompt + completion only, no audio. Cached by `(prompt_hash, completion_hash)`.
  - **task_floor**: WER floor for ASR / chrF floor for S2TT / exact-match for T2 JSON — prevents the model from gaming the judge.
  - **repetition_penalty**: `−α · (n-gram_repetition_ratio)` where α tuned so the penalty caps at 0.2.
- **Training prompts:** 50% paraphrased-instruction versions of standard tasks, 50% synthetic novel tasks (T1–T10). This explicitly RLs the Surprisal-relevant distribution.
- Matches the 2026 track's "Instruction Following" theme directly.

**R2 — Repetition + format repair (fallback)**
- Reward = `−repetition_penalty + length_ratio_reward + format_adherence`
- Pure heuristic; no judge LLM needed.
- Solves NAVER's known disfluency loops; smaller novelty but safe.
- Fall back to this if R1 shows reward-hacking in the first 50 steps.

**R3 — COMET-RL on S2TT (bonus if time)**
- Reward = COMET-22 + chrF floor + length ratio
- Direct MCIF-metric optimization.
- Risky (COMET is gameable, needs mWERSegmenter in rollout loop). Skip unless R1 finishes with days to spare.

### Anti-collapse settings (from GRPO community practice; refine after Wolfe blog)

- Clip-Higher (DAPO-style asymmetric ε, e.g., ε_low=0.2, ε_high=0.28) to preserve exploration.
- Dynamic Sampling (DAPO): drop groups with zero reward variance.
- Token-level loss averaging (Dr.GRPO): divide by batch tokens, not per-sequence.
- Entropy bonus: small (1e-3) to delay entropy collapse.
- Rollout temperature 0.9.
- Stratify groups by language (never mix en/de/it/zh in one group) — prevents language-mode collapse.

### Infrastructure

- **Trainer:** `trl.GRPOTrainer` subclassed to pass `input_features` through rollout.
- **Rollouts:** HF generation initially (skip vLLM; infra simpler). Upgrade to vLLM only if rollouts bottleneck training.
- **Batch / rollout layout:** prompts per step = 8, G = 4 rollouts each = 32 completions per optimizer step. LoRA grad accumulation 4 → effective ~128 completions per update.
- **Judge:** Qwen3-4B frozen, served via vLLM on a separate GPU (or via `transformers` batch if GPU-constrained). Cache hit rate should exceed 60% by step 100.

### Budget

- Target 300 steps. Stop early on any of:
  - KL(π_θ || π_ref) > 3.0
  - Reward mean plateaus for 50 steps
  - Judge cache hit rate drops (indicates mode drift)
- Wall time: ~8–14 hrs on 2× H200 (slower than SFT due to rollout cost).
- Mon–Thu window fits two 300-step runs if we need to retry.

---

## 9. Timeline — Fri 2026-04-17 → Fri 2026-04-24

All times Eastern / local.

### Phase 1 — Weekend (Fri eve → Mon AM): first 3 stages + draft

| Day | Time | Action | GPUs |
|---|---|---|---|
| **Fri** | evening | Download Gemma-4-31B + Gemma-4-E4B-it; start feature precompute (S2TSUM, ACHAP); draft 10 generator prompts; generate paraphrased MCIF dev | 1 A100 for precompute + CPU work |
| **Fri** | late evening | Throughput smoke (500 samples per generator instance); adjust batch/TP | 2 A100 |
| **Fri** | evening | Launch **two parallel gen scripts**: `gen_text.sh` (T1–T7) on 2× H200, `gen_audio.sh` (T8–T10) on 2× A100, simultaneously | 2 H200 + 2 A100, DP replicas |
| **Fri** | late evening | Audio gen finishes first (~2 hrs); A100s freed for precompute / paraphrased-MCIF backfill | — |
| **Fri** | ~midnight | Text gen finishes (~3–4 hrs); all 50k synthetic samples ready | — |
| **Sat** | afternoon | Launch **V-novel Stage 3** on 2× H200 DDP (or 2× A100 DDP) and **V-base Stage 3** on 2× remaining GPUs in parallel | 4 GPUs |
| **Sat** | evening | Stage 3 both variants done; MCIF + paraphrased-MCIF eval | 1 H200 |
| **Sun** | all day | Write paper: intro, related work, method (Stages 1/2/3), results, ablation | — |
| **Mon** | AM | ✅ **Draft of first 3 stages ready** | — |

### Phase 2 — Mon → Thu: Stage 4 + final draft

| Day | Action |
|---|---|
| **Mon PM / Tue** | Implement GRPOTrainer subclass, judge caching, reward composition. Smoke test on 20 prompts. |
| **Tue PM / Wed** | Run Stage 4 R1 training (300 steps, ~8–14 hrs). Monitor entropy, KL, judge cache hit. Abort & fall back to R2 if reward hacking detected in first 50 steps. |
| **Wed evening** | Eval Stage 4 on MCIF + paraphrased MCIF. |
| **Thu** | Integrate Stage 4 results into paper. Polish. |
| **Fri 04-24** | ✅ **Submit paper** |

### Hard deadlines

- **Sat 2026-04-18 end of day:** Stage 3 V-novel trained and evaluated, OR rollback to V-base as our sole Stage 3.
- **Wed 2026-04-22 end of day:** Stage 4 results in hand, OR we ship the paper with only Stages 1–3.
- **Fri 2026-04-24 EoD UTC-12:** System paper deadline.

---

## 10. Open risks & mitigations

| Risk | Mitigation |
|---|---|
| `google/gemma-4-31B` doesn't exist / wrong URL | Fall back to `google/gemma-3-27b-it` (confirmed real, solid text generator). |
| vLLM doesn't support `Gemma4ForConditionalGeneration` yet | Run with `transformers` + `accelerate`; throughput drops ~30%, still fits. |
| Gemma-4-31B too large for 1× A100 80GB bf16 | Use TP=2 (2 A100s per instance), 2 instances across 4 A100s. |
| Gemma-4-E4B-it audio outputs are noisy / low quality | Quality-gate: audit 50 samples after the first 500 generated; if poor, drop T8–T10 and fall back to 7-task text-only synthetic. |
| Stage 3 1-epoch overtrains (NAVER saw 2K steps already saturating) | Save every 500 steps; pick best checkpoint at eval, not last. |
| Stage 3 ICL collator OOMs with 2 audio segments per prompt | Cap exemplar audio at 15 s; disable ICL for any batch where max-seq > 4096. |
| Synthetic data generation runs > 9 hrs | Scale pool from 5k → 4k per task (40k total) and extend Stage 3 launch by the same margin. |
| Stage 4 GRPO destabilizes | Early-abort rules in §8; fall back to R2 repetition-repair; worst case, ship Stages 1–3 only. |
| Paraphrased MCIF dev prompts are too divergent from originals | Human-audit 100 paraphrases on Sat AM before eval; regenerate rejects with stricter paraphrase prompt. |

---

## 11. What's drafted next

- **Turn N+1 (this weekend's first code work):**
  1. `prompts/synthetic/T*_*.txt` — 20 prompt files (10 tasks × 2 variants)
  2. `scripts/download_generator_models.sh`
  3. `scripts/benchmark_gen_throughput.py`
  4. `scripts/generate_synthetic.py` — vLLM wrapper with prompt-variant rotation, post-filters, JSONL writer
  5. `scripts/build_paraphrased_mcif.py`

- **Turn N+2:**
  6. `configs/stage3_surpriseaware.yaml`
  7. ICL-aware collator patch (`src/data/collator.py`)
  8. Synthetic dataset loader (`src/data/synthetic.py`)

- **Turn N+3 (after Wolfe GRPO Tricks digest):**
  9. Fill §8 with specific values taken from Wolfe's post.
  10. `src/train_grpo.py` — GRPOTrainer subclass for SpeechLLM.
  11. `configs/stage4_grpo.yaml`.
