# GRPO (Stage 4) — Brainstorming & Design Log

**Status:** brainstorming in progress. This doc is a running log of the
decisions we've committed to, the open questions, and the reasoning behind
each choice. Intended as a handoff so a future turn can pick it up without
re-deriving anything.

Plan reference: `docs/stage3_stage4_plan.md` §8.

---

## 1. Scope and framing (locked)

- **When:** after Stage 3 SFT (V-novel checkpoint).
- **What it's fixing:** residual *instruction-following* failures — model
  sometimes ignores the instruction, drifts to wrong language, repeats, or
  produces the right content in the wrong format. SFT cross-entropy can't
  directly optimize these.
- **Why RL and not more SFT:** we already have gold references for every
  task. The gap is *behaviour at inference time* (repetition, drift, format).
  A rubric-aware judge scores "did the rollout follow the instruction" in
  a way that's not easy to teach with token-level loss.
- **Separate folder:** all GRPO code lives under `grpo/` (training scripts,
  rubrics, reward functions, configs). Not mixed into `src/`.

---

## 2. Reward family (decided)

Primary reward: **R1 — Instruction-adherence judge** (per plan §8).

```
reward = 0.65 * judge_norm       (Prometheus-7B-v2.0, soft-scored)
       + 0.25 * task_floor       (deterministic metric per task)
       - 0.10 * hygiene_penalty  (repetition + length + language)
```

Clipped to `[-0.2, 1.0]`. Weights are starting guesses — tune on the first
50-step smoke run.

Fallback: **R2 — Repetition + format repair.** Auto-trigger if the first 50
training steps show reward-hacking signatures:
- Mean judge score climbs above 0.85 but floor score drops below 0.4.
- fastText language-match rate drops below 95% on translation tasks.
- n-gram repetition ratio climbs above 0.25 on average.

If any of those fire, drop the judge weight to 0 and switch to hand-coded
`R2 = -repetition - |log(length_ratio)| - lang_mismatch`.

R3 (COMET on S2TT) is **deferred** — outside our scope for the
2026-04-24 deadline.

---

## 3. Training prompt distribution (decided — Option A)

50% standard paraphrased-instruction prompts + 50% synthetic T1–T10.

| Bucket | Share | Source |
|---|---|---|
| Paraphrased MCIF / standard | 50% | `data/paraphrased_mcif_dev.jsonl` (built via Qwen3-4B paraphrase pass over MCIF instructions) |
| Synthetic T1–T10 | 50% | `data/synthetic/<task>/gemma-4-31B/part-*.jsonl` |

Rationale: the held-out Surprisal task on IWSLT 2026 is explicitly described
as "novel phrasing of instruction-following." Option A exercises both
reworded-standard-task prompts and wholly-new synthetic tasks — directly
mirrors the distribution we expect at eval.

Option B (pure synthetic) was rejected — too narrow. Option C (standard
only) was rejected — just polishes existing behaviour, no novelty claim.

---

## 4. Judge model (decided)

**Prometheus-7B-v2.0** — `prometheus-eval/prometheus-7b-v2.0` on HuggingFace.

Why:
- Specifically trained for rubric-based absolute grading and relative
  grading — not a general chat model coerced into evaluation.
- Less gameable than a generic Qwen3/Gemma judge (trained with adversarial
  preference data).
- 7B bf16 (~14 GB) fits on a single A100 80GB with plenty of room for vLLM
  batch serving.
- License permissive.

### What Prometheus takes

Absolute-grading mode:
```
INPUT:  instruction + response + reference_answer + score_rubric
OUTPUT: feedback (paragraph of reasoning) + score ∈ {1,2,3,4,5}
```

It is **text-only** — never sees the audio. Fine, because the reference
answer is derived from the audio.

### Judge score → continuous reward

Discrete {1,2,3,4,5} is bad for group-relative advantages in GRPO. Solution:
compute **expected score** from logprobs over the 5 score tokens.

```python
# after Prometheus forward pass, at the position where the score token sits
logprobs = model.get_logprobs_at(score_position)
p = softmax([logprobs[tok_1], logprobs[tok_2], ..., logprobs[tok_5]])
expected_score = 1*p[0] + 2*p[1] + 3*p[2] + 4*p[3] + 5*p[4]
judge_norm = (expected_score - 1) / 4   # ∈ [0, 1]
```

Check: vLLM exposes prompt logprobs via `SamplingParams(logprobs=N)`.
If this turns out brittle, fallback is two samples with different
temperatures, averaged — coarser but works.

---

## 5. Per-task reward routing (decided)

### 5.1 The task label IS available

**Verified**: `src/data/multitask.py:57` stamps `item["_task"] = task_name`
on every sample. The collator propagates it as `batch["_tasks"]`
(`src/data/collator.py:42-44`). `src/train.py:281, 632` already pops
`_tasks` in `compute_loss` for per-task loss logging.

Task names in the training data:
`asr`, `s2tt_de`, `s2tt_it`, `s2tt_zh`, `sqa_en`, `sqa_de`, `sqa_it`,
`sqa_zh`, `s2tsum_en`, `achap_en`, `synth_T1`…`synth_T10`, plus
text-interleave partners (`mt_de`, `qa_en`, …).

### 5.2 The reference answer IS available at a consistent location

**Verified**: every dataset module builds ChatML messages via
`format_chat_messages(instruction, response)` or
`format_text_only_messages(instruction, response, source_text)` from
`src/data/utils.py`. Both emit:

```python
[
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": response},   # ← always last
]
```

Checked modules: `asr.py:50-52`, `s2tt.py:51-54`, `synthetic.py:119-123`
(text T1–T7), `synthetic.py:170-173` (audio T8–T10).

So the GRPO reward function can do:
```python
reference = feature["messages"][-1]["content"]
```
No plumbing changes to the dataset modules.

### 5.3 Task → rubric + floor map

Five rubric templates cover all 16 task names. Stored in
`grpo/rubrics/{translation,gist,extraction,numeric,tone}.txt`.

| Task name | Rubric template | Floor metric | Skip judge? |
|---|---|---|---|
| `asr` | — | 1 − WER | **yes** (WER is the ground truth) |
| `s2tt_de`, `s2tt_it`, `s2tt_zh` | translation | chrF | no |
| `sqa_en`, `sqa_de`, `sqa_it`, `sqa_zh` | numeric (if answer has digits) else extraction | token-F1 | no |
| `s2tsum_en` | gist | chrF | no |
| `achap_en` | gist | chrF | no |
| `synth_T1` (keywords) | extraction | Jaccard on comma-split sets | no |
| `synth_T2` (NER) | extraction | JSON parses AND ≥1 entity match | no |
| `synth_T3` (gist_en) | gist | chrF | no |
| `synth_T4` (topic) | extraction | chrF | no |
| `synth_T5` (numeric_qa) | numeric | number-extract + exact match | no |
| `synth_T6` (gist_de) | translation | chrF | no |
| `synth_T7` (gist_zh) | translation | chrF | no |
| `synth_T8` (tone_en) | tone | chrF | no |
| `synth_T9` (tone_de) | tone | chrF | no |
| `synth_T10` (tone_zh) | tone | chrF | no |

**Special case: ASR.** Skip the judge call entirely. WER vs the reference
is the canonical metric, a judge adds no information and costs a judge call
per rollout. For ASR, `reward = (1 - WER) - hygiene_penalty`.

---

## 6. The five rubrics

Each rubric is a short prose block in Prometheus-v2.0's native format.
Plain text, stored under `grpo/rubrics/`. Sketches:

### `translation.txt`
```
Score the translation quality of the response against the reference.
5: Fluent, idiomatic target-language output that conveys the full meaning
   of the reference. All named entities and numeric values preserved.
4: Minor fluency or detail issues, core meaning intact.
3: Meaning preserved but awkward phrasing or one small omission.
2: Significant content loss or noticeably broken target-language grammar.
1: Wrong meaning, wrong target language, or catastrophic omission.
```

### `gist.txt`
```
Score whether the response captures the main claim of the source as
a single sentence without enumeration.
5: One sentence, the single most central claim, names/numbers preserved.
4: One sentence, correct claim, slightly wordy.
3: Correct claim but runs long or mildly enumerative.
2: Multiple sentences or missed claim, partial coverage only.
1: Wrong claim or filler summary.
```

### `extraction.txt` (keywords / NER / topic)
```
Score whether the response returns the requested format and items that
match the reference.
5: Correct format, full recall on reference items, no hallucinated items.
4: Correct format, minor recall gap OR one hallucinated item.
3: Correct format but significant recall gap.
2: Wrong format OR only one or two correct items.
1: Empty, invalid format, or unrelated items.
```

### `numeric.txt` (numeric QA / T5)
```
Score whether the response gives the correct numeric answer with units.
5: Correct number, correct unit, cleanly stated.
4: Correct number, missing or implicit unit.
3: Near-correct number (unit conversion or small rounding).
2: Wrong number but right quantity being asked about.
1: No number or unrelated answer.
```

### `tone.txt` (T8 / T9 / T10)
```
Score whether the response describes the speaker's tone, pace, and
delivery style in 1–2 sentences in the target language.
5: Accurate and specific description, target language, 1–2 sentences.
4: Accurate but generic description.
3: Partially accurate OR 1 sentence is fine but slightly off.
2: Vague OR wrong target language.
1: Contradicts the reference or empty.
```

---

## 7. Hygiene penalty (details)

Orthogonal to correctness — these are output-shape problems that would
still score on the rubric but indicate degenerate behaviour. Three pieces,
each in `[0, 1]`, summed and clipped to 1.0 before the `0.10 *` weight.

Maximum total deduction: **0.10**. Small by design — meant to discourage
garbage, not dominate learning.

### 7.1 Repetition penalty
```python
def repetition_penalty(text):
    toks = text.split()
    if len(toks) < 6: return 0.0
    trigrams = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
    rep_ratio = 1 - len(set(trigrams)) / len(trigrams)
    # Clean text: 0.05–0.15. Penalize above 0.3.
    return max(0.0, min(1.0, (rep_ratio - 0.3) / 0.5))
```
Catches: "the the the", "yes yes yes", repeated phrases, infinite loops.

### 7.2 Length abuse penalty
```python
def length_penalty(rollout, reference):
    r = max(1, len(rollout.split()))
    g = max(1, len(reference.split()))
    ratio = math.log(r / g)
    # Allow 0.5× to 2× naturally; penalize beyond
    return max(0.0, min(1.0, (abs(ratio) - 0.7) / 1.5))
```
Catches: model producing 500 tokens when reference is 20, or dropping to
5 tokens when reference is 50.

### 7.3 Language mismatch penalty
Applies only to tasks with a fixed output language:
`s2tt_de|it|zh`, `synth_T6`, `synth_T7`, `synth_T9`, `synth_T10`.
```python
def lang_penalty(text, target_lang):
    import fasttext
    model = fasttext.load_model("lid.176.bin")
    lang, _ = model.predict(text.replace("\n", " "), k=1)
    detected = lang[0].replace("__label__", "")
    return 1.0 if detected != target_lang else 0.0
```
Flat 1.0 penalty on mismatch. This is the one hygiene term that often
fires at step 0 because Qwen's base tendency is to answer in English.

---

## 8. Reward cache (decided)

Every rollout costs a Prometheus call. Budget napkin math:
- 8 prompts/step × G=4 rollouts × 300 steps = 9600 judge calls.
- 7B at bf16, single A100, batch 32: ~0.5 s / call including prefill.
- Naïve: 9600 × 0.5 s = ~80 min of serial judge time.

With caching on `(prompt_hash, completion_hash) → (expected_score)`:
- Hit rate by step 100 expected 30–60% (easy prompts + converging policy
  produce repeat rollouts).
- Effective judge time drops to ~40–50 min total.

Cache implementation: sqlite file at `grpo/cache/judge.db`, `prompt_hash`
and `completion_hash` are MD5 of the exact strings sent to Prometheus.
Store `expected_score` (float) and `raw_score` (int) for debugging.
Cache key also includes `rubric_name` to avoid cross-rubric pollution.

---

## 9. Group-relative dynamics

### 9.1 The flat-reward problem
If all G=4 rollouts in a group get the same reward, advantage = 0 and no
gradient. Two mitigations:
- **DAPO dynamic sampling**: drop groups with `σ(reward) < 0.05` from the
  optimizer step. Requires tracking group variance before loss aggregation.
- **Clip-Higher** (DAPO): asymmetric PPO clipping (`ε_low=0.2, ε_high=0.28`)
  to preserve exploration on high-reward tails.

### 9.2 Rollout diversity knobs
- `temperature=0.9` at rollout time.
- Entropy bonus `1e-3` to delay entropy collapse.
- KL-to-reference β ∈ `[0.005, 0.02]` against the V-novel SFT checkpoint.
- Stratify groups by target language — never mix `en/de/it/zh` in one
  group (prevents language-mode collapse).

---

## 10. Open questions (to resolve before coding)

### Q1. vLLM logprobs for expected-score computation
Does `vllm.LLM.generate(..., SamplingParams(logprobs=N))` return logprobs
at a specified position cleanly enough to read the {1..5} token probs?
**Action:** write a 20-line smoke script that runs Prometheus on one
example and extracts the logprobs. If clean, commit to expected-score.
If flaky, fall back to 2-sample averaging.

### Q2. Rollout infra — HF generate vs vLLM
Plan §8 currently says "HF generation initially, vLLM if bottleneck."
But for 8 prompts × 4 rollouts = 32 generations per step, HF loop time
could be 5–10 s/step just for rollouts. vLLM-in-training is messier but
much faster.
**Lean:** start HF, measure, upgrade only if rollouts are >60% of step time.

### Q3. Judge serving — separate process or in-process?
- Separate vLLM server (OpenAI-compat API on a dedicated GPU) → cleanest
  isolation, but adds network overhead and an extra process to manage.
- In-process `vllm.LLM(...)` held by the reward fn → one fewer moving part,
  but shares GPU with training.
**Lean:** separate process. We have 4+ GPUs available during Stage 4 and
the clean interface is worth it.

### Q4. Reference trust for synthetic data
Our Gemma-generated T1–T10 references are ~95% clean, ~5% imperfect
(enumeration leakage in T7, a few meta-refs in T3). Two choices:
- Trust the reference, let the rubric do its job.
- Weight synthetic rewards 0.5× to reflect lower reference quality.
**Lean:** trust. Prometheus's rubric grades the rollout against the
rubric + reference; an imperfect reference just raises the noise floor,
doesn't bias direction.

### Q5. Group size G
Plan says G=4. Compute-bound. Could we go G=6 or G=8 if rollouts are fast?
**Lean:** start G=4, raise to G=6 if step time < 8 s.

---

## 11. File layout (proposed)

```
grpo/
├── rubrics/
│   ├── translation.txt
│   ├── gist.txt
│   ├── extraction.txt
│   ├── numeric.txt
│   └── tone.txt
├── cache/
│   └── judge.db              # sqlite, created on first run
├── reward/
│   ├── __init__.py
│   ├── compose.py            # top-level compute_reward(task, prompt, rollout, reference)
│   ├── judge.py              # Prometheus client + expected-score extraction
│   ├── floors.py             # WER, chrF, Jaccard, F1, etc.
│   └── hygiene.py            # repetition, length, language-id
├── trainer/
│   ├── __init__.py
│   ├── speech_grpo_trainer.py  # subclass of trl.GRPOTrainer
│   └── dataloader.py           # reuses src/data/multitask.py with a GRPO collator
├── configs/
│   ├── stage4_grpo.yaml
│   └── stage4_grpo_r2_fallback.yaml
└── scripts/
    ├── train_grpo.sh         # launcher
    ├── eval_grpo.sh          # MCIF + paraphrased-MCIF eval
    └── smoke_reward.py       # one-example end-to-end reward-fn check
```

Kept strictly out of `src/` so the SFT code path stays untouched.

---

## 12. Resumption checklist

When we circle back to implement:

- [ ] Resolve Q1 (vLLM logprobs smoke test) — decides expected-score vs
      sample-average.
- [ ] Resolve Q2/Q3 (rollout + judge infra).
- [ ] Write the 5 rubric files.
- [ ] Implement `grpo/reward/floors.py` (WER, chrF, Jaccard, F1, number match).
- [ ] Implement `grpo/reward/hygiene.py` (3 penalty functions).
- [ ] Implement `grpo/reward/judge.py` (Prometheus client + cache).
- [ ] Implement `grpo/reward/compose.py` (top-level reward function).
- [ ] Write `grpo/trainer/speech_grpo_trainer.py` (TRL subclass with
      `input_features` passed through rollout).
- [ ] Write `grpo/configs/stage4_grpo.yaml`.
- [ ] Write `grpo/scripts/smoke_reward.py` — verify per-task reward routing
      on one example of each task name.
- [ ] Run 50-step smoke; check for reward-hacking triggers (§2).
- [ ] Full 300-step run.

---

## 13. Brainstorming log (chronological notes to self)

- **2026-04-19:** Decided Option A (50/50 paraphrased-standard + synthetic).
  Decided Prometheus-7B-v2.0 as judge. Composed reward as
  `0.65*judge + 0.25*floor - 0.10*hygiene`. Picked five rubric families
  mapped over 16 task names. Verified via code inspection that (a) `_task`
  flows end-to-end through the SFT pipeline and (b) reference answer lives
  at `messages[-1]["content"]` consistently across all task modules.
  Deferred Q1–Q5 to implementation time.
- *(next session)*: fill in rollout infra decisions after Q1–Q3 are
  resolved; draft the rubric files; sketch `speech_grpo_trainer.py`
  subclass surface.
