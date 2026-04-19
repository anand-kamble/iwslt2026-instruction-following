# ICL (1-shot overlay) — Implementation Spec

**Status:** designed, not yet built. This doc is the handoff so a future turn
can pick it up cold and implement without re-deriving the design.

**Goal:** add an in-context 1-shot overlay to Stage 3 training so the model
is exposed during SFT to the prompt format it will see on the held-out IWSLT
2026 *Surprisal* task (organizers explicitly say the surprise task is
"solvable via in-context / 1-shot learning", en→de / en→zh only).

Plan reference: `docs/stage3_stage4_plan.md` §5 "ICL overlay (Layer 2 —
collator coin flip)".

---

## 1. Behaviour we want

For each **sampled batch** from a short-audio task, with probability **p = 0.25**,
reformat every sample in the batch as a 1-shot prompt:

```
[user ]  <audio_exemplar>  + instruction_same_task
[asst ]  answer_for_exemplar              ← masked, no loss
[user ]  <audio_target>    + instruction_same_task
[asst ]  answer_for_target                ← trained target (only this turn)
```

### Eligibility

| Task | ICL-eligible? | Why |
|---|---|---|
| asr, s2tt_{de,it,zh}, sqa_{en,de,it,zh} | yes | ≤ ~20 s clips |
| synth_T1..T10 | yes | T1/T2/T4/T5 use CoVoST audio; T3/T6/T7 are text-only but still benefit from a text-1-shot; T8/T9/T10 are audio tone tasks, ≤20 s |
| s2tsum_en, achap_en | **no** | ~6 min audio; ICL would double that |
| text-interleave partners (mt_*, qa_*) | **no** | they're the text half of a pair, zero-shot by design |

Constants to put in code:

```python
ICL_ELIGIBLE_TASK_PREFIXES = (
    "asr", "s2tt_", "sqa_",
    "synth_T1", "synth_T2", "synth_T3", "synth_T4", "synth_T5",
    "synth_T6", "synth_T7", "synth_T8", "synth_T9", "synth_T10",
)
ICL_MAX_EXEMPLAR_AUDIO_S = 15         # default
ICL_MAX_EXEMPLAR_AUDIO_S_AUDIO_TASKS = 20   # T8–T10
```

### Exemplar rules

- Same task as target.
- Different sample (compare by the task-local dataset index).
- Exemplar audio ≤ 15 s (20 s for T8–T10).
- Draw is reproducible: seeded RNG keyed off (step, batch_idx).
- Exemplar's assistant turn is masked out of loss (handled naturally by the
  existing "last assistant header" label-masking logic in
  `src/data/collator.py:163-177`).

---

## 2. Architectural decisions (locked)

These were debated and decided — do not relitigate on resume.

1. **Coin flip is batch-level, not per-sample.** The existing
   `TemperatureSampler` (`src/data/multitask.py`) already yields
   homogeneous-by-task batches when `batch_size > 1`. One coin flip per
   batch preserves shape uniformity and keeps the collator simple. In
   expectation 25% of samples = 25% of batches, so this matches the plan.
2. **The coin flip lives in the collator, not the sampler and not a dataset
   wrapper.** Sampler's job ends at picking indices; the collator has the
   context to read `_task`, splice exemplars, and build the 4-turn prompt.
3. **Exemplar pool is built once at init, ~500 indices per eligible task.**
   We store indices, not materialized audio features. Audio is loaded lazily
   through the same code path as targets.
4. **Synthetic T3/T6/T7 (text-only) also get ICL.** They become a pure
   text 1-shot (two transcripts back-to-back). The collator's text-only
   branch handles this with no special case — just let `audio_path is None`
   flow through.
5. **Label masking is unchanged.** The collator currently masks everything
   before the *last* `<|im_start|>assistant\n` header; for a 4-turn ICL
   prompt that naturally keeps only the target's assistant turn trainable.
6. **Sequence length:** bump `max_seq_length` to 6144 in the Stage 3 config
   for ICL runs. Alternative (let tokenizer truncate) is fine too because
   truncation happens at the *front* (exemplar side) and the target is last.
   Pick one at implementation time — default 6144 is safest.

---

## 3. Files to add / change

### 3.1 `src/data/icl.py` (new, ~80–120 lines)

Exports:
- `ICL_ELIGIBLE_TASK_PREFIXES`, `ICL_MAX_EXEMPLAR_AUDIO_S`,
  `ICL_MAX_EXEMPLAR_AUDIO_S_AUDIO_TASKS` — constants.
- `is_icl_eligible(task_name: str) -> bool` — helper.
- `class IclExemplarPool`:
  ```
  def __init__(self, task_datasets: dict[str, Dataset], pool_size=500,
               max_audio_s=15, max_audio_s_audio_tasks=20, seed=42): ...
  def draw(self, task: str, exclude_local_idx: int, rng: random.Random) -> dict: ...
  def has(self, task: str) -> bool: ...
  ```
  - At construction, for each task in `task_datasets` whose name is
    `is_icl_eligible`, scan up to a few thousand candidate samples and keep
    the first `pool_size` whose duration meets the cap (or all samples
    unconditionally for text-only tasks with no audio).
  - Duration filter needs to be cheap: read from dataset metadata where
    possible (HF datasets expose `audio.array` + sampling_rate lazily —
    compute `len(array)/sr` without decoding the full file when the backend
    supports it). If decoding is forced, cache per-index durations in a
    small JSON next to the dataset (`data/.icl_durations/<task>.json`).
  - `draw` loops until `retry_idx != exclude_local_idx`.

### 3.2 `src/data/collator.py` (extend `SpeechTextCollator`)

New `__init__` args (all optional, defaults preserve current behaviour):
```python
icl_pool: IclExemplarPool | None = None
icl_prob: float = 0.25
icl_seed: int = 42
```

Add a private `self._icl_rng = random.Random(icl_seed)` in `__post_init__`
(advance per call).

In `__call__`, BEFORE the existing audio/text processing:

```python
# --- ICL decision (batch-level) --------------------------------------
shared_task = features[0].get("_task")
use_icl = (
    self.icl_pool is not None
    and shared_task is not None
    and is_icl_eligible(shared_task)
    and all(f.get("_task") == shared_task for f in features)
    and self._icl_rng.random() < self.icl_prob
    and self.icl_pool.has(shared_task)
)

if use_icl:
    for f in features:
        ex = self.icl_pool.draw(
            shared_task,
            exclude_local_idx=f.get("_local_idx", -1),
            rng=self._icl_rng,
        )
        f["_icl_exemplar"] = ex
```

**Audio branch changes (only for ICL batches):**
- For each feature with `_icl_exemplar`, load exemplar audio using the same
  try/except block, then target audio. Append both to `audios` so the final
  `input_features` tensor has `2 * B` rows in order
  `[ex_0, tg_0, ex_1, tg_1, ..., ex_{B-1}, tg_{B-1}]`.
- Same for `input_features_attention_mask`.
- The SpeechLLM model side must know how to consume 2·B audio rows for B
  text samples — see §3.3 below.

**Text branch changes (only for ICL batches):**
- Build a 4-turn message list per feature:
  ```python
  if "_icl_exemplar" in f:
      ex = f["_icl_exemplar"]
      f["messages"] = [
          {"role": "user",      "content": SPEECH_PLACEHOLDER + ex["user_text"]},
          {"role": "assistant", "content": ex["answer"]},
          {"role": "user",      "content": SPEECH_PLACEHOLDER + f["user_text"]},
          {"role": "assistant", "content": f["answer"]},
      ]
  ```
  Exact construction depends on how each task dataset populates the
  messages. Best pattern: do this transform *after* the per-task dataset
  has already produced its 2-turn messages, by stripping and re-assembling.
  Easier alternative: have `IclExemplarPool.draw()` return a
  pre-extracted `(user_text, answer)` pair and require each task's
  `__getitem__` to surface the same. This forces a small interface
  alignment across dataset modules.

**Label masking:** unchanged. Lines 163–177 already find the LAST
`<|im_start|>assistant\n` header and mask everything before it — so the
exemplar's assistant turn falls into the masked region. Verify with a
unit test, don't modify.

### 3.3 `src/model/speech_llm.py` — OPEN QUESTION

**Status: UNVERIFIED.** Before implementing, inspect `speech_llm.py` and
answer: does `SPEECH_PLACEHOLDER` insertion handle **multiple placeholder
tokens** in a single prompt?

Two cases:

- **Case A (already multi-placeholder).** The model code replaces *every*
  occurrence of the placeholder token with the next speech embedding in
  sequence. ICL works for free — just make sure the collator emits audio
  rows in exemplar-then-target order to match placeholder order.

- **Case B (single-placeholder only).** The current code assumes exactly
  one speech segment per prompt. We need to generalize
  `insert_speech_embeds(input_ids, speech_embeds)` to walk all placeholder
  positions and splice in successive embeddings. Approximate diff: ~20
  lines, same function.

**First action on resume:** grep for `SPEECH_PLACEHOLDER` in
`src/model/speech_llm.py` and determine which case we're in.

### 3.4 Dataset-module interface tweak

Each ICL-eligible task dataset (`asr.py`, `s2tt.py`, `sqa.py`,
`synthetic.py`) should include on every sample:

- `_local_idx: int` — the per-task index (for exemplar exclusion).
- `user_text: str` — the user turn content WITHOUT the `SPEECH_PLACEHOLDER`
  prefix (the collator adds that back when building messages).
- `answer: str` — the assistant turn target.

If these fields aren't already present, add them in each module's
`__getitem__`. Prefer surface changes over restructuring — the existing
`messages` field can stay.

### 3.5 Config — `configs/stage3_surpriseaware.yaml`

Add an `icl:` block:
```yaml
icl:
  enabled: true
  prob: 0.25
  pool_size: 500
  max_exemplar_audio_seconds: 15
  max_exemplar_audio_seconds_audio_tasks: 20
  seed: 42

data:
  max_seq_length: 6144   # bumped from 4096 to fit 1-shot prompts
```

### 3.6 Trainer entry point — `src/train.py` (or wherever collator is built)

After datasets are loaded but before trainer construction:
```python
icl_cfg = config.get("icl", {})
icl_pool = None
if icl_cfg.get("enabled", False):
    from src.data.icl import IclExemplarPool
    icl_pool = IclExemplarPool(
        task_datasets=all_datasets,
        pool_size=icl_cfg.get("pool_size", 500),
        max_audio_s=icl_cfg.get("max_exemplar_audio_seconds", 15),
        max_audio_s_audio_tasks=icl_cfg.get(
            "max_exemplar_audio_seconds_audio_tasks", 20
        ),
        seed=icl_cfg.get("seed", 42),
    )

collator = SpeechTextCollator(
    processor=...,
    tokenizer=...,
    max_seq_length=config["data"].get("max_seq_length", 4096),
    icl_pool=icl_pool,
    icl_prob=icl_cfg.get("prob", 0.25),
    icl_seed=icl_cfg.get("seed", 42),
)
```

---

## 4. Sanity tests before full training

1. **Unit:** construct pool from a tiny CoVoST2 subset; draw 10 times; assert
   `exclude_local_idx` is never returned.
2. **Unit:** build a collator with `icl_prob=1.0`, pass 2 samples; assert
   `input_features.shape[0] == 4`, `messages` has 4 turns, `labels[:3_turns]
   == -100`.
3. **Integration:** forward-pass one ICL batch through the SpeechLLM; assert
   loss is finite, gradient flows only to target turn (check via
   `labels != -100` count).
4. **Smoke:** 50-step training run with `icl.prob=1.0` (force every batch
   to be ICL). Loss should decrease and not NaN.
5. **A/B:** 500-step runs with `icl.prob=0.0` vs `0.25` on same seed; eval
   both on MCIF dev and paraphrased MCIF dev. Expect parity or improvement
   on paraphrased, no regression on standard.

---

## 5. Resumption checklist

When you come back to build this, work in this order:

- [ ] **Verify model multi-placeholder case** (grep `SPEECH_PLACEHOLDER` in
      `src/model/speech_llm.py`). Decide Case A vs Case B. Fix model if B.
- [ ] Write `src/data/icl.py` with `IclExemplarPool` and `is_icl_eligible`.
- [ ] Extend `SpeechTextCollator.__init__` + `__call__` per §3.2.
- [ ] Thread `user_text` / `answer` / `_local_idx` through each task's
      `__getitem__` (only ICL-eligible tasks).
- [ ] Add `icl:` block to `configs/stage3_surpriseaware.yaml` and bump
      `max_seq_length` to 6144.
- [ ] Wire pool construction in `src/train.py`.
- [ ] Run unit tests in §4 step 1–3.
- [ ] Run smoke (§4 step 4).
- [ ] Launch the A/B (§4 step 5) as part of Stage 3.

---

## 6. Things to explicitly NOT do

- Do **not** modify `TemperatureSampler`. The homogeneous-batch behaviour
  we need is already there.
- Do **not** build a separate `IclDataset` wrapper — the collator is the
  right layer.
- Do **not** precompute or cache exemplar audio features ahead of training.
  Lazy loading through the existing collator path is cheap enough.
- Do **not** mix tasks across exemplar and target (rule: same-task).
- Do **not** remove the existing last-assistant-header masking — it's
  exactly what makes 4-turn ICL work without special-casing labels.
