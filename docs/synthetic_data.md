# Synthetic Data for IWSLT 2026 Instruction Following

This document describes the synthetic instruction-tuning corpus we built for
the IWSLT 2026 Instruction Following submission. It covers the task list,
source datasets, generator models, scripts, and file layout.

## 1. Why synthetic data?

The IWSLT IF track evaluates how well a speech-LLM follows *arbitrary*
instructions about an audio clip (keywords, gist, NER, numeric QA, tone,
etc.). The released training data only covers a narrow slice; we need a much
wider distribution of (audio, instruction, answer) tuples so the model learns
to read and comply with the instruction instead of memorizing the task.

We generate this data by prompting large instruction-tuned LLMs with a
transcript (and, for audio-aware tasks, the audio clip itself) and collecting
answers for 10 different instruction types.

## 2. Task inventory (T1–T10)

| ID  | Name       | Language | Modality | What the model must produce | Generator | Source |
|-----|------------|----------|----------|-----------------------------|-----------|--------|
| T1  | keywords   | en       | text     | 5–8 comma-separated keyphrases | Gemma-4-31B | CoVoST2 |
| T2  | ner        | en       | text     | JSON with `PER / ORG / LOC / DATE / NUM` entity lists | Gemma-4-31B | CoVoST2 |
| T3  | gist_en    | en       | text     | One-sentence English core claim (8–25 words) | Gemma-4-31B | **YTSeg** |
| T4  | topic      | en       | text     | 1–3 word topic label | Gemma-4-31B | CoVoST2 |
| T5  | numeric_qa | en       | text     | `Q: …\nA: …` pair whose answer is a number | Gemma-4-31B | CoVoST2 |
| T6  | gist_de    | de       | text     | One-sentence German gist of an English transcript | Gemma-4-31B | **YTSeg** |
| T7  | gist_zh    | zh       | text     | One-sentence Simplified-Chinese gist of an English transcript | Gemma-4-31B | **YTSeg** |
| T8  | tone_en    | en       | audio    | One-sentence English description of the speaker's delivery / tone | Gemma-4-E4B-it | CoVoST2 |
| T9  | tone_de    | de       | audio    | Same, in German | Gemma-4-E4B-it | CoVoST2 |
| T10 | tone_zh    | zh       | audio    | Same, in Chinese | Gemma-4-E4B-it | CoVoST2 |

Why YTSeg for T3/T6/T7: gist = *compression of the main claim*. A single
CoVoST2 sentence has nothing to compress, so the model collapses gist into
paraphrase. YTSeg's YouTube chapters (100–1500 words) give the generator a
real multi-sentence speech where one claim dominates — so the output is a
genuine summarization, not a rewording.

## 3. Source datasets

### CoVoST2 (`facebook/covost2`)
- Single-sentence Common-Voice utterances paired with audio clips.
- Used for T1/T2/T4/T5 (single-sentence text tasks) and T8/T9/T10 (audio
  tasks — we need the waveform for tone analysis).
- We pull the `en_de` config and take the English side.

### YTSeg (`retkowski/ytseg`)
- 19,299 YouTube videos with pre-shipped reference sentence lists + chapter
  boundary markers (`target_binary_ref`) and chapter timestamps/titles.
- License CC-BY-NC-SA. Transcripts only — no audio required for T3/T6/T7
  which are text-only.
- We slice each video's sentence list by chapter boundaries and keep chunks
  of 100–1500 words. One chapter = one training example.
- No per-sentence timestamps and no sub-chapter segmentation is needed for
  the text gist tasks.

### Why not NUTSHELL / ACHAP?
NUTSHELL (academic talks + abstracts) would have been ideal for gist but
ships abstracts only — transcripts would need a full Whisper pass, which
was too expensive given the 2026-04-24 deadline. YTSeg's pre-shipped
transcripts made it the pragmatic choice.

## 4. Generator models

### Gemma-4-31B (text tasks T1–T7)
- Base model, no instruction tuning.
- Driven by few-shot prompting (2 variants per task, A and B, each with
  3 in-context demos including one NONE / refusal demo).
- `max_model_len=4096`, bf16, served via vLLM.

### Gemma-4-E4B-it (audio tasks T8–T10)
- Audio-instruction-tuned 4B model.
- Driven by `llm.chat()` with an attached audio clip.
- Used only for T8/T9/T10 since they require listening to the waveform,
  not just reading a transcript.

Local weights under `models/` (downloaded by
`scripts/download_generator_models.sh`).

## 5. Prompts

One prompt per (task, variant) — 20 files total:

```
prompts/synthetic/
  T1_keywords_gemma_{A,B}.txt
  T2_ner_gemma_{A,B}.txt
  T3_gist_en_gemma_{A,B}.txt
  T4_topic_gemma_{A,B}.txt
  T5_numeric_qa_gemma_{A,B}.txt
  T6_gist_de_gemma_{A,B}.txt
  T7_gist_zh_gemma_{A,B}.txt
  T8_tone_en_gemma4e4b_{A,B}.txt
  T9_tone_de_gemma4e4b_{A,B}.txt
  T10_tone_zh_gemma4e4b_{A,B}.txt
```

Each file contains: the instruction header, 2–3 short rules, 2–3 in-context
demos (one of which is usually a NONE / refusal example so the model learns
to reject scaffolding-only chunks), and a `{{TRANSCRIPT}}` slot that is
string-substituted by the driver.

T3/T6/T7 prompts use chapter-length demos (climate study / transformer
curriculum / sponsor-read NONE; ECB rate cut / solar panels / Brilliant
sponsor NONE) to teach the long-input → one-sentence compression behaviour.
T7 additionally carries an explicit "no enumeration, ≤40 CJK chars" clause
to keep Chinese outputs tight.

## 6. Driver script

`scripts/generate_synthetic.py` is the single driver used by both the text
and audio launchers. Key responsibilities:

- Load a source pool (CoVoST2 or YTSeg) into `data/synthetic/source_pool*.jsonl`.
- For each task, for each prompt variant A/B, format the prompt per record
  and issue a batch of vLLM completions (text) or `llm.chat()` calls (audio).
- Apply per-task reject filters (word/char bounds, CJK presence, JSON
  validity for NER, numeric presence for T5, refusal/thinking-leak detection)
  and split output into `part-*.jsonl` (kept) and `part-*.jsonl.rejects`
  (discarded).
- Stamp each record with id/task/model/prompt-variant/decoding/source/
  transcript/answer/raw_answer/finish_reason/audit fields for traceability.

### CLI flags worth knowing
- `--tasks T1,T3,T6` — comma-separated task ids.
- `--modality {text,audio}` — picks which task table to use.
- `--samples-per-task N` — target accepted count per task.
- `--replica-id / --num-replicas` — data-parallel shard split.
- `--source {covost2,ytseg}` — source pool builder.
- `--source-pool-cache PATH` — override auto cache location.

## 7. Launcher scripts

### Text tasks (T1–T7)

```bash
# Full run (5000/task) on CoVoST2 for T1,T2,T4,T5
bash scripts/gen_text.sh

# Gist tasks on long YTSeg chapters
SOURCE=ytseg TASKS="T3,T6,T7" bash scripts/gen_text.sh

# Smoke (500/task)
SMOKE=1 TASKS="T1,T3" bash scripts/gen_text.sh

# Use specific GPUs as DP replicas
GPUS="0,1" bash scripts/gen_text.sh
```

Default: 2 data-parallel replicas, one per GPU, 90% GPU memory each,
`max-model-len=4096`, `max-num-seqs=64`. Outputs go to
`data/synthetic/<task>/gemma-4-31B/part-<replica>.jsonl`.

### Audio tasks (T8–T10)

```bash
bash scripts/gen_audio.sh                    # full 5000/task
SMOKE=1 bash scripts/gen_audio.sh            # 500/task
TASKS="T8" GPUS="2,3" bash scripts/gen_audio.sh
```

Gemma-4-E4B-it is small (~16 GB bf16), so several replicas fit on one GPU.
Outputs go to `data/synthetic/<task>/gemma-4-E4B-it/part-<replica>.jsonl`.

### Environment prerequisites
```bash
cd /work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following
source .venv_gen/bin/activate
module load cuda/12.8
```
`.venv_gen` is a generation-only env with torch 2.10 + vllm 0.19 (separate
from the training env so vllm's pins don't clash with the trainer).

## 8. Output layout and counts

```
data/synthetic/
├── source_pool.jsonl             # CoVoST2 cache
├── source_pool_ytseg.jsonl       # YTSeg chapter cache
├── T1_keywords/gemma-4-31B/part-0.jsonl         # 5898 accepted
├── T2_ner/gemma-4-31B/part-0.jsonl              # 5281
├── T3_gist_en/gemma-4-31B/part-0.jsonl          # 5101  (YTSeg)
├── T4_topic/gemma-4-31B/part-0.jsonl            # 5670
├── T5_numeric_qa/gemma-4-31B/part-0.jsonl       # 4937
├── T6_gist_de/gemma-4-31B/part-0.jsonl          # 6199  (YTSeg)
├── T7_gist_zh/gemma-4-31B/part-0.jsonl          # 5587  (YTSeg)
├── T8_tone_en/gemma-4-E4B-it/part-0.jsonl       # 7500
├── T9_tone_de/gemma-4-E4B-it/part-0.jsonl       # 7335
└── T10_tone_zh/gemma-4-E4B-it/part-0.jsonl      # 7452
```

Total accepted: ~66k records across 10 instruction types. Each task's
`*.jsonl.rejects` sibling holds the filtered-out generations with
`audit.filter_reasons` populated for post-hoc analysis.

## 9. Per-record schema

```json
{
  "id": "synth_T3_r0_000000",
  "task": "T3_gist_en",
  "source_model": "gemma-4-31B",
  "prompt_variant": "A",
  "decoding": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 120},
  "source": "ytseg",
  "audio_path": null,
  "transcript": "…",
  "answer": "One-sentence gist goes here.",
  "raw_answer": "…",
  "finish_reason": "stop",
  "stop_reason": null,
  "n_output_tokens": 27,
  "generation_timestamp": "2026-04-18T22:14:…",
  "audit": {"rejected": false, "filter_reasons": []}
}
```

## 10. Quality checks we ran

- **Yield** per task: 66–83 % of requested samples passed filters; the rest
  were `none_output` (model correctly refused ad/filler chunks) or
  length-bound failures.
- **Empty / ultra-short answers**: 0 across T3/T6/T7.
- **Exact-duplicate answers**: 4 in T3 (out of 5101), 0 in T6/T7.
- **finish_reason**: 16,884 / 16,887 natural stop across T3+T6+T7; 3
  length-cut records (all complete sentences).
- **Length distributions** (accepted): T3 median 22 words, T6 median 22
  words, T7 median 37 CJK chars — all within prompt targets.
- **Residual issues** (~1–3 % of records): occasional enumeration on
  product-tutorial chunks and a small number of meta-references ("The
  transcript describes…") in T3. Not a corpus-level problem.
