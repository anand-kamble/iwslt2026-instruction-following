#!/usr/bin/env python
"""Synthetic-data generator driver (vLLM + Gemma-4).

Loads a source pool of English (audio, transcript) pairs, then for each target
task runs two prompt variants (A/B) through the configured generator model
and writes filtered JSONL under data/synthetic/<task>/<source_model_tag>/.

Invoked by scripts/gen_text.sh and scripts/gen_audio.sh as:
    python -m scripts.generate_synthetic --model ... --tasks T1,T2 --modality text \
        --samples-per-task 500 --replica-id 0 --num-replicas 2 ...

Text tasks (T1-T7) run through the Gemma-4-31B base model via few-shot
completion. Audio tasks (T8-T10) run through the Gemma-4-E4B-it
instruction-tuned audio model via llm.chat() with an attached audio clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vllm import LLM, SamplingParams

# -------- Task definitions ---------------------------------------------------

TEXT_TASKS = {
    "T1": {"name": "keywords",    "max_tokens": 120, "prompt_stem": "T1_keywords_gemma"},
    "T2": {"name": "ner",         "max_tokens": 160, "prompt_stem": "T2_ner_gemma"},
    "T3": {"name": "gist_en",     "max_tokens": 120, "prompt_stem": "T3_gist_en_gemma"},
    "T4": {"name": "topic",       "max_tokens": 40,  "prompt_stem": "T4_topic_gemma"},
    "T5": {"name": "numeric_qa",  "max_tokens": 80,  "prompt_stem": "T5_numeric_qa_gemma"},
    "T6": {"name": "gist_de",     "max_tokens": 160, "prompt_stem": "T6_gist_de_gemma"},
    "T7": {"name": "gist_zh",     "max_tokens": 220, "prompt_stem": "T7_gist_zh_gemma"},
}

AUDIO_TASKS = {
    "T8":  {"name": "tone_en",  "max_tokens": 120, "prompt_stem": "T8_tone_en_gemma4e4b"},
    "T9":  {"name": "tone_de",  "max_tokens": 160, "prompt_stem": "T9_tone_de_gemma4e4b"},
    "T10": {"name": "tone_zh",  "max_tokens": 160, "prompt_stem": "T10_tone_zh_gemma4e4b"},
}

REFUSAL_PATTERNS = re.compile(
    r"\b(i (?:cannot|can't|won't)|as an ai|i'm an ai|i am an ai|sorry, (?:but )?i)\b",
    re.IGNORECASE,
)
THINK_PATTERNS = re.compile(r"<think>|\bLet me (?:think|reason)\b", re.IGNORECASE)


@dataclass
class SourceRecord:
    id: str
    transcript: str
    audio_path: str | None
    source: str  # "covost2" | "ytseg"


# -------- Source pool --------------------------------------------------------

def build_source_pool_covost2(n: int, cache_path: Path) -> list[SourceRecord]:
    """Load EN transcripts from fixie-ai/covost2 and cache a JSONL pool.

    For text-only tasks we don't need the audio bytes; we just need the
    transcript and a stable id. audio_path is stored as an HF-cache pointer
    string (not a real filesystem path) so downstream schema has a slot.
    """
    if cache_path.exists():
        pool = [SourceRecord(**json.loads(l)) for l in cache_path.read_text().splitlines()]
        if len(pool) >= n:
            return pool[:n]

    from datasets import Audio, load_dataset

    print(f"[source] loading fixie-ai/covost2 en_de train (target n={n})", flush=True)
    ds = load_dataset("fixie-ai/covost2", "en_de", split="train")
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    pool: list[SourceRecord] = []
    seen_hashes: set[str] = set()
    for i, ex in enumerate(ds):
        if len(pool) >= n:
            break
        txt = (ex.get("sentence") or "").strip()
        if len(txt.split()) < 5:
            continue
        h = hashlib.md5(txt.encode()).hexdigest()[:10]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        pool.append(SourceRecord(
            id=f"covost2_{i:06d}",
            transcript=txt,
            audio_path=f"hf://fixie-ai/covost2/en_de/train/{i}",
            source="covost2",
        ))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(r.__dict__) for r in pool))
    print(f"[source] cached {len(pool)} records to {cache_path}", flush=True)
    return pool


def build_source_pool_ytseg(
    n: int,
    cache_path: Path,
    min_words: int = 100,
    max_words: int = 1500,
) -> list[SourceRecord]:
    """Load chapter-sized English transcript chunks from retkowski/ytseg.

    YTSeg ships `text_ref` as a flat sentence list and `target_binary_ref` as a
    same-length 0/1 string where 1 marks a chapter boundary. We reconstruct
    chapters by splitting the sentence list at each boundary, join sentences
    back to prose, and keep only chunks in [min_words, max_words] so the gist
    task has real compression work to do without blowing the 4096 prompt budget.
    """
    if cache_path.exists():
        pool = [SourceRecord(**json.loads(l)) for l in cache_path.read_text().splitlines()]
        if len(pool) >= n:
            return pool[:n]

    from datasets import load_dataset

    print(f"[source] loading retkowski/ytseg text train (target n={n})", flush=True)
    ds = load_dataset("retkowski/ytseg", "text", split="train")

    pool: list[SourceRecord] = []
    seen_hashes: set[str] = set()
    n_schema_skip = 0
    for ex in ds:
        if len(pool) >= n:
            break
        sentences = ex.get("text_ref") or []
        binary = (ex.get("target_binary_ref") or "").lstrip("|=")
        video_id = ex.get("video_id") or ""
        if not sentences or not binary or len(sentences) != len(binary):
            n_schema_skip += 1
            continue

        # Chapter starts = positions where binary == '1'. Ensure index 0 is
        # treated as the first chapter start even if binary doesn't mark it.
        starts = [i for i, c in enumerate(binary) if c == "1"]
        if not starts or starts[0] != 0:
            starts = [0] + starts
        boundaries = starts + [len(sentences)]

        for k in range(len(boundaries) - 1):
            chunk_sents = sentences[boundaries[k]:boundaries[k + 1]]
            chunk_text = " ".join(s.strip() for s in chunk_sents if s and s.strip()).strip()
            chunk_text = re.sub(r"\s+", " ", chunk_text)
            if not chunk_text:
                continue
            word_count = len(chunk_text.split())
            if word_count < min_words or word_count > max_words:
                continue
            h = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pool.append(SourceRecord(
                id=f"ytseg_{video_id}_{k:02d}",
                transcript=chunk_text,
                audio_path=None,
                source="ytseg",
            ))
            if len(pool) >= n:
                break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(r.__dict__) for r in pool))
    print(f"[source] cached {len(pool)} ytseg chapter chunks to {cache_path} "
          f"(schema-skipped {n_schema_skip} videos)", flush=True)
    return pool


_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    "fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|"
    "half|quarter|third|percent"
)
_T5_NUMERIC_RE = re.compile(r"\d|\b(?:" + _NUMBER_WORDS + r")\b", re.IGNORECASE)
# CJK Unified Ideographs — used to sanity-check Chinese-language outputs.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def filter_for_task(task_id: str, records: list[SourceRecord]) -> list[SourceRecord]:
    if task_id == "T5":
        return [r for r in records if _T5_NUMERIC_RE.search(r.transcript)]
    return records


def shard(records: list, replica_id: int, num_replicas: int) -> list:
    return records[replica_id::num_replicas]


# -------- Audio pool ---------------------------------------------------------

@dataclass
class AudioSample:
    rec: SourceRecord
    audio: "np.ndarray"  # type: ignore[name-defined]
    sampling_rate: int


def _decode_mp3_bytes_pyav(mp3_bytes: bytes, target_sr: int = 16000):
    """Decode MP3 bytes to mono float32 numpy at target sample rate using PyAV.

    PyAV bundles ffmpeg in its wheels, so this works on hosts without system
    ffmpeg. We explicitly resample + downmix to mono since the Gemma-4 audio
    encoder expects 16 kHz mono.
    """
    import io as _io
    import av
    import numpy as np

    buf = _io.BytesIO(mp3_bytes)
    with av.open(buf) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream in MP3 bytes.")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                chunks.append(out_frame.to_ndarray())
        # flush
        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray())
    if not chunks:
        return np.zeros((0,), dtype=np.float32), target_sr
    arr = np.concatenate(chunks, axis=-1)
    # AudioResampler returns shape (1, n_samples) for mono fltp
    if arr.ndim == 2:
        arr = arr.squeeze(0)
    return arr.astype(np.float32, copy=False), target_sr


def build_audio_pool_covost2(n: int) -> list[AudioSample]:
    """Decode N (audio, transcript) pairs from CoVoST2 en_de train at 16kHz.

    CoVoST2 stores audio as MP3 bytes. We pass `Audio(decode=False)` to
    keep HF datasets from trying torchcodec (not installable without system
    ffmpeg on this cluster), then decode each MP3 ourselves via PyAV.
    """
    from datasets import Audio, load_dataset

    print(f"[audio_source] decoding fixie-ai/covost2 en_de train (target n={n})", flush=True)
    ds = load_dataset("fixie-ai/covost2", "en_de", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: list[AudioSample] = []
    seen: set[str] = set()
    n_fail = 0
    for i, ex in enumerate(ds):
        if len(samples) >= n:
            break
        txt = (ex.get("sentence") or "").strip()
        if len(txt.split()) < 5:
            continue
        h = hashlib.md5(txt.encode()).hexdigest()[:10]
        if h in seen:
            continue
        seen.add(h)
        audio_blob = ex["audio"]
        try:
            arr, sr = _decode_mp3_bytes_pyav(audio_blob["bytes"], target_sr=16000)
        except Exception as exc:
            n_fail += 1
            if n_fail <= 3:
                print(f"[audio_source] decode failed for idx={i}: {exc}", flush=True)
            continue
        if arr.size < 16000 // 2:  # < 0.5s
            continue
        samples.append(AudioSample(
            rec=SourceRecord(
                id=f"covost2_{i:06d}",
                transcript=txt,
                audio_path=f"hf://fixie-ai/covost2/en_de/train/{i}",
                source="covost2",
            ),
            audio=arr,
            sampling_rate=sr,
        ))
    print(f"[audio_source] decoded {len(samples)} audio records "
          f"(skipped {n_fail} decode failures)", flush=True)
    return samples


def render_audio_chat_prompt(tokenizer, prompt_text: str) -> str:
    """Render the Gemma-4 chat template for a single-turn [audio + text] user
    message, leaving the `<|audio|>` placeholder in place for vLLM to expand."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio"},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


# -------- Prompt loading -----------------------------------------------------

def load_prompt(prompts_dir: Path, stem: str, variant: str) -> str:
    p = prompts_dir / f"{stem}_{variant}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing prompt template: {p}")
    return p.read_text()


def clean_transcript(raw: str) -> str:
    t = raw.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    t = t.replace('""', '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fill(template: str, *, transcript: str) -> str:
    return template.replace("{{TRANSCRIPT}}", clean_transcript(transcript))


# -------- Filtering ----------------------------------------------------------

_HALLUCINATION_MARKERS = ("\nTRANSCRIPT:", "\nExample ", "\nDemonstration", "\n\n")


def extract_answer(raw: str) -> str:
    """Trim raw output at the first hallucinated new-demo marker, then return
    the first non-empty line. Handles the base-model failure mode where the
    model emits a blank line and starts fabricating a new few-shot example."""
    if not raw:
        return ""
    trimmed = raw
    for marker in _HALLUCINATION_MARKERS:
        idx = trimmed.find(marker)
        if idx >= 0:
            trimmed = trimmed[:idx]
    for line in trimmed.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def soft_normalize(text: str) -> str:
    t = extract_answer(text)
    t = t.strip().strip("`\"'")
    t = re.sub(r"\s+", " ", t)
    return t


def reject_reasons(task_id: str, text: str) -> list[str]:
    """Return a list of rejection reasons for a normalized generation.

    Hard rule from the user: no empty / NONE / semantically-null rows for
    any task. We drop on structure (invalid JSON, wrong scaffold) and on
    content (empty entity lists, missing Chinese characters, word count
    wildly outside the task spec).
    """
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped or len(stripped) < 3:
        reasons.append("too_short")
    if stripped.upper() == "NONE":
        reasons.append("none_output")
    if REFUSAL_PATTERNS.search(text):
        reasons.append("refusal")
    if THINK_PATTERNS.search(text):
        reasons.append("reasoning_leak")

    # Structural/content checks only make sense if we have non-empty text.
    if "none_output" in reasons or "too_short" in reasons:
        return reasons

    word_count = len(stripped.split())

    if task_id == "T1":
        items = [p.strip() for p in stripped.split(",") if p.strip()]
        if len(items) < 2:
            reasons.append("t1_too_few_items")

    elif task_id == "T2":
        try:
            parsed = json.loads(stripped)
        except Exception:
            reasons.append("t2_not_json")
        else:
            if not isinstance(parsed, dict):
                reasons.append("t2_not_object")
            else:
                total_entities = sum(
                    len(v) if isinstance(v, list) else 0 for v in parsed.values()
                )
                if total_entities == 0:
                    reasons.append("t2_empty_entities")

    elif task_id == "T3":
        if word_count < 6 or word_count > 40:
            reasons.append("t3_bad_word_count")

    elif task_id == "T4":
        if word_count < 1 or word_count > 5:
            reasons.append("t4_bad_word_count")

    elif task_id == "T5":
        if "Q:" not in stripped or "A:" not in stripped:
            reasons.append("t5_missing_scaffold")
        else:
            answer_part = stripped.split("A:", 1)[1]
            # Accept digits OR spelled-out numbers ("five", "twenty") —
            # both are valid numeric answers the model will emit at train time.
            if not _T5_NUMERIC_RE.search(answer_part):
                reasons.append("t5_no_number_in_answer")

    elif task_id == "T6":
        if word_count < 6 or word_count > 40:
            reasons.append("t6_bad_word_count")

    elif task_id == "T7":
        if not _CJK_RE.search(stripped):
            reasons.append("t7_no_chinese")
        # One-sentence Chinese gist should be compact. A Chinese char is
        # roughly equivalent to an English word for length purposes, so >120
        # chars = the model is enumerating features instead of compressing.
        cjk_chars = len(_CJK_RE.findall(stripped))
        if cjk_chars > 120:
            reasons.append("t7_too_long")

    elif task_id == "T8":
        if word_count < 5 or word_count > 80:
            reasons.append("t8_bad_length")

    elif task_id == "T9":
        if word_count < 5 or word_count > 80:
            reasons.append("t9_bad_length")

    elif task_id == "T10":
        if not _CJK_RE.search(stripped):
            reasons.append("t10_no_chinese")

    return reasons


# -------- Main loop ----------------------------------------------------------

INITIAL_BUFFER = 1.5   # first pass oversamples by 50% to absorb NONE/refusal rejects
REFILL_BUFFER = 3.0    # refills use a larger cushion — still-needed * 3


def run_task(
    *,
    llm: LLM,
    task_id: str,
    prompts_dir: Path,
    source_pool: list[SourceRecord],
    samples_per_task: int,
    replica_id: int,
    num_replicas: int,
    output_dir: Path,
    source_model_tag: str,
) -> int:
    """Generate until we have `target_per_replica` kept rows, refilling from
    the shard when rejects eat into yield. Pool must be pre-sized with buffer
    (see BUFFER_FACTOR in main)."""
    spec = TEXT_TASKS[task_id]
    eligible = filter_for_task(task_id, source_pool)
    if len(eligible) != len(source_pool):
        print(f"[{task_id}] filtered source pool: {len(source_pool)} -> {len(eligible)} "
              f"records eligible for this task", flush=True)
    my_records = shard(eligible, replica_id, num_replicas)
    target_per_replica = samples_per_task // num_replicas

    tmpl_a = load_prompt(prompts_dir, spec["prompt_stem"], "A")
    tmpl_b = load_prompt(prompts_dir, spec["prompt_stem"], "B")

    out_subdir = output_dir / f"{task_id}_{spec['name']}" / source_model_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"part-{replica_id}.jsonl"
    # Truncate outputs at start so oversample loop can append cleanly.
    out_path.open("w").close()
    out_path.with_suffix(out_path.suffix + ".rejects").open("w").close()

    stop_strings = ["\nTRANSCRIPT:", "\nExample ", "\nDemonstration"]
    sampling_temp07 = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=spec["max_tokens"],
                                     seed=1234, stop=stop_strings)
    sampling_temp09 = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=spec["max_tokens"],
                                     seed=1235, stop=stop_strings)

    total_kept = 0
    total_rejected = 0
    cursor = 0
    chunk_idx = 0
    while total_kept < target_per_replica and cursor < len(my_records):
        remaining = target_per_replica - total_kept
        factor = INITIAL_BUFFER if chunk_idx == 0 else REFILL_BUFFER
        chunk_size = min(int(math.ceil(remaining * factor)), len(my_records) - cursor)
        chunk_recs = my_records[cursor:cursor + chunk_size]
        cursor += chunk_size

        prompts_batch: list[str] = []
        meta_batch: list[dict] = []
        sampling_pairs: list[SamplingParams] = []
        for j, rec in enumerate(chunk_recs):
            global_i = total_kept + total_rejected + j
            variant = "A" if global_i % 2 == 0 else "B"
            template = tmpl_a if variant == "A" else tmpl_b
            user_text = fill(template, transcript=rec.transcript).rstrip()
            prompts_batch.append(user_text)
            temperature = 0.7 if variant == "A" else 0.9
            meta_batch.append({"rec": rec, "variant": variant, "temperature": temperature})
            sampling_pairs.append(sampling_temp07 if variant == "A" else sampling_temp09)

        t0 = time.time()
        outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_pairs, use_tqdm=True)
        dt = time.time() - t0
        print(f"[{task_id}] replica {replica_id}: chunk {chunk_idx} generated "
              f"{len(outputs)} in {dt:.1f}s ({len(outputs)/max(dt,1e-3):.1f} samples/s)",
              flush=True)

        # Retry pass for outputs that emitted no content (base-model early-stop).
        retry_idxs = [i for i, o in enumerate(outputs) if not extract_answer(o.outputs[0].text)]
        if retry_idxs:
            retry_sampling = SamplingParams(
                temperature=1.0, top_p=0.95, max_tokens=spec["max_tokens"],
                seed=4242 + chunk_idx, stop=stop_strings,
            )
            retry_prompts = [prompts_batch[i] + " " for i in retry_idxs]
            t1 = time.time()
            retry_outs = llm.generate(prompts=retry_prompts, sampling_params=retry_sampling, use_tqdm=False)
            print(f"[{task_id}] replica {replica_id}: retried {len(retry_idxs)} empty outputs "
                  f"in {time.time() - t1:.1f}s", flush=True)
            for slot, new_out in zip(retry_idxs, retry_outs):
                if extract_answer(new_out.outputs[0].text):
                    outputs[slot] = new_out

        n_kept, n_rejected = _write_outputs(
            task_id=task_id, spec=spec, outputs=outputs, meta_batch=meta_batch,
            out_path=out_path, replica_id=replica_id, source_model_tag=source_model_tag,
            id_offset=total_kept + total_rejected, append=True,
        )
        total_kept += n_kept
        total_rejected += n_rejected
        chunk_idx += 1

    if total_kept < target_per_replica:
        print(f"[{task_id}] replica {replica_id}: WARN pool exhausted with "
              f"{total_kept}/{target_per_replica} kept (rejected {total_rejected}, "
              f"shard {len(my_records)}). Increase --source-pool-size.", flush=True)
    else:
        print(f"[{task_id}] replica {replica_id}: kept {total_kept} "
              f"(target {target_per_replica}) | rejected {total_rejected} "
              f"| {chunk_idx} chunk(s) | written to {out_path}", flush=True)
    return total_kept


def _write_outputs(
    *,
    task_id: str,
    spec: dict,
    outputs,
    meta_batch: list[dict],
    out_path: Path,
    replica_id: int,
    source_model_tag: str,
    id_offset: int = 0,
    append: bool = False,
) -> tuple[int, int]:
    """Write kept rows to `out_path` and rejected rows to `<out_path>.rejects`.

    Only rows that pass reject_reasons land in the main JSONL — the sidecar
    preserves the audit trail without contaminating the training data. Call
    with `append=True` across chunks so the oversample loop can refill without
    truncating previous writes; `id_offset` keeps row ids monotonic.
    """
    rejects_path = out_path.with_suffix(out_path.suffix + ".rejects")
    mode = "a" if append else "w"
    n_written = 0
    n_rejected = 0
    reject_counts: dict[str, int] = {}
    with out_path.open(mode) as fh, rejects_path.open(mode) as rej_fh:
        for out, meta in zip(outputs, meta_batch):
            raw = out.outputs[0].text
            finish_reason = getattr(out.outputs[0], "finish_reason", None)
            stop_reason = getattr(out.outputs[0], "stop_reason", None)
            n_tokens = len(getattr(out.outputs[0], "token_ids", []))
            norm = soft_normalize(raw)
            reasons = reject_reasons(task_id, norm)
            idx = id_offset + n_written + n_rejected
            row = {
                "id": f"synth_{task_id}_r{replica_id}_{idx:06d}",
                "task": f"{task_id}_{spec['name']}",
                "source_model": source_model_tag,
                "prompt_variant": meta["variant"],
                "decoding": {"temperature": meta["temperature"], "top_p": 0.9, "max_tokens": spec["max_tokens"]},
                "source": meta["rec"].source,
                "audio_path": meta["rec"].audio_path,
                "transcript": meta["rec"].transcript,
                "answer": norm,
                "raw_answer": raw,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "n_output_tokens": n_tokens,
                "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "audit": {"rejected": bool(reasons), "filter_reasons": reasons},
            }
            if reasons:
                rej_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rejected += 1
                for r in reasons:
                    reject_counts[r] = reject_counts.get(r, 0) + 1
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
    if reject_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(reject_counts.items()))
        print(f"[{task_id}] reject breakdown: {summary}", flush=True)
    return n_written, n_rejected


def run_audio_task(
    *,
    llm: LLM,
    task_id: str,
    prompts_dir: Path,
    audio_pool: list[AudioSample],
    samples_per_task: int,
    replica_id: int,
    num_replicas: int,
    output_dir: Path,
    source_model_tag: str,
) -> int:
    """Audio-modality twin of run_task — oversamples + refills from the shard
    until target_per_replica kept rows are written."""
    spec = AUDIO_TASKS[task_id]
    my_samples = shard(audio_pool, replica_id, num_replicas)
    target_per_replica = samples_per_task // num_replicas

    tmpl_a = load_prompt(prompts_dir, spec["prompt_stem"], "A").strip()
    tmpl_b = load_prompt(prompts_dir, spec["prompt_stem"], "B").strip()

    tokenizer = llm.get_tokenizer()
    prompt_a = render_audio_chat_prompt(tokenizer, tmpl_a)
    prompt_b = render_audio_chat_prompt(tokenizer, tmpl_b)

    out_subdir = output_dir / f"{task_id}_{spec['name']}" / source_model_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"part-{replica_id}.jsonl"
    out_path.open("w").close()
    out_path.with_suffix(out_path.suffix + ".rejects").open("w").close()

    sampling_temp07 = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=spec["max_tokens"], seed=1234)
    sampling_temp09 = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=spec["max_tokens"], seed=1235)

    total_kept = 0
    total_rejected = 0
    cursor = 0
    chunk_idx = 0
    while total_kept < target_per_replica and cursor < len(my_samples):
        remaining = target_per_replica - total_kept
        factor = INITIAL_BUFFER if chunk_idx == 0 else REFILL_BUFFER
        chunk_size = min(int(math.ceil(remaining * factor)), len(my_samples) - cursor)
        chunk_samples = my_samples[cursor:cursor + chunk_size]
        cursor += chunk_size

        prompts_batch: list[dict] = []
        meta_batch: list[dict] = []
        sampling_pairs: list[SamplingParams] = []
        for j, sample in enumerate(chunk_samples):
            global_i = total_kept + total_rejected + j
            variant = "A" if global_i % 2 == 0 else "B"
            rendered = prompt_a if variant == "A" else prompt_b
            prompts_batch.append({
                "prompt": rendered,
                "multi_modal_data": {"audio": (sample.audio, sample.sampling_rate)},
            })
            meta_batch.append({
                "rec": sample.rec,
                "variant": variant,
                "temperature": 0.7 if variant == "A" else 0.9,
            })
            sampling_pairs.append(sampling_temp07 if variant == "A" else sampling_temp09)

        t0 = time.time()
        outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_pairs, use_tqdm=True)
        dt = time.time() - t0
        print(f"[{task_id}] replica {replica_id}: chunk {chunk_idx} generated "
              f"{len(outputs)} in {dt:.1f}s ({len(outputs)/max(dt,1e-3):.2f} samples/s)",
              flush=True)

        n_kept, n_rejected = _write_outputs(
            task_id=task_id, spec=spec, outputs=outputs, meta_batch=meta_batch,
            out_path=out_path, replica_id=replica_id, source_model_tag=source_model_tag,
            id_offset=total_kept + total_rejected, append=True,
        )
        total_kept += n_kept
        total_rejected += n_rejected
        chunk_idx += 1

    if total_kept < target_per_replica:
        print(f"[{task_id}] replica {replica_id}: WARN pool exhausted with "
              f"{total_kept}/{target_per_replica} kept (rejected {total_rejected}, "
              f"shard {len(my_samples)}). Increase --source-pool-size.", flush=True)
    else:
        print(f"[{task_id}] replica {replica_id}: kept {total_kept} "
              f"(target {target_per_replica}) | rejected {total_rejected} "
              f"| {chunk_idx} chunk(s) | written to {out_path}", flush=True)
    return total_kept


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True, help="Comma-separated task ids, e.g. T1,T2")
    ap.add_argument("--modality", choices=["text", "audio"], required=True)
    ap.add_argument("--samples-per-task", type=int, required=True)
    ap.add_argument("--replica-id", type=int, required=True)
    ap.add_argument("--num-replicas", type=int, required=True)
    ap.add_argument("--source-model-tag", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--source-pool-cache", default=None,
                    help="JSONL cache of source records; defaults to data/synthetic/source_pool.jsonl")
    ap.add_argument("--source-pool-size", type=int, default=0,
                    help="How many (audio, transcript) pairs to pre-sample from CoVoST2. "
                         "Defaults to samples-per-task (so each task pulls enough).")
    ap.add_argument("--source", choices=["covost2", "ytseg"], default="covost2",
                    help="Which corpus to draw transcripts from. covost2 = short single "
                         "sentences; ytseg = YouTube chapter chunks (100-1500 words each) "
                         "for gist tasks that need real compression signal.")
    return ap.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.modality == "text":
        for t in tasks:
            if t not in TEXT_TASKS:
                raise SystemExit(f"Unknown text task: {t}. Known: {list(TEXT_TASKS)}")
    else:
        for t in tasks:
            if t not in AUDIO_TASKS:
                raise SystemExit(f"Unknown audio task: {t}. Known: {list(AUDIO_TASKS)}")

    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)

    # Pool is oversized so every task has headroom to refill after NONE/refusal
    # rejects without exhausting its shard. When T5 is in the task list we need
    # a MUCH larger pool: empirically only ~10% of CoVoST2 transcripts contain
    # numbers (the T5 filter), and ~45% of those still produce NONE — so T5's
    # usable yield is ~5% of pool. Non-T5 tasks need ~3x at worst.
    POOL_BUFFER_DEFAULT = 3
    POOL_BUFFER_T5 = 20  # 1/(0.10 * 0.55) ≈ 18, rounded up for safety
    needs_t5 = "T5" in tasks
    buffer = POOL_BUFFER_T5 if needs_t5 and args.modality == "text" else POOL_BUFFER_DEFAULT
    if args.modality == "text":
        default_cache_name = "source_pool.jsonl" if args.source == "covost2" else f"source_pool_{args.source}.jsonl"
        cache = Path(args.source_pool_cache or (output_dir / default_cache_name))
        pool_size = args.source_pool_size or args.samples_per_task * buffer
        if needs_t5:
            print(f"[init] T5 is in tasks — bumping pool to {pool_size} (~{buffer}x) "
                  f"to compensate for numeric-filter + NONE attrition", flush=True)
        if args.source == "ytseg":
            source_pool = build_source_pool_ytseg(n=pool_size, cache_path=cache)
        else:
            source_pool = build_source_pool_covost2(n=pool_size, cache_path=cache)
        audio_pool: list[AudioSample] = []
    else:
        pool_size = args.source_pool_size or args.samples_per_task * buffer
        audio_pool = build_audio_pool_covost2(n=pool_size)
        source_pool = []

    print(f"[init] loading vLLM model {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print("[init] vLLM loaded", flush=True)

    total = 0
    for task_id in tasks:
        if args.modality == "text":
            total += run_task(
                llm=llm,
                task_id=task_id,
                prompts_dir=prompts_dir,
                source_pool=source_pool,
                samples_per_task=args.samples_per_task,
                replica_id=args.replica_id,
                num_replicas=args.num_replicas,
                output_dir=output_dir,
                source_model_tag=args.source_model_tag,
            )
        else:
            total += run_audio_task(
                llm=llm,
                task_id=task_id,
                prompts_dir=prompts_dir,
                audio_pool=audio_pool,
                samples_per_task=args.samples_per_task,
                replica_id=args.replica_id,
                num_replicas=args.num_replicas,
                output_dir=output_dir,
                source_model_tag=args.source_model_tag,
            )

    print(f"[done] replica {args.replica_id} kept {total} samples across {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
