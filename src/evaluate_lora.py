"""Evaluate a Stage 2 (LoRA-only) checkpoint on text-only MT and SQA tasks.

Following NAVER IWSLT 2025 (arxiv 2506.01808) Stage B evaluation:
- MT: evaluate on CoVoST2 validation split using reference transcripts as source text.
  Metrics: BLEU4 and COMET.
- SQA: evaluate on LibriSQA validation split using reference transcripts.
  Metrics: BERTScore (F1).

No audio is used — reference English transcripts are provided as text input.

Run all commands from this repository root (iwslt2026-instruction-following),
not from a parent folder that may contain another src/ tree with an older script.

Usage (LoRA checkpoint):
    python src/evaluate_lora.py --config configs/stage2_text_lora.yaml \
        --checkpoint ./outputs/stage2_text_lora
    python src/evaluate_lora.py --config configs/stage2_text_lora.yaml \
        --checkpoint ./outputs/stage2_text_lora --tasks mt --mt_langs de zh

Usage (base pretrained LLM only, no lora_adapters/):
    python src/evaluate_lora.py --config configs/stage2_text_lora.yaml --base
"""

import argparse
import os

import torch

from src.model.build_model import load_llm_from_checkpoint, load_llm_with_lora, get_device
from src.config import load_config
from src.data.utils import TASK_PROMPTS, UNANSWERABLE, format_text_only_messages


def load_lora_checkpoint(checkpoint_dir: str, config: dict = None, device: str = None):
    """Load a Stage 2 checkpoint: base LLM + LoRA adapters, no projector."""
    if device is None:
        device = get_device()

    config = config or {}
    training_config = config.get("training", {})
    llm_config = config.get("model", {})

    lora_path = os.path.join(checkpoint_dir, "lora_adapters")
    llm, tokenizer = load_llm_from_checkpoint(
        checkpoint_dir=lora_path,
        base_model_name=llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507"),
        load_in_4bit=training_config.get("load_in_4bit", False),
        load_in_8bit=training_config.get("load_in_8bit", False),
        is_trainable=False,
        device=device,
    )
    llm.eval()

    return llm, tokenizer


def load_base_llm(config: dict, device: str = None):
    """Load the pretrained LLM from the hub with no LoRA (for baseline eval)."""
    if device is None:
        device = get_device()

    training_config = config.get("training", {})
    llm_config = config.get("model", {})
    model_name = llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507")

    llm, tokenizer = load_llm_with_lora(
        model_name=model_name,
        load_in_4bit=training_config.get("load_in_4bit", False),
        load_in_8bit=training_config.get("load_in_8bit", False),
        no_lora=True,
        device=device,
    )
    llm.eval()
    return llm, tokenizer


def evaluate_mt(llm, tokenizer, target_lang: str, max_samples: int = None):
    """Evaluate MT on CoVoST2 validation split with reference transcripts.

    Returns dict with BLEU and COMET scores.
    """
    from datasets import load_dataset
    import sacrebleu

    covost2_configs = {"de": "en_de", "zh": "en_zh-CN"}
    if target_lang not in covost2_configs:
        print(f"  Skipping MT {target_lang}: CoVoST2 doesn't support EN->{target_lang}")
        return {}

    print(f"  Loading CoVoST2 validation ({target_lang})...")
    ds = load_dataset("fixie-ai/covost2", covost2_configs[target_lang], split="validation")
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    instruction = TASK_PROMPTS["s2tt"][target_lang]
    device = next(llm.parameters()).device
    predictions = []
    references = []
    sources = []
    total = len(ds)

    for i, example in enumerate(ds):
        if i % 50 == 0:
            print(f"  [{i}/{total}] Generating...", flush=True)

        source_text = example["sentence"]  # English transcript
        reference = example["translation"]  # Target language reference

        messages = format_text_only_messages(instruction, source_text=source_text)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = llm.generate(
                **text_inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        new_tokens = output_ids[0][text_inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(output_text)
        references.append(reference)
        sources.append(source_text)

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    scores = {"BLEU4": bleu.score}
    print(f"  BLEU4: {bleu.score:.2f}")

    # COMET
    try:
        from comet import download_model, load_from_checkpoint

        print("  Computing COMET...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(sources, predictions, references)
        ]
        comet_output = comet_model.predict(comet_data, batch_size=32, gpus=1 if device == "cuda" else 0)
        comet_score = comet_output.system_score
        scores["COMET"] = comet_score
        print(f"  COMET: {comet_score:.4f}")
    except Exception as e:
        print(f"  COMET unavailable: {e}")

    return scores


def evaluate_sqa(llm, tokenizer, language: str = "en", max_samples: int = None):
    """Evaluate SQA on LibriSQA validation split with reference context.

    Returns dict with BERTScore (F1).
    """
    from src.data.sqa import load_librisqa_source_dataset

    print(f"  Loading LibriSQA validation ({language})...")
    ds = load_librisqa_source_dataset("partI.test")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    device = next(llm.parameters()).device
    predictions = []
    references = []
    total = len(ds)

    for i, example in enumerate(ds):
        if i % 50 == 0:
            print(f"  [{i}/{total}] Generating...", flush=True)

        question = example.get("question", "")
        answer = example.get("answer", "")
        transcript = example.get("text", "")

        if not answer or answer.lower().strip() in ("unanswerable", "not answerable"):
            answer = UNANSWERABLE[language]

        prompt_template = TASK_PROMPTS["sqa"][language]
        instruction = prompt_template.replace("{QUESTION}", question)

        # Provide transcript as source text (reference transcript)
        source_text = transcript if transcript else None
        messages = format_text_only_messages(instruction, source_text=source_text)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = llm.generate(
                **text_inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        new_tokens = output_ids[0][text_inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(output_text)
        references.append(answer)

    # BERTScore
    try:
        from bert_score import score as bert_score

        print("  Computing BERTScore...")
        P, R, F1 = bert_score(predictions, references, lang=language, rescale_with_baseline=True)
        avg_f1 = F1.mean().item()
        scores = {"BERTScore-F1": avg_f1}
        print(f"  BERTScore F1: {avg_f1:.4f}")
    except Exception as e:
        print(f"  BERTScore unavailable: {e}")
        scores = {}

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 LoRA checkpoint (text-only MT + SQA, following NAVER Stage B)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Stage 2 checkpoint dir (must contain lora_adapters/). Not used with --base.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate the pretrained LLM from config (model.llm) with no LoRA weights.",
    )
    parser.add_argument("--tasks", nargs="+", default=["mt", "sqa"],
                        choices=["mt", "sqa"], help="Which tasks to evaluate")
    parser.add_argument("--mt_langs", nargs="+", default=["de", "zh"],
                        help="Target languages for MT evaluation")
    parser.add_argument("--sqa_lang", type=str, default="en",
                        help="Language for SQA evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per eval set (for quick testing)")
    args = parser.parse_args()

    if not args.base and not args.checkpoint:
        parser.error("Provide --checkpoint for a LoRA run, or pass --base for pretrained LLM only.")
    if args.base and args.checkpoint:
        print("Note: --base is set; ignoring --checkpoint and loading hub base weights only.\n")

    config = load_config(args.config)

    print("=" * 60)
    print("Stage 2 LoRA Evaluation (text-only, NAVER Stage B)")
    print("=" * 60)
    if args.base:
        print("Mode: base pretrained LLM (no LoRA)")
        print(f"Model: {config.get('model', {}).get('llm', 'Qwen/Qwen3-4B-Instruct-2507')}")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks}")
    if "mt" in args.tasks:
        print(f"MT languages: {args.mt_langs}")
    if "sqa" in args.tasks:
        print(f"SQA language: {args.sqa_lang}")
    print()

    # Load model
    print("Loading model...")
    if args.base:
        llm, tokenizer = load_base_llm(config)
    else:
        llm, tokenizer = load_lora_checkpoint(args.checkpoint, config)

    all_scores = {}

    # MT evaluation
    if "mt" in args.tasks:
        for lang in args.mt_langs:
            print(f"\n=== MT EN->{lang} ===")
            scores = evaluate_mt(llm, tokenizer, lang, args.max_samples)
            all_scores[f"mt_{lang}"] = scores

    # SQA evaluation
    if "sqa" in args.tasks:
        print(f"\n=== SQA ({args.sqa_lang}) ===")
        scores = evaluate_sqa(llm, tokenizer, args.sqa_lang, args.max_samples)
        all_scores[f"sqa_{args.sqa_lang}"] = scores

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task_name, scores in all_scores.items():
        print(f"  {task_name}: {scores}")


if __name__ == "__main__":
    main()
