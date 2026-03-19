"""Generate synthetic multilingual SQA data by translating LibriSQA.

Uses SeamlessM4T text translation + COMETKiwi quality filtering.

Usage:
    python -m src.data.translate_sqa --output_dir data/sqa_translated --languages de it zh
    python -m src.data.translate_sqa --output_dir data/sqa_translated --languages de --comet_threshold 0.85
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


def filter_with_cometkiwi(sources, translations, threshold=0.85):
    """Filter translations using COMETKiwi (reference-free quality estimation).

    Returns indices of pairs that pass the quality threshold.
    """
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    model.eval()

    data = [{"src": src, "mt": mt} for src, mt in zip(sources, translations)]
    output = model.predict(data, batch_size=64, gpus=1 if torch.cuda.is_available() else 0)

    scores = output.scores
    passed = [i for i, score in enumerate(scores) if score >= threshold]
    print(f"COMETKiwi filtering: {len(passed)}/{len(scores)} passed (threshold={threshold})")
    return passed, scores


def main():
    parser = argparse.ArgumentParser(description="Translate LibriSQA for multilingual SQA")
    parser.add_argument("--output_dir", type=str, default="data/sqa_translated")
    parser.add_argument("--languages", nargs="+", default=["de", "it", "zh"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--comet_threshold", type=float, default=0.85)
    parser.add_argument("--skip_filtering", action="store_true", help="Skip COMETKiwi filtering")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SeamlessM4T for text translation...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
    model = model.to(args.device)
    model.eval()

    print("Loading LibriSQA...")
    ds = load_dataset("ZihanZhao/LibriSQA", split="partI.train")

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

        # COMETKiwi quality filtering
        if not args.skip_filtering:
            print("Running COMETKiwi quality filtering on questions...")
            passed_q, scores_q = filter_with_cometkiwi(
                questions, translated_questions, threshold=args.comet_threshold
            )
            print("Running COMETKiwi quality filtering on answers...")
            passed_a, scores_a = filter_with_cometkiwi(
                answers, translated_answers, threshold=args.comet_threshold
            )
            # Keep only indices that passed BOTH question and answer filtering
            passed_both = sorted(set(passed_q) & set(passed_a))
            print(f"Pairs passing both filters: {len(passed_both)}/{len(questions)}")
        else:
            passed_both = list(range(len(questions)))

        # Save as JSONL (only filtered pairs)
        output_path = Path(args.output_dir) / f"librisqa_{lang}.jsonl"
        count = 0
        with open(output_path, "w") as f:
            for i in passed_both:
                record = {
                    "id": i,
                    "question": translated_questions[i],
                    "answer": translated_answers[i],
                    "original_question": questions[i],
                    "original_answer": answers[i],
                    "language": lang,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"Saved {count} translated pairs to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
