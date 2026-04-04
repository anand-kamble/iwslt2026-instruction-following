"""Evaluate a Stage 2 (LoRA-only) checkpoint using the base LLM + LoRA, no projector or audio.

This is a text-only evaluation: MCIF prompts are fed as plain text without
audio input. Useful for measuring the LoRA-adapted LLM's instruction-following
ability in isolation.

Usage:
    python src/evaluate_lora.py --config configs/stage1_alignment.yaml \
        --checkpoint ./outputs/stage2_text_lora
    python src/evaluate_lora.py --config configs/stage1_alignment.yaml \
        --checkpoint ./outputs/stage2_text_lora --languages en de
"""

import argparse
import json
import os
import subprocess

import torch
import yaml

from src.model.build_model import load_llm_from_checkpoint, get_device
from src.data.utils import format_text_only_messages


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


def run_inference_on_mcif(llm, tokenizer, task: str, language: str, track: str = "short"):
    """Run text-only inference on MCIF validation set and write output XML."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    config_name = f"{track}_fixedprompt"
    ds = load_dataset("FBK-MT/MCIF", config_name, split="test")

    prompt_key = f"prompt_{language}"
    samples = []
    total = len(ds)
    device = next(llm.parameters()).device

    for i, example in enumerate(ds):
        if i % 10 == 0:
            print(f"  [{i}/{total}] Processing...", flush=True)
        instruction = example[prompt_key]

        # Text-only: no speech placeholder, no audio
        messages = format_text_only_messages(instruction)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            output_ids = llm.generate(
                **text_inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Strip the input prompt — generate() returns full sequence with input_ids
        new_tokens = output_ids[0][text_inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        samples.append(OutputSample(id=example["id"], value=output_text))

    # Write output XML
    output_path = f"outputs/eval_lora_{task}_{language}_{track}.xml"
    os.makedirs("outputs", exist_ok=True)
    write_output(samples, track=track, language=language, output_name=task, output=output_path)

    return output_path


def run_mcif_eval(output_path: str, track: str, language: str):
    """Run MCIF evaluation on the output file."""
    cmd = ["mcif_eval", "-t", track, "-l", language, "-s", output_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Evaluation error: {result.stderr}")
        return {}

    try:
        parsed = json.loads(result.stdout)
        if parsed.get("state") == "OK":
            print(f"Scores for {language}:")
            for metric, score in parsed["scores"].items():
                print(f"  {metric}: {score:.4f}")
            return parsed["scores"]
        else:
            print(f"Evaluation failed: {parsed.get('reason', 'unknown')}")
            return {}
    except json.JSONDecodeError:
        print(result.stdout)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 LoRA checkpoint (text-only, no audio)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 2 checkpoint dir (must contain lora_adapters/)")
    parser.add_argument("--languages", nargs="+", default=["en", "de", "it", "zh"])
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config.get("task", "multitask")
    print(f"Evaluating LoRA-only (text, no audio) | task={task} track={args.track}")

    # Load model (base LLM + LoRA, no projector, no speech encoder)
    llm, tokenizer = load_lora_checkpoint(args.checkpoint, config)

    # Run evaluation for each language
    all_scores = {}
    for language in args.languages:
        print(f"\n=== Evaluating language={language} ===")
        output_path = run_inference_on_mcif(llm, tokenizer, task, language, args.track)
        print(f"Output written to: {output_path}")

        scores = run_mcif_eval(output_path, args.track, language)
        all_scores[language] = scores

    # Summary
    print("\n=== Summary ===")
    for lang, scores in all_scores.items():
        print(f"  {lang}: {scores}")


if __name__ == "__main__":
    main()
