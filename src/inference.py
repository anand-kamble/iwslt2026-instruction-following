"""Inference and submission generation for IWSLT 2026.

Usage:
    python src/inference.py --checkpoint ./outputs/asr --input_dir ./test_data --output_dir ./submission
"""

import argparse
import os
from pathlib import Path

import torch
import yaml

from src.data.utils import load_audio, format_chat_messages, TASK_PROMPTS
from src.evaluate import load_checkpoint


def run_inference(
    model,
    processor,
    tokenizer,
    audio_path: str,
    instruction: str,
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a single audio + instruction pair."""
    # Load and process audio
    waveform, sr = load_audio(audio_path)
    audio_inputs = processor(
        audios=waveform.squeeze(0).numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    audio_inputs = {k: v.to(model.llm.device) for k, v in audio_inputs.items()}

    # Tokenize instruction
    messages = format_chat_messages(instruction)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_inputs = tokenizer(text, return_tensors="pt").to(model.llm.device)

    # Generate
    output_ids = model.generate(
        input_features=audio_inputs["input_features"],
        input_features_attention_mask=audio_inputs["attention_mask"],
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def process_submission_input(
    model,
    processor,
    tokenizer,
    input_dir: str,
    output_dir: str,
    task: str,
    language: str = "en",
):
    """Process all inputs in a directory and generate MCIF-formatted output."""
    from mcif.io import OutputSample, write_output

    input_path = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    audio_files = sorted(input_path.glob("*.wav"))

    for i, audio_file in enumerate(audio_files):
        instruction = TASK_PROMPTS.get(task, {}).get(language, "")

        # Check for per-sample instructions (e.g., SQA questions)
        instruction_file = audio_file.with_suffix(".txt")
        if instruction_file.exists():
            instruction = instruction_file.read_text().strip()

        print(f"[{i+1}/{len(audio_files)}] {audio_file.name}")
        output_text = run_inference(model, processor, tokenizer, str(audio_file), instruction)
        samples.append(OutputSample(id=i, value=output_text))

    # Determine track from audio length heuristic
    track = "short"  # default; override with --track if needed

    output_file = os.path.join(output_dir, f"{task}_{language}.xml")
    write_output(samples, track=track, language=language, output_name=task, output=output_file)
    print(f"Submission output: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run inference for submission")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["asr", "s2tt", "sqa", "s2tsum", "achap"])
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, processor, tokenizer = load_checkpoint(args.checkpoint)

    process_submission_input(
        model, processor, tokenizer,
        args.input_dir, args.output_dir,
        args.task, args.language,
    )


if __name__ == "__main__":
    main()
