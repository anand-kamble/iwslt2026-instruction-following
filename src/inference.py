"""Inference and submission generation for IWSLT 2026.

Usage:
    # Single file inference
    python src/inference.py --checkpoint ./outputs/stage3 --input_dir ./test_data --output_dir ./submission --task asr

    # MCIF submission (all languages)
    python src/inference.py --checkpoint ./outputs/stage3 --mcif --output_dir ./submission --languages en de it zh
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.utils import load_audio, format_chat_messages, TASK_PROMPTS
from src.evaluate import load_checkpoint


def run_inference(
    model,
    processor,
    tokenizer,
    audio_path: str = None,
    instruction: str = "",
    audio_array=None,
    sample_rate: int = 16000,
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a single audio + instruction pair.

    Accepts either audio_path (loads from file) or audio_array (numpy array).
    """
    if audio_array is not None:
        audio_inputs = processor(
            audios=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
    elif audio_path is not None:
        waveform, sr = load_audio(audio_path)
        audio_inputs = processor(
            audios=waveform.squeeze(0).numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )
    else:
        raise ValueError("Either audio_path or audio_array must be provided")

    audio_inputs = {k: v.to(model.llm.device) for k, v in audio_inputs.items()}

    # Tokenize instruction
    messages = format_chat_messages(instruction)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_inputs = tokenizer(text, return_tensors="pt").to(model.llm.device)

    # Generate (greedy decoding for deterministic output)
    output_ids = model.generate(
        input_features=audio_inputs["input_features"],
        input_features_attention_mask=audio_inputs["attention_mask"],
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_mcif_submission(model, processor, tokenizer, output_dir, languages, track="short"):
    """Generate submission XML files for all languages from MCIF dataset."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:
        print(f"\n=== Generating {track}/{lang} ===")
        config_name = f"{track}_fixedprompt"
        ds = load_dataset("FBK-MT/MCIF", config_name, split="test")
        prompt_key = f"prompt_{lang}"

        samples = []
        for example in tqdm(ds, desc=f"{lang}"):
            instruction = example[prompt_key]
            output_text = run_inference(
                model, processor, tokenizer,
                instruction=instruction,
                audio_array=example["audio"]["array"],
                sample_rate=example["audio"].get("sampling_rate", 16000),
                max_new_tokens=512,
            )
            samples.append(OutputSample(id=example["id"], value=output_text))

        output_path = os.path.join(output_dir, f"{track}_{lang}.xml")
        write_output(samples, track=track, language=lang,
                     output_name="iwslt2026", output=output_path)
        print(f"Written: {output_path} ({len(samples)} samples)")


def process_submission_input(
    model, processor, tokenizer,
    input_dir: str, output_dir: str,
    task: str, language: str = "en",
):
    """Process all inputs in a directory and generate MCIF-formatted output."""
    from mcif.io import OutputSample, write_output

    input_path = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    audio_files = sorted(input_path.glob("*.wav"))

    for i, audio_file in enumerate(audio_files):
        # Per-sample instruction file takes priority
        instruction_file = audio_file.with_suffix(".txt")
        if instruction_file.exists():
            instruction = instruction_file.read_text().strip()
        elif task == "custom":
            raise FileNotFoundError(
                f"Custom/surprise task requires per-sample instruction file: {instruction_file}"
            )
        else:
            instruction = TASK_PROMPTS.get(task, {}).get(language, "")

        print(f"[{i+1}/{len(audio_files)}] {audio_file.name}")
        output_text = run_inference(
            model, processor, tokenizer,
            audio_path=str(audio_file), instruction=instruction,
        )
        samples.append(OutputSample(id=i, value=output_text))

    track = "short"
    output_file = os.path.join(output_dir, f"{task}_{language}.xml")
    write_output(samples, track=track, language=language, output_name=task, output=output_file)
    print(f"Submission output: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run inference for submission")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Config YAML for model params")
    parser.add_argument("--mcif", action="store_true", help="Generate submission from MCIF dataset")
    parser.add_argument("--languages", nargs="+", default=["en", "de", "it", "zh"])
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="asr",
                        choices=["asr", "s2tt", "sqa", "s2tsum", "achap", "custom"])
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, processor, tokenizer = load_checkpoint(args.checkpoint, config)

    if args.mcif:
        run_mcif_submission(
            model, processor, tokenizer,
            args.output_dir, args.languages, args.track,
        )
    elif args.input_dir:
        process_submission_input(
            model, processor, tokenizer,
            args.input_dir, args.output_dir,
            args.task, args.language,
        )
    else:
        print("Specify --mcif for MCIF submission or --input_dir for file-based inference")


if __name__ == "__main__":
    main()
