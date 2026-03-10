"""Evaluation entry point using MCIF evaluation tools.

Usage:
    python src/evaluate.py --config configs/asr.yaml --checkpoint ./outputs/asr
    python src/evaluate.py --config configs/asr.yaml --checkpoint ./outputs/asr --language en
"""

import argparse
import os
import subprocess

import torch
import yaml

from src.model.build_model import load_speech_encoder, build_projector, SPEECH_DIM, QWEN3_4B_DIM
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load a trained SpeechLLM from a checkpoint directory."""
    from unsloth import FastLanguageModel

    # Load speech encoder
    speech_encoder, processor = load_speech_encoder(device=device)

    # Load LLM with LoRA adapters
    lora_path = os.path.join(checkpoint_dir, "lora_adapters")
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(llm)

    # Load projector
    projector = build_projector(device=device)
    projector_path = os.path.join(checkpoint_dir, "projector.pt")
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    projector.eval()

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)
    return model, processor, tokenizer


def run_inference_on_mcif(model, processor, tokenizer, task: str, language: str, track: str = "short"):
    """Run inference on MCIF validation set and write output XML."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    # Load MCIF dataset
    config_name = f"{track}_fixedprompt"
    ds = load_dataset("FBK-MT/MCIF", config_name, split="test")

    prompt_key = f"prompt_{language}"
    samples = []

    for example in ds:
        instruction = example[prompt_key]

        # Process audio
        audio_inputs = processor(
            audios=example["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        audio_inputs = {k: v.to(model.llm.device) for k, v in audio_inputs.items()}

        # Tokenize instruction (with speech placeholder)
        messages = format_chat_messages(instruction)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = tokenizer(text, return_tensors="pt").to(model.llm.device)

        # Generate
        output_ids = model.generate(
            input_features=audio_inputs["input_features"],
            input_features_attention_mask=audio_inputs["attention_mask"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        samples.append(OutputSample(id=example["id"], value=output_text))

    # Write output XML
    output_path = f"outputs/eval_{task}_{language}_{track}.xml"
    os.makedirs("outputs", exist_ok=True)
    write_output(samples, track=track, language=language, output_name=task, output=output_path)

    return output_path


def run_mcif_eval(output_path: str, track: str, language: str):
    """Run MCIF evaluation on the output file."""
    cmd = ["mcif_eval", "-t", track, "-l", language, "-s", output_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Evaluation error: {result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on IWSLT tasks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config["task"]
    print(f"Evaluating task={task} language={args.language} track={args.track}")

    # Load model
    model, processor, tokenizer = load_checkpoint(args.checkpoint)

    # Run inference
    output_path = run_inference_on_mcif(
        model, processor, tokenizer, task, args.language, args.track
    )
    print(f"Output written to: {output_path}")

    # Run evaluation
    print("\n--- Evaluation Results ---")
    run_mcif_eval(output_path, args.track, args.language)


if __name__ == "__main__":
    main()
