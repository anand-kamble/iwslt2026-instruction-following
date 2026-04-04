"""Evaluation entry point using MCIF evaluation tools.

Usage:
    python src/evaluate.py --config configs/stage3_multimodal.yaml --checkpoint ./outputs/stage3
    python src/evaluate.py --config configs/stage3_multimodal.yaml --checkpoint ./outputs/stage3 --languages en de it zh
"""

import argparse
import json
import os
import shutil
import subprocess

import torch
import yaml

from src.model.build_model import load_speech_encoder, build_projector
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_checkpoint(checkpoint_dir: str, config: dict = None, device: str = None):
    """Load a trained SpeechLLM from a checkpoint directory."""
    from src.model.build_model import load_llm_from_checkpoint, get_device

    if device is None:
        device = get_device()

    config = config or {}
    training_config = config.get("training", {})
    llm_config = config.get("model", {})

    # Load speech encoder
    speech_encoder, processor = load_speech_encoder(device=device)

    # Load LLM with LoRA adapters
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

    # Load projector with correct type and downsample factor
    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 3),
        projector_type=training_config.get("projector_type", "transformer"),
        device=device,
    )
    projector_path = os.path.join(checkpoint_dir, "projector.pt")
    projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
    projector.eval()

    # Ensure speech placeholder token is in tokenizer
    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER])
        llm.resize_token_embeddings(len(tokenizer))

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)
    return model, processor, tokenizer


def check_mwer_segmenter():
    """Check if mWERSegmenter is available (required for S2TT COMET scoring)."""
    if shutil.which("mwerSegmenter") is None and os.getenv("MWERSEGMENTER_ROOT") is None:
        print(
            "WARNING: mWERSegmenter not found. S2TT COMET scoring will fail.\n"
            "Install from: https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz\n"
            "Set MWERSEGMENTER_ROOT or add to PATH."
        )
        return False
    return True


def run_inference_on_mcif(model, processor, tokenizer, task: str, language: str, track: str = "short"):
    """Run inference on MCIF validation set and write output XML."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

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

        # Generate (greedy decoding for deterministic eval)
        output_ids = model.generate(
            input_features=audio_inputs["input_features"],
            input_features_attention_mask=audio_inputs["attention_mask"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
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
    parser = argparse.ArgumentParser(description="Evaluate model on IWSLT tasks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--languages", nargs="+", default=["en", "de", "it", "zh"])
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config.get("task", "multitask")
    print(f"Evaluating task={task} track={args.track}")

    # Check mWERSegmenter for non-English S2TT COMET scoring
    if any(lang != "en" for lang in args.languages):
        if not check_mwer_segmenter():
            print("Continuing without mWERSegmenter — S2TT COMET scores will be unavailable.")

    # Load model
    model, processor, tokenizer = load_checkpoint(args.checkpoint, config)

    # Run evaluation for each language
    all_scores = {}
    for language in args.languages:
        print(f"\n=== Evaluating language={language} ===")
        output_path = run_inference_on_mcif(
            model, processor, tokenizer, task, language, args.track
        )
        print(f"Output written to: {output_path}")

        scores = run_mcif_eval(output_path, args.track, language)
        all_scores[language] = scores

    # Summary
    print("\n=== Summary ===")
    for lang, scores in all_scores.items():
        print(f"  {lang}: {scores}")


if __name__ == "__main__":
    main()
