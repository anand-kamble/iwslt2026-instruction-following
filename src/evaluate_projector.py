"""Evaluate a Stage 1 (projector-only) checkpoint using the base LLM without LoRA.

Usage:
    python src/evaluate_projector.py --config configs/stage1_alignment.yaml \
        --checkpoint ./outputs/stage1_alignment
    python src/evaluate_projector.py --config configs/stage1_alignment_sqa.yaml \
        --checkpoint ./outputs/stage1_alignment --data-dir /path/to/mcif
    python src/evaluate_projector.py --config configs/stage1_alignment.yaml \
        --checkpoint ./outputs/stage1_alignment \
        --data-dir ./MCIF_DATA --output-dir ./evaluation
    python src/evaluate_projector.py --config configs/stage1_alignment.yaml \
        --base --output-name seamless_base
"""

import argparse
import json
import os
import subprocess

import torch
import yaml
from transformers import AutoProcessor, SeamlessM4Tv2Model

from src.model.build_model import load_speech_encoder, load_llm_with_lora, build_projector, get_device
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.utils import TASK_PROMPTS, format_chat_messages, load_audio


def resolve_audio_path(data_dir: str, audio_field: str) -> str:
    """Resolve MCIF audio path robustly for both prefixed and non-prefixed entries.

    Some MCIF rows store paths like `MCIF_DATA/SHORT_AUDIOS/...`, while callers may
    already pass `--data-dir .../MCIF_DATA`. This helper avoids duplicated
    `.../MCIF_DATA/MCIF_DATA/...` paths.
    """
    if os.path.isabs(audio_field):
        return audio_field

    candidates = [os.path.join(data_dir, audio_field)]
    prefix = "MCIF_DATA/"
    if audio_field.startswith(prefix):
        candidates.append(os.path.join(data_dir, audio_field[len(prefix):]))

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_projector_checkpoint(checkpoint_dir: str, config: dict = None, device: str = None):
    """Load a Stage 1 checkpoint: base LLM (no LoRA) + trained projector."""
    if device is None:
        device = get_device()

    config = config or {}
    training_config = config.get("training", {})
    llm_config = config.get("model", {})

    # Load speech encoder
    speech_encoder, processor = load_speech_encoder(device=device)

    # Load base LLM without LoRA
    llm, tokenizer = load_llm_with_lora(
        model_name=llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507"),
        load_in_4bit=training_config.get("load_in_4bit", False),
        load_in_8bit=training_config.get("load_in_8bit", False),
        no_lora=True,
        device=device,
    )
    llm.eval()

    # Add speech placeholder token
    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER])
        llm.resize_token_embeddings(len(tokenizer))

    # Load trained projector
    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 3),
        projector_type=training_config.get("projector_type", "transformer"),
        device=device,
    )
    projector_path = os.path.join(checkpoint_dir, "projector.pt")
    projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
    projector.eval()

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)
    return model, processor, tokenizer


def load_seamless_baseline(device: str = None):
    """Load SeamlessM4T-v2-large baseline model for direct speech ASR/ST."""
    if device is None:
        device = get_device()

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, processor


def run_inference_on_mcif_seamless(
    model,
    processor,
    language: str,
    track: str = "short",
    data_dir: str = ".",
    output_dir: str = "outputs",
    output_name: str = "seamless_base",
    task: str = "multitask",
):
    """Run Seamless baseline inference on MCIF and write output XML."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    tgt_lang_map = {
        "en": "eng",
        "de": "deu",
        "it": "ita",
        "zh": "cmn",
    }
    if language not in tgt_lang_map:
        raise ValueError(f"Unsupported language for Seamless baseline: {language}")

    def _to_token_batch(generated):
        """Normalize Seamless generate() outputs into List[List[int]] for batch_decode."""
        # Some transformer versions return tensors, others return tuples/lists.
        if isinstance(generated, (tuple, list)):
            if len(generated) == 0:
                return [[]]
            generated = generated[0]

        if torch.is_tensor(generated):
            # If we receive logits/probabilities (float), convert to ids by argmax.
            if generated.dtype.is_floating_point:
                generated = generated.argmax(dim=-1)
            if generated.dim() == 1:
                generated = generated.unsqueeze(0)
            elif generated.dim() > 2:
                # Repeatedly squeeze singleton leading dims if present.
                while generated.dim() > 2 and generated.shape[0] == 1:
                    generated = generated.squeeze(0)
                if generated.dim() > 2:
                    generated = generated.argmax(dim=-1)
                if generated.dim() == 1:
                    generated = generated.unsqueeze(0)
            return generated.long().cpu().tolist()

        # Fallback for nested Python lists.
        if isinstance(generated, list):
            if len(generated) == 0:
                return [[]]
            if isinstance(generated[0], list):
                first = generated[0]
                if len(first) > 0 and isinstance(first[0], float):
                    # Looks like per-token distributions; take argmax per token.
                    return [[int(max(range(len(step)), key=lambda i: step[i])) for step in first]]
                return [[int(x) for x in first]]
            if isinstance(generated[0], float):
                return [[int(x) for x in generated]]
            return [[int(x) for x in generated]]

        raise TypeError(f"Unsupported generate() output type: {type(generated)}")

    config_name = f"{track}_fixedprompt"
    ds = load_dataset("FBK-MT/MCIF", config_name, split="test")
    samples = []
    total = len(ds)
    device = next(model.parameters()).device

    for i, example in enumerate(ds):
        if i % 10 == 0:
            print(f"  [{i}/{total}] Processing...", flush=True)

        audio_path = resolve_audio_path(data_dir, example["audio"])
        waveform, _ = load_audio(audio_path)
        audio_array = waveform.squeeze(0).numpy()

        audio_inputs = processor(
            audios=audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang_map[language],
                max_new_tokens=512,
                do_sample=False,
            )
        token_batch = _to_token_batch(output_ids)
        output_text = processor.batch_decode(token_batch, skip_special_tokens=True)[0]
        samples.append(OutputSample(id=example["id"], value=output_text))

    output_path = os.path.join(output_dir, f"{output_name}_{language}_{track}.xml")
    os.makedirs(output_dir, exist_ok=True)
    write_output(samples, track=track, language=language, output_name=task, output=output_path)
    return output_path


def run_inference_on_mcif(
    model,
    processor,
    tokenizer,
    task: str,
    language: str,
    track: str = "short",
    data_dir: str = ".",
    output_dir: str = "outputs",
    output_name: str = "projector",
):
    """Run inference on MCIF validation set and write output XML."""
    from datasets import load_dataset
    from mcif.io import OutputSample, write_output

    config_name = f"{track}_fixedprompt"
    ds = load_dataset("FBK-MT/MCIF", config_name, split="test")

    prompt_key = f"prompt_{language}"
    samples = []
    total = len(ds)

    for i, example in enumerate(ds):
        if i % 10 == 0:
            print(f"  [{i}/{total}] Processing...", flush=True)
        instruction = example[prompt_key]

        # Load audio from file path
        audio_path = resolve_audio_path(data_dir, example["audio"])
        waveform, _ = load_audio(audio_path)
        audio_array = waveform.squeeze(0).numpy()

        # Process audio
        audio_inputs = processor(
            audios=audio_array,
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
    output_path = os.path.join(output_dir, f"{output_name}_{language}_{track}.xml")
    os.makedirs(output_dir, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 projector checkpoint (no LoRA)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Stage 1 checkpoint dir (must contain projector.pt). Required unless --base is set.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Use SeamlessM4T-v2-large baseline directly (no projector, no LLM).",
    )
    parser.add_argument("--languages", nargs="+", default=["en", "de", "it", "zh"])
    parser.add_argument("--track", type=str, default="short", choices=["short", "long"])
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following/MCIF_DATA",
        help="Base directory for MCIF audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following/evaluation",
        help="Directory where evaluation XML outputs are saved",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Base name for XML outputs (e.g. stage1_alignment -> stage1_alignment_en_short.xml).",
    )
    args = parser.parse_args()
    if not args.base and not args.checkpoint:
        parser.error("Provide --checkpoint for projector evaluation, or pass --base for Seamless baseline.")
    if args.base and args.checkpoint:
        print("Note: --base is set; ignoring --checkpoint and using SeamlessM4T-v2-large baseline.\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config.get("task", "asr")
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    output_name = args.output_name or (f"{config_stem}_base" if args.base else config_stem)
    mode_name = "seamless-base" if args.base else "projector-only"
    print(f"Evaluating {mode_name} | task={task} track={args.track}")
    print(f"Output name: {output_name}")
    if args.base:
        model, processor = load_seamless_baseline()
    else:
        # Load model (base LLM + trained projector, no LoRA)
        model, processor, tokenizer = load_projector_checkpoint(args.checkpoint, config)

    # Run evaluation for each language
    all_scores = {}
    for language in args.languages:
        print(f"\n=== Evaluating language={language} ===")
        if args.base:
            output_path = run_inference_on_mcif_seamless(
                model,
                processor,
                language,
                args.track,
                args.data_dir,
                args.output_dir,
                output_name,
                task,
            )
        else:
            output_path = run_inference_on_mcif(
                model,
                processor,
                tokenizer,
                task,
                language,
                args.track,
                args.data_dir,
                args.output_dir,
                output_name,
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
