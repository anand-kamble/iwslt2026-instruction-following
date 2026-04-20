"""Extract LoRA adapters from a trained full checkpoint.

Computes delta between:
- Stage 3 trained checkpoint
- Base model + Stage 2 LoRA (the starting point for Stage 3)

This extracts the weights learned during Stage 3 training.

Usage:
    python src/extract_lora.py --checkpoint outputs/stage3_multimodal_v1_4xh100/checkpoint-1800 \
        --stage2_lora outputs/stage2_text_lora/lora_adapters \
        --output outputs/stage3_multimodal_v1_4xh100/checkpoint-1800/lora_adapters
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_lora_adapters(
    checkpoint_dir: str,
    output_dir: str,
    base_model_name: str,
    stage2_lora_dir: str = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    """Extract LoRA adapters by computing delta between trained checkpoint and base+stage2."""
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    base_state = {k: v.to(torch.bfloat16) for k, v in base_model.state_dict().items()}
    del base_model

    if stage2_lora_dir:
        print(f"Loading Stage 2 LoRA: {stage2_lora_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model = PeftModel.from_pretrained(base_model, stage2_lora_dir)
        base_model = base_model.merge_and_unload()
        base_state = {k: v.to(torch.bfloat16) for k, v in base_model.state_dict().items()}
        del base_model
    else:
        print("No Stage 2 LoRA provided, using base model only")

    print(f"Loading trained checkpoint: {checkpoint_dir}")
    trained_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    trained_state = trained_model.state_dict()

    print("Computing delta (trained - starting point)...")
    lora_state = {}
    for key in trained_state:
        if key not in base_state:
            print(f"  New key in trained: {key}")
            continue

        delta = trained_state[key].to(torch.bfloat16) - base_state[key]
        if delta.abs().max() > 1e-8:
            lora_state[key] = delta.to(torch.float16)
        del delta

    print(f"Saving {len(lora_state)} adapter tensors...")
    os.makedirs(output_dir, exist_ok=True)

    adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(lora_state, adapter_model_path)

    lora_config = {
        "base_model_name_or_path": base_model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None,
        "fan_in_fan_out": False,
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    print(f"Tokenizer to: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Adapter saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA adapters from trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for lora_adapters")
    parser.add_argument("--stage2_lora", type=str, default=None,
                        help="Stage 2 LoRA adapters directory (optional)")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    args = parser.parse_args()

    extract_lora_adapters(
        args.checkpoint,
        args.output,
        args.base_model,
        args.stage2_lora,
        args.lora_r,
        args.lora_alpha,
    )


if __name__ == "__main__":
    main()