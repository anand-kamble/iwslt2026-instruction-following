"""Training entry point for IWSLT instruction-following model.

Usage:
    python src/train.py --config configs/asr.yaml
    python src/train.py --config configs/asr.yaml --output_dir ./outputs/asr-v1
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from transformers import TrainingArguments, Trainer

from src.config import load_config
from src.model.build_model import load_speech_encoder, load_llm_with_lora, build_projector
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.collator import SpeechTextCollator


def load_task_datasets(config: dict):
    """Load training and eval datasets based on task config."""
    task = config["task"]

    if task == "asr":
        from src.data.asr import load_asr_dataset
        train_datasets = []
        for ds_name in config["data"]["train"]:
            train_datasets.append(load_asr_dataset(ds_name, "train"))
        return train_datasets

    elif task == "s2tt":
        from src.data.s2tt import load_s2tt_dataset
        train_datasets = []
        for ds_name in config["data"]["train"]:
            for lang in config["target_languages"]:
                try:
                    train_datasets.append(load_s2tt_dataset(ds_name, lang, "train"))
                except ValueError:
                    print(f"Skipping {ds_name} for {lang} (not available)")
        return train_datasets

    elif task == "sqa":
        from src.data.sqa import load_sqa_dataset
        return [load_sqa_dataset("train")]

    elif task == "s2tsum":
        from src.data.s2tsum import load_s2tsum_dataset
        return [load_s2tsum_dataset("train")]

    elif task == "achap":
        from src.data.achap import load_achap_dataset
        return [load_achap_dataset("train")]

    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(description="Train IWSLT instruction-following model")
    parser.add_argument("--config", type=str, required=True, help="Path to task config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config.get("paths", {}).get("output_dir", "./outputs")
    output_dir = os.path.join(output_dir, config["task"])

    print(f"Task: {config['task']}")
    print(f"Output: {output_dir}")

    # 1. Load models
    print("Loading speech encoder...")
    speech_encoder, processor = load_speech_encoder(
        config.get("model", {}).get("sfm", "facebook/seamless-m4t-v2-large")
    )

    print("Loading LLM with LoRA...")
    llm_config = config.get("model", {})
    training_config = config.get("training", {})
    llm, tokenizer = load_llm_with_lora(
        model_name=llm_config.get("llm", "unsloth/Qwen3-4B-Instruct-2507"),
        max_seq_length=training_config.get("max_seq_length", 4096),
        load_in_4bit=training_config.get("load_in_4bit", True),
        lora_r=training_config.get("lora_r", 32),
        lora_alpha=training_config.get("lora_alpha", 32),
    )

    # Add speech placeholder token to tokenizer
    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
        llm.resize_token_embeddings(len(tokenizer))

    print("Building projector...")
    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 1),
    )

    # 2. Build SpeechLLM
    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)

    # 3. Load data
    print("Loading datasets...")
    train_datasets = load_task_datasets(config)

    if len(train_datasets) > 1:
        from datasets import concatenate_datasets
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f"Training samples: {len(train_dataset)}")

    # 4. Collator
    collator = SpeechTextCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=training_config.get("max_seq_length", 4096),
    )

    # 5. Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        bf16=training_config.get("bf16", True),
        optim=training_config.get("optim", "adamw_8bit"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        seed=training_config.get("seed", 42),
        report_to="none",
        save_total_limit=3,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 7. Save
    print(f"Saving to {output_dir}")
    # Save projector
    torch.save(projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    # Save LoRA adapters
    llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))

    # Push to hub if configured
    hub_config = config.get("hub", {})
    if hub_config.get("push_to_hub", False) and hub_config.get("organization", "YOUR_HF_ORG") != "YOUR_HF_ORG":
        repo_id = f"{hub_config['organization']}/iwslt2026-{config['task']}"
        print(f"Pushing to {repo_id}...")
        llm.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    print("Done!")


if __name__ == "__main__":
    main()
