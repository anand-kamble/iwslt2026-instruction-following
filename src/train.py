"""Training entry point for IWSLT instruction-following model.

Supports 3-stage training:
    Stage 1: python src/train.py --config configs/stage1_alignment.yaml
    Stage 2: python src/train.py --config configs/stage2_text_lora.yaml
    Stage 3: python src/train.py --config configs/stage3_multimodal.yaml \
             --stage1_checkpoint outputs/stage1_asr --stage2_checkpoint outputs/stage2_text_lora
"""

import argparse
import os

import torch
from transformers import TrainingArguments, Trainer

from src.config import load_config
from src.model.build_model import load_speech_encoder, load_llm_with_lora, build_projector, get_device
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.collator import SpeechTextCollator


def get_precision_args():
    """Auto-select precision: bf16 on CUDA (H200/A100), fp32 on MPS/CPU.

    H200 has native bf16 support with excellent throughput.
    MPS does not support bf16 — use fp32.
    """
    device = get_device()
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return {"bf16": True}
        return {"fp16": True}
    # MPS and CPU: no mixed precision (model already in fp16, compute in fp32)
    return {}


def load_task_datasets(config: dict):
    """Load training datasets based on task config."""
    task = config["task"]

    if task in ("multitask", "text_lora"):
        return None

    if task == "asr":
        from src.data.asr import load_asr_dataset
        max_samples = config.get("data", {}).get("max_samples")
        train_datasets = []
        for ds_name in config["data"]["train"]:
            train_datasets.append(load_asr_dataset(ds_name, "train", max_samples=max_samples))
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
        languages = config.get("languages", ["en"])
        train_datasets = []
        for lang in languages:
            train_datasets.append(load_achap_dataset("train", lang))
        return train_datasets

    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(description="Train IWSLT instruction-following model")
    parser.add_argument("--config", type=str, required=True, help="Path to task config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Load projector weights from Stage 1")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Load LoRA weights from Stage 2")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    text_only = training_config.get("text_only", False)
    no_lora = training_config.get("no_lora", False)

    output_dir = args.output_dir or config.get("paths", {}).get("output_dir", "./outputs")
    task_label = config["task"]
    stage = config.get("stage", "")
    if stage:
        task_label = f"stage{stage}_{task_label}"
    output_dir = os.path.join(output_dir, task_label)

    print(f"Task: {config['task']} | Stage: {stage} | Text-only: {text_only} | No-LoRA: {no_lora}")
    print(f"Output: {output_dir}")

    # Setup wandb — env vars take precedence over YAML config
    wandb_config = config.get("wandb", {})
    wandb_project = os.environ.get("WANDB_PROJECT") or wandb_config.get("project")

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if not os.environ.get("WANDB_ENTITY") and wandb_config.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_config["entity"]
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        report_to = "wandb"
        from datetime import datetime
        run_name = f"{task_label}_{datetime.now().strftime('%m%d_%H%M')}"
    else:
        report_to = "tensorboard"
        run_name = None

    # ===== Stage 2: Text-only LoRA training (no audio) =====
    if text_only:
        print("=== Stage 2: Text-only LoRA training ===")
        llm_config = config.get("model", {})
        llm, tokenizer = load_llm_with_lora(
            model_name=llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507"),
            load_in_4bit=training_config.get("load_in_4bit", False),
            load_in_8bit=training_config.get("load_in_8bit", False),
            no_lora=False,
            lora_r=training_config.get("lora_r", 16),
            lora_alpha=training_config.get("lora_alpha", 16),
        )

        from src.data.text_only import load_text_only_datasets
        from datasets import concatenate_datasets

        train_datasets = load_text_only_datasets(config)
        if not train_datasets:
            raise RuntimeError("No text-only datasets loaded")
        train_dataset = concatenate_datasets(train_datasets)
        print(f"Text-only training samples: {len(train_dataset)}")

        collator = SpeechTextCollator(
            processor=None, tokenizer=tokenizer,
            max_seq_length=training_config.get("max_seq_length", 4096),
            text_only=True,
        )

        max_steps = training_config.get("max_steps", -1)
        num_epochs = training_config.get("num_epochs", 2) if max_steps <= 0 else 100

        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            learning_rate=float(training_config.get("learning_rate", 3e-4)),
            warmup_ratio=float(training_config.get("warmup_ratio", 0.05)),
            warmup_steps=training_config.get("warmup_steps") or 0,
            logging_steps=training_config.get("logging_steps", 50),
            save_steps=training_config.get("save_steps", 500),
            optim=training_config.get("optim", "adamw_torch"),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            seed=training_config.get("seed", 42),
            report_to=report_to,
            logging_dir=training_config.get("logging_dir", os.path.join(output_dir, "logs")),
            save_total_limit=3,
            save_safetensors=False,
            remove_unused_columns=False,
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            dataloader_pin_memory=True,
            tf32=True,
            **get_precision_args(),
        )

        trainer = Trainer(
            model=llm, args=training_args,
            train_dataset=train_dataset, data_collator=collator,
        )

        # Log full config to wandb (HF Trainer only logs TrainingArguments)
        if report_to == "wandb":
            import wandb
            if wandb.run is not None:
                wandb.config.update({"pipeline": config}, allow_val_change=True)

        try:
            trainer.train()
        finally:
            if report_to == "wandb":
                import wandb
                wandb.finish()

        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        print(f"Stage 2 LoRA saved to {output_dir}/lora_adapters")
        return

    # ===== Stage 1 or 3: Speech + LLM pipeline =====
    print("Loading speech encoder...")
    speech_encoder, processor = load_speech_encoder(
        config.get("model", {}).get("sfm", "facebook/seamless-m4t-v2-large")
    )

    # Load LLM — conditionally from Stage 2 checkpoint or fresh
    llm_config = config.get("model", {})
    if args.stage2_checkpoint:
        from src.model.build_model import load_llm_from_checkpoint
        lora_path = os.path.join(args.stage2_checkpoint, "lora_adapters")
        print(f"Loading LLM with Stage 2 LoRA from {lora_path}")
        llm, tokenizer = load_llm_from_checkpoint(
            checkpoint_dir=lora_path,
            base_model_name=llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507"),
            load_in_4bit=training_config.get("load_in_4bit", False),
            load_in_8bit=training_config.get("load_in_8bit", False),
        )
    else:
        print("Loading LLM...")
        llm, tokenizer = load_llm_with_lora(
            model_name=llm_config.get("llm", "Qwen/Qwen3-4B-Instruct-2507"),
            load_in_4bit=training_config.get("load_in_4bit", False),
            load_in_8bit=training_config.get("load_in_8bit", False),
            no_lora=no_lora,
            lora_r=training_config.get("lora_r", 16),
            lora_alpha=training_config.get("lora_alpha", 16),
        )

    # Add speech placeholder token
    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
        llm.resize_token_embeddings(len(tokenizer))

    print("Building projector...")
    projector = build_projector(
        downsample_factor=training_config.get("downsample_factor", 3),
        projector_type=training_config.get("projector_type", "transformer"),
    )

    # Load projector checkpoint if provided (from Stage 0 or previous Stage 1)
    if args.stage1_checkpoint:
        projector_path = os.path.join(args.stage1_checkpoint, "projector.pt")
        print(f"Loading projector from {projector_path}")
        state_dict = torch.load(projector_path, map_location="cpu", weights_only=True)
        missing, unexpected = projector.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (randomly initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

    # Freeze LLM in Stage 1 (no LoRA — only projector trains)
    if no_lora:
        print("Freezing LLM (Stage 1: projector-only training)")
        for param in llm.parameters():
            param.requires_grad = False

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)

    # Load data
    print("Loading datasets...")
    sampler = None

    if config["task"] == "multitask":
        from src.data.multitask import MultiTaskDataset, TemperatureSampler, load_all_task_datasets
        task_datasets = load_all_task_datasets(config)
        train_dataset = MultiTaskDataset(task_datasets)
        sampling_config = config.get("sampling", {})
        temperature = sampling_config.get("temperature", 0.5)
        sampler = TemperatureSampler(
            train_dataset, temperature=temperature,
            seed=training_config.get("seed", 42),
        )
        print(f"Multi-task sampling: temperature={temperature}")
    else:
        train_datasets = load_task_datasets(config)
        if len(train_datasets) > 1:
            from datasets import concatenate_datasets
            train_dataset = concatenate_datasets(train_datasets)
        else:
            train_dataset = train_datasets[0]

    print(f"Training samples: {len(train_dataset)}")

    collator = SpeechTextCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=training_config.get("max_seq_length", 4096),
    )

    # Training args — support max_steps
    max_steps = training_config.get("max_steps", -1)
    num_epochs = training_config.get("num_epochs", 3) if max_steps <= 0 else 100

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.05)),
        warmup_steps=training_config.get("warmup_steps") or 0,
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        optim=training_config.get("optim", "adamw_torch"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        seed=training_config.get("seed", 42),
        report_to=report_to,
        logging_dir=training_config.get("logging_dir", os.path.join(output_dir, "logs")),
        save_total_limit=3,
        save_safetensors=False,  # avoid shared tensor error with tied weights
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        tf32=True,
        **get_precision_args(),
    )

    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=collator,
    )

    if sampler is not None:
        trainer = Trainer(**trainer_kwargs)
        trainer._get_train_sampler = lambda *args, **kwargs: sampler
    else:
        trainer = Trainer(**trainer_kwargs)

    # Differential LR for Stage 3
    lr_projector = training_config.get("lr_projector")
    lr_lora = training_config.get("lr_lora")
    if lr_projector and lr_lora and not no_lora:

        def create_optimizer_with_param_groups():
            projector_params = list(model.projector.parameters())
            projector_ids = {id(p) for p in projector_params}
            other_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in projector_ids
            ]

            # Use fused AdamW on CUDA for better kernel fusion
            use_fused = get_device() == "cuda"
            from torch.optim import AdamW
            optimizer = AdamW(
                [
                    {"params": projector_params, "lr": float(lr_projector)},
                    {"params": other_params, "lr": float(lr_lora)},
                ],
                fused=use_fused,
            )
            trainer.optimizer = optimizer
            return optimizer

        trainer.create_optimizer = create_optimizer_with_param_groups

    # Log full config to wandb
    if report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.config.update({"pipeline": config}, allow_val_change=True)

    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    finally:
        if report_to == "wandb":
            import wandb
            wandb.finish()

    # Save
    print(f"Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    if not no_lora:
        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))

    # Push to hub if configured
    hub_config = config.get("hub", {})
    if hub_config.get("push_to_hub", False) and hub_config.get("organization", "YOUR_HF_ORG") != "YOUR_HF_ORG":
        repo_id = f"{hub_config['organization']}/iwslt2026-{task_label}"
        print(f"Pushing to {repo_id}...")
        if not no_lora:
            llm.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)

    print("Done!")


if __name__ == "__main__":
    main()
