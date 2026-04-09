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
from transformers import TrainingArguments, Trainer, TrainerCallback

from src.config import load_config
from src.model.build_model import (
    load_speech_encoder,
    load_llm_with_lora,
    build_projector,
    get_device,
)
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.collator import SpeechTextCollator


class WandbConfigCallback(TrainerCallback):
    """Logs the full pipeline config to wandb once the run is initialized."""

    def __init__(self, config: dict):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb

        if wandb.run is not None:
            wandb.config.update({"pipeline": self.config}, allow_val_change=True)


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
    parser.add_argument(
        "--stage1_checkpoint", type=str, default=None, help="Load projector weights from Stage 1"
    )
    parser.add_argument(
        "--stage2_checkpoint", type=str, default=None, help="Load LoRA weights from Stage 2"
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="Dir with pre-extracted speech features (from extract_features.py)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    text_only = training_config.get("text_only", False)
    no_lora = training_config.get("no_lora", False)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = args.output_dir or config.get("paths", {}).get("output_dir", "./outputs")
    task_label = config["task"]
    stage = config.get("stage", "")
    if stage:
        task_label = f"stage{stage}_{task_label}"
    output_dir = os.path.join(output_dir, config_name)

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

        run_name = f"{config_name}_{datetime.now().strftime('%m%d_%H%M')}"
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
            lora_r=training_config.get("lora_r", 8),
            lora_alpha=training_config.get("lora_alpha", 16),
            target_modules=training_config.get("lora_target_modules"),
        )

        from src.data.text_only import load_text_only_datasets

        task_datasets = load_text_only_datasets(config)
        if not task_datasets:
            raise RuntimeError("No text-only datasets loaded")

        # Use MultiTaskDataset + TemperatureSampler when sampling ratios are
        # configured (NAVER Stage B: MT 0.6, QA 0.4), otherwise concatenate.
        sampling_config = config.get("sampling", {})
        stage2_ratios = sampling_config.get("ratios")
        stage2_sampler = None

        if stage2_ratios:
            from src.data.multitask import MultiTaskDataset, TemperatureSampler

            train_dataset = MultiTaskDataset(task_datasets)
            stage2_sampler = TemperatureSampler(
                train_dataset,
                ratios=stage2_ratios,
                seed=training_config.get("seed", 42),
                batch_size=training_config.get("per_device_train_batch_size", 8),
            )
            print(f"Text-only sampling: ratios={stage2_ratios}")
        else:
            from datasets import concatenate_datasets

            train_dataset = concatenate_datasets(list(task_datasets.values()))

        print(f"Text-only training samples: {len(train_dataset)}")

        collator = SpeechTextCollator(
            processor=None,
            tokenizer=tokenizer,
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
            weight_decay=float(training_config.get("weight_decay", 0.0)),
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
            dataloader_pin_memory=torch.cuda.is_available(),
            tf32=torch.cuda.is_available(),
            **get_precision_args(),
        )

        callbacks = [WandbConfigCallback(config)] if report_to == "wandb" else []

        trainer = Trainer(
            model=llm,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
        if stage2_sampler is not None:
            trainer._get_train_sampler = lambda *args, **kwargs: stage2_sampler

            from collections import defaultdict

            class _Stage2LossCallback(TrainerCallback):
                def __init__(self):
                    self.task_losses = defaultdict(list)

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not self.task_losses:
                        return
                    import wandb

                    if wandb.run is None:
                        return
                    task_logs = {}
                    for task, losses in self.task_losses.items():
                        task_logs[f"train/loss_{task}"] = sum(losses) / len(losses)
                    wandb.log(task_logs, step=state.global_step)
                    self.task_losses.clear()

            stage2_loss_cb = _Stage2LossCallback()
            trainer.add_callback(stage2_loss_cb)

            original_compute_loss = trainer.compute_loss

            def compute_loss_with_task_tracking(
                model, inputs, return_outputs=False, num_items_in_batch=None
            ):
                tasks = inputs.pop("_tasks", None)
                result = original_compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                if tasks:
                    loss = result[0] if return_outputs else result
                    unique = set(t for t in tasks if t)
                    task = next(iter(unique)) if len(unique) == 1 else "mixed"
                    stage2_loss_cb.task_losses[task].append(loss.detach().item())
                return result

            trainer.compute_loss = compute_loss_with_task_tracking

        trainer.train()

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
            lora_r=training_config.get("lora_r", 8),
            lora_alpha=training_config.get("lora_alpha", 16),
            target_modules=training_config.get("lora_target_modules"),
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

        # Wrap with precomputed features if available.
        # Only wrap tasks that actually have .pt files (speech tasks).
        # Text-only tasks (mt_*, qa_*) have no audio to precompute.
        if args.precomputed_dir:
            from src.data.precomputed import PrecomputedWrapper

            wrapped = 0
            for task_name in task_datasets:
                task_dir = os.path.join(args.precomputed_dir, task_name)
                has_features = os.path.isdir(task_dir) and any(
                    f.endswith(".pt") for f in os.listdir(task_dir)
                )
                if has_features:
                    task_datasets[task_name] = PrecomputedWrapper(
                        task_datasets[task_name], task_dir
                    )
                    wrapped += 1
            if wrapped:
                print(f"Using precomputed features for {wrapped}/{len(task_datasets)} tasks")
            else:
                print(f"WARNING: No precomputed features found in {args.precomputed_dir}")

        train_dataset = MultiTaskDataset(task_datasets)
        sampling_config = config.get("sampling", {})
        temperature = sampling_config.get("temperature", 0.5)
        ratios = sampling_config.get("ratios")

        # Paired text interleaving (NAVER Stage C): text_interleave maps
        # speech tasks to their text equivalents (e.g. s2tt_de → mt_de).
        # The sampler yields a text batch after each paired speech batch.
        text_interleave = config.get("text_interleave", {})
        text_pairs = {speech: text for speech, text in text_interleave.items()}

        sampler = TemperatureSampler(
            train_dataset,
            temperature=temperature,
            ratios=ratios,
            text_pairs=text_pairs if text_pairs else None,
            seed=training_config.get("seed", 42),
            batch_size=training_config.get("per_device_train_batch_size", 1),
        )
        if text_pairs:
            print(f"Multi-task sampling: text pairs={text_pairs}")
        if ratios:
            print(f"Multi-task sampling: explicit ratios={ratios}")
        else:
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
        weight_decay=float(training_config.get("weight_decay", 0.0)),
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
        dataloader_pin_memory=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        **get_precision_args(),
    )

    callbacks = [WandbConfigCallback(config)] if report_to == "wandb" else []

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
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
                p for p in model.parameters() if p.requires_grad and id(p) not in projector_ids
            ]

            assert projector_params, "No projector parameters found — is projector frozen?"
            assert other_params, "No LoRA parameters found — is LoRA disabled?"

            # Use fused AdamW on CUDA for better kernel fusion
            use_fused = get_device() == "cuda"
            from torch.optim import AdamW

            wd = float(training_config.get("weight_decay", 0.0))
            # LoRA first so HF Trainer's default `learning_rate` metric shows
            # the LoRA LR (cosine decay) instead of the projector LR (constant).
            optimizer = AdamW(
                [
                    {"params": other_params, "lr": float(lr_lora)},
                    {"params": projector_params, "lr": float(lr_projector)},
                ],
                weight_decay=wd,
                fused=use_fused,
            )
            trainer.optimizer = optimizer
            return optimizer

        trainer.create_optimizer = create_optimizer_with_param_groups

        # Log per-group learning rates to wandb (must use wandb.log directly
        # because WandbCallback runs before this callback in the chain)
        class PerGroupLRCallback(TrainerCallback):
            """Log per-parameter-group learning rates separately."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                if trainer.optimizer is None:
                    return
                import wandb

                if wandb.run is None:
                    return
                lr_logs = {}
                for i, name in enumerate(["lora", "projector"]):
                    if i < len(trainer.optimizer.param_groups):
                        lr_logs[f"train/lr_{name}"] = trainer.optimizer.param_groups[i]["lr"]
                if lr_logs:
                    wandb.log(lr_logs, step=state.global_step)

        trainer.add_callback(PerGroupLRCallback())

        # Per-group LR scheduling: constant for projector, main schedule for LoRA
        lr_scheduler_projector = training_config.get("lr_scheduler_projector")
        if lr_scheduler_projector == "constant":
            import math
            from torch.optim.lr_scheduler import LambdaLR

            sched_type = training_config.get("lr_scheduler_type", "cosine")

            def create_scheduler_per_group(num_training_steps, optimizer=None):
                opt = optimizer or trainer.optimizer
                warmup_steps = training_args.get_warmup_steps(num_training_steps)

                def projector_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    return 1.0

                def lora_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(
                        max(1, num_training_steps - warmup_steps)
                    )
                    if sched_type == "cosine":
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    elif sched_type == "linear":
                        return max(0.0, 1.0 - progress)
                    return 1.0  # fallback: constant

                scheduler = LambdaLR(opt, [lora_lambda, projector_lambda])
                trainer.lr_scheduler = scheduler
                return scheduler

            trainer.create_scheduler = create_scheduler_per_group

    # Per-task loss tracking for multitask training
    if config["task"] == "multitask":
        from collections import defaultdict

        class MultitaskLossCallback(TrainerCallback):
            """Track and log per-task training losses to wandb."""

            def __init__(self):
                self.task_losses = defaultdict(list)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not self.task_losses:
                    return
                import wandb

                if wandb.run is None:
                    return
                task_logs = {}
                for task, losses in self.task_losses.items():
                    task_logs[f"train/loss_{task}"] = sum(losses) / len(losses)
                wandb.log(task_logs, step=state.global_step)
                self.task_losses.clear()

        multitask_cb = MultitaskLossCallback()
        trainer.add_callback(multitask_cb)

        original_compute_loss = trainer.compute_loss

        def compute_loss_with_task_tracking(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            tasks = inputs.pop("_tasks", None)
            result = original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            if tasks:
                loss = result[0] if return_outputs else result
                unique = set(t for t in tasks if t)
                task = next(iter(unique)) if len(unique) == 1 else "mixed"
                multitask_cb.task_losses[task].append(loss.detach().item())
            return result

        trainer.compute_loss = compute_loss_with_task_tracking

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save
    print(f"Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    if not no_lora:
        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))

    # Push to hub if configured
    hub_config = config.get("hub", {})
    if (
        hub_config.get("push_to_hub", False)
        and hub_config.get("organization", "YOUR_HF_ORG") != "YOUR_HF_ORG"
    ):
        repo_id = f"{hub_config['organization']}/iwslt2026-{task_label}"
        print(f"Pushing to {repo_id}...")
        if not no_lora:
            llm.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)

    print("Done!")


if __name__ == "__main__":
    main()
