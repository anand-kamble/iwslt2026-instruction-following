"""Local smoke training on MacBook (CPU/MPS, no Unsloth, no bitsandbytes).

Runs 10 training steps with a tiny LLM (Qwen3-0.6B) and mock speech encoder
to verify the full training pipeline works end-to-end.

Usage:
    uv run python scripts/local_smoke_train.py
    uv run python scripts/local_smoke_train.py --steps 5 --stage 1
    uv run python scripts/local_smoke_train.py --steps 5 --stage 2
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.adapters import TransformerProjector
from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER
from src.data.collator import SpeechTextCollator
from src.data.utils import format_chat_messages, format_text_only_messages


class MockEncoder(torch.nn.Module):
    """Mock speech encoder that returns random features."""
    def forward(self, input_features, attention_mask=None, return_dict=True):
        B = input_features.shape[0]
        class Out:
            def __init__(self, h): self.last_hidden_state = h
        return Out(torch.randn(B, 50, 1024))


class MockProcessor:
    def __call__(self, audios, sampling_rate=16000, return_tensors="pt",
                 padding=True, return_attention_mask=True):
        if isinstance(audios, list):
            B = len(audios)
        else:
            B = 1
        return {
            "input_features": torch.randn(B, 200, 160),
            "attention_mask": torch.ones(B, 200, dtype=torch.long),
        }


def make_fake_dataset(n=20, text_only=False):
    """Create a tiny fake dataset."""
    from datasets import Dataset
    if text_only:
        records = []
        for i in range(n):
            if i % 2 == 0:
                records.append({"messages": format_text_only_messages(
                    "Translate to German.", f"Das ist Satz {i}.")})
            else:
                records.append({"messages": format_text_only_messages(
                    f"Answer: what is {i}?", f"The answer is {i}.")})
        return Dataset.from_list(records)
    else:
        import numpy as np
        records = []
        for i in range(n):
            records.append({
                "audio": np.random.randn(16000).astype(np.float32),  # 1 sec
                "messages": format_chat_messages(
                    "Transcribe the English content.", f"This is sentence {i}."),
            })
        return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    output_dir = f"/tmp/iwslt_smoke_stage{args.stage}"

    print(f"=== Local smoke training: Stage {args.stage}, {args.steps} steps ===")
    print(f"Output: {output_dir}")

    # Load tiny LLM
    print("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True, dtype=torch.float32)

    if args.stage == 2:
        # Stage 2: text-only LoRA
        print("Stage 2: Text-only LoRA training")
        lora_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

        dataset = make_fake_dataset(n=20, text_only=True)
        collator = SpeechTextCollator(
            processor=None, tokenizer=tokenizer,
            max_seq_length=256, text_only=True)

        training_args = TrainingArguments(
            output_dir=output_dir, max_steps=args.steps,
            per_device_train_batch_size=2, learning_rate=3e-4,
            logging_steps=1, save_steps=999, save_strategy="no",
            report_to="none", remove_unused_columns=False, use_cpu=True)

        trainer = Trainer(model=llm, args=training_args,
                          train_dataset=dataset, data_collator=collator)
        trainer.train()

        llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        print(f"Stage 2 done! LoRA saved to {output_dir}/lora_adapters")

    else:
        # Stage 1 or 3: Speech + LLM
        if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
            tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
            llm.resize_token_embeddings(len(tokenizer))

        if args.stage == 3:
            # Add LoRA for Stage 3
            lora_config = LoraConfig(
                r=16, lora_alpha=16, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            llm = get_peft_model(llm, lora_config)
            llm.print_trainable_parameters()

        llm_dim = llm.config.hidden_size if not hasattr(llm, 'peft_config') else llm.model.config.hidden_size
        projector = TransformerProjector(speech_dim=1024, llm_dim=llm_dim, downsample_factor=3)
        encoder = MockEncoder()
        processor = MockProcessor()

        model = SpeechLLM(encoder, projector, llm, tokenizer)

        dataset = make_fake_dataset(n=20, text_only=False)
        collator = SpeechTextCollator(
            processor=processor, tokenizer=tokenizer,
            max_seq_length=256)

        training_args = TrainingArguments(
            output_dir=output_dir, max_steps=args.steps,
            per_device_train_batch_size=2, learning_rate=1e-4,
            logging_steps=1, save_steps=999, save_strategy="no",
            report_to="none", remove_unused_columns=False,
            use_cpu=True, gradient_checkpointing=False)

        trainer = Trainer(model=model, args=training_args,
                          train_dataset=dataset, data_collator=collator)
        trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        torch.save(projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        print(f"Stage {args.stage} done! Projector saved to {output_dir}/projector.pt")

    print("Smoke training completed successfully!")


if __name__ == "__main__":
    main()
