"""Custom data collator for speech + text multimodal training."""

import torch
from dataclasses import dataclass

from src.model.speech_llm import SPEECH_PLACEHOLDER


@dataclass
class SpeechTextCollator:
    """Collator that handles both audio features and tokenized text.

    Each sample in the batch should have:
        - "audio": raw audio waveform (numpy array or tensor, 16kHz mono)
        - "messages": list of chat messages (with SPEECH_PLACEHOLDER)

    The collator:
        1. Processes audio through the Seamless M4T v2 processor
        2. Tokenizes messages via the LLM tokenizer
        3. Creates labels (masking everything except the assistant response)
    """

    processor: object  # SeamlessM4T processor
    tokenizer: object  # Qwen tokenizer
    max_seq_length: int = 4096

    def __call__(self, features: list[dict]) -> dict:
        # Process audio
        audios = [f["audio"] for f in features]
        audio_inputs = self.processor(
            audios=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        # Tokenize text (full conversation including response)
        texts = []
        for f in features:
            text = self.tokenizer.apply_chat_template(
                f["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Create labels: mask everything except assistant response
        labels = text_inputs["input_ids"].clone()
        for i, f in enumerate(features):
            # Find the assistant response start
            # Tokenize just the prompt (without assistant response)
            prompt_messages = [m for m in f["messages"] if m["role"] != "assistant"]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(
                prompt_text, truncation=True, max_length=self.max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Mask prompt tokens in labels
            labels[i, :prompt_len] = -100

        # Mask padding in labels
        labels[text_inputs["attention_mask"] == 0] = -100

        return {
            "input_features": audio_inputs["input_features"],
            "input_features_attention_mask": audio_inputs["attention_mask"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
        }
