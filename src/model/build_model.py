"""Model construction: Seamless M4T v2 speech encoder + projection + Qwen3-4B.

The speech encoder is loaded from the full Seamless M4T v2 model and frozen.
The LLM is loaded via Unsloth with LoRA adapters.
The projection layer bridges the speech encoder (1024-dim) to the LLM (2560-dim).
"""

import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

from src.model.adapters import SpeechProjector

SPEECH_DIM = 1024
QWEN3_4B_DIM = 2560


def load_speech_encoder(model_name: str = "facebook/seamless-m4t-v2-large", device: str = "cuda"):
    """Load and freeze the Seamless M4T v2 speech encoder.

    Returns the speech encoder module and the audio processor.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    full_model = SeamlessM4Tv2Model.from_pretrained(model_name)

    speech_encoder = full_model.speech_encoder
    del full_model
    torch.cuda.empty_cache()

    speech_encoder = speech_encoder.to(device)
    speech_encoder.eval()
    for param in speech_encoder.parameters():
        param.requires_grad = False

    return speech_encoder, processor


def load_llm_with_lora(
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
):
    """Load Qwen3-4B-Instruct via Unsloth with LoRA adapters.

    Returns the model and tokenizer.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )

    return model, tokenizer


def build_projector(
    speech_dim: int = SPEECH_DIM,
    llm_dim: int = QWEN3_4B_DIM,
    downsample_factor: int = 1,
    device: str = "cuda",
) -> SpeechProjector:
    """Build the trainable projection layer between speech encoder and LLM."""
    projector = SpeechProjector(
        speech_dim=speech_dim,
        llm_dim=llm_dim,
        downsample_factor=downsample_factor,
    )
    return projector.to(device)
