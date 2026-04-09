"""Model construction: Seamless M4T v2 speech encoder + projection + Qwen3-4B.

The speech encoder is loaded from the full Seamless M4T v2 model and frozen.
The LLM is loaded with optional LoRA adapters and quantization.
The projection layer bridges the speech encoder (1024-dim) to the LLM (2560-dim).

Supports both CUDA and MPS (Apple Silicon):
- CUDA: optional 8-bit/4-bit quantization via bitsandbytes
- MPS/CPU: float16/float32, no quantization
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, SeamlessM4Tv2Model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.model.adapters import SpeechProjector, TransformerProjector

SPEECH_DIM = 1024
QWEN3_4B_DIM = 2560


def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_speech_encoder(model_name: str = "facebook/seamless-m4t-v2-large", device: str = None):
    """Load and freeze the Seamless M4T v2 speech encoder.

    Returns the speech encoder module and the audio processor.
    Loads in float16 on CUDA for memory efficiency (encoder is frozen anyway).
    """
    if device is None:
        device = get_device()

    processor = AutoProcessor.from_pretrained(model_name)

    # Load in float16 on CUDA — encoder is frozen, no precision needed
    load_dtype = torch.float16 if device == "cuda" else torch.float32
    full_model = SeamlessM4Tv2Model.from_pretrained(
        model_name, torch_dtype=load_dtype,
    )

    speech_encoder = full_model.speech_encoder
    del full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speech_encoder = speech_encoder.to(device)
    speech_encoder.eval()
    for param in speech_encoder.parameters():
        param.requires_grad = False

    return speech_encoder, processor


def load_llm_with_lora(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    no_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
    device: str = None,
):
    """Load LLM with optional LoRA adapters and quantization.

    On CUDA: supports 8-bit/4-bit quantization via bitsandbytes.
    On MPS/CPU: loads in float16 (no quantization, bitsandbytes not supported).
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Quantization only works on CUDA
    use_quantization = (load_in_4bit or load_in_8bit) and device == "cuda"

    model_kwargs = {
        "trust_remote_code": True,
    }

    # Enable Flash Attention 2 on CUDA for faster attention
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if use_quantization:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # Use bf16 on CUDA (matches training autocast dtype, better range than fp16)
        # Fall back to fp16 on MPS/CPU
        if device == "cuda" and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Move to device if not already placed (device_map handles CUDA, manual for MPS/CPU)
    if not use_quantization and device != "cuda":
        model = model.to(device)

    # Prepare quantized model for training: casts LayerNorm to float32,
    # enables input gradients on the quantized model.
    # Ref: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    if not no_lora:
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_llm_from_checkpoint(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    is_trainable: bool = True,
    device: str = None,
):
    """Load LLM with LoRA adapters from a saved checkpoint.

    Used to load Stage 2 LoRA weights for Stage 3, or for inference.
    """
    from peft import PeftModel

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    use_quantization = (load_in_4bit or load_in_8bit) and device == "cuda"
    model_kwargs = {"trust_remote_code": True}

    # Enable Flash Attention 2 on CUDA
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if use_quantization:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

    # Load base model, resize embeddings BEFORE loading LoRA (to match saved shapes)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    # Prepare quantized model for training (same as load_llm_with_lora)
    if use_quantization and is_trainable:
        base_model = prepare_model_for_kbit_training(base_model)

    # Apply saved LoRA adapters (trainable for continued fine-tuning)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=is_trainable)

    # Move to device if not already placed (device_map handles CUDA, manual for MPS/CPU)
    if not use_quantization and device != "cuda":
        model = model.to(device)

    return model, tokenizer


def build_projector(
    speech_dim: int = SPEECH_DIM,
    llm_dim: int = QWEN3_4B_DIM,
    downsample_factor: int = 3,
    projector_type: str = "transformer",
    device: str = None,
) -> nn.Module:
    """Build the trainable projection layer between speech encoder and LLM.

    Args:
        projector_type: "transformer" (4-layer Transformer, ~39M params) or
                        "linear" (2-layer MLP, ~5M params, legacy).

    Projector stays in float32 for training stability. The SpeechLLM forward
    pass casts projector output to the LLM's dtype before merging.
    """
    if device is None:
        device = get_device()

    if projector_type == "transformer":
        projector = TransformerProjector(
            speech_dim=speech_dim,
            llm_dim=llm_dim,
            downsample_factor=downsample_factor,
        )
    else:
        projector = SpeechProjector(
            speech_dim=speech_dim,
            llm_dim=llm_dim,
            downsample_factor=downsample_factor,
        )
    return projector.to(device)
