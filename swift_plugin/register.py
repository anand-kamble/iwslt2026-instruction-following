"""SWIFT plugin for IWSLT 2026 Speech-LLM.

Registers:
- SpeechLLMConfig / SpeechLLMForConditionalGeneration (PreTrainedModel)
- SpeechLLMLoader (ModelLoader)
- SpeechLLMTemplate (Template)
- Model arch, model meta, template meta

Usage:
    swift sft --external_plugins swift_plugin/register.py --model_type speech_llm ...
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    SeamlessM4Tv2Model,
)

from swift.model import (
    Model,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch,
)
from swift.model.utils import use_submodel_func
from swift.template import Template, TemplateMeta, register_template, StdTemplateInputs
import torchaudio
from swift.utils import get_logger

from src.model.adapters import TransformerProjector

logger = get_logger()

SPEECH_PLACEHOLDER = "<|speech|>"


# ============================================================================
# 1. Model Config & Architecture
# ============================================================================

class SpeechLLMConfig(PretrainedConfig):
    model_type = "speech_llm"

    def __init__(
        self,
        sfm_model_name: str = "facebook/seamless-m4t-v2-large",
        llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        speech_dim: int = 1024,
        llm_dim: int = 2560,
        downsample_factor: int = 3,
        projector_num_layers: int = 4,
        projector_num_heads: int = 8,
        projector_ffn_dim: int = 2048,
        **kwargs,
    ):
        self.sfm_model_name = sfm_model_name
        self.llm_model_name = llm_model_name
        self.speech_dim = speech_dim
        self.llm_dim = llm_dim
        self.downsample_factor = downsample_factor
        self.projector_num_layers = projector_num_layers
        self.projector_num_heads = projector_num_heads
        self.projector_ffn_dim = projector_ffn_dim
        super().__init__(**kwargs)


class SpeechLLMForConditionalGeneration(PreTrainedModel):
    config_class = SpeechLLMConfig
    _no_split_modules = ["TransformerProjector"]

    def __init__(self, config: SpeechLLMConfig):
        super().__init__(config)
        self.projector = TransformerProjector(
            speech_dim=config.speech_dim,
            llm_dim=config.llm_dim,
            downsample_factor=config.downsample_factor,
            num_layers=config.projector_num_layers,
            num_heads=config.projector_num_heads,
            ffn_dim=config.projector_ffn_dim,
        )
        # speech_encoder and llm are set by SpeechLLMLoader.get_model()
        self.speech_encoder = None
        self.llm = None

    def get_input_embeddings(self):
        if self.llm is None:
            return None
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        if self.llm is None:
            return None
        return self.llm.get_output_embeddings()

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if self.llm is not None:
            self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if self.llm is not None:
            self.llm.gradient_checkpointing_disable()


# ============================================================================
# 2. Model Arch Registration
# ============================================================================

register_model_arch(
    MultiModelKeys(
        'speech_llm',
        language_model=['llm.model', 'llm.lm_head'],
        vision_tower=['speech_encoder'],
        aligner=['projector'],
    )
)


# ============================================================================
# 3. Model Loader
# ============================================================================

class SpeechLLMLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            return SpeechLLMConfig.from_pretrained(model_dir)
        return SpeechLLMConfig()

    def get_processor(self, model_dir: str, config: PretrainedConfig):
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name, trust_remote_code=True
        )
        # Store audio processor as attribute on tokenizer for Template access
        audio_processor = AutoProcessor.from_pretrained(config.sfm_model_name)
        tokenizer.audio_processor = audio_processor
        return tokenizer

    def get_model(
        self,
        model_dir: str,
        config: PretrainedConfig,
        processor,
        model_kwargs,
    ) -> PreTrainedModel:
        logger.info("Building SpeechLLM...")

        # 1. Create model shell (initializes projector)
        model = SpeechLLMForConditionalGeneration(config)

        # 2. Load speech encoder (frozen)
        logger.info(f"Loading speech encoder: {config.sfm_model_name}")
        full_sfm = SeamlessM4Tv2Model.from_pretrained(config.sfm_model_name)
        model.speech_encoder = full_sfm.speech_encoder
        del full_sfm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.speech_encoder.eval()
        for p in model.speech_encoder.parameters():
            p.requires_grad = False

        # 3. Load LLM
        logger.info(f"Loading LLM: {config.llm_model_name}")
        # Pass through model_kwargs from SWIFT (torch_dtype, device_map, quantization, etc.)
        llm_kwargs = {k: v for k, v in model_kwargs.items() if k != 'config'}
        model.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
            **llm_kwargs,
        )

        # 4. Add speech placeholder token
        if SPEECH_PLACEHOLDER not in processor.get_vocab():
            processor.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
            model.llm.resize_token_embeddings(len(processor))

        # 5. Load Stage 1 projector checkpoint if provided
        stage1_ckpt = os.environ.get('STAGE1_CHECKPOINT')
        if stage1_ckpt:
            projector_path = os.path.join(stage1_ckpt, 'projector.pt')
            if os.path.exists(projector_path):
                logger.info(f"Loading Stage 1 projector from {projector_path}")
                state_dict = torch.load(projector_path, map_location='cpu')
                missing, unexpected = model.projector.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing projector keys: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected projector keys: {unexpected}")

        # 6. Load Stage 2 LoRA checkpoint if provided
        stage2_ckpt = os.environ.get('STAGE2_CHECKPOINT')
        if stage2_ckpt:
            from peft import PeftModel
            lora_path = stage2_ckpt
            if os.path.isdir(os.path.join(stage2_ckpt, 'lora_adapters')):
                lora_path = os.path.join(stage2_ckpt, 'lora_adapters')
            logger.info(f"Loading Stage 2 LoRA from {lora_path}")
            model.llm = PeftModel.from_pretrained(
                model.llm, lora_path, is_trainable=True,
            )
            # Resize again if needed after LoRA loading
            if len(processor) > model.llm.config.vocab_size:
                model.llm.resize_token_embeddings(len(processor))

        # Delegate forward/generate/gradient_checkpointing to llm submodule
        use_submodel_func(model, 'llm')

        logger.info("SpeechLLM built successfully")
        return model


# ============================================================================
# 4. Model Registration
# ============================================================================

register_model(
    ModelMeta(
        'speech_llm',
        [ModelGroup([
            Model('local/speech-llm', 'local/speech-llm'),
        ])],
        SpeechLLMLoader,
        template='speech_llm',
        is_multimodal=True,
        model_arch='speech_llm',
        architectures=['SpeechLLMForConditionalGeneration'],
        requires=['transformers>=4.45', 'torchaudio'],
        tags=['audio'],
    )
)


# ============================================================================
# 5. Template
# ============================================================================

class SpeechLLMTemplate(Template):
    use_model = True

    def replace_tag(
        self,
        media_type: Literal['image', 'video', 'audio'],
        index: int,
        inputs: StdTemplateInputs,
    ) -> List[Union[str, List[int]]]:
        if media_type == 'audio':
            audio_path = inputs.audios[index]
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            inputs.audios[index] = waveform.squeeze(0).numpy()
            # Remove the <audio> tag from text — speech will be prepended in _post_encode
            return []
        return []

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)

        if not inputs.audios or len(inputs.audios) == 0:
            return encoded

        # Process audio through SeamlessM4T processor
        audio_processor = self.processor.audio_processor
        audio_data = inputs.audios[0]

        audio_inputs = audio_processor(
            audio=audio_data,
            sampling_rate=16000,
            return_tensors='pt',
        )
        encoded['input_features'] = audio_inputs['input_features'].squeeze(0)
        if 'attention_mask' in audio_inputs:
            encoded['input_features_attention_mask'] = audio_inputs['attention_mask'].squeeze(0)

        return encoded

    def _post_encode(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')

        # Ensure input_ids is a tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        # Get text embeddings (handles both [seq_len] and [batch, seq_len])
        embed_tokens = model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        if input_features is None:
            return {'inputs_embeds': inputs_embeds}

        dtype = inputs_embeds.dtype

        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features)
        input_features = input_features.to(device=device, dtype=dtype)

        feat_mask = inputs.get('input_features_attention_mask')
        if feat_mask is not None:
            if not isinstance(feat_mask, torch.Tensor):
                feat_mask = torch.tensor(feat_mask)
            feat_mask = feat_mask.to(device=device)

        # Run frozen speech encoder
        with torch.no_grad():
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if feat_mask is not None and feat_mask.dim() == 1:
                feat_mask = feat_mask.unsqueeze(0)

            speech_out = model.speech_encoder(
                input_features=input_features,
                attention_mask=feat_mask,
                return_dict=True,
            )
            speech_features = speech_out.last_hidden_state

        # Project to LLM dim
        speech_embeds = model.projector(speech_features)  # (1, T', llm_dim)
        speech_embeds = speech_embeds.to(dtype=dtype)

        # Handle both unbatched [seq, dim] and batched [batch, seq, dim]
        had_batch_dim = inputs_embeds.dim() == 3
        if had_batch_dim:
            # Process each sample in batch
            batch_size = inputs_embeds.shape[0]
            new_embeds = []
            new_labels_list = []
            labels = inputs.get('labels')
            if labels is not None and not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long).to(device)

            for b in range(batch_size):
                text_emb = inputs_embeds[b]  # (seq, dim)
                speech_emb = speech_embeds[b] if speech_embeds.shape[0] > b else speech_embeds[0]
                # Prepend speech before text: [speech] [text]
                merged = torch.cat([speech_emb, text_emb], dim=0)
                new_embeds.append(merged)

                if labels is not None:
                    lab = labels[b]
                    speech_labels = torch.full(
                        (speech_emb.shape[0],), -100, dtype=lab.dtype, device=device,
                    )
                    new_labels_list.append(torch.cat([speech_labels, lab]))

            # Pad to same length
            max_len = max(e.shape[0] for e in new_embeds)
            llm_dim = new_embeds[0].shape[1]
            padded_embeds = torch.zeros(batch_size, max_len, llm_dim, device=device, dtype=dtype)
            if new_labels_list:
                padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
            for b in range(batch_size):
                sl = new_embeds[b].shape[0]
                padded_embeds[b, :sl] = new_embeds[b]
                if new_labels_list:
                    padded_labels[b, :sl] = new_labels_list[b]

            result = {'inputs_embeds': padded_embeds}
            if new_labels_list:
                inputs['labels'] = padded_labels
        else:
            # Unbatched: [seq, dim]
            speech_emb = speech_embeds.squeeze(0)
            merged = torch.cat([speech_emb, inputs_embeds], dim=0)
            result = {'inputs_embeds': merged}

            if 'labels' in inputs:
                labels = inputs['labels']
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long).to(device)
                speech_labels = torch.full(
                    (speech_emb.shape[0],), -100, dtype=labels.dtype, device=device,
                )
                inputs['labels'] = torch.cat([speech_labels, labels])

        return result

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        input_features = [
            b['input_features'] for b in batch
            if b.get('input_features') is not None
        ]
        feat_masks = [
            b['input_features_attention_mask'] for b in batch
            if b.get('input_features_attention_mask') is not None
        ]
        if input_features:
            # Pad to same length and stack
            max_len = max(f.shape[0] for f in input_features)
            padded = []
            padded_masks = []
            for i, feat in enumerate(input_features):
                if feat.shape[0] < max_len:
                    pad = torch.zeros(max_len - feat.shape[0], *feat.shape[1:], dtype=feat.dtype)
                    feat = torch.cat([feat, pad], dim=0)
                padded.append(feat)
                if feat_masks:
                    mask = feat_masks[i]
                    if mask.shape[0] < max_len:
                        mask_pad = torch.zeros(max_len - mask.shape[0], dtype=mask.dtype)
                        mask = torch.cat([mask, mask_pad], dim=0)
                    padded_masks.append(mask)
            res['input_features'] = torch.stack(padded)
            if padded_masks:
                res['input_features_attention_mask'] = torch.stack(padded_masks)
        return res


# ============================================================================
# 6. Template Registration
# ============================================================================

register_template(
    TemplateMeta(
        'speech_llm',
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        default_system=None,
        stop_words=['<|endoftext|>'],
        template_cls=SpeechLLMTemplate,
    )
)
