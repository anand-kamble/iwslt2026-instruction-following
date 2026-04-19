"""SpeechLLM: end-to-end model combining speech encoder + projector + LLM.

During training:
1. Audio → Seamless M4T v2 encoder (frozen) → speech features (B, T, 1024)
2. Speech features → projector (trainable) → LLM embeddings (B, T', 2560)
3. Concatenate [speech_embeds, instruction_embeds] → Qwen3-4B (LoRA) → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Special token to mark where speech features are inserted in the prompt
SPEECH_PLACEHOLDER = "<|speech|>"


class SpeechLLM(nn.Module):
    """Wraps speech encoder, projector, and LLM for end-to-end forward pass."""

    def __init__(self, speech_encoder, projector, llm, tokenizer):
        super().__init__()
        self.speech_encoder = speech_encoder  # frozen
        self.projector = projector  # trainable
        self.llm = llm  # LoRA trainable
        self.tokenizer = tokenizer

    @property
    def device(self):
        """Return device of the LLM for Trainer compatibility."""
        return next(self.llm.parameters()).device

    @property
    def dtype(self):
        """Return dtype of the LLM for Trainer compatibility.

        For quantized models, parameters are stored in uint8/int8 but compute
        happens in the configured compute dtype (bfloat16). Fall back to the
        first floating-point parameter's dtype.
        """
        for p in self.llm.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
        return torch.float32

    @staticmethod
    def _has_trainable_params(module: nn.Module) -> bool:
        return any(p.requires_grad for p in module.parameters())

    def train(self, mode: bool = True):
        """Keep frozen submodules in eval mode even during training.

        Trainer calls `model.train()` each step/epoch, which recursively flips
        all children to train mode. The speech encoder is always frozen and
        should stay deterministic (dropout disabled). In Stage 1, the LLM may
        also be fully frozen and should likewise remain in eval mode.
        """
        super().train(mode)

        # Frozen speech encoder should never run with dropout enabled.
        self.speech_encoder.eval()

        # Stage 1 projector-only training freezes the whole LLM.
        if not self._has_trainable_params(self.llm):
            self.llm.eval()
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the LLM."""
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate gradient checkpointing to the LLM."""
        self.llm.gradient_checkpointing_disable()

    @staticmethod
    def _strip_to_loss_if_training(outputs, labels):
        """Avoid returning full logits during training to reduce DDP memory."""
        if labels is None:
            return outputs
        if not isinstance(outputs, dict) and not hasattr(outputs, "loss"):
            return outputs
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return {"loss": loss}

    def _get_embed_tokens(self):
        """Get the embedding layer, handling both PEFT-wrapped and vanilla models."""
        # PEFT/Unsloth: llm.model.model.embed_tokens
        # Vanilla: llm.model.embed_tokens
        base = self.llm
        if hasattr(base, "model") and hasattr(base.model, "model"):
            return base.model.model.embed_tokens
        elif hasattr(base, "model"):
            return base.model.embed_tokens
        raise AttributeError("Cannot find embed_tokens in LLM structure")

    def _compute_speech_output_lengths(self, input_lengths):
        """Compute speech encoder output lengths from input feature lengths.

        Accounts for the adapter's convolutional subsampling inside the
        SeamlessM4Tv2 speech encoder.
        """
        config = self.speech_encoder.config
        kernel_size = getattr(config, "adaptor_kernel_size", 3)
        stride = getattr(config, "adaptor_stride", 2)
        num_layers = getattr(config, "num_adapter_layers", 1)
        pad = kernel_size // 2

        lengths = input_lengths.float()
        for _ in range(num_layers):
            lengths = torch.floor((lengths + 2 * pad - kernel_size) / stride + 1)
        return lengths.long()

    @torch.no_grad()
    def encode_speech(self, input_features, attention_mask=None):
        """Extract speech features from audio using the frozen encoder.

        Args:
            input_features: (batch, seq_len, 160) from processor
            attention_mask: (batch, seq_len)

        Returns:
            (speech_features, speech_lengths) where speech_lengths is
            the number of valid (non-padding) frames per sample.
        """
        output = self.speech_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        speech_features = output.last_hidden_state

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            speech_lengths = self._compute_speech_output_lengths(input_lengths)
            speech_lengths = speech_lengths.clamp(max=speech_features.shape[1])
        else:
            speech_lengths = torch.full(
                (speech_features.shape[0],), speech_features.shape[1],
                device=speech_features.device, dtype=torch.long,
            )

        return speech_features, speech_lengths

    def forward(
        self,
        input_features=None,
        input_features_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """Forward pass for training.

        Supports two modes:
        - Speech + text: replaces SPEECH_PLACEHOLDER tokens with projected speech features
        - Text-only: standard LLM forward (for interleaved text batches in Stage 3)

        Args:
            input_features: (batch, audio_seq, 160) — audio mel features (None for text-only)
            input_features_attention_mask: (batch, audio_seq) — audio mask
            input_ids: (batch, text_seq) — tokenized instruction + response
            attention_mask: (batch, text_seq) — text attention mask
            labels: (batch, text_seq) — labels for language modeling loss
        """
        device = self.device

        # Text-only forward (no audio) — used for interleaved text batches
        if input_features is None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if self.training:
                return self._strip_to_loss_if_training(outputs, labels)
            return outputs

        if input_features.device != device:
            input_features = input_features.to(device)
            input_features_attention_mask = input_features_attention_mask.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)

        # 1. Encode speech (skip if pre-extracted features provided)
        speech_dim = getattr(self.speech_encoder.config, "hidden_size", 1024)
        if input_features.shape[-1] == speech_dim:
            # Pre-extracted encoder outputs — skip the frozen encoder
            speech_features = input_features
            speech_lengths = input_features_attention_mask.sum(dim=1)
        else:
            speech_features, speech_lengths = self.encode_speech(
                input_features, input_features_attention_mask
            )

        # 2. Project to LLM space (projector masks padding internally)
        target_dtype = self.dtype
        speech_embeds, speech_lengths = self.projector(speech_features, speech_lengths)
        speech_embeds = speech_embeds.to(target_dtype)

        # 3. Get text embeddings from LLM (already in target_dtype)
        text_embeds = self._get_embed_tokens()(input_ids)

        # 4. Find speech placeholder positions and replace with speech embeddings
        inputs_embeds, new_attention_mask, new_labels = self._merge_speech_and_text(
            speech_embeds, text_embeds, input_ids, attention_mask, labels,
            speech_lengths=speech_lengths,
        )

        # 5. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )
        if self.training:
            return self._strip_to_loss_if_training(outputs, new_labels)
        return outputs

    def _merge_speech_and_text(
        self, speech_embeds, text_embeds, input_ids, attention_mask, labels,
        speech_lengths=None,
    ):
        """Replace speech placeholder tokens with actual speech embeddings.

        For each sample in the batch, find the SPEECH_PLACEHOLDER token(s)
        and replace them with the projected speech embeddings. Only valid
        (non-padding) speech frames are inserted.

        Uses F.pad + torch.stack for proper autograd tracking (no in-place ops
        on untracked tensors).
        """
        speech_token_id = self.tokenizer.convert_tokens_to_ids(SPEECH_PLACEHOLDER)
        if speech_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(
                f"{SPEECH_PLACEHOLDER} not in tokenizer vocabulary. "
                "Call tokenizer.add_tokens([SPEECH_PLACEHOLDER]) first."
            )
        batch_size = text_embeds.shape[0]
        device = text_embeds.device

        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for i in range(batch_size):
            # Find placeholder positions
            placeholder_mask = input_ids[i] == speech_token_id
            placeholder_indices = placeholder_mask.nonzero(as_tuple=True)[0]

            if len(placeholder_indices) == 0:
                # No speech — just use text
                new_embeds_list.append(text_embeds[i])
                new_mask_list.append(attention_mask[i])
                if labels is not None:
                    new_labels_list.append(labels[i])
                continue

            # Per-sample valid speech length (exclude padding frames)
            sample_speech_len = (
                speech_lengths[i].item() if speech_lengths is not None
                else speech_embeds.shape[1]
            )

            # Replace first placeholder with speech embeddings, remove the rest
            first_pos = placeholder_indices[0].item()
            num_placeholders = len(placeholder_indices)

            # Parts: [text_before_placeholder] [valid_speech_embeds] [text_after_placeholders]
            before = text_embeds[i, :first_pos]
            after = text_embeds[i, first_pos + num_placeholders :]

            merged = torch.cat([before, speech_embeds[i, :sample_speech_len], after], dim=0)
            new_embeds_list.append(merged)

            # Rebuild attention mask
            mask_before = attention_mask[i, :first_pos]
            mask_speech = torch.ones(sample_speech_len, device=device, dtype=attention_mask.dtype)
            mask_after = attention_mask[i, first_pos + num_placeholders :]
            new_mask_list.append(torch.cat([mask_before, mask_speech, mask_after]))

            # Rebuild labels (speech tokens get -100 = ignored)
            if labels is not None:
                labels_before = labels[i, :first_pos]
                labels_speech = torch.full(
                    (sample_speech_len,), -100, device=device, dtype=labels.dtype
                )
                labels_after = labels[i, first_pos + num_placeholders :]
                new_labels_list.append(
                    torch.cat([labels_before, labels_speech, labels_after])
                )

        # Pad using F.pad + torch.stack for proper gradient tracking
        max_len = max(e.shape[0] for e in new_embeds_list)

        padded_embeds = torch.stack([
            F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in new_embeds_list
        ])
        padded_mask = torch.stack([
            F.pad(m, (0, max_len - m.shape[0])) for m in new_mask_list
        ])
        padded_labels = (
            torch.stack([
                F.pad(l, (0, max_len - l.shape[0]), value=-100)
                for l in new_labels_list
            ])
            if labels is not None and new_labels_list
            else None
        )

        return padded_embeds, padded_mask, padded_labels

    @torch.inference_mode()
    def generate(self, input_features, input_features_attention_mask, input_ids, attention_mask, **generate_kwargs):
        """Generate text from speech + instruction input."""
        speech_features, speech_lengths = self.encode_speech(input_features, input_features_attention_mask)
        # Cast to projector dtype (fp32) — projector trained in fp32 for stability
        speech_features = speech_features.to(next(self.projector.parameters()).dtype)
        speech_embeds, speech_lengths = self.projector(speech_features, speech_lengths)
        speech_embeds = speech_embeds.to(self.dtype)

        text_embeds = self._get_embed_tokens()(input_ids)

        inputs_embeds, new_attention_mask, _ = self._merge_speech_and_text(
            speech_embeds, text_embeds, input_ids, attention_mask, labels=None,
            speech_lengths=speech_lengths,
        )

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            **generate_kwargs,
        )
        return outputs
