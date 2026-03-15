"""SpeechLLM: end-to-end model combining speech encoder + projector + LLM.

During training:
1. Audio → Seamless M4T v2 encoder (frozen) → speech features (B, T, 1024)
2. Speech features → projector (trainable) → LLM embeddings (B, T', 2560)
3. Concatenate [speech_embeds, instruction_embeds] → Qwen3-4B (LoRA) → output
"""

import torch
import torch.nn as nn

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
        """Return dtype of the LLM for Trainer compatibility."""
        return next(self.llm.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the LLM."""
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate gradient checkpointing to the LLM."""
        self.llm.gradient_checkpointing_disable()

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

    @torch.no_grad()
    def encode_speech(self, input_features, attention_mask=None):
        """Extract speech features from audio using the frozen encoder.

        Args:
            input_features: (batch, seq_len, 160) from processor
            attention_mask: (batch, seq_len)

        Returns:
            (batch, T_down, 1024) speech features after adapter
        """
        output = self.speech_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state

    def forward(
        self,
        input_features,
        input_features_attention_mask,
        input_ids,
        attention_mask,
        labels=None,
    ):
        """Forward pass for training.

        This replaces SPEECH_PLACEHOLDER tokens in the input with projected
        speech features from the encoder.

        Args:
            input_features: (batch, audio_seq, 160) — audio mel features
            input_features_attention_mask: (batch, audio_seq) — audio mask
            input_ids: (batch, text_seq) — tokenized instruction + response
            attention_mask: (batch, text_seq) — text attention mask
            labels: (batch, text_seq) — labels for language modeling loss
        """
        # Move inputs to model device (MPS/CUDA) if needed
        device = self.device
        if input_features.device != device:
            input_features = input_features.to(device)
            input_features_attention_mask = input_features_attention_mask.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)

        # 1. Encode speech
        speech_features = self.encode_speech(
            input_features, input_features_attention_mask
        )
        speech_embeds = self.projector(speech_features)  # (B, T', llm_dim)

        # 2. Get text embeddings from LLM
        text_embeds = self._get_embed_tokens()(input_ids)  # (B, S, llm_dim)

        # 3. Find speech placeholder positions and replace with speech embeddings
        inputs_embeds, new_attention_mask, new_labels = self._merge_speech_and_text(
            speech_embeds, text_embeds, input_ids, attention_mask, labels
        )

        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )
        return outputs

    def _merge_speech_and_text(
        self, speech_embeds, text_embeds, input_ids, attention_mask, labels
    ):
        """Replace speech placeholder tokens with actual speech embeddings.

        For each sample in the batch, find the SPEECH_PLACEHOLDER token(s)
        and replace them with the projected speech embeddings.
        """
        speech_token_id = self.tokenizer.convert_tokens_to_ids(SPEECH_PLACEHOLDER)
        batch_size = text_embeds.shape[0]
        speech_len = speech_embeds.shape[1]
        llm_dim = text_embeds.shape[2]
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

            # Replace first placeholder with speech embeddings, remove the rest
            first_pos = placeholder_indices[0].item()

            # Parts: [text_before_placeholder] [speech_embeds] [text_after_placeholders]
            before = text_embeds[i, :first_pos]
            after = text_embeds[i, first_pos + len(placeholder_indices) :]

            merged = torch.cat([before, speech_embeds[i], after], dim=0)
            new_embeds_list.append(merged)

            # Rebuild attention mask
            mask_before = attention_mask[i, :first_pos]
            mask_speech = torch.ones(speech_len, device=device, dtype=attention_mask.dtype)
            mask_after = attention_mask[i, first_pos + len(placeholder_indices) :]
            new_mask_list.append(torch.cat([mask_before, mask_speech, mask_after]))

            # Rebuild labels (speech tokens get -100 = ignored)
            if labels is not None:
                labels_before = labels[i, :first_pos]
                labels_speech = torch.full(
                    (speech_len,), -100, device=device, dtype=labels.dtype
                )
                labels_after = labels[i, first_pos + len(placeholder_indices) :]
                new_labels_list.append(
                    torch.cat([labels_before, labels_speech, labels_after])
                )

        # Pad to same length
        max_len = max(e.shape[0] for e in new_embeds_list)

        padded_embeds = torch.zeros(batch_size, max_len, llm_dim, device=device)
        padded_mask = torch.zeros(batch_size, max_len, device=device, dtype=attention_mask.dtype)
        padded_labels = (
            torch.full((batch_size, max_len), -100, device=device, dtype=torch.long)
            if labels is not None
            else None
        )

        for i in range(batch_size):
            seq_len = new_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = new_embeds_list[i]
            padded_mask[i, :seq_len] = new_mask_list[i]
            if padded_labels is not None and new_labels_list:
                padded_labels[i, :seq_len] = new_labels_list[i]

        return padded_embeds, padded_mask, padded_labels

    @torch.inference_mode()
    def generate(self, input_features, input_features_attention_mask, input_ids, attention_mask, **generate_kwargs):
        """Generate text from speech + instruction input."""
        speech_features = self.encode_speech(input_features, input_features_attention_mask)
        speech_embeds = self.projector(speech_features)

        text_embeds = self._get_embed_tokens()(input_ids)

        inputs_embeds, new_attention_mask, _ = self._merge_speech_and_text(
            speech_embeds, text_embeds, input_ids, attention_mask, labels=None
        )

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            **generate_kwargs,
        )
        return outputs
