"""Adapter layers between Seamless M4T v2 speech encoder and Qwen3 LLM.

The speech encoder outputs 1024-dim features at ~62.5 tokens/sec.
Qwen3-4B has a hidden_size of 2560.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpeechProjector(nn.Module):
    """Projects speech encoder outputs into the LLM's embedding space.

    Architecture: Linear -> GELU -> Linear with optional downsampling
    to reduce the sequence length of speech tokens.
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        llm_dim: int = 2560,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        if downsample_factor > 1:
            self.downsample = nn.Conv1d(
                speech_dim,
                speech_dim,
                kernel_size=downsample_factor,
                stride=downsample_factor,
            )
        else:
            self.downsample = None

        self.proj = nn.Sequential(
            nn.Linear(speech_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(
        self, speech_features: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project speech features to LLM embedding space.

        Args:
            speech_features: (batch, seq_len, speech_dim)
            lengths: (batch,) valid frame count per sample

        Returns:
            (projected_features, updated_lengths)
        """
        x = speech_features

        if self.downsample is not None:
            # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.downsample(x)
            x = x.transpose(1, 2)
            if lengths is not None:
                lengths = lengths // self.downsample_factor

        return self.proj(x), lengths


class TransformerProjector(nn.Module):
    """Projects speech encoder outputs into the LLM's embedding space.

    Architecture: Frame averaging downsample -> Transformer encoder -> Linear projection.
    Based on NAVER IWSLT 2025 constrained system projector.
    Uses frame averaging (not Conv1d) for MPS compatibility.
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        llm_dim: int = 2560,
        downsample_factor: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=speech_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-layer norm, matching NAVER IWSLT 2025 (arxiv 2506.01808)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(speech_dim, llm_dim)

    def forward(
        self, speech_features: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project speech features to LLM embedding space.

        Args:
            speech_features: (batch, seq_len, speech_dim)
            lengths: (batch,) valid frame count per sample

        Returns:
            (projected_features, updated_lengths)
        """
        x = speech_features

        # Frame averaging downsample (MPS-safe, matches NAVER's approach)
        if self.downsample_factor > 1:
            B, T, D = x.shape
            T_down = T // self.downsample_factor
            x = x[:, :T_down * self.downsample_factor].reshape(
                B, T_down, self.downsample_factor, D
            ).mean(dim=2)
            if lengths is not None:
                lengths = lengths // self.downsample_factor

        # Mask padding in self-attention so real positions don't attend to padding
        if lengths is not None:
            T_curr = x.shape[1]
            # nn.TransformerEncoder convention: True = position to ignore
            padding_mask = torch.arange(T_curr, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        return self.output_proj(x), lengths
