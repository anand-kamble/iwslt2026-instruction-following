"""Adapter layers between Seamless M4T v2 speech encoder and Qwen3 LLM.

The speech encoder outputs 1024-dim features at ~62.5 tokens/sec.
Qwen3-4B has a hidden_size of 2560.
"""

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

    def forward(self, speech_features: torch.Tensor) -> torch.Tensor:
        """Project speech features to LLM embedding space.

        Args:
            speech_features: (batch, seq_len, speech_dim)

        Returns:
            (batch, seq_len // downsample_factor, llm_dim)
        """
        x = speech_features

        if self.downsample is not None:
            # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.downsample(x)
            x = x.transpose(1, 2)

        return self.proj(x)
