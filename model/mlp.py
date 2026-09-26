"""
SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.
Proposed by Shazeer (2020), "GLU Variants Improve Transformer".
Used in LLaMA, Mistral, Qwen, and Gemma.

Replaces standard 2-layer GELU MLP with a 3-matrix gated formulation:
    SwiGLU(x) = (SiLU(W_gate * x) * (W_up * x)) * W_down
Provides higher parameter expressivity and better training loss convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TinyGPTConfig


class SwiGLU(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.w_up = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.w_down = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_size).
        Returns:
            Output tensor (batch, seq_len, hidden_size).
        """
        # Element-wise product of SiLU-activated gate and up-projection
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
