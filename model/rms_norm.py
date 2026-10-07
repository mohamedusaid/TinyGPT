"""
Root Mean Square Normalization (RMSNorm).
Proposed by Zhang and Sennrich (2019). Modern zero-mean bias-free normalization layer.

Unlike LayerNorm, RMSNorm does not shift activations by their mean and has no bias vector:
    y = (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: Dimension of the hidden features (d_model).
            eps: Small epsilon added to variance to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate root mean square along last dimension
        # x.pow(2).mean(-1, keepdim=True) calculates (1/d) * sum(x_i^2)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor scaled by learnable parameter gamma (weight).
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
