"""
Rotary Position Embedding (RoPE).
Proposed by Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding".
Used in LLaMA, Mistral, Qwen, Gemma, and modern MNC architectures.

RoPE encodes relative token positions directly into the attention query and key vectors
via complex number rotation:
    q_rot = q * cos(m * theta) + rotate_half(q) * sin(m * theta)
"""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input:
    [-x2, x1] where x = [x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        """
        Args:
            dim: Dimension of each attention head (head_dim, e.g. 64). Must be even.
            max_position_embeddings: Maximum sequence length to precompute cache for.
            base: Base theta frequency (standard 10000.0).
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: theta_i = base ^ (-2 * (i - 1) / dim)
        # Shape: (dim // 2,)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin caches for fast indexing
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to match full head_dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cached cosine and sine tables for the given sequence slice.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim) to infer device/dtype.
            seq_len: Current sequence length.
            offset: Starting position offset (used during KV-cached generation).

        Returns:
            cos, sin tensors of shape (1, 1, seq_len, head_dim)
        """
        end_pos = offset + seq_len
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.max_position_embeddings * 2))
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        cos = self.cos_cached[offset:end_pos].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[offset:end_pos].to(dtype=x.dtype, device=x.device)

        # Unsqueeze to broadcast across (batch, num_heads, seq_len, head_dim)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to queries and keys.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Precomputed cosine tensor of shape (1, 1, seq_len, head_dim)
        sin: Precomputed sine tensor of shape (1, 1, seq_len, head_dim)

    Returns:
        Rotated (q, k) with identical shapes.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
