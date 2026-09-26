"""
Grouped-Query Attention (GQA) with PyTorch Scaled Dot-Product Attention (SDPA).
Proposed by Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints".
Used in LLaMA 3, Mistral, Gemma 2, and Qwen 2.5.

Allows multiple Query heads to share a single Key/Value head (e.g. 16 Q heads share 4 KV heads = 4:1 ratio).
This saves 75% of KV cache memory and memory-bandwidth during autoregressive inference decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TinyGPTConfig
from model.rotary import apply_rotary_pos_emb
from model.kv_cache import KVCache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands Key/Value heads across the GQA group ratio:
    (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.gqa_ratio = config.gqa_ratio

        # Linear projections
        self.w_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.w_k = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.w_v = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.w_o = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input activations (batch, seq_len, hidden_size).
            cos: RoPE cosine table (1, 1, seq_len, head_dim).
            sin: RoPE sine table (1, 1, seq_len, head_dim).
            kv_cache: Optional KVCache instance for autoregressive generation.

        Returns:
            Attention output tensor (batch, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Reshape to multi-head format: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Position Embedding (RoPE)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Handle KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
            # When generating step-by-step (seq_len == 1) and attending to past tokens,
            # causal masking is not needed because the single query attends to all prior keys.
            is_causal = (seq_len > 1)
        else:
            is_causal = True

        # 5. Expand Key and Value heads for GQA (if num_kv_heads < num_heads)
        k = repeat_kv(k, self.gqa_ratio)
        v = repeat_kv(v, self.gqa_ratio)

        # 6. Scaled Dot-Product Attention (SDPA)
        # PyTorch chooses optimal backend: MemoryEfficient (Cutlass) on T4, FlashAttention-2 on Ampere/Hopper
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )

        # 7. Reshape and project out
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.w_o(attn_output)
