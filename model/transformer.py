"""
Full TinyGPT-500M Causal Language Model.
Includes TransformerBlock with pre-norm RMSNorm, GQA with native SDPA, SwiGLU MLP,
scaled residual projection initialization, and untied LM head.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TinyGPTConfig
from model.rms_norm import RMSNorm
from model.rotary import RotaryEmbedding
from model.attention import GroupedQueryAttention
from model.mlp import SwiGLU
from model.kv_cache import KVCache


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyGPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        # Pre-norm Self Attention with residual connection
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(norm_x, cos=cos, sin=sin, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-norm SwiGLU MLP with residual connection
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x


class TinyGPT500M(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary Positional Embedding (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # 30 Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final RMS Normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head (Language Modeling Head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Scale residual projections for deep networks:
        # Scale by 1 / sqrt(2 * num_layers) to prevent activation growth
        residual_scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
        for name, param in self.named_parameters():
            if "w_o.weight" in name or "w_down.weight" in name:
                with torch.no_grad():
                    param.data.mul_(residual_scale)

    def _init_weights(self, module: nn.Module):
        """
        Initializes weights using normal distribution with mean 0.0 and std 0.02.
        Biases are initialized to zero if present.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> CausalLMOutput:
        """
        Forward pass for next-token prediction and cross-entropy loss computation.

        Args:
            input_ids: Token ID tensor of shape (batch, seq_len).
            targets: Optional ground-truth next token IDs of shape (batch, seq_len).
            kv_cache: Optional KVCache instance for inference.

        Returns:
            CausalLMOutput with logits and optional loss.
        """
        batch_size, seq_len = input_ids.shape

        # Sequence offset for KV cache
        offset = kv_cache.seq_len if kv_cache is not None else 0

        # Embed tokens
        h = self.embed_tokens(input_ids)

        # Get precomputed RoPE frequencies for current position slice
        cos, sin = self.rotary_emb(h, seq_len=seq_len, offset=offset)

        # Pass through all 30 transformer layers
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training and kv_cache is None:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, cos, sin, None, use_reentrant=False
                )
            else:
                h = layer(h, cos=cos, sin=sin, kv_cache=kv_cache)

        # Final normalization
        h = self.norm(h)

        # Compute next-token logits
        logits = self.lm_head(h)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Shift targets are handled by caller, or compute direct cross entropy:
            # Flatten to (batch * seq_len, vocab_size) and (batch * seq_len,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss)

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
