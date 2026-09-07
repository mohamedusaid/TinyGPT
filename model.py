"""
Efficient Tiny GPT
------------------

Goal:
    Build a real GPT-style language model that can be trained
    efficiently on CPU.

Architecture:
    GPT tokenizer
    -> token embeddings
    -> positional embeddings
    -> Transformer blocks
        -> causal self-attention
        -> residual connection
        -> LayerNorm
        -> feed-forward network
        -> residual connection
    -> final LayerNorm
    -> language-model head
    -> logits
    -> cross entropy

This version is designed for learning and CPU experimentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        context_size=64,
        dropout=0.0
    ):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_size = embed_dim // num_heads

        # One projection produces Q, K and V together.
        #
        # This is faster than having three separate operations.

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=False
        )

        # Output projection

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(dropout)

        # Causal mask

        mask = torch.tril(
            torch.ones(
                context_size,
                context_size
            )
        )

        self.register_buffer(
            "mask",
            mask
        )


    def forward(self, x):

        B, T, C = x.shape

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ----------------------------------------------------
        # Split into heads
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        # Shapes:

        # q = B × heads × T × head_size
        # k = B × heads × T × head_size
        # v = B × heads × T × head_size


        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_size)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        weights = F.softmax(
            scores,
            dim=-1
        )

        weights = self.dropout(weights)

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        out = weights @ v

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            C
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        out = self.proj(out)

        return out


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        dropout=0.0
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        context_size=64,
        dropout=0.0
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_size=context_size,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = FeedForward(
            embed_dim=embed_dim,
            dropout=dropout
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        x = x + self.attention(
            self.ln1(x)
        )

        # ----------------------------------------------------
        # Feed-forward
        # ----------------------------------------------------

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# ============================================================
# GPT MODEL
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size=50257,
        context_size=64,
        embedding_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        embed_dim=None
    ):

        super().__init__()

        if embed_dim is not None:
            embedding_size = embed_dim

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # ----------------------------------------------------
        # Position embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_size
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embedding_size,
                num_heads=num_heads,
                context_size=context_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            embedding_size
        )

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------
        #
        # The token embedding matrix and LM head share weights.
        #
        # This saves a huge number of parameters.
        #

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        # Initialize weights

        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        # Clip context window if input exceeds context_size
        if T > self.context_size:
            idx = idx[:, -self.context_size:]
            T = self.context_size

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_vectors = self.token_embedding(idx)

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_vectors = (
            self.position_embedding(positions)
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        x = (
            token_vectors
            + position_vectors
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.ln_f(x)

        # ----------------------------------------------------
        # LM head
        # ----------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

            return logits, loss

        return logits


# ============================================================
# ARCHITECTURE SELF-TEST
# ============================================================

if __name__ == "__main__":

    print("Testing TinyGPT architecture...")

    model = TinyGPT()

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Total parameters:",
        f"{num_parameters:,}"
    )

    dummy_input = torch.randint(0, 50257, (2, 32))
    dummy_targets = torch.randint(0, 50257, (2, 32))

    logits = model(dummy_input)
    print("Forward (inference) logits shape:", logits.shape)

    logits, loss = model(dummy_input, dummy_targets)
    print(f"Forward (training) loss: {loss.item():.4f}")

    print("Self-test passed!")