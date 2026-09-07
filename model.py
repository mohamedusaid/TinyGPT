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
import tiktoken


# ============================================================
# 1. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# 2. CONFIGURATION
# ============================================================

vocab_size = 50257

context_size = 64

embedding_size = 128

num_heads = 4

num_layers = 2

dropout = 0.0

batch_size = 8

learning_rate = 3e-4

max_steps = 3000

eval_interval = 100

eval_batches = 10

generate_tokens = 200


# ============================================================
# 3. TOKENIZER
# ============================================================

enc = tiktoken.get_encoding("gpt2")

EOS_TOKEN_ID = 50256


# ============================================================
# 4. LOAD STORY
# ============================================================

with open("content.txt", "r", encoding="utf-8") as f:
    text = f.read()

print()
print("Story characters:", len(text))


# Tokenize

tokens = enc.encode(text)

# Add EOS token to explicitly mark the end of the story
tokens.append(EOS_TOKEN_ID)

tokens = torch.tensor(tokens, dtype=torch.long)

print("Number of tokens:", len(tokens))

print("First 20 tokens:")
print(tokens[:20])


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(0.9 * len(tokens))

train_data = tokens[:split]
val_data = tokens[split:]

print()
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# ============================================================
# 6. BATCH CREATION
# ============================================================

def get_batch(data):

    # Random starting positions

    ix = torch.randint(
        0,
        len(data) - context_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + context_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + context_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# ============================================================
# 7. CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()

        assert embedding_size % num_heads == 0

        self.num_heads = num_heads

        self.head_size = embedding_size // num_heads

        # One projection produces Q, K and V together.
        #
        # This is faster than having three separate operations.

        self.qkv = nn.Linear(
            embedding_size,
            embedding_size * 3,
            bias=False
        )

        # Output projection

        self.proj = nn.Linear(
            embedding_size,
            embedding_size
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
# 8. FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 9. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(
            embedding_size
        )

        self.ffn = FeedForward()


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
# 10. GPT MODEL
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(num_layers)
            ]
        )

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

        x = self.blocks(x)

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


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = TinyGPT().to(device)


# ============================================================
# 12. PARAMETER COUNT
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("Model configuration:")
print("Vocabulary:", vocab_size)
print("Context:", context_size)
print("Embedding:", embedding_size)
print("Heads:", num_heads)
print("Layers:", num_layers)

print()
print(
    "Total parameters:",
    f"{num_parameters:,}"
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)


# ============================================================
# 14. EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(eval_batches):

            x, y = get_batch(data)

            _, loss = model(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# 15. INITIAL TEST
# ============================================================

x, y = get_batch(train_data)

logits, loss = model(
    x,
    y
)

print()
print("Batch input shape:")
print(x.shape)

print()
print("Batch target shape:")
print(y.shape)

print()
print("Logits shape:")
print(logits.shape)

print()
print("Initial loss:")
print(loss.item())


# ============================================================
# 16. TRAINING
# ============================================================

print()
print("Starting training...")
print()


for step in range(max_steps):

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    x, y = get_batch(train_data)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = model(
        x,
        y
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )


# ============================================================
# 17. FINAL EVALUATION
# ============================================================

losses = estimate_loss()

print()
print("Training finished.")

print()
print(
    "Final Train Loss:",
    losses["train"]
)

print(
    "Final Validation Loss:",
    losses["val"]
)


# ============================================================
# 18. TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens
):

    model.eval()

    # Tokenize prompt

    encoded = enc.encode(prompt)

    idx = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # Generate one token at a time
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # Keep only context window

        idx_cond = idx[:, -context_size:]

        # Forward

        logits, _ = model(
            idx_cond
        )

        # Last position

        logits = logits[:, -1, :]

        # Convert to probabilities

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # Sample

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Stop when the model generates EOS
        if next_token.item() == EOS_TOKEN_ID:
            print("EOS generated - stopping.")
            break

        # Append

        idx = torch.cat(
            (
                idx,
                next_token
            ),
            dim=1
        )

    # Decode

    output = enc.decode(
        idx[0].tolist()
    )

    model.train()

    return output


# ============================================================
# 19. GENERATE
# ============================================================

print()
print("Generated text:")
print()

prompt = (
    "Beneath a bruised and violet sky,"
)

generated = generate(
    model,
    prompt,
    generate_tokens
)

print(generated)


# ============================================================
# 20. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "context_size": context_size,
            "embedding_size": embedding_size,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
    },
    "tiny_gpt.pt"
)

print()
print("Model saved as tiny_gpt.pt")