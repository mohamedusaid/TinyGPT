import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 50257
CONTEXT_SIZE = 64
EMBEDDING_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 2


# ============================================================
# ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.num_heads = NUM_HEADS
        self.head_size = EMBEDDING_SIZE // NUM_HEADS

        # IMPORTANT:
        # The trained model uses bias=False
        self.qkv = nn.Linear(
            EMBEDDING_SIZE,
            EMBEDDING_SIZE * 3,
            bias=False
        )

        self.proj = nn.Linear(
            EMBEDDING_SIZE,
            EMBEDDING_SIZE
        )

        # IMPORTANT:
        # The checkpoint contains mask with shape [64, 64]
        mask = torch.tril(
            torch.ones(
                CONTEXT_SIZE,
                CONTEXT_SIZE
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

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

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

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = q @ k.transpose(
            -2,
            -1
        )

        scores = scores / (
            self.head_size ** 0.5
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Attention weights
        # ----------------------------------------------------

        weights = F.softmax(
            scores,
            dim=-1
        )

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

        # Output projection

        out = self.proj(out)

        return out


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                EMBEDDING_SIZE,
                EMBEDDING_SIZE * 4
            ),

            nn.GELU(),

            nn.Linear(
                EMBEDDING_SIZE * 4,
                EMBEDDING_SIZE
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            EMBEDDING_SIZE
        )

        self.attention = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(
            EMBEDDING_SIZE
        )

        self.ffn = FeedForward()

    def forward(self, x):

        # Attention residual

        x = x + self.attention(
            self.ln1(x)
        )

        # FFN residual

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# ============================================================
# TINY GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_SIZE
        )

        # ----------------------------------------------------
        # Position embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            CONTEXT_SIZE,
            EMBEDDING_SIZE
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(

            *[
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            EMBEDDING_SIZE
        )

        # ----------------------------------------------------
        # Output head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            EMBEDDING_SIZE,
            VOCAB_SIZE,
            bias=False
        )

    def forward(self, idx):

        B, T = idx.shape

        # Keep within context length

        if T > CONTEXT_SIZE:

            idx = idx[:, -CONTEXT_SIZE:]

            T = CONTEXT_SIZE

        # Token embeddings

        token_embeddings = self.token_embedding(idx)

        # Position embeddings

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Combine

        x = (
            token_embeddings
            + position_embeddings
        )

        # Transformer

        x = self.blocks(x)

        # Final normalization

        x = self.ln_f(x)

        # Vocabulary logits

        logits = self.lm_head(x)

        return logits


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

enc = tiktoken.get_encoding(
    "gpt2"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    "tiny_gpt.pt",
    map_location=device
)

print("Checkpoint keys:")

for key in checkpoint.keys():

    print(" ", key)


# ============================================================
# CREATE MODEL
# ============================================================

model = TinyGPT().to(device)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print("Model loaded successfully.")

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.7,
    top_k=20
):

    # Encode prompt

    token_ids = enc.encode(
        prompt
    )

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # Keep last 64 tokens

        idx_cond = idx[:, -CONTEXT_SIZE:]

        # Forward pass

        logits = model(
            idx_cond
        )

        # Only last position

        logits = logits[:, -1, :]

        # Temperature

        logits = logits / temperature

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        if top_k is not None:

            values, indices = torch.topk(
                logits,
                min(
                    top_k,
                    logits.size(-1)
                )
            )

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            logits = filtered_logits

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Sample next token
        # ----------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append

        idx = torch.cat(
            [
                idx,
                next_token
            ],
            dim=1
        )

    # Decode

    return enc.decode(
        idx[0].tolist()
    )


# ============================================================
# QUESTION LOOP
# ============================================================

print()
print("=" * 60)
print("TinyGPT Question Answering")
print("=" * 60)

print()
print("Ask a question about the story.")
print("Type 'exit' to quit.")
print()


while True:

    question = input(
        "You: "
    ).strip()

    if question.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print(
            "Goodbye!"
        )

        break

    if not question:

        continue

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = (
        "Question: "
        + question
        + "\nAnswer:"
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = generate(
        prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=20
    )

    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    if "Answer:" in result:

        answer = result.split(
            "Answer:",
            1
        )[1]

    else:

        answer = result

    print()
    print(
        "TinyGPT:",
        answer.strip()
    )
    print()