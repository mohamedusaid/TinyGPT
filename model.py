import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 1. CONFIGURATION
# ============================================================

vocab_size = 50257

embedding_size = 128
context_size = 32

num_heads = 4
num_layers = 2

batch_size = 16

learning_rate = 3e-4
weight_decay = 0.01

training_steps = 3000

eval_interval = 200

generation_length = 300

temperature = 0.8
top_k = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# 2. LOAD STORY
# ============================================================

with open("content.txt", "r", encoding="utf-8") as f:
    story = f.read()

print("\nStory characters:", len(story))


# ============================================================
# 3. TOKENIZER
# ============================================================

encoding = tiktoken.get_encoding("gpt2")

tokens = encoding.encode(story)

tokens = torch.tensor(tokens, dtype=torch.long)

print("Number of tokens:", len(tokens))

print("First 20 tokens:")
print(tokens[:20])


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(len(tokens) * 0.9)

train_tokens = tokens[:split_index]
val_tokens = tokens[split_index:]

print("\nTraining tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))


# ============================================================
# 5. GET BATCH
# ============================================================

def get_batch(data):

    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - context_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + context_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + context_size + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


# ============================================================
# 6. MULTI-HEAD SELF ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = embedding_size // num_heads

        self.qkv = nn.Linear(
            embedding_size,
            embedding_size * 3
        )

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.num_heads = num_heads
        self.head_size = head_size

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_size,
                    context_size
                )
            )
        )

    def forward(self, x):

        batch, tokens_count, channels = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ------------------------------------------------
        # Reshape into heads
        # ------------------------------------------------

        q = q.view(
            batch,
            tokens_count,
            self.num_heads,
            self.head_size
        )

        k = k.view(
            batch,
            tokens_count,
            self.num_heads,
            self.head_size
        )

        v = v.view(
            batch,
            tokens_count,
            self.num_heads,
            self.head_size
        )

        # Move heads before sequence dimension

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ------------------------------------------------
        # Attention scores
        # ------------------------------------------------

        scores = q @ k.transpose(-2, -1)

        scores = scores / (self.head_size ** 0.5)

        # ------------------------------------------------
        # Causal masking
        # ------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:tokens_count, :tokens_count] == 0,
            float("-inf")
        )

        # ------------------------------------------------
        # Softmax
        # ------------------------------------------------

        attention = F.softmax(
            scores,
            dim=-1
        )

        # ------------------------------------------------
        # Weighted values
        # ------------------------------------------------

        output = attention @ v

        # ------------------------------------------------
        # Concatenate heads
        # ------------------------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous()

        output = output.view(
            batch,
            tokens_count,
            channels
        )

        # ------------------------------------------------
        # Output projection
        # ------------------------------------------------

        output = self.projection(output)

        return output


# ============================================================
# 7. FEED FORWARD NETWORK
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
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 8. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = MultiHeadAttention()

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward()

    def forward(self, x):

        # Attention residual

        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed-forward residual

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


# ============================================================
# 9. TINY GPT MODEL
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # ------------------------------------------------
        # Token embeddings
        # ------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # ------------------------------------------------
        # Position embeddings
        # ------------------------------------------------

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_size
        )

        # ------------------------------------------------
        # Transformer blocks
        # ------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(num_layers)
            ]
        )

        # ------------------------------------------------
        # Final normalization
        # ------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # ------------------------------------------------
        # Language model head
        # ------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens, targets=None):

        batch, tokens_count = tokens.shape

        # ------------------------------------------------
        # Token embeddings
        # ------------------------------------------------

        token_vectors = self.token_embedding(tokens)

        # ------------------------------------------------
        # Position embeddings
        # ------------------------------------------------

        positions = torch.arange(
            tokens_count,
            device=tokens.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        # ------------------------------------------------
        # Combine
        # ------------------------------------------------

        x = token_vectors + position_vectors

        # ------------------------------------------------
        # Transformer
        # ------------------------------------------------

        x = self.blocks(x)

        # ------------------------------------------------
        # Final LayerNorm
        # ------------------------------------------------

        x = self.final_layer_norm(x)

        # ------------------------------------------------
        # Vocabulary logits
        # ------------------------------------------------

        logits = self.lm_head(x)

        # ------------------------------------------------
        # Loss
        # ------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

        return logits, loss


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = TinyGPT().to(device)

print("\nNumber of parameter tensors:", len(
    list(model.parameters())
))

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Total parameters:", total_parameters)


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# ============================================================
# 12. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(data):

    model.eval()

    x, y = get_batch(data)

    logits, loss = model(
        x,
        y
    )

    model.train()

    return loss.item()


# ============================================================
# 13. FIRST BATCH TEST
# ============================================================

x, y = get_batch(train_tokens)

print("\nBatch input shape:")
print(x.shape)

print("\nBatch target shape:")
print(y.shape)


# ============================================================
# 14. INITIAL LOSS
# ============================================================

logits, loss = model(
    x,
    y
)

print("\nLogits shape:")
print(logits.shape)

print("\nInitial loss:")
print(loss.item())


# ============================================================
# 15. TRAINING
# ============================================================

print("\nStarting training...\n")

for step in range(training_steps):

    # Get training batch

    x, y = get_batch(train_tokens)

    # Forward pass

    logits, loss = model(
        x,
        y
    )

    # Clear old gradients

    optimizer.zero_grad()

    # Backpropagation

    loss.backward()

    # Update parameters

    optimizer.step()

    # Print progress

    if step % 100 == 0:

        validation_loss = evaluate(
            val_tokens
        )

        print(
            f"Step {step:4d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {validation_loss:.4f}"
        )


# ============================================================
# 16. FINAL EVALUATION
# ============================================================

final_train_loss = evaluate(
    train_tokens
)

final_val_loss = evaluate(
    val_tokens
)

print("\nTraining finished.")

print(
    f"Final training loss: "
    f"{final_train_loss:.4f}"
)

print(
    f"Final validation loss: "
    f"{final_val_loss:.4f}"
)


# ============================================================
# 17. GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate(
    starting_tokens,
    max_new_tokens,
    temperature=1.0,
    top_k=None
):

    model.eval()

    tokens = starting_tokens.clone().to(device)

    for _ in range(max_new_tokens):

        # ------------------------------------------------
        # Keep only context window
        # ------------------------------------------------

        context = tokens[
            -context_size:
        ]

        # Add batch dimension

        context = context.unsqueeze(0)

        # ------------------------------------------------
        # Model prediction
        # ------------------------------------------------

        logits, _ = model(context)

        # Last position

        logits = logits[:, -1, :]

        # ------------------------------------------------
        # Temperature
        # ------------------------------------------------

        logits = logits / temperature

        # ------------------------------------------------
        # Top-k sampling
        # ------------------------------------------------

        if top_k is not None:

            values, indices = torch.topk(
                logits,
                top_k
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

        # ------------------------------------------------
        # Convert to probabilities
        # ------------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # ------------------------------------------------
        # Sample next token
        # ------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        next_token = next_token.squeeze(0)

        # ------------------------------------------------
        # Append
        # ------------------------------------------------

        tokens = torch.cat(
            [
                tokens,
                next_token
            ]
        )

    model.train()

    return tokens


# ============================================================
# 18. GENERATE STORY
# ============================================================

print("\nGenerating text...\n")

# Start with first 8 tokens from the story

starting_tokens = train_tokens[:8]

generated_tokens = generate(
    starting_tokens,
    generation_length,
    temperature=temperature,
    top_k=top_k
)

generated_text = encoding.decode(
    generated_tokens.tolist()
)

print(generated_text)


# ============================================================
# 19. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "config": {
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "context_size": context_size,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
    },
    "tiny_gpt.pt"
)

print("\nModel saved as tiny_gpt.pt")