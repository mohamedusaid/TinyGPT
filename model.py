import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 1. Configuration
# ============================================================

vocab_size = 50257
embedding_size = 64
context_size = 8

num_heads = 4
head_size = embedding_size // num_heads

batch_size = 16
learning_rate = 0.001
training_steps = 2000


# ============================================================
# 2. Load story
# ============================================================

with open("content.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Story characters:", len(text))


# ============================================================
# 3. Tokenizer
# ============================================================

tokenizer = tiktoken.get_encoding("gpt2")

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

print("Number of tokens:", len(tokens))
print("First 20 tokens:")
print(tokens[:20])


# ============================================================
# 4. Train / validation split
# ============================================================

split = int(len(tokens) * 0.9)

train_tokens = tokens[:split]
val_tokens = tokens[split:]

print("\nTraining tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))


# ============================================================
# 5. Create a random batch
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

    return x, y


# ============================================================
# 6. Model parameters
# ============================================================

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

position_embedding = nn.Embedding(
    context_size,
    embedding_size
)


# ============================================================
# 7. Attention weights
# ============================================================

Wq = nn.Parameter(
    torch.randn(
        num_heads,
        embedding_size,
        head_size
    ) * 0.02
)

Wk = nn.Parameter(
    torch.randn(
        num_heads,
        embedding_size,
        head_size
    ) * 0.02
)

Wv = nn.Parameter(
    torch.randn(
        num_heads,
        embedding_size,
        head_size
    ) * 0.02
)


# ============================================================
# 8. Output projection
# ============================================================

Wo = nn.Parameter(
    torch.randn(
        embedding_size,
        embedding_size
    ) * 0.02
)


# ============================================================
# 9. LayerNorm
# ============================================================

layer_norm = nn.LayerNorm(
    embedding_size
)

final_layer_norm = nn.LayerNorm(
    embedding_size
)


# ============================================================
# 10. Feed-Forward Network
# ============================================================

ffn = nn.Sequential(
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


# ============================================================
# 11. Language model head
# ============================================================

lm_head = nn.Linear(
    embedding_size,
    vocab_size,
    bias=False
)


# ============================================================
# 12. Collect parameters
# ============================================================

parameters = [

    token_embedding.weight,

    position_embedding.weight,

    Wq,
    Wk,
    Wv,
    Wo,

    layer_norm.weight,
    layer_norm.bias,

    final_layer_norm.weight,
    final_layer_norm.bias,

]

parameters += list(ffn.parameters())
parameters += list(lm_head.parameters())


print("\nNumber of parameter tensors:", len(parameters))

total_parameters = sum(
    p.numel()
    for p in parameters
)

print(
    "Total parameters:",
    total_parameters
)


# ============================================================
# 13. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    parameters,
    lr=learning_rate
)


# ============================================================
# 14. Forward pass
# ============================================================

def forward(x, targets=None):

    # x shape:
    #
    # batch_size × context_size
    #
    # Example:
    #
    # 16 × 8

    B, T = x.shape


    # --------------------------------------------------------
    # Token embeddings
    # --------------------------------------------------------

    token_vectors = token_embedding(x)

    # B × T × 64


    # --------------------------------------------------------
    # Position embeddings
    # --------------------------------------------------------

    positions = torch.arange(
        T,
        device=x.device
    )

    position_vectors = position_embedding(
        positions
    )

    # T × 64


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X = token_vectors + position_vectors

    # B × T × 64


    # --------------------------------------------------------
    # Q, K, V
    # --------------------------------------------------------

    Q = torch.einsum(
        "btd,hdm->bhtm",
        X,
        Wq
    )

    K = torch.einsum(
        "btd,hdm->bhtm",
        X,
        Wk
    )

    V = torch.einsum(
        "btd,hdm->bhtm",
        X,
        Wv
    )

    # B × heads × T × head_size


    # --------------------------------------------------------
    # Attention scores
    # --------------------------------------------------------

    scores = Q @ K.transpose(
        -2,
        -1
    )

    # B × heads × T × T


    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scores = scores / (
        head_size ** 0.5
    )


    # --------------------------------------------------------
    # Causal mask
    # --------------------------------------------------------

    mask = torch.tril(
        torch.ones(
            T,
            T,
            device=x.device
        )
    )

    scores = scores.masked_fill(
        mask == 0,
        float("-inf")
    )


    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    attention_weights = torch.softmax(
        scores,
        dim=-1
    )


    # --------------------------------------------------------
    # Attention output
    # --------------------------------------------------------

    head_outputs = attention_weights @ V

    # B × heads × T × head_size


    # --------------------------------------------------------
    # Concatenate heads
    # --------------------------------------------------------

    output = head_outputs.transpose(
        1,
        2
    ).contiguous()

    # B × T × heads × head_size


    output = output.view(
        B,
        T,
        embedding_size
    )

    # B × T × 64


    # --------------------------------------------------------
    # Output projection
    # --------------------------------------------------------

    projected_output = output @ Wo

    # B × T × 64


    # --------------------------------------------------------
    # First residual
    # --------------------------------------------------------

    output = projected_output + X


    # --------------------------------------------------------
    # LayerNorm
    # --------------------------------------------------------

    output = layer_norm(output)


    # --------------------------------------------------------
    # Feed Forward Network
    # --------------------------------------------------------

    ffn_output = ffn(output)


    # --------------------------------------------------------
    # Second residual
    # --------------------------------------------------------

    output = ffn_output + output


    # --------------------------------------------------------
    # Final LayerNorm
    # --------------------------------------------------------

    output = final_layer_norm(output)


    # --------------------------------------------------------
    # Language model head
    # --------------------------------------------------------

    logits = lm_head(output)

    # B × T × vocab_size


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = None

    if targets is not None:

        loss = F.cross_entropy(
            logits.view(
                -1,
                vocab_size
            ),
            targets.view(-1)
        )


    return logits, loss


# ============================================================
# 15. Test one batch before training
# ============================================================

x, y = get_batch(train_tokens)

logits, loss = forward(
    x,
    y
)

print("\nBatch input shape:")
print(x.shape)

print("\nBatch target shape:")
print(y.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nInitial loss:")
print(loss.item())


# ============================================================
# 16. Training loop
# ============================================================

print("\nStarting training...\n")


for step in range(training_steps):

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    x, y = get_batch(
        train_tokens
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = forward(
        x,
        y
    )


    # --------------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------------

    optimizer.zero_grad()


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step % 100 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# 17. Validation loss
# ============================================================

with torch.no_grad():

    x, y = get_batch(
        val_tokens
    )

    _, val_loss = forward(
        x,
        y
    )


print("\nTraining finished.")

print(
    "Validation loss:",
    val_loss.item()
)


# ============================================================
# 18. Generate text
# ============================================================

def generate(
    start_text,
    max_new_tokens=100
):

    # --------------------------------------------------------
    # Convert starting text to tokens
    # --------------------------------------------------------

    tokens = torch.tensor(
        tokenizer.encode(start_text),
        dtype=torch.long
    )


    for _ in range(max_new_tokens):

        # Keep only recent context
        context = tokens[
            -context_size:
        ]


        # Add batch dimension

        x = context.unsqueeze(0)


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.no_grad():

            logits, _ = forward(x)


        # ----------------------------------------------------
        # Last position
        # ----------------------------------------------------

        logits = logits[
            0,
            -1
        ]


        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )


        # ----------------------------------------------------
        # Pick most likely token
        # ----------------------------------------------------

        next_token = torch.argmax(
            probabilities
        )


        # ----------------------------------------------------
        # Append token
        # ----------------------------------------------------

        tokens = torch.cat(
            [
                tokens,
                next_token.unsqueeze(0)
            ]
        )


    # --------------------------------------------------------
    # Convert tokens back to text
    # --------------------------------------------------------

    return tokenizer.decode(
        tokens.tolist()
    )


# ============================================================
# 19. Test generation
# ============================================================

print("\nGenerated text:\n")

print(
    generate(
        "The",
        max_new_tokens=100
    )
)