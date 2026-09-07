import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 1. SETTINGS
# ============================================================

embedding_size = 64
context_size = 8

num_heads = 4
head_size = embedding_size // num_heads

batch_size = 16

learning_rate = 0.001

training_steps = 2000

eval_interval = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# 2. LOAD STORY
# ============================================================

with open("content.txt", "r", encoding="utf-8") as f:
    story = f.read()

print("\nStory characters:", len(story))


# ============================================================
# 3. GPT-2 TOKENIZER
# ============================================================

encoding = tiktoken.get_encoding("gpt2")

original_tokens = encoding.encode(story)

print("Number of tokens:", len(original_tokens))

print("First 20 tokens:")
print(torch.tensor(original_tokens[:20]))


# ============================================================
# 4. CREATE SMALL VOCABULARY
# ============================================================
#
# GPT-2 has 50,257 possible token IDs.
#
# But our story only uses a small subset of them.
#
# Example:
#
# Original GPT token:
#
#     44379
#
# becomes:
#
#     local token 123
#
# This dramatically reduces the output layer.
# ============================================================

unique_tokens = sorted(set(original_tokens))

small_vocab_size = len(unique_tokens)

print("\nOriginal GPT vocabulary:", encoding.n_vocab)
print("Story vocabulary:", small_vocab_size)


# Original GPT token ID -> local token ID
token_to_local = {
    token: i
    for i, token in enumerate(unique_tokens)
}


# Local token ID -> original GPT token ID
local_to_token = {
    i: token
    for i, token in enumerate(unique_tokens)
}


# Convert story tokens to local IDs
local_tokens = [
    token_to_local[token]
    for token in original_tokens
]

tokens = torch.tensor(
    local_tokens,
    dtype=torch.long
)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(0.9 * len(tokens))

train_tokens = tokens[:split]
val_tokens = tokens[split:]

print("\nTraining tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))


# ============================================================
# 6. BATCH CREATION
# ============================================================

def get_batch(data):

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
# 7. TINY GPT MODEL
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            small_vocab_size,
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
        # Attention weights
        # ----------------------------------------------------

        self.Wq = nn.Parameter(
            torch.randn(
                num_heads,
                embedding_size,
                head_size
            ) * 0.02
        )

        self.Wk = nn.Parameter(
            torch.randn(
                num_heads,
                embedding_size,
                head_size
            ) * 0.02
        )

        self.Wv = nn.Parameter(
            torch.randn(
                num_heads,
                embedding_size,
                head_size
            ) * 0.02
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.Wo = nn.Parameter(
            torch.randn(
                embedding_size,
                embedding_size
            ) * 0.02
        )

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        self.layer_norm = nn.LayerNorm(
            embedding_size
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        self.ffn = nn.Sequential(

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

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_size,
            small_vocab_size,
            bias=False
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, inputs, targets=None):

        batch_size_current = inputs.shape[0]
        sequence_length = inputs.shape[1]

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_vectors = self.token_embedding(inputs)

        # Shape:
        #
        # [batch, time, embedding]
        #
        # Example:
        #
        # [16, 8, 64]

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=inputs.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        X = token_vectors + position_vectors

        # ----------------------------------------------------
        # Q K V
        # ----------------------------------------------------

        Q = torch.einsum(
            "btd,hdm->bhtm",
            X,
            self.Wq
        )

        K = torch.einsum(
            "btd,hdm->bhtm",
            X,
            self.Wk
        )

        V = torch.einsum(
            "btd,hdm->bhtm",
            X,
            self.Wv
        )

        # Shapes:
        #
        # Q = [batch, heads, time, head_size]
        #
        # [16, 4, 8, 16]

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # Shape:
        #
        # [batch, heads, time, time]
        #
        # [16, 4, 8, 8]

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scores = scores / (head_size ** 0.5)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=inputs.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Attention output
        # ----------------------------------------------------

        head_outputs = attention_weights @ V

        # ----------------------------------------------------
        # Concatenate heads
        # ----------------------------------------------------

        output = head_outputs.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size_current,
            sequence_length,
            embedding_size
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        projected_output = output @ self.Wo

        # ----------------------------------------------------
        # First residual
        # ----------------------------------------------------

        output = projected_output + X

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        output = self.layer_norm(output)

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        ffn_output = self.ffn(output)

        # ----------------------------------------------------
        # Second residual
        # ----------------------------------------------------

        output = ffn_output + output

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        output = self.final_layer_norm(output)

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        logits = self.lm_head(output)

        # Shape:
        #
        # [batch, time, small_vocab_size]

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, small_vocab_size),
                targets.view(-1)
            )

        return logits, loss


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = TinyGPT().to(device)

print("\nNumber of parameter tensors:",
      len(list(model.parameters())))

print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# 9. TEST ONE BATCH
# ============================================================

x, y = get_batch(train_tokens)

print("\nBatch input shape:")
print(x.shape)

print("\nBatch target shape:")
print(y.shape)


logits, loss = model(x, y)

print("\nLogits shape:")
print(logits.shape)

print("\nInitial loss:")
print(loss.item())


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    train_losses = []
    val_losses = []

    for _ in range(10):

        x, y = get_batch(train_tokens)

        _, loss = model(x, y)

        train_losses.append(
            loss.item()
        )

    for _ in range(10):

        x, y = get_batch(val_tokens)

        _, loss = model(x, y)

        val_losses.append(
            loss.item()
        )

    model.train()

    return (
        sum(train_losses) / len(train_losses),
        sum(val_losses) / len(val_losses)
    )


# ============================================================
# 12. TRAINING
# ============================================================

print("\nStarting training...\n")


for step in range(training_steps):

    # --------------------------------------------------------
    # Get training batch
    # --------------------------------------------------------

    x, y = get_batch(train_tokens)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits, loss = model(x, y)

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

    if step % eval_interval == 0:

        train_loss, val_loss = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )


print("\nTraining finished.")


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

train_loss, val_loss = estimate_loss()

print("\nFinal Train Loss:")
print(train_loss)

print("\nFinal Validation Loss:")
print(val_loss)


# ============================================================
# 14. GENERATION
# ============================================================

@torch.no_grad()
def generate(start_text, max_new_tokens=300):

    model.eval()

    # --------------------------------------------------------
    # Encode starting text using GPT-2 tokenizer
    # --------------------------------------------------------

    start_tokens = encoding.encode(start_text)

    # Only keep tokens that exist in our story vocabulary.
    #
    # If a token wasn't present in the story, we cannot use
    # it with our reduced vocabulary.
    #
    # Therefore use the first known token as fallback.

    local_start = []

    for token in start_tokens:

        if token in token_to_local:

            local_start.append(
                token_to_local[token]
            )

    if len(local_start) == 0:

        local_start = [
            local_tokens[0]
        ]

    generated = torch.tensor(
        local_start,
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        # Keep context window
        input_tokens = generated[-context_size:]

        # Add batch dimension
        input_tokens = input_tokens.unsqueeze(0)

        # Model prediction
        logits, _ = model(
            input_tokens
        )

        # Last position
        logits = logits[:, -1, :]

        # Convert to probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Greedy prediction
        next_token = torch.argmax(
            probabilities,
            dim=-1
        )

        # Append
        generated = torch.cat(
            [
                generated,
                next_token
            ]
        )

    # --------------------------------------------------------
    # Convert local IDs back to GPT token IDs
    # --------------------------------------------------------

    original_generated_tokens = [
        local_to_token[int(token)]
        for token in generated
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    text = encoding.decode(
        original_generated_tokens
    )

    model.train()

    return text


# ============================================================
# 15. GENERATE STORY
# ============================================================

print("\n\nGenerated text:\n")

generated_text = generate(
    story[:100],
    max_new_tokens=300
)

print(generated_text)