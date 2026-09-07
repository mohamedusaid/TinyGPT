"""
Training Script for Tiny GPT
----------------------------
Trains a TinyGPT language model on story text and saves the checkpoint.
"""

import torch
import torch.nn.functional as F
from model import TinyGPT
from dataset import get_tokenizer, load_dataset, train_val_split, get_batch


# ============================================================
# 1. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


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

EOS_TOKEN_ID = 50256

DATA_FILE = "content.txt"

CHECKPOINT_FILE = "tiny_gpt.pt"


# ============================================================
# EVALUATION & GENERATION HELPERS
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data):

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(eval_batches):

            x, y = get_batch(
                data,
                batch_size=batch_size,
                context_size=context_size,
                device=device
            )

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


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    max_new_tokens
):

    model.eval()

    # Tokenize prompt

    encoded = tokenizer.encode(prompt)

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

        logits = model(
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

    output = tokenizer.decode(
        idx[0].tolist()
    )

    model.train()

    return output


# ============================================================
# MAIN TRAINING ROUTINE
# ============================================================

def train():

    print("Device:", device)

    # --------------------------------------------------------
    # Load story & tokenize
    # --------------------------------------------------------

    enc = get_tokenizer("gpt2")

    tokens = load_dataset(
        DATA_FILE,
        eos_token_id=EOS_TOKEN_ID
    )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print()
    print("Story characters:", len(text))
    print("Number of tokens:", len(tokens))

    print("First 20 tokens:")
    print(tokens[:20])

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_data, val_data = train_val_split(tokens, train_ratio=0.9)

    print()
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TinyGPT(
        vocab_size=vocab_size,
        context_size=context_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # --------------------------------------------------------
    # Initial test
    # --------------------------------------------------------

    x, y = get_batch(
        train_data,
        batch_size=batch_size,
        context_size=context_size,
        device=device
    )

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

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("Starting training...")
    print()

    for step in range(max_steps):

        # Get batch
        x, y = get_batch(
            train_data,
            batch_size=batch_size,
            context_size=context_size,
            device=device
        )

        # Forward
        logits, loss = model(
            x,
            y
        )

        # Backward
        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        # Update
        optimizer.step()

        # Print progress
        if step % eval_interval == 0:

            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            print(
                f"Step {step:4d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f}"
            )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    losses = estimate_loss(
        model,
        train_data,
        val_data
    )

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

    # --------------------------------------------------------
    # Sample generation
    # --------------------------------------------------------

    print()
    print("Generated text:")
    print()

    prompt = (
        "Beneath a bruised and violet sky,"
    )

    generated = generate(
        model,
        prompt,
        enc,
        generate_tokens
    )

    print(generated)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

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
        CHECKPOINT_FILE
    )

    print()
    print(f"Model saved as {CHECKPOINT_FILE}")


if __name__ == "__main__":
    train()
