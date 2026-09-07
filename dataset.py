"""
Dataset & Tokenization Pipeline for Tiny GPT
--------------------------------------------
Handles tokenizer initialization, dataset reading, sequence splitting,
and batch generation for training.
"""

import torch
import tiktoken


# ============================================================
# TOKENIZER
# ============================================================

def get_tokenizer(encoding_name="gpt2"):
    """Returns the tiktoken encoding instance."""
    return tiktoken.get_encoding(encoding_name)


# ============================================================
# LOAD STORY & DATASET
# ============================================================

def load_dataset(file_path="content.txt", eos_token_id=50256):
    """
    Reads the text file, encodes tokens with GPT-2 tokenizer,
    appends EOS_TOKEN_ID, and returns a 1D LongTensor.
    """
    enc = get_tokenizer()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    tokens = enc.encode(text)

    # Add EOS token to explicitly mark the end of the story
    if eos_token_id is not None:
        tokens.append(eos_token_id)

    return torch.tensor(tokens, dtype=torch.long)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def train_val_split(tokens, train_ratio=0.9):
    """Splits a 1D token tensor into train and validation sets."""
    split = int(train_ratio * len(tokens))

    train_data = tokens[:split]
    val_data = tokens[split:]

    return train_data, val_data


# ============================================================
# BATCH CREATION
# ============================================================

def get_batch(data, batch_size=8, context_size=64, device="cpu"):
    """
    Samples random chunks of length context_size from data
    and generates (x, y) input-target pairs.
    """
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


if __name__ == "__main__":

    print("Testing dataset pipeline...")

    tokens = load_dataset("content.txt")

    print()
    print("Number of tokens:", len(tokens))

    print("First 20 tokens:")
    print(tokens[:20])

    train_data, val_data = train_val_split(tokens)

    print()
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    x, y = get_batch(train_data, batch_size=4, context_size=64)
    print("Batch x shape:", x.shape)
    print("Batch y shape:", y.shape)

    assert x.shape == (4, 64) and y.shape == (4, 64)
    print("Dataset self-test passed!")
