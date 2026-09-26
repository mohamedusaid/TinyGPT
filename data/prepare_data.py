"""
Data Preparation & Tokenization Pipeline for TinyGPT-500M.
Tokenizes text sources with GPT-2 tokenizer into memory-mapped uint16 binary shards (.bin).
Eliminates padding waste by packing sequences delimited with <|endoftext|>.
"""

import argparse
import os
import sys
import numpy as np
import tiktoken

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tokenize_and_save_shards(
    text_data: list[str],
    output_dir: str,
    split_name: str = "train",
    shard_size: int = 10_000_000,
):
    """
    Tokenizes a list of documents and writes them to uint16 .bin files.

    Args:
        text_data: List of raw string documents.
        output_dir: Target directory to save shards.
        split_name: 'train' or 'val'.
        shard_size: Number of tokens per shard file (default 10M tokens = 20 MB).
    """
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = 50256

    print(f"Tokenizing {len(text_data):,} documents for '{split_name}' split...")

    current_shard = []
    shard_idx = 0
    total_tokens = 0

    for doc in text_data:
        if not doc.strip():
            continue
        tokens = enc.encode(doc, allowed_special={"<|endoftext|>"})
        # Append EOS token to delimit documents
        tokens.append(eos_token_id)
        current_shard.extend(tokens)
        total_tokens += len(tokens)

        # Write shard when threshold is reached
        while len(current_shard) >= shard_size:
            shard_tokens = np.array(current_shard[:shard_size], dtype=np.uint16)
            shard_filename = os.path.join(
                output_dir, f"{split_name}_shard_{shard_idx:04d}.bin"
            )
            shard_tokens.tofile(shard_filename)
            print(
                f"  Saved {shard_filename} ({len(shard_tokens):,} tokens, {os.path.getsize(shard_filename) / 1024 / 1024:.2f} MB)"
            )
            shard_idx += 1
            current_shard = current_shard[shard_size:]

    # Write remaining tokens as final shard
    if current_shard:
        shard_tokens = np.array(current_shard, dtype=np.uint16)
        shard_filename = os.path.join(
            output_dir, f"{split_name}_shard_{shard_idx:04d}.bin"
        )
        shard_tokens.tofile(shard_filename)
        print(
            f"  Saved {shard_filename} ({len(shard_tokens):,} tokens, {os.path.getsize(shard_filename) / 1024 / 1024:.2f} MB)"
        )

    print(f"Finished '{split_name}': {total_tokens:,} total tokens saved across {shard_idx + 1} shard(s).\n")
    return total_tokens


def generate_synthetic_corpus(num_samples: int = 1000) -> list[str]:
    """Generates synthetic multi-domain STEM, science, and coding text for dry-run testing."""
    templates = [
        "In artificial intelligence and deep learning, transformer architectures use self-attention mechanisms to model long-range token dependencies in natural language processing tasks.",
        "The fundamental theorem of calculus connects differentiation and integration, showing that integration can be reversed by differentiation.",
        "Python is a high-level, general-purpose programming language that emphasizes code readability with the use of significant indentation.",
        "In physics, thermodynamics studies heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter.",
        "Grouped-Query Attention (GQA) reduces key-value memory bandwidth consumption during autoregressive transformer inference by sharing key and value heads across multiple query heads.",
        "Quantum computing harnesses the phenomena of quantum mechanics, such as superposition and entanglement, to perform computations exponentially faster for specific problems.",
        "Operating systems manage computer hardware and software resources, providing common services for computer programs through process scheduling and memory virtualization.",
    ]
    docs = []
    for i in range(num_samples):
        # Create varied paragraphs
        t = templates[i % len(templates)]
        doc = f"Article {i + 1}:\n{t}\nKey concept: {t[:40]}...\nConclusion: Understanding these fundamentals is essential for scientific inquiry."
        docs.append(doc)
    return docs


def main():
    parser = argparse.ArgumentParser(description="Prepare data shards for TinyGPT-500M")
    parser.add_argument("--output_dir", type=str, default="data_shards", help="Output directory")
    parser.add_argument("--input_file", type=str, default=None, help="Optional text file to tokenize")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio (default 5%)")
    parser.add_argument("--shard_size", type=int, default=1_000_000, help="Tokens per shard file")
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        print(f"Loading input text file: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]
    else:
        print("No input text file specified. Generating synthetic STEM corpus for pipeline verification...")
        docs = generate_synthetic_corpus(num_samples=2500)

    # Train / Val Split
    np.random.seed(42)
    indices = np.random.permutation(len(docs))
    val_count = max(1, int(len(docs) * args.val_ratio))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_docs = [docs[i] for i in train_indices]
    val_docs = [docs[i] for i in val_indices]

    tokenize_and_save_shards(train_docs, output_dir=os.path.join(args.output_dir, "train"), split_name="train", shard_size=args.shard_size)
    tokenize_and_save_shards(val_docs, output_dir=os.path.join(args.output_dir, "val"), split_name="val", shard_size=args.shard_size)
    print("Data preparation complete! Shards are ready in:", args.output_dir)


if __name__ == "__main__":
    main()
