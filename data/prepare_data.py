"""
Data Preparation & Tokenization Pipeline for TinyGPT-500M.
Genuinely streaming: tokenizes input text incrementally line-by-line without loading
the entire 100M+ token corpus into RAM.
Writes uint16 binary shards (.bin) directly to disk.
"""

import argparse
import os
import sys
from typing import Iterator
import numpy as np
import tiktoken

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def stream_documents_from_file(file_path: str) -> Iterator[str]:
    """
    Streams documents from a text file line-by-line.
    Documents are separated by one or more blank lines.
    Never loads the entire file into RAM.
    """
    buffer = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    yield "\n".join(buffer)
                    buffer = []
            else:
                buffer.append(stripped)
        if buffer:
            yield "\n".join(buffer)


def stream_synthetic_documents(num_samples: int = 2500) -> Iterator[str]:
    """Generates synthetic multi-domain STEM text on the fly without storing all in RAM."""
    templates = [
        "In artificial intelligence and deep learning, transformer architectures use self-attention mechanisms to model long-range token dependencies in natural language processing tasks.",
        "The fundamental theorem of calculus connects differentiation and integration, showing that integration can be reversed by differentiation.",
        "Python is a high-level, general-purpose programming language that emphasizes code readability with the use of significant indentation.",
        "In physics, thermodynamics studies heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter.",
        "Grouped-Query Attention (GQA) reduces key-value memory bandwidth consumption during autoregressive transformer inference by sharing key and value heads across multiple query heads.",
        "Quantum computing harnesses the phenomena of quantum mechanics, such as superposition and entanglement, to perform computations exponentially faster for specific problems.",
        "Operating systems manage computer hardware and software resources, providing common services for computer programs through process scheduling and memory virtualization.",
    ]
    for i in range(num_samples):
        t = templates[i % len(templates)]
        yield f"Article {i + 1}:\n{t}\nKey concept: {t[:40]}...\nConclusion: Understanding these fundamentals is essential for scientific inquiry."


class StreamingShardWriter:
    """
    Buffers tokens in memory only up to shard_size, then writes directly to disk
    and flushes the buffer. Peak RAM is strictly bounded.
    """

    def __init__(self, output_dir: str, split_name: str, shard_size: int = 10_000_000):
        self.output_dir = output_dir
        self.split_name = split_name
        self.shard_size = shard_size
        self.shard_idx = 0
        self.total_tokens = 0
        self.buffer = []
        os.makedirs(output_dir, exist_ok=True)

    def add_tokens(self, tokens: list[int]):
        self.buffer.extend(tokens)
        self.total_tokens += len(tokens)

        # Flush full shards to disk immediately
        while len(self.buffer) >= self.shard_size:
            shard_tokens = np.array(self.buffer[: self.shard_size], dtype=np.uint16)
            shard_path = os.path.join(
                self.output_dir, f"{self.split_name}_shard_{self.shard_idx:04d}.bin"
            )
            shard_tokens.tofile(shard_path)
            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            print(
                f"  [{self.split_name}] Flushed {shard_path} ({len(shard_tokens):,} tokens, {size_mb:.2f} MB)"
            )
            self.shard_idx += 1
            self.buffer = self.buffer[self.shard_size :]

    def close(self):
        """Flushes any remaining tokens to the final shard."""
        if self.buffer:
            shard_tokens = np.array(self.buffer, dtype=np.uint16)
            shard_path = os.path.join(
                self.output_dir, f"{self.split_name}_shard_{self.shard_idx:04d}.bin"
            )
            shard_tokens.tofile(shard_path)
            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            print(
                f"  [{self.split_name}] Flushed final {shard_path} ({len(shard_tokens):,} tokens, {size_mb:.2f} MB)"
            )
            self.shard_idx += 1
            self.buffer = []

        print(
            f"  Finished {self.split_name}: {self.total_tokens:,} tokens saved across {self.shard_idx} shard(s)."
        )


def process_streaming_data(
    doc_iterator: Iterator[str],
    output_dir: str,
    val_ratio: float = 0.05,
    shard_size: int = 10_000_000,
):
    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = 50256

    train_writer = StreamingShardWriter(
        os.path.join(output_dir, "train"), "train", shard_size=shard_size
    )
    val_writer = StreamingShardWriter(
        os.path.join(output_dir, "val"), "val", shard_size=shard_size
    )

    # Use deterministic stride for train/val split without loading all documents
    val_interval = max(1, int(1.0 / val_ratio)) if val_ratio > 0 else 0
    doc_count = 0

    print(f"Beginning streaming tokenization (shard size: {shard_size:,} tokens)...")

    for doc in doc_iterator:
        if not doc.strip():
            continue

        tokens = enc.encode(doc, allowed_special={"<|endoftext|>"})
        tokens.append(eos_token_id)

        # Route to validation if doc_count matches interval
        if val_interval > 0 and (doc_count % val_interval == 0):
            val_writer.add_tokens(tokens)
        else:
            train_writer.add_tokens(tokens)

        doc_count += 1
        if doc_count % 1000 == 0:
            print(
                f"  Processed {doc_count:,} documents | "
                f"Train tokens: {train_writer.total_tokens:,} | "
                f"Val tokens: {val_writer.total_tokens:,}"
            )

    train_writer.close()
    val_writer.close()

    total_all = train_writer.total_tokens + val_writer.total_tokens
    print(f"\nCompleted streaming data prep: {total_all:,} total tokens processed from {doc_count:,} documents.")
    return train_writer.total_tokens, val_writer.total_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare streaming data shards for TinyGPT-500M")
    parser.add_argument("--output_dir", type=str, default="data_shards", help="Output directory")
    parser.add_argument("--input_file", type=str, default=None, help="Path to raw text file")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio (default 5%)")
    parser.add_argument("--shard_size", type=int, default=1_000_000, help="Tokens per shard file")
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        print(f"Streaming from input text file: {args.input_file}")
        doc_stream = stream_documents_from_file(args.input_file)
    else:
        print("No input text file specified. Streaming synthetic STEM documents...")
        doc_stream = stream_synthetic_documents(num_samples=2500)

    process_streaming_data(
        doc_stream,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
