"""
Multi-Discipline Streaming Pre-training Data Ingestion Pipeline for TinyGPT-500M.
Streams high-quality educational, scientific, coding, and general knowledge text
from the HuggingFaceTB/smollm-corpus (Cosmopedia-v2, FineWeb-Edu-dedup, Python-Edu).
Tokenizes incrementally line-by-line and writes directly to uint16 binary shards
without downloading massive parquet files to disk.
"""

import argparse
import itertools
import os
import sys
import time
import numpy as np
import tiktoken

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.prepare_data import StreamingShardWriter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_streaming_dataset_iterators():
    """
    Initializes streaming iterators from the premier sub-1B educational & multi-language corpus:
      - Cosmopedia-v2: Synthetic textbooks, STEM, humanities, multi-language algorithms (~30%)
      - FineWeb-Edu-dedup: High-quality web educational articles (~30%)
      - Python-Edu: Clean Python code, algorithms, and documentation (~15%)
      - Multi-Code & Reasoning: OpenHermes multi-language programming (C, C++, Java, Python) & logic (~15%)
      - Everyday Conversations: Real multi-turn chat dialogues for natural conversation flow (~10%)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required for streaming pre-training data.")
        print("Please run: pip install datasets")
        sys.exit(1)

    print("Connecting to Hugging Face streaming endpoints...")
    print("  [1/5] Streaming subset: Cosmopedia-v2 (Textbooks, STEM, World facts)")
    cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

    print("  [2/5] Streaming subset: FineWeb-Edu-dedup (Curated educational web)")
    fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)

    print("  [3/5] Streaming subset: Python-Edu (Algorithms & clean code)")
    py_edu = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)

    print("  [4/5] Streaming subset: Multi-Code & Reasoning (C, C++, Java, Python, logic)")
    multi_code = load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True)

    print("  [5/5] Streaming subset: Everyday Conversations (Conversational flow & dialogues)")
    chat_dialogues = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)

    return iter(cosmo), iter(fineweb), iter(py_edu), iter(multi_code), iter(chat_dialogues)


def stream_multi_discipline_documents(total_tokens_target: int):
    """
    Interleaves documents from the 5 subsets in a balanced multi-discipline ratio:
      - 2 Cosmopedia docs (~28.5%)
      - 2 FineWeb-Edu docs (~28.5%)
      - 1 Python-Edu doc (~14.3%)
      - 1 Multi-Code doc (C, C++, Java, Python algorithms) (~14.3%)
      - 1 Everyday Conversation doc (~14.3%)
    """
    cosmo_iter, fineweb_iter, py_iter, multi_code_iter, chat_iter = get_streaming_dataset_iterators()

    recipe_pattern = ["cosmo", "fineweb", "code_py", "cosmo", "fineweb", "code_multi", "chat"]
    pattern_cycle = itertools.cycle(recipe_pattern)

    doc_count = 0
    for source in pattern_cycle:
        try:
            if source == "cosmo":
                sample = next(cosmo_iter)
                text = sample.get("text", "").strip()
            elif source == "fineweb":
                sample = next(fineweb_iter)
                text = sample.get("text", "").strip()
            elif source == "code_py":
                sample = next(py_iter)
                text = sample.get("text", "").strip()
            elif source == "code_multi":
                sample = next(multi_code_iter)
                # OpenHermes format: list of messages or text
                if "messages" in sample:
                    turns = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in sample["messages"]]
                    text = "\n\n".join(turns).strip()
                else:
                    text = sample.get("text", "").strip()
            elif source == "chat":
                sample = next(chat_iter)
                if "messages" in sample:
                    turns = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in sample["messages"]]
                    text = "\n\n".join(turns).strip()
                else:
                    text = sample.get("text", "").strip()
            else:
                continue

            if text:
                doc_count += 1
                yield text

        except StopIteration:
            break
        except Exception:
            # Tolerant to occasional transient network hiccups on streaming
            time.sleep(1)
            continue


def build_pretrain_shards(
    output_dir: str = "data_shards",
    total_tokens: int = 100_000_000,
    shard_size: int = 5_000_000,
    val_ratio: float = 0.02,
):
    print("=" * 80)
    print("       USAID AI (500M) MULTI-DISCIPLINE & MULTI-LANGUAGE DATA INGESTION")
    print("                 Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Total Tokens: {total_tokens:,} tokens ({total_tokens / 1e6:.1f}M)")
    print(f"Tokens Per Shard:    {shard_size:,} tokens ({shard_size / 1e6:.1f}M)")
    print(f"Validation Split:    {val_ratio * 100:.1f}%")
    print(f"Destination:         {output_dir}")
    print("=" * 80 + "\n")

    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = 50256

    train_writer = StreamingShardWriter(
        os.path.join(output_dir, "train"), "train", shard_size=shard_size
    )
    val_writer = StreamingShardWriter(
        os.path.join(output_dir, "val"), "val", shard_size=shard_size
    )

    val_interval = max(1, int(1.0 / val_ratio)) if val_ratio > 0 else 0
    doc_stream = stream_multi_discipline_documents(total_tokens_target=total_tokens)

    start_time = time.time()
    total_tokens_written = 0
    doc_index = 0

    print("Starting streaming tokenization...")
    for doc_text in doc_stream:
        # Tokenize document with EOS
        tokens = enc.encode(doc_text, allowed_special={"<|endoftext|>"})
        tokens.append(eos_token_id)
        token_len = len(tokens)

        # Route to train or val
        if val_interval > 0 and (doc_index % val_interval == val_interval - 1):
            val_writer.add_tokens(tokens)
        else:
            train_writer.add_tokens(tokens)

        total_tokens_written += token_len
        doc_index += 1

        # Live telemetry every 500k tokens
        if total_tokens_written % 500_000 < token_len:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens_written / max(elapsed, 1e-5)
            pct = (total_tokens_written / total_tokens) * 100
            remaining_tokens = max(0, total_tokens - total_tokens_written)
            eta_seconds = remaining_tokens / max(tok_per_sec, 1)
            eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds))

            print(
                f"  [{pct:5.1f}%] Processed {total_tokens_written:,} / {total_tokens:,} tokens | "
                f"Docs: {doc_index:,} | Speed: {tok_per_sec:,.0f} tok/s | ETA: {eta_str}"
            )

        if total_tokens_written >= total_tokens:
            print(f"\nReached target token budget of {total_tokens:,} tokens!")
            break

    train_writer.close()
    val_writer.close()

    total_time = time.time() - start_time
    total_final = train_writer.total_tokens + val_writer.total_tokens

    print("\n" + "=" * 80)
    print("                    DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total Tokens Saved:      {total_final:,}")
    print(f"  - Training Tokens:     {train_writer.total_tokens:,} ({train_writer.shard_idx} shards)")
    print(f"  - Validation Tokens:   {val_writer.total_tokens:,} ({val_writer.shard_idx} shards)")
    print(f"Total Documents:         {doc_index:,}")
    print(f"Total Elapsed Time:      {time.strftime('%Hh %Mm %Ss', time.gmtime(total_time))}")
    print(f"Average Stream Speed:    {total_final / max(total_time, 1):,.0f} tokens/second")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and shard multi-discipline pretraining data")
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=100_000_000,
        help="Target number of tokens to stream (e.g. 50000000, 100000000, 200000000)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5_000_000,
        help="Number of tokens per binary shard file (default: 5,000,000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_shards",
        help="Directory to save shards (default: data_shards)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="Validation split ratio (default: 0.02 = 2%)",
    )
    args = parser.parse_args()

    build_pretrain_shards(
        output_dir=args.output_dir,
        total_tokens=args.total_tokens,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
