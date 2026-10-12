"""
Interactive SFT Training Data Inspector for Usaid AI (500M).
Created by Mohamed Usaid.

Allows viewing and searching the exact instruction-response pairs
used to align Usaid AI (500M).

Usage:
    python scripts/view_sft_samples.py               # View sample identity & coding pairs
    python scripts/view_sft_samples.py --stream 5    # Stream 5 live samples from SmolTalk
    python scripts/view_sft_samples.py --search "C++"# Search examples by keyword
"""

import argparse
import json
import os
import sys

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.download_sft_data import USAID_AI_IDENTITY_EXAMPLES, MULTI_LANG_CODE_EXAMPLES


def print_banner():
    print("=" * 80)
    print("           USAID AI (500M) — SFT DATASET INSPECTION CONSOLE")
    print("                       Created by Mohamed Usaid")
    print("=" * 80)


def display_pair(idx: int, category: str, q: str, a: str):
    print(f"\n[{category} #{idx + 1}]")
    print(f"User Prompt:\n  {q}")
    print(f"\nAssistant Response:\n  {a}")
    print("-" * 80)


def view_curated(category: str = "all", limit: int = 5, search: str = ""):
    print_banner()
    items = []

    if category in ("all", "identity"):
        for item in USAID_AI_IDENTITY_EXAMPLES:
            items.append(("Identity & Persona", item["q"], item["a"]))

    if category in ("all", "code"):
        for item in MULTI_LANG_CODE_EXAMPLES:
            items.append(("Multi-Language Coding", item["q"], item["a"]))

    if search:
        search_lower = search.lower()
        items = [it for it in items if search_lower in it[1].lower() or search_lower in it[2].lower()]
        print(f"Found {len(items)} matching examples for query: '{search}'")

    print(f"Showing up to {min(limit, len(items))} examples (Total available: {len(items)}):")
    for idx, (cat, q, a) in enumerate(items[:limit]):
        display_pair(idx, cat, q, a)


def stream_smoltalk_samples(num_samples: int = 5):
    print_banner()
    print(f"Streaming {num_samples} sample instructions from Hugging Face (HuggingFaceTB/smoltalk)...\n")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)
        for idx, sample in enumerate(ds):
            if idx >= num_samples:
                break
            messages = sample.get("messages", [])
            print(f"\n[SmolTalk Everyday-Conversations Sample #{idx + 1}] (Turns: {len(messages)})")
            for m in messages:
                role = m.get("role", "").capitalize()
                content = m.get("content", "").strip()
                print(f"{role}:\n  {content}\n")
            print("-" * 80)
    except Exception as e:
        print(f"Error streaming from Hugging Face: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SFT training dataset for Usaid AI")
    parser.add_argument("--category", choices=["all", "identity", "code"], default="all", help="Category to display")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--search", type=str, default="", help="Search query keyword")
    parser.add_argument("--stream", type=int, default=0, help="Stream N live samples from Hugging Face SmolTalk")
    args = parser.parse_args()

    if args.stream > 0:
        stream_smoltalk_samples(args.stream)
    else:
        view_curated(category=args.category, limit=args.limit, search=args.search)


if __name__ == "__main__":
    main()
