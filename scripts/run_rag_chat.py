"""
Interactive RAG Chat CLI for Usaid AI (500M).
Created by Mohamed Usaid.
Grounds conversational answers in real-time verified knowledge from the knowledge_base/ directory.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from rag.knowledge_base import KnowledgeBase
from rag.retriever import BM25Retriever
from rag.pipeline import RAGPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Usaid AI (500M) + RAG Knowledge Grounding")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt). Defaults to usaid_ai_500m.pt or best_tinygpt_500m.pt",
    )
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default="knowledge_base",
        help="Directory containing documents for RAG context",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt query (if omitted, starts interactive chat session)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of reference passages to retrieve",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Generation temperature",
    )
    return parser.parse_args()


def resolve_checkpoint(ckpt_arg: str | None) -> str:
    candidates = [
        ckpt_arg,
        "checkpoints/usaid_ai_500m.pt",
        "checkpoints/best_usaid_ai_500m.pt",
        "checkpoints/best_tinygpt_500m.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def print_banner():
    print("=" * 80)
    print("             USAID AI (500M) — RETRIEVAL-AUGMENTED CHAT SYSTEM")
    print("                         Created by Mohamed Usaid")
    print("=" * 80)
    print("  Commands: type 'exit' or 'quit' to end session | 'sources' to toggle context")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    print_banner()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. Load Knowledge Base & Retriever
    print(f"Indexing knowledge documents from '{args.knowledge_dir}'...")
    kb = KnowledgeBase(args.knowledge_dir)
    retriever = BM25Retriever(kb)

    # 2. Resolve & Load Model Checkpoint
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    tokenizer = GPT2Tokenizer()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading Usaid AI weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_dict = checkpoint.get("config", {})
        config = TinyGPTConfig.from_dict(config_dict) if config_dict else TinyGPTConfig()
        model = TinyGPT500M(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Note: Checkpoint not found. Instantiating fresh Usaid AI (500M) model for interface demo...")
        model = TinyGPT500M(TinyGPTConfig())

    model.to(dtype=target_dtype, device=device)
    model.eval()

    # 3. Initialize RAG Pipeline
    pipeline = RAGPipeline(model=model, tokenizer=tokenizer, retriever=retriever)

    # 4. Handle Single Prompt Mode
    if args.prompt:
        print(f"\n[User Query]: {args.prompt}")
        answer, passages, _ = pipeline.query(args.prompt, top_k=args.top_k, temperature=args.temperature)

        print("\n[Retrieved Reference Context]:")
        for i, (p, score) in enumerate(passages, 1):
            print(f"  ({i}) {os.path.basename(p.source)} (BM25: {score:.2f}): {p.text[:120]}...")

        print(f"\n[Usaid AI]:\n{answer}\n")
        return

    # 5. Interactive Chat Session
    show_sources = True
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("\nThank you for using Usaid AI. Session ended.")
                break
            if user_input.lower() == "sources":
                show_sources = not show_sources
                print(f"  [Context Display: {'ON' if show_sources else 'OFF'}]")
                continue

            answer, passages, reason = pipeline.query(
                user_input, top_k=args.top_k, temperature=args.temperature
            )

            if show_sources and passages:
                print("\n  [Verified Sources]:")
                for i, (p, score) in enumerate(passages, 1):
                    preview = p.text.replace("\n", " ")[:100]
                    print(f"    [{i}] {os.path.basename(p.source)} (Relevance: {score:.2f}) -> \"{preview}...\"")
                print()

            print(f"Usaid AI: {answer}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


if __name__ == "__main__":
    main()
