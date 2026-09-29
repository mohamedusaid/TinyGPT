"""
Continuous Interactive Multi-Turn Chat Console for Usaid AI (500M).
Created by Mohamed Usaid.

Features:
  - Continuous multi-turn conversation loop (does NOT exit after one reply).
  - Real-time token streaming with live typing effect.
  - Multi-turn conversational memory with automatic context window sliding.
  - In-session commands: 'clear'/'reset' to wipe history, 'quit'/'exit' to leave.
  - Optional RAG knowledge grounding toggle ('rag on' / 'rag off').
"""

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from inference.generate import generate
from rag.knowledge_base import KnowledgeBase
from rag.retriever import BM25Retriever
from rag.pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous Interactive Chat with Usaid AI (500M) — Created by Mohamed Usaid"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint .pt file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (lower = more deterministic, higher = more creative)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=120,
        help="Maximum new tokens per assistant turn",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG knowledge base grounding by default",
    )
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        default="knowledge_base",
        help="Directory containing RAG knowledge documents",
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
    raise FileNotFoundError(
        "Could not find any trained checkpoint! Please ensure 'checkpoints/usaid_ai_500m.pt' exists."
    )


def print_banner(model_name: str, params: int, device: torch.device, rag_active: bool):
    print("=" * 80)
    print("                  USAID AI (500M) — CONTINUOUS CHAT CONSOLE")
    print("                           Created by Mohamed Usaid")
    print("=" * 80)
    print(f"  Model:       {model_name} ({params:,} parameters)")
    print(f"  Device:      {device} [torch.float32 if CPU else torch.float16]")
    print(f"  RAG Engine:  {'ENABLED (Grounding Active)' if rag_active else 'DISABLED (Pure Model)'}")
    print("-" * 80)
    print("  Commands:")
    print("    - Type your message and press [Enter] to chat")
    print("    - 'clear' or 'reset' : Wipe conversation history to start fresh")
    print("    - 'rag on' / 'rag off': Toggle RAG knowledge retrieval")
    print("    - 'exit' or 'quit'   : Close the session")
    print("=" * 80 + "\n")


def build_conversation_prompt(history: list[tuple[str, str]], new_user_msg: str, rag_context: str | None = None) -> str:
    """
    Constructs the conversational prompt format conforming to SFT training:
    User: <msg>\n\nAssistant: <msg><|endoftext|>\n\nUser: <new>\n\nAssistant:
    """
    prompt = ""
    if rag_context:
        prompt += (
            "You are Usaid AI, a helpful and concise AI assistant created by Mohamed Usaid.\n"
            f"Reference Context:\n{rag_context}\n\n"
        )

    for user_turn, asst_turn in history:
        prompt += f"User: {user_turn}\n\nAssistant: {asst_turn}<|endoftext|>\n\n"

    prompt += f"User: {new_user_msg}\n\nAssistant: "
    return prompt


def run_continuous_chat():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. Load Checkpoint
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading Usaid AI weights from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config_dict = checkpoint.get("config", {})
        model_name = checkpoint.get("model_name", "Usaid AI")
    else:
        state_dict = checkpoint
        config_dict = {}
        model_name = "Usaid AI"

    config = TinyGPTConfig.from_dict(config_dict) if config_dict else TinyGPTConfig()
    model = TinyGPT500M(config)
    model.load_state_dict(state_dict)
    model.to(dtype=target_dtype, device=device)
    model.eval()

    tokenizer = GPT2Tokenizer()

    # 2. Setup Optional RAG Retriever
    rag_enabled = args.rag
    retriever = None
    if os.path.exists(args.knowledge_dir):
        try:
            kb = KnowledgeBase(args.knowledge_dir)
            retriever = BM25Retriever(kb)
        except Exception as e:
            print(f"Notice: RAG indexer skipped ({e}). Running in pure model mode.")

    print_banner(model_name, model.count_parameters(), device, rag_enabled)

    # 3. Multi-Turn Interactive Conversation Loop
    history: list[tuple[str, str]] = []
    max_history_turns = 4  # Keep last 4 exchanges to comfortably fit 1024 context window

    while True:
        try:
            user_input = input("\nYou > ").strip()

            if not user_input:
                continue

            # Command: Exit
            if user_input.lower() in ["quit", "exit", "q", ":q"]:
                print("\nUsaid AI > Goodbye! Have a wonderful day.\n")
                break

            # Command: Clear Memory
            if user_input.lower() in ["clear", "reset"]:
                history.clear()
                print("\n[Conversation history cleared. Fresh session started.]")
                continue

            # Command: Toggle RAG
            if user_input.lower() in ["rag on", "rag enable"]:
                rag_enabled = True
                print("\n[RAG Knowledge Grounding: ENABLED]")
                continue
            if user_input.lower() in ["rag off", "rag disable"]:
                rag_enabled = False
                print("\n[RAG Knowledge Grounding: DISABLED]")
                continue

            # Optional RAG Context Retrieval
            rag_context = None
            if rag_enabled and retriever:
                passages = retriever.retrieve(user_input, top_k=2, min_score=3.0)
                if passages:
                    chunks_text = [f"[{i+1}] {p.text.strip()}" for i, (p, score) in enumerate(passages)]
                    rag_context = "\n".join(chunks_text)
                    print(f"  [RAG: Grounded with {len(passages)} verified source passage(s)]")

            # Build Formatted Prompt with History
            full_prompt = build_conversation_prompt(history, user_input, rag_context)

            print("\nUsaid AI > ", end="", flush=True)

            accumulated_reply = []
            stop_detected = False

            def stream_callback(token_str: str):
                nonlocal stop_detected
                if stop_detected:
                    return

                # Stop early if assistant attempts to simulate the next user prompt
                if "\nUser:" in token_str or "User:" in token_str:
                    stop_detected = True
                    return

                sys.stdout.write(token_str)
                sys.stdout.flush()
                accumulated_reply.append(token_str)

            # Generate response
            start_t = time.time()
            _, reason = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream_callback=stream_callback,
            )

            # Clean and store the turn in multi-turn history
            raw_response = "".join(accumulated_reply).strip()
            # Remove any trailing User: hallucination if present
            clean_response = raw_response.split("User:")[0].strip()

            if clean_response:
                history.append((user_input, clean_response))
                if len(history) > max_history_turns:
                    history.pop(0)

            print()

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting continuous chat.")
            break


if __name__ == "__main__":
    run_continuous_chat()
