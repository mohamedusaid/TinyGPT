"""
Interactive Terminal Chat & Generation Interface for TinyGPT-500M.
Streams generated tokens in real-time to stdout with generation speed telemetry.
"""

import sys
import time
import torch

from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from inference.generate import generate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def start_chat_session(
    model: TinyGPT500M,
    tokenizer: GPT2Tokenizer,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 150,
):
    print("=" * 80)
    print("                  USAID AI (500M) CONVERSATIONAL CONSOLE")
    print("                         Created by Mohamed Usaid")
    print("=" * 80)
    print("Type your prompt and press Enter. Type 'quit' or 'exit' to terminate.")
    print(f"Sampling Parameters: Temperature={temperature}, Top-p={top_p}, Max Tokens={max_new_tokens}")
    print("-" * 80 + "\n")

    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit"]:
                print("Exiting interactive session.")
                break

            print("\nUsaid AI > ", end="", flush=True)

            token_count = [0]
            start_time = time.time()

            def stream_token(token_str: str):
                sys.stdout.write(token_str)
                sys.stdout.flush()
                token_count[0] += 1

            formatted_prompt = f"User: {prompt}\n\nAssistant: "
            full_text, reason = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream_callback=stream_token,
            )

            elapsed = max(time.time() - start_time, 1e-4)
            tok_s = token_count[0] / elapsed
            print(f"\n\n[Stats: {token_count[0]} tokens generated in {elapsed:.2f}s ({tok_s:.1f} tok/s) | Stop: {reason}]")

        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
            break
