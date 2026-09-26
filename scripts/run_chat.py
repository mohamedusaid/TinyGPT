"""
Main CLI entrypoint to run generation or interactive chat with TinyGPT-500M.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from inference.generate import generate
from inference.chat import start_chat_session


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with TinyGPT-500M")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_tinygpt_500m.pt",
        help="Path to trained checkpoint .pt file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate completion for (non-interactive mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
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
        default=100,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu' (auto-detected if None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    tokenizer = GPT2Tokenizer()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using scripts/run_pretrain.py.")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config_dict = checkpoint.get("config", {})
    if config_dict:
        config = TinyGPTConfig.from_dict(config_dict)
    else:
        config = TinyGPTConfig()

    model = TinyGPT500M(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully ({model.count_parameters():,} parameters) on {device}.")

    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("Generating completion...\n")
        full_text, reason = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream_callback=lambda token: sys.stdout.write(token),
        )
        print(f"\n\n[Finished generation. Reason: {reason}]")
    else:
        start_chat_session(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
