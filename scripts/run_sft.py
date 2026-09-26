"""
Main CLI entrypoint to launch Supervised Fine-Tuning (SFT) on TinyGPT-500M.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from sft.sft_dataset import SFTDataset
from sft.train_sft import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT on TinyGPT-500M")
    parser.add_argument(
        "--base_model",
        type=str,
        default="checkpoints/best_tinygpt_500m.pt",
        help="Path to pre-trained base model checkpoint",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/sample_instructions.json",
        help="Path to JSON/JSONL instruction dataset",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train_sft.json",
        help="Path to SFT configuration JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load SFT config
    with open(args.train_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = GPT2Tokenizer()

    # Create dummy sample instructions if data_file does not exist
    if not os.path.exists(args.data_file):
        os.makedirs(os.path.dirname(args.data_file) or ".", exist_ok=True)
        sample_data = [
            {
                "instruction": "Explain the concept of quantum superposition in simple terms.",
                "response": "Quantum superposition is a fundamental principle of quantum mechanics where a physical system can exist in multiple states simultaneously until it is measured.",
            },
            {
                "instruction": "Write a Python function to calculate the factorial of a number.",
                "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            },
            {
                "instruction": "What is Grouped-Query Attention (GQA)?",
                "response": "Grouped-Query Attention is an attention mechanism where multiple query heads share a single key-value head, drastically reducing memory consumption during token generation.",
            },
        ]
        with open(args.data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)
        print(f"Created sample instructions file at: {args.data_file}")

    dataset = SFTDataset(
        args.data_file,
        tokenizer=tokenizer,
        sequence_length=config.get("sequence_length", 1024),
    )

    # Load base model
    if os.path.exists(args.base_model):
        checkpoint = torch.load(
            args.base_model, map_location="cpu", weights_only=False
        )
        model_config = TinyGPTConfig.from_dict(checkpoint.get("config", {}))
        model = TinyGPT500M(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded base model weights from {args.base_model}")
    else:
        print(
            f"Base model checkpoint {args.base_model} not found. Instantiating fresh 2-layer model for test..."
        )
        model = TinyGPT500M(TinyGPTConfig(num_hidden_layers=2))

    trainer = SFTTrainer(model=model, train_dataset=dataset, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
