"""
Main CLI entrypoint to launch Supervised Fine-Tuning (SFT) for Usaid AI (500M).
Created by Mohamed Usaid.
Aligns base pretraining checkpoints on multi-turn dialogue, coding, and identity instructions.
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
from scripts.download_sft_data import build_sft_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT for Usaid AI (500M)")
    parser.add_argument(
        "--base_model",
        type=str,
        default="checkpoints/best_tinygpt_500m.pt",
        help="Path to pre-trained base model checkpoint",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/sft_instructions.jsonl",
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

    # Generate Usaid AI instruction dataset if data_file does not exist
    if not os.path.exists(args.data_file):
        print(f"Instruction dataset '{args.data_file}' not found. Generating Usaid AI SFT instructions...")
        build_sft_dataset(output_file=args.data_file, num_samples=5000)

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
