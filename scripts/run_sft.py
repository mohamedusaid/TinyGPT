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


def resolve_base_model_path(path: str) -> str:
    """
    Resolves the base model checkpoint path, automatically searching cloud input directories
    and common checkpoint folders if the exact path does not exist.
    """
    if os.path.exists(path):
        return path

    candidates = [
        "checkpoints/best_tinygpt_500m.pt",
        "checkpoints/checkpoint_step_002000.pt",
        "/kaggle/working/tiny-gpt-500m/checkpoints/best_tinygpt_500m.pt",
        "/kaggle/working/checkpoints/best_tinygpt_500m.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Discovered base model at: {c}")
            return c

    # Recursively search cloud storage mounts if available
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "best_tinygpt_500m.pt" in files:
                found = os.path.join(root, "best_tinygpt_500m.pt")
                print(f"Auto-discovered base model in cloud storage at: {found}")
                return found

    raise FileNotFoundError(
        f"Base model checkpoint not found at '{path}'. "
        "Please attach your trained model checkpoint in cloud storage, "
        "or ensure best_tinygpt_500m.pt is present in checkpoints/."
    )


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

    # Load base model with strict verification and auto-discovery
    base_model_path = resolve_base_model_path(args.base_model)
    print(f"Loading base model weights from: {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config_dict = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config_dict = {}

    model_config = TinyGPTConfig.from_dict(config_dict) if config_dict else TinyGPTConfig()
    model = TinyGPT500M(model_config)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded {model.count_parameters():,} parameters into Usaid AI architecture!")

    trainer = SFTTrainer(model=model, train_dataset=dataset, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
