"""
Main CLI entrypoint to launch TinyGPT-500M pretraining.
Supports resumption from checkpoint and fast dry-run micro-benchmarks.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.dataset import BinaryShardedDataset
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain TinyGPT-500M")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_500m.json",
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train_pretrain.json",
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_shards",
        help="Directory containing train/ and val/ binary shards",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Runs a 5-step micro-test with synthetic shards to verify pipeline",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps in training config",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Override eval_interval in training config",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Override save_interval in training config",
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        type=int,
        default=None,
        help="Maximum periodic checkpoints to keep on disk (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    if os.path.exists(args.model_config):
        model_config = TinyGPTConfig.from_pretrained(args.model_config)
    else:
        print(f"Warning: {args.model_config} not found. Using default TinyGPTConfig.")
        model_config = TinyGPTConfig()

    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = json.load(f)

    if args.steps is not None:
        train_config["max_steps"] = args.steps
    if args.eval_interval is not None:
        train_config["eval_interval"] = args.eval_interval
    if args.save_interval is not None:
        train_config["save_interval"] = args.save_interval
    if args.max_checkpoints_to_keep is not None:
        train_config["max_checkpoints_to_keep"] = args.max_checkpoints_to_keep

    # 2. Dry-Run Setup
    if args.dry_run:
        print(">>> RUNNING IN DRY-RUN VERIFICATION MODE <<<")
        train_config["max_steps"] = args.steps if args.steps is not None else 5
        train_config["eval_interval"] = 3
        train_config["save_interval"] = 5
        train_config["gradient_accumulation_steps"] = 2
        train_config["micro_batch_size"] = 1
        train_config["sequence_length"] = 64
        # Use 2 layers for fast CPU verification in dry-run
        model_config.num_hidden_layers = 2
        model_config.context_size = 64

        # Generate temporary multi-shard test data
        dry_dir = os.path.join(REPO_ROOT, "tests", "dry_shards")
        os.makedirs(os.path.join(dry_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(dry_dir, "val"), exist_ok=True)
        import numpy as np

        # Create 3 train shards and 2 val shards to verify multi-shard sampling
        for s_idx in range(3):
            np.random.randint(0, 50257, size=2048, dtype=np.uint16).tofile(
                os.path.join(dry_dir, "train", f"train_{s_idx:03d}.bin")
            )
        for s_idx in range(2):
            np.random.randint(0, 50257, size=1024, dtype=np.uint16).tofile(
                os.path.join(dry_dir, "val", f"val_{s_idx:03d}.bin")
            )
        args.data_dir = dry_dir

    # 3. Load Datasets
    train_path = os.path.join(args.data_dir, "train")
    val_path = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training shard directory '{train_path}' does not exist. "
            f"Please run data/prepare_data.py first or specify --data_dir."
        )

    print(f"Loading training dataset from: {train_path}")
    train_dataset = BinaryShardedDataset(
        train_path, sequence_length=train_config.get("sequence_length", 1024)
    )
    print(f"  Found {train_dataset.total_tokens:,} training tokens across {len(train_dataset.files)} shard(s):")
    for f_idx, (f_name, f_len) in enumerate(zip(train_dataset.files, train_dataset.shard_lengths)):
        print(f"    - Shard {f_idx}: {os.path.basename(f_name)} ({f_len:,} tokens)")

    val_dataset = None
    if os.path.exists(val_path):
        val_dataset = BinaryShardedDataset(
            val_path, sequence_length=train_config.get("sequence_length", 1024)
        )
        print(f"  Found {val_dataset.total_tokens:,} validation tokens across {len(val_dataset.files)} shard(s):")
        for f_idx, (f_name, f_len) in enumerate(zip(val_dataset.files, val_dataset.shard_lengths)):
            print(f"    - Shard {f_idx}: {os.path.basename(f_name)} ({f_len:,} tokens)")

    # 4. Instantiate Model
    print("Instantiating TinyGPT model...")
    model = TinyGPT500M(model_config)
    print(f"Total model parameters: {model.count_parameters():,}")

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
    )

    # 6. Resume from Checkpoint if requested
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = trainer.checkpoint_manager.load(
            args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if trainer.use_scaler else None,
            device=trainer.device.type,
        )
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    # 7. Start Pretraining
    trainer.train(start_step=start_step, best_val_loss=best_val_loss)


if __name__ == "__main__":
    main()
