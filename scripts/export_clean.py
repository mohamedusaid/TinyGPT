"""
Clean Exporter for TinyGPT-500M.
Strips out AdamW optimizer buffers (momentum and velocity) and saves model weights
in FP16 precision. Shrinks the checkpoint from 5.59 GB down to ~953 MB for fast
downloading and lightweight local inference.
"""

import argparse
import os
import sys
import torch


def export_clean_checkpoint(checkpoint_path: str, output_path: str, fp16: bool = True):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading full checkpoint from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw_state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    if fp16:
        print("Casting model weights to float16 (half precision)...")
        clean_state = {k: v.half() for k, v in raw_state.items()}
    else:
        print("Keeping model weights in float32...")
        clean_state = raw_state

    clean_dict = {
        "model_state_dict": clean_state,
        "config": config,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving clean lightweight model to: {output_path}...")
    torch.save(clean_dict, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    size_gb = size_mb / 1024
    print(f"Export successful!")
    print(f"Clean Model Path: {output_path}")
    print(f"Clean Model Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Create lightweight model checkpoint without optimizer buffers")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_tinygpt_500m.pt",
        help="Path to full 5.59 GB checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/tinygpt_500m_clean.pt",
        help="Path for lightweight output model",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Save in float32 (default is float16, ~953 MB)",
    )
    args = parser.parse_args()

    export_clean_checkpoint(args.checkpoint, args.output, fp16=not args.fp32)


if __name__ == "__main__":
    main()
