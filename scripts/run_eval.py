"""
Main CLI entrypoint to evaluate TinyGPT-500M checkpoints.
Computes validation perplexity and zero-shot benchmark reasoning accuracy.
"""

import argparse
import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from data.dataset import BinaryShardedDataset
from eval.perplexity import evaluate_perplexity
from eval.benchmarks import evaluate_multiple_choice


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyGPT-500M")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_tinygpt_500m.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data_shards/val",
        help="Directory with validation .bin shards",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of validation batches to evaluate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = TinyGPTConfig.from_dict(checkpoint.get("config", {}))

    model = TinyGPT500M(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer()

    # 1. Perplexity Evaluation
    if os.path.exists(args.val_data):
        print(f"\nEvaluating Perplexity on {args.val_data}...")
        dataset = BinaryShardedDataset(args.val_data, sequence_length=config.context_size)
        res = evaluate_perplexity(
            model=model,
            dataset=dataset,
            num_batches=args.batches,
            device=device,
        )
        print(f"  Validation Loss: {res['loss']:.4f}")
        print(f"  Perplexity:      {res['perplexity']:.2f}")

    # 2. Benchmark Reasoning Evaluation
    sample_benchmarks = [
        {
            "prompt": "Question: In Python, which keyword defines a function?",
            "choices": ["func", "def", "lambda", "function"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: What is the primary benefit of Grouped-Query Attention?",
            "choices": [
                "Reduces memory consumption of KV cache",
                "Increases training loss",
                "Removes positional embeddings",
                "Eliminates weight decay",
            ],
            "answer_idx": 0,
        },
    ]

    print("\nRunning Zero-Shot Multiple Choice Benchmark Sample...")
    bench_res = evaluate_multiple_choice(
        model=model,
        tokenizer=tokenizer,
        questions=sample_benchmarks,
        device=device,
    )
    print(f"  Score: {bench_res['correct']}/{bench_res['total']} ({bench_res['accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
