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


BENCHMARK_CATEGORIES = {
    "Python & Software Engineering": [
        {
            "prompt": "Question: In Python, which keyword defines a function?",
            "choices": ["func", "def", "lambda", "function"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: What is the output data type of 5 / 2 in Python 3?",
            "choices": ["int", "float", "double", "str"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: Which Python data structure is mutable and ordered?",
            "choices": ["tuple", "set", "list", "frozenset"],
            "answer_idx": 2,
        },
        {
            "prompt": "Question: In Python, what does the 'len()' function return?",
            "choices": ["Memory address", "Number of elements", "Variable type", "Hash value"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: Which keyword in Python handles exceptions?",
            "choices": ["catch", "try", "rescue", "trap"],
            "answer_idx": 1,
        },
    ],
    "ML & Transformer Architecture": [
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
        {
            "prompt": "Question: What does RoPE stand for in modern Transformer models?",
            "choices": [
                "Rotary Position Embedding",
                "Recurrent Output Position Encoding",
                "Randomized Optimization Path Estimator",
                "Residual Operation Processing Engine",
            ],
            "answer_idx": 0,
        },
        {
            "prompt": "Question: Which normalization layer replaces LayerNorm in modern LLMs like Llama and Usaid AI?",
            "choices": ["BatchNorm", "RMSNorm", "InstanceNorm", "WeightNorm"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: What activation function combines Swish and Gated Linear Units?",
            "choices": ["ReLU", "GeLU", "SwiGLU", "Sigmoid"],
            "answer_idx": 2,
        },
        {
            "prompt": "Question: What is the parameter scale of Usaid AI?",
            "choices": ["100 Million", "500 Million", "7 Billion", "70 Billion"],
            "answer_idx": 1,
        },
    ],
    "World Knowledge & Science": [
        {
            "prompt": "Question: What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer_idx": 2,
        },
        {
            "prompt": "Question: Which planet in our solar system is known as the Red Planet?",
            "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: What chemical element has the symbol 'O' on the periodic table?",
            "choices": ["Gold", "Osmium", "Oxygen", "Iron"],
            "answer_idx": 2,
        },
        {
            "prompt": "Question: What is the boiling point of water at standard sea level atmospheric pressure in Celsius?",
            "choices": ["50 degrees", "100 degrees", "200 degrees", "0 degrees"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: Who created Usaid AI?",
            "choices": ["OpenAI", "Google", "Mohamed Usaid", "Meta"],
            "answer_idx": 2,
        },
    ],
    "Logic & Arithmetic": [
        {
            "prompt": "Question: What is the result of 10 plus 15?",
            "choices": ["20", "25", "30", "150"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: If all humans are mortal, and Socrates is human, then Socrates is:",
            "choices": ["Immortal", "Mortal", "Greek", "Unknown"],
            "answer_idx": 1,
        },
        {
            "prompt": "Question: Which number is an even prime number?",
            "choices": ["2", "3", "5", "7"],
            "answer_idx": 0,
        },
        {
            "prompt": "Question: If a train travels at 60 miles per hour, how many miles does it travel in 2 hours?",
            "choices": ["30", "60", "120", "180"],
            "answer_idx": 2,
        },
        {
            "prompt": "Question: Which word is the opposite of 'hot'?",
            "choices": ["Warm", "Cold", "Spicy", "Bright"],
            "answer_idx": 1,
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyGPT-500M / Usaid AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/usaid_ai_500m.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--eval_perplexity",
        action="store_true",
        help="Whether to evaluate perplexity on binary dataset shards",
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

    print("=" * 80)
    print("           USAID AI (500M) — COMPREHENSIVE BENCHMARK EVALUATOR")
    print("                       Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Loading checkpoint from: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = TinyGPTConfig.from_dict(checkpoint.get("config", {}))

    model = TinyGPT500M(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer()

    # 1. Perplexity Evaluation (Optional)
    if args.eval_perplexity and os.path.exists(args.val_data):
        print(f"\n[1/2] Evaluating Perplexity on {args.val_data}...")
        dataset = BinaryShardedDataset(args.val_data, sequence_length=config.context_size)
        res = evaluate_perplexity(
            model=model,
            dataset=dataset,
            num_batches=args.batches,
            device=device,
        )
        print(f"  Validation Loss: {res['loss']:.4f}")
        print(f"  Perplexity:      {res['perplexity']:.2f}")

    # 2. Comprehensive Zero-Shot Benchmark Suite
    print(f"\nRunning Zero-Shot Multi-Domain Reasoning Benchmark (20 Questions)...")
    print("-" * 80)

    total_correct = 0
    total_questions = 0
    category_results = {}

    for cat_name, questions in BENCHMARK_CATEGORIES.items():
        print(f"\nEvaluating Category: {cat_name} ({len(questions)} items)...")
        cat_res = evaluate_multiple_choice(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            device=device,
        )
        category_results[cat_name] = cat_res
        total_correct += cat_res["correct"]
        total_questions += cat_res["total"]
        print(f"  --> Score: {cat_res['correct']}/{cat_res['total']} ({cat_res['accuracy']:.1f}%)")

    overall_acc = (total_correct / total_questions) * 100.0

    print("\n" + "=" * 80)
    print("                         FINAL BENCHMARK SCORECARD")
    print("=" * 80)
    for cat_name, res in category_results.items():
        print(f"  {cat_name:<35} : {res['correct']}/{res['total']} ({res['accuracy']:.1f}%)")
    print("-" * 80)
    print(f"  {'OVERALL ZERO-SHOT BENCHMARK ACCURACY':<35} : {total_correct}/{total_questions} ({overall_acc:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
