"""
Package GPT-2 Tokenizer assets into exported Hugging Face model directory.
Ensures zero-friction loading via AutoTokenizer.from_pretrained().
"""

import os
import sys
from transformers import AutoTokenizer


def package_tokenizer(output_dir: str = "exported_usaid_ai_aligned"):
    print(f"Exporting Hugging Face tokenizer assets to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)
    
    print(f"Successfully saved tokenizer assets into '{output_dir}'.")
    print("Files created:")
    for f in os.listdir(output_dir):
        if "tokenizer" in f or "vocab" in f or "merges" in f or "special_tokens" in f:
            print(f"  - {f}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "exported_usaid_ai_aligned"
    package_tokenizer(target)
