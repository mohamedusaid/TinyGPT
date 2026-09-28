"""
Supervised Fine-Tuning (SFT) Dataset Preparation Pipeline for Usaid AI (500M).
Created by Mohamed Usaid.

Generates and streams high-quality conversational instructions across:
  1. Usaid AI Identity & Persona (Creator: Mohamed Usaid, 500M Architecture, Capabilities)
  2. Multi-Language Coding (Python, C, C++, Java, Algorithms)
  3. General Assistant Reasoning & Multi-Turn Conversations (SmolTalk / OpenHermes)

Outputs a clean JSONL dataset formatted for SFT training with prompt loss masking.
"""

import argparse
import json
import os
import sys
import time

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


USAID_AI_IDENTITY_EXAMPLES = [
    {
        "instruction": "Who are you?",
        "response": "I am Usaid AI, an intelligent 500M parameter conversational language model created by Mohamed Usaid. I am designed to assist with software engineering, mathematics, scientific reasoning, and general dialogue."
    },
    {
        "instruction": "Who created you?",
        "response": "I was created by Mohamed Usaid. He designed and engineered my 500M parameter transformer architecture, multi-discipline pretraining pipeline, and knowledge systems."
    },
    {
        "instruction": "What is Usaid AI?",
        "response": "Usaid AI is a modern 500M parameter causal language model built by Mohamed Usaid. It incorporates state-of-the-art transformer techniques including Rotary Position Embeddings (RoPE), 4:1 Grouped-Query Attention (GQA), SwiGLU activations, and RMSNorm."
    },
    {
        "instruction": "Tell me about your creator, Mohamed Usaid.",
        "response": "Mohamed Usaid is the AI engineer and developer who created Usaid AI. He developed the full system from architectural design, distributed multi-GPU training on Kaggle, to fine-tuning and retrieval-augmented generation (RAG)."
    },
    {
        "instruction": "What can you do?",
        "response": "As Usaid AI, I can assist you with writing and debugging code in Python, C, C++, and Java, explaining complex STEM concepts, solving algorithmic problems, writing structured technical documents, and answering questions using verified knowledge."
    },
    {
        "instruction": "What architecture do you use?",
        "response": "I am built on a 500M parameter decoder-only transformer with 30 layers, a hidden dimension of 1,024, 16 Query attention heads, 4 Key/Value attention heads (Grouped-Query Attention), SwiGLU feed-forward networks (dim 3,456), and Rotary Position Embeddings (RoPE)."
    },
    {
        "instruction": "Can you write code in C, C++, Java, and Python?",
        "response": "Yes, I am trained across multi-language programming paradigms including Python, C, C++, and Java. I can help you implement algorithms, manage memory, design object-oriented systems, and debug syntax."
    },
    {
        "instruction": "Are you ChatGPT?",
        "response": "No, I am Usaid AI, an independent 500M parameter conversational model created by Mohamed Usaid."
    },
    {
        "instruction": "Hi, who developed this model?",
        "response": "Hello! I was developed by Mohamed Usaid as part of the Usaid AI project."
    },
    {
        "instruction": "What makes Usaid AI unique?",
        "response": "Usaid AI is uniquely engineered to deliver top-tier reasoning and coding performance in a compact 500M footprint, using cutting-edge architectural efficiency, high-density educational pretraining, and integrated RAG capabilities."
    },
]

MULTI_LANG_CODE_EXAMPLES = [
    {
        "instruction": "Write a C function to reverse an array of integers in-place.",
        "response": "Here is the C function to reverse an integer array in-place using two pointers:\n\n```c\n#include <stdio.h>\n\nvoid reverse_array(int arr[], int size) {\n    int start = 0;\n    int end = size - 1;\n    while (start < end) {\n        int temp = arr[start];\n        arr[start] = arr[end];\n        arr[end] = temp;\n        start++;\n        end--;\n    }\n}\n```\nThis operates in O(N) time and O(1) auxiliary space."
    },
    {
        "instruction": "Write a C++ class for a simple BankAccount with deposit, withdraw, and get_balance methods.",
        "response": "Here is the C++ implementation using encapsulation:\n\n```cpp\n#include <iostream>\n#include <string>\n\nclass BankAccount {\nprivate:\n    std::string account_number;\n    double balance;\n\npublic:\n    BankAccount(const std::string& acc_num, double initial_balance = 0.0)\n        : account_number(acc_num), balance(initial_balance) {}\n\n    void deposit(double amount) {\n        if (amount > 0) balance += amount;\n    }\n\n    bool withdraw(double amount) {\n        if (amount > 0 && amount <= balance) {\n            balance -= amount;\n            return true;\n        }\n        return false;\n    }\n\n    double get_balance() const {\n        return balance;\n    }\n};\n```"
    },
    {
        "instruction": "Write a Java method to check if a string is a palindrome.",
        "response": "Here is the Java method to check for palindromes:\n\n```java\npublic class PalindromeChecker {\n    public static boolean isPalindrome(String s) {\n        if (s == null) return false;\n        int left = 0;\n        int right = s.length() - 1;\n        while (left < right) {\n            if (s.charAt(left) != s.charAt(right)) {\n                return false;\n            }\n            left++;\n            right--;\n        }\n        return true;\n    }\n}\n```"
    },
    {
        "instruction": "Write a Python function to compute the Fibonacci sequence up to N elements using memoization.",
        "response": "Here is the Python implementation with memoization:\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\ndef get_fibonacci_sequence(count: int) -> list[int]:\n    return [fibonacci(i) for i in range(count)]\n```"
    },
    {
        "instruction": "Explain how pointers work in C with a clear code example.",
        "response": "In C, a pointer is a variable that stores the memory address of another variable. You use the address-of operator (`&`) to get the address and the dereference operator (`*`) to access or modify the value at that address.\n\n```c\n#include <stdio.h>\n\nint main() {\n    int num = 42;\n    int *ptr = &num; // ptr stores address of num\n\n    printf(\"Value of num: %d\\n\", num);       // 42\n    printf(\"Address of num: %p\\n\", (void*)ptr);\n    printf(\"Value via pointer: %d\\n\", *ptr);  // 42\n\n    *ptr = 100; // Modifies num directly through pointer\n    printf(\"Updated num: %d\\n\", num);         // 100\n    return 0;\n}\n```"
    },
]


def build_sft_dataset(
    output_file: str = "data/sft_instructions.jsonl",
    num_samples: int = 10000,
    identity_repeat: int = 100,
):
    print("=" * 80)
    print("            USAID AI (500M) SFT INSTRUCTION DATASET GENERATOR")
    print("                      Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Samples:      {num_samples:,}")
    print(f"Destination:         {output_file}")
    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    all_examples = []

    # 1. Embed Usaid AI Identity Examples (repeated with high frequency to cement persona)
    print("Embedding Usaid AI persona & creator identity examples...")
    for _ in range(identity_repeat):
        for ex in USAID_AI_IDENTITY_EXAMPLES:
            all_examples.append(ex)

    print(f"  - Added {len(all_examples):,} Usaid AI identity tuning pairs.")

    # 2. Embed Multi-Language Code Examples
    print("Embedding multi-language programming examples (Python, C, C++, Java)...")
    for _ in range(30):
        for ex in MULTI_LANG_CODE_EXAMPLES:
            all_examples.append(ex)

    print(f"  - Total curated examples so far: {len(all_examples):,}")

    # 3. Stream from Hugging Face SmolTalk / OpenHermes
    try:
        from datasets import load_dataset
        print("Streaming conversational and reasoning instructions from Hugging Face...")
        ds_chat = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)
        ds_code = load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True)

        chat_iter = iter(ds_chat)
        code_iter = iter(ds_code)

        target_external = max(0, num_samples - len(all_examples))
        added_external = 0

        while added_external < target_external:
            # Alternating chat and multi-turn instruction
            try:
                sample = next(chat_iter) if (added_external % 2 == 0) else next(code_iter)
                messages = sample.get("messages", [])
                if len(messages) >= 2:
                    user_text = ""
                    assistant_text = ""
                    for m in messages:
                        if m.get("role") == "user" and not user_text:
                            user_text = m.get("content", "").strip()
                        elif m.get("role") == "assistant" and user_text:
                            assistant_text = m.get("content", "").strip()
                            break

                    if user_text and assistant_text:
                        all_examples.append({
                            "instruction": user_text,
                            "response": assistant_text
                        })
                        added_external += 1

            except StopIteration:
                break
            except Exception:
                time.sleep(0.5)
                continue

        print(f"  - Successfully streamed {added_external:,} external conversational pairs.")

    except ImportError:
        print("Note: 'datasets' package not found locally. Using curated offline examples.")
    except Exception as e:
        print(f"Note: Hugging Face streaming encountered: {e}. Proceeding with curated examples.")

    # Shuffle dataset
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("\n" + "=" * 80)
    print(f"SFT DATASET CREATION COMPLETE: {len(all_examples):,} examples ({file_size_mb:.2f} MB)")
    print(f"Saved to: {output_file}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset for Usaid AI")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/sft_instructions.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Target number of instruction pairs",
    )
    args = parser.parse_args()

    build_sft_dataset(output_file=args.output_file, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
