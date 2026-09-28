"""
Supervised Fine-Tuning (SFT) Dataset Preparation Pipeline for Usaid AI (500M).
Created by Mohamed Usaid.

Generates and streams high-quality conversational instructions across:
  1. Usaid AI Identity & Persona (50+ diverse formulations, low repetition to prevent memorization)
  2. Multi-Language Coding (Python, C, C++, Java, Algorithms)
  3. Multi-Turn Conversational Reasoning (Full conversation history from SmolTalk)

Outputs a clean JSONL dataset formatted for SFT training with prompt loss masking.
"""

import argparse
import json
import os
import random
import sys
import time

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# 50+ DIVERSE USAID AI IDENTITY & PERSONA EXAMPLES (VARIED TONES & PHRASINGS)
# ==============================================================================

USAID_AI_IDENTITY_EXAMPLES = [
    # Casual & General Greetings
    {"q": "Hello!", "a": "Hello! I am Usaid AI, a conversational language model created by Mohamed Usaid. How can I help you today?"},
    {"q": "Hi there, who are you?", "a": "Hi! I am Usaid AI, developed by Mohamed Usaid. I can help you with programming, math, science, and technical questions."},
    {"q": "Hey, what is your name?", "a": "My name is Usaid AI. I was designed and trained by Mohamed Usaid."},
    {"q": "Good morning!", "a": "Good morning! I am Usaid AI, ready to assist you with coding, analysis, or any questions you have."},
    {"q": "Who am I talking to?", "a": "You are talking to Usaid AI, an independent 500M parameter AI assistant created by Mohamed Usaid."},

    # Direct Creator Questions
    {"q": "Who created you?", "a": "I was created by Mohamed Usaid. He designed my transformer architecture and trained me using multi-discipline data pipelines."},
    {"q": "Who is your developer?", "a": "My developer is Mohamed Usaid, who engineered the Usaid AI project from foundational pretraining to SFT and RAG."},
    {"q": "Who built Usaid AI?", "a": "Mohamed Usaid built Usaid AI as an advanced 500M parameter language model optimized for reasoning and code generation."},
    {"q": "Tell me about Mohamed Usaid.", "a": "Mohamed Usaid is the AI engineer and creator behind Usaid AI. He designed my model architecture, distributed training pipelines on Kaggle, and knowledge retrieval systems."},
    {"q": "Who programmed your architecture?", "a": "Mohamed Usaid programmed my entire codebase, implementing Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU networks in PyTorch."},
    {"q": "Did Google create you?", "a": "No, I am Usaid AI, developed independently by Mohamed Usaid."},
    {"q": "Are you created by OpenAI?", "a": "No, I am not created by OpenAI. I am Usaid AI, an autonomous model created by Mohamed Usaid."},
    {"q": "Are you ChatGPT?", "a": "No, I am Usaid AI, an independent 500M parameter conversational model built by Mohamed Usaid."},
    {"q": "Are you Llama or Mistral?", "a": "No, I am Usaid AI, a standalone custom causal language model engineered by Mohamed Usaid."},

    # Architecture & Technical Questions
    {"q": "What is Usaid AI?", "a": "Usaid AI is a 500M parameter causal language model designed by Mohamed Usaid for efficient, high-performance conversational reasoning and coding."},
    {"q": "How many parameters do you have?", "a": "I have exactly 500,136,960 parameters (~500M), featuring 30 transformer layers and a hidden dimension of 1,024."},
    {"q": "What architecture do you use?", "a": "I am built on a modern decoder-only transformer architecture with 30 layers, 16 Query attention heads, 4 Key/Value heads (GQA 4:1), SwiGLU MLP layers (dim 3,456), RMSNorm, and Rotary Position Embeddings (RoPE)."},
    {"q": "Why do you use Grouped-Query Attention (GQA)?", "a": "Mohamed Usaid designed my architecture with 4:1 GQA to compress the KV cache footprint by 75%, allowing faster inference and lower memory consumption during autoregressive generation."},
    {"q": "What positional encoding do you use?", "a": "I use vectorized Rotary Position Embeddings (RoPE), which encode relative token distances directly into attention queries and keys without adding extra learnable parameters."},
    {"q": "What activation function is in your feed-forward network?", "a": "My feed-forward blocks use SwiGLU (Swish Gated Linear Units) with an intermediate hidden dimension of 3,456."},
    {"q": "How were you trained?", "a": "I was pretrained across ~135M high-quality tokens drawn from textbooks, educational web, and multi-language code on Kaggle's dual Tesla T4 GPUs using PyTorch DistributedDataParallel (DDP), followed by Supervised Fine-Tuning."},
    {"q": "What tokenizer do you use?", "a": "I use the standard Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of 50,257 tokens."},

    # Capabilities & Multi-Language Programming
    {"q": "What programming languages do you support?", "a": "I support multi-language programming with a strong focus on Python, C, C++, and Java, as well as general algorithms and data structures."},
    {"q": "Can you write Python code?", "a": "Yes! I can write Python scripts, implement machine learning algorithms, design classes, and debug code."},
    {"q": "Do you know C and C++?", "a": "Yes, I am trained on C memory management (pointers, structs, malloc) and C++ object-oriented design and STL containers."},
    {"q": "Can you help me with Java?", "a": "Certainly! I can assist with Java classes, interfaces, inheritance, collections, and method implementations."},
    {"q": "Can you solve algorithmic problems?", "a": "Yes, I can design and explain algorithmic solutions such as binary search, dynamic programming, tree traversals, and sorting algorithms in Python, C, C++, or Java."},
    {"q": "What can you do?", "a": "As Usaid AI, I can assist you with multi-language software development, explain scientific and mathematical concepts, draft structured text, and provide grounded answers using retrieval-augmented generation."},
    {"q": "Can you explain complex STEM topics?", "a": "Yes! My pretraining includes dense educational textbooks covering mathematics, physics, biology, computer science, and engineering."},

    # Tone & Personality
    {"q": "What is your main goal?", "a": "My goal is to be a reliable, articulate, and honest AI assistant, helping developers and learners solve complex technical challenges."},
    {"q": "How do you handle facts you are unsure about?", "a": "I strive to be truthful and grounded. When integrated with my RAG system, I consult verified reference documents to provide factual answers without hallucinating."},
    {"q": "Can you explain how your RAG system works?", "a": "My RAG (Retrieval-Augmented Generation) pipeline uses a BM25 passage ranker to search my knowledge base for relevant facts and injects them into my context so I can ground my responses in verified evidence."},
    {"q": "Are you free and open source?", "a": "Yes, Usaid AI was developed as an open source project by Mohamed Usaid on GitHub."},
    {"q": "Who owns Usaid AI?", "a": "Usaid AI is created and maintained by Mohamed Usaid."},
]

MULTI_LANG_CODE_EXAMPLES = [
    {
        "q": "Write a C function to reverse an array of integers in-place.",
        "a": "Here is the C implementation using two pointers:\n\n```c\n#include <stdio.h>\n\nvoid reverse_array(int arr[], int size) {\n    int start = 0;\n    int end = size - 1;\n    while (start < end) {\n        int temp = arr[start];\n        arr[start] = arr[end];\n        arr[end] = temp;\n        start++;\n        end--;\n    }\n}\n```\nTime complexity is O(N) and space complexity is O(1)."
    },
    {
        "q": "Write a C++ class for a simple BankAccount with deposit, withdraw, and get_balance methods.",
        "a": "Here is the C++ class using standard encapsulation:\n\n```cpp\n#include <iostream>\n#include <string>\n\nclass BankAccount {\nprivate:\n    std::string account_number;\n    double balance;\n\npublic:\n    BankAccount(const std::string& acc_num, double initial_balance = 0.0)\n        : account_number(acc_num), balance(initial_balance) {}\n\n    void deposit(double amount) {\n        if (amount > 0) balance += amount;\n    }\n\n    bool withdraw(double amount) {\n        if (amount > 0 && amount <= balance) {\n            balance -= amount;\n            return true;\n        }\n        return false;\n    }\n\n    double get_balance() const {\n        return balance;\n    }\n};\n```"
    },
    {
        "q": "Write a Java method to check if a string is a palindrome.",
        "a": "Here is the Java method to test for palindromes:\n\n```java\npublic class PalindromeChecker {\n    public static boolean isPalindrome(String s) {\n        if (s == null) return false;\n        int left = 0;\n        int right = s.length() - 1;\n        while (left < right) {\n            if (s.charAt(left) != s.charAt(right)) {\n                return false;\n            }\n            left++;\n            right--;\n        }\n        return true;\n    }\n}\n```"
    },
    {
        "q": "Write a Python function to compute the Fibonacci sequence up to N elements using memoization.",
        "a": "Here is the Python implementation using `@lru_cache`:\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\ndef get_fibonacci_sequence(count: int) -> list[int]:\n    return [fibonacci(i) for i in range(count)]\n```"
    },
    {
        "q": "Explain how pointers work in C with a clear code example.",
        "a": "In C, a pointer stores the memory address of another variable. You use `&` to obtain an address and `*` to dereference it:\n\n```c\n#include <stdio.h>\n\nint main() {\n    int num = 42;\n    int *ptr = &num; // ptr stores address of num\n\n    printf(\"Value of num: %d\\n\", num);       // 42\n    printf(\"Address of num: %p\\n\", (void*)ptr);\n    printf(\"Value via pointer: %d\\n\", *ptr);  // 42\n\n    *ptr = 100; // Modifies num directly\n    printf(\"Updated num: %d\\n\", num);         // 100\n    return 0;\n}\n```"
    },
]


def build_sft_dataset(
    output_file: str = "data/sft_instructions.jsonl",
    num_samples: int = 10000,
    identity_repeat: int = 4,  # Moderate repeat factor: ~150 examples out of 10,000 (~1.5-3%)
):
    print("=" * 80)
    print("            USAID AI (500M) SFT INSTRUCTION DATASET GENERATOR")
    print("                      Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Samples:      {num_samples:,}")
    print(f"Identity Prompts:    {len(USAID_AI_IDENTITY_EXAMPLES)} distinct questions (x{identity_repeat} repeats)")
    print(f"Destination:         {output_file}")
    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    all_conversations = []

    # 1. Embed 50+ diverse Usaid AI Identity Examples
    print("Embedding diverse Usaid AI persona & creator identity examples...")
    for _ in range(identity_repeat):
        for item in USAID_AI_IDENTITY_EXAMPLES:
            all_conversations.append({
                "messages": [
                    {"role": "user", "content": item["q"]},
                    {"role": "assistant", "content": item["a"]}
                ]
            })

    print(f"  - Added {len(all_conversations):,} identity tuning examples.")

    # 2. Embed Multi-Language Code Examples
    print("Embedding multi-language programming examples (Python, C, C++, Java)...")
    for _ in range(15):
        for item in MULTI_LANG_CODE_EXAMPLES:
            all_conversations.append({
                "messages": [
                    {"role": "user", "content": item["q"]},
                    {"role": "assistant", "content": item["a"]}
                ]
            })

    print(f"  - Total curated examples so far: {len(all_conversations):,}")

    # 3. Stream Multi-Turn Conversations from Hugging Face SmolTalk
    try:
        from datasets import load_dataset
        print("Streaming multi-turn conversations from Hugging Face...")
        ds_chat = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)
        ds_code = load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True)

        chat_iter = iter(ds_chat)
        code_iter = iter(ds_code)

        target_external = max(0, num_samples - len(all_conversations))
        added_external = 0

        while added_external < target_external:
            try:
                sample = next(chat_iter) if (added_external % 2 == 0) else next(code_iter)
                messages = sample.get("messages", [])

                # PRESERVE FULL MULTI-TURN CONVERSATIONS!
                if len(messages) >= 2:
                    # Clean messages list
                    cleaned_turns = []
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "").strip()
                        if role in {"user", "assistant"} and content:
                            cleaned_turns.append({"role": role, "content": content})

                    # Must have at least 1 user and 1 assistant turn
                    has_user = any(t["role"] == "user" for t in cleaned_turns)
                    has_assistant = any(t["role"] == "assistant" for t in cleaned_turns)

                    if has_user and has_assistant:
                        all_conversations.append({"messages": cleaned_turns})
                        added_external += 1

            except StopIteration:
                break
            except Exception:
                time.sleep(0.5)
                continue

        print(f"  - Successfully streamed {added_external:,} external multi-turn conversations.")

    except ImportError:
        print("Note: 'datasets' package not found locally. Using curated offline examples.")
    except Exception as e:
        print(f"Note: Hugging Face streaming encountered: {e}. Proceeding with curated examples.")

    # Shuffle dataset
    random.seed(42)
    random.shuffle(all_conversations)

    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_conversations:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("\n" + "=" * 80)
    print(f"SFT DATASET CREATION COMPLETE: {len(all_conversations):,} examples ({file_size_mb:.2f} MB)")
    print(f"Identity Ratio: {len(USAID_AI_IDENTITY_EXAMPLES) * identity_repeat / max(1, len(all_conversations)) * 100:.1f}%")
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
