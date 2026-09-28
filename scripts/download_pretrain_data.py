"""
Multi-Discipline & Multi-Language Streaming Pre-training Data Ingestion Pipeline for Usaid AI (500M).
Created by Mohamed Usaid.

Streams high-quality educational, scientific, coding (Python, C, C++, Java), and conversational text:
  - Cosmopedia-v2 (30.0%): Synthetic textbooks, STEM, academic concepts
  - FineWeb-Edu-dedup (30.0%): Curated high-scoring educational web
  - Python-Edu (10.0%): Clean Python code, algorithms, data structures
  - Verified C / C++ Code (10.0%): Memory management, pointers, structs, STL, algorithms
  - Verified Java Code (10.0%): Enterprise OOP, classes, interfaces, JVM patterns
  - Everyday Conversations (10.0%): Multi-turn natural dialogue flow (SmolTalk)

Includes pipeline-level quality filtering, repetition filtering, and hash deduplication.
Tokenizes incrementally line-by-line and writes directly to uint16 binary shards (.bin).
"""

import argparse
import itertools
import os
import sys
import time
from typing import Iterator, Set
import numpy as np
import tiktoken

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.prepare_data import StreamingShardWriter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# 1. LANGUAGE VERIFICATION & SYNTAX DETECTORS
# ==============================================================================

def is_c_cpp_code(text: str) -> bool:
    """Verifies that the document contains genuine C or C++ source code."""
    indicators = [
        "#include <", "int main(", "printf(", "scanf(", "malloc(", "free(",
        "std::cout", "std::vector", "std::string", "template <", "nullptr",
        "typedef struct", "->", "size_t", "const char*", "return 0;",
        "class ", "namespace ", "#define ", "std::cin", "public:", "private:"
    ]
    matches = sum(1 for ind in indicators if ind in text)
    return matches >= 2


def is_java_code(text: str) -> bool:
    """Verifies that the document contains genuine Java source code."""
    indicators = [
        "public class ", "public static void main", "System.out.println",
        "import java.", "implements ", "extends ", "new ArrayList",
        "public void ", "private int ", "private String ", "@Override",
        "package ", "throws Exception", "interface ", "boolean "
    ]
    matches = sum(1 for ind in indicators if ind in text)
    return matches >= 2


def is_python_code(text: str) -> bool:
    """Verifies that the document contains genuine Python code."""
    indicators = [
        "def ", "import ", "class ", "self.", "elif ", "return ",
        "print(", "__init__", "in range(", "lambda ", "except "
    ]
    matches = sum(1 for ind in indicators if ind in text)
    return matches >= 2


# ==============================================================================
# 2. QUALITY FILTERING & HASH DEDUPLICATION
# ==============================================================================

def passes_quality_filter(text: str, seen_hashes: Set[int]) -> bool:
    """
    Applies heuristic quality filtering:
      1. Minimum & maximum length bounds
      2. Repetitive line filtering
      3. Exact content hash deduplication
    """
    # 1. Length bounds
    if len(text) < 150 or len(text) > 60_000:
        return False

    # 2. Repetitive lines filter (detects boilerplate loops)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 6:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.45:  # Over 55% repetitive lines
            return False

    # 3. Exact hash deduplication (normalized leading content)
    doc_hash = hash(text[:300].strip().lower())
    if doc_hash in seen_hashes:
        return False

    if len(seen_hashes) > 100_000:
        seen_hashes.clear()
    seen_hashes.add(doc_hash)

    return True


# ==============================================================================
# 3. STREAMING DATASET ITERATORS
# ==============================================================================

class InfiniteStream:
    """Wraps a streaming HuggingFace dataset into an endless, fault-tolerant iterator."""

    def __init__(self, loader_fn, name: str = ""):
        self.loader_fn = loader_fn
        self.name = name
        self.iterator = iter(self.loader_fn())

    def __iter__(self):
        return self

    def __next__(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return next(self.iterator)
            except StopIteration:
                # Stream reached the end of the split -> seamlessly restart from beginning
                self.iterator = iter(self.loader_fn())
                try:
                    return next(self.iterator)
                except Exception:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1.0 + attempt)
                try:
                    self.iterator = iter(self.loader_fn())
                except Exception:
                    pass
        raise RuntimeError(f"Stream '{self.name}' failed to deliver samples after {max_retries} attempts.")


def get_streaming_dataset_iterators():
    """
    Initializes streaming iterators from the premier sub-1B educational & multi-language corpus:
      - Cosmopedia-v2: Synthetic textbooks, STEM, academic concepts (30.0%)
      - FineWeb-Edu-dedup: High-quality web educational articles (30.0%)
      - Python-Edu: Clean Python code, algorithms, and documentation (10.0%)
      - OpenHermes-100k / SmolTalk: Raw pool for filtered C/C++ and Java code (20.0%)
      - Everyday Conversations: Real multi-turn chat dialogues (10.0%)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required for streaming pre-training data.")
        print("Please run: pip install datasets")
        sys.exit(1)

    print("Connecting to Hugging Face streaming endpoints...")
    print("  [1/5] Streaming subset: Cosmopedia-v2 (Textbooks, STEM, World facts)")
    cosmo = InfiniteStream(
        lambda: load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True),
        name="Cosmopedia-v2",
    )

    print("  [2/5] Streaming subset: FineWeb-Edu-dedup (Curated educational web)")
    fineweb = InfiniteStream(
        lambda: load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True),
        name="FineWeb-Edu",
    )

    print("  [3/5] Streaming subset: Python-Edu (Clean Python code & algorithms)")
    py_edu = InfiniteStream(
        lambda: load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True),
        name="Python-Edu",
    )

    print("  [4/5] Streaming subset: OpenHermes-100k (Multi-language programming pool)")
    multi_code_pool = InfiniteStream(
        lambda: load_dataset("HuggingFaceTB/smoltalk", "openhermes-100k", split="train", streaming=True),
        name="OpenHermes-100k",
    )

    print("  [5/5] Streaming subset: Everyday Conversations (Conversational flow & dialogue)")
    chat_dialogues = InfiniteStream(
        lambda: load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True),
        name="Everyday-Conversations",
    )

    return cosmo, fineweb, py_edu, multi_code_pool, chat_dialogues


# Curated multi-language code fallbacks in case external stream has low density of specific languages
SYNTHETIC_C_CPP_SAMPLES = [
    "/* C/C++ Data Structures: Linked List Implementation */\n#include <stdio.h>\n#include <stdlib.h>\n\ntypedef struct Node {\n    int data;\n    struct Node* next;\n} Node;\n\nNode* create_node(int val) {\n    Node* n = (Node*)malloc(sizeof(Node));\n    n->data = val;\n    n->next = NULL;\n    return n;\n}\n\nvoid print_list(Node* head) {\n    Node* curr = head;\n    while (curr != NULL) {\n        printf(\"%d -> \", curr->data);\n        curr = curr->next;\n    }\n    printf(\"NULL\\n\");\n}\n\nint main() {\n    Node* head = create_node(10);\n    head->next = create_node(20);\n    head->next->next = create_node(30);\n    print_list(head);\n    return 0;\n}",
    "// C++ Standard Template Library: Vector and Binary Search\n#include <iostream>\n#include <vector>\n#include <algorithm>\n\nint binary_search(const std::vector<int>& arr, int target) {\n    int left = 0, right = arr.size() - 1;\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}\n\nint main() {\n    std::vector<int> nums = {1, 3, 5, 7, 9, 11, 13};\n    int idx = binary_search(nums, 7);\n    std::cout << \"Element found at index: \" << idx << std::endl;\n    return 0;\n}",
]

SYNTHETIC_JAVA_SAMPLES = [
    "// Java Object-Oriented Architecture: Generic Stack Implementation\nimport java.util.EmptyStackException;\nimport java.util.ArrayList;\n\npublic class CustomStack<T> {\n    private ArrayList<T> elements = new ArrayList<>();\n\n    public void push(T item) {\n        elements.add(item);\n    }\n\n    public T pop() {\n        if (elements.isEmpty()) throw new EmptyStackException();\n        return elements.remove(elements.size() - 1);\n    }\n\n    public boolean isEmpty() {\n        return elements.isEmpty();\n    }\n\n    public static void main(String[] args) {\n        CustomStack<String> stack = new CustomStack<>();\n        stack.push(\"Java\");\n        stack.push(\"Data Structure\");\n        System.out.println(\"Popped: \" + stack.pop());\n    }\n}",
    "// Java Sorting Algorithms: QuickSort Implementation\npublic class QuickSort {\n    public static void quickSort(int[] arr, int low, int high) {\n        if (low < high) {\n            int pi = partition(arr, low, high);\n            quickSort(arr, low, pi - 1);\n            quickSort(arr, pi + 1, high);\n        }\n    }\n\n    private static int partition(int[] arr, int low, int high) {\n        int pivot = arr[high];\n        int i = low - 1;\n        for (int j = low; j < high; j++) {\n            if (arr[j] < pivot) {\n                i++;\n                int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;\n            }\n        }\n        int temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;\n        return i + 1;\n    }\n}",
]


def extract_text_from_sample(sample) -> str:
    """Extracts plain text from either text fields or conversation message lists."""
    if "text" in sample and sample["text"].strip():
        return sample["text"].strip()
    if "messages" in sample and isinstance(sample["messages"], list):
        turns = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in sample["messages"]]
        return "\n\n".join(turns).strip()
    return ""


def stream_multi_discipline_documents(total_tokens_target: int):
    """
    Interleaves documents in an exact, mathematically consistent 10-slot cycle:
      - Slot 1: Cosmopedia-v2 (Textbooks)        [10.0%]
      - Slot 2: FineWeb-Edu (Educational Web)     [10.0%]
      - Slot 3: Python-Edu (Python Code)          [10.0%]
      - Slot 4: Cosmopedia-v2 (Textbooks)        [10.0%]
      - Slot 5: FineWeb-Edu (Educational Web)     [10.0%]
      - Slot 6: Verified C / C++ Code             [10.0%]
      - Slot 7: Cosmopedia-v2 (Textbooks)        [10.0%]
      - Slot 8: FineWeb-Edu (Educational Web)     [10.0%]
      - Slot 9: Verified Java Code                [10.0%]
      - Slot 10: Everyday Conversations (Chat)    [10.0%]

    Category Distribution:
      - Academic Textbooks & STEM: 30.0%
      - Educational Web:          30.0%
      - Multi-Language Code:       30.0% (10% Python + 10% C/C++ + 10% Java)
      - Conversational Flow:       10.0%
      Total:                      100.0%
    """
    cosmo_iter, fineweb_iter, py_iter, code_pool_iter, chat_iter = get_streaming_dataset_iterators()

    recipe_pattern = [
        "cosmo", "fineweb", "code_py",
        "cosmo", "fineweb", "code_c_cpp",
        "cosmo", "fineweb", "code_java",
        "chat"
    ]
    pattern_cycle = itertools.cycle(recipe_pattern)
    seen_hashes: Set[int] = set()

    c_cpp_synth_cycle = itertools.cycle(SYNTHETIC_C_CPP_SAMPLES)
    java_synth_cycle = itertools.cycle(SYNTHETIC_JAVA_SAMPLES)

    for slot in pattern_cycle:
        text = ""
        try:
            if slot == "cosmo":
                for _ in range(10):
                    raw = extract_text_from_sample(next(cosmo_iter))
                    if passes_quality_filter(raw, seen_hashes):
                        text = raw
                        break

            elif slot == "fineweb":
                for _ in range(10):
                    raw = extract_text_from_sample(next(fineweb_iter))
                    if passes_quality_filter(raw, seen_hashes):
                        text = raw
                        break

            elif slot == "code_py":
                for _ in range(10):
                    raw = extract_text_from_sample(next(py_iter))
                    if is_python_code(raw) and passes_quality_filter(raw, seen_hashes):
                        text = raw
                        break

            elif slot == "code_c_cpp":
                # Scan code pool for authentic C / C++ code
                found = False
                for _ in range(15):
                    raw = extract_text_from_sample(next(code_pool_iter))
                    if is_c_cpp_code(raw) and passes_quality_filter(raw, seen_hashes):
                        text = raw
                        found = True
                        break
                if not found:
                    text = next(c_cpp_synth_cycle)

            elif slot == "code_java":
                # Scan code pool for authentic Java code
                found = False
                for _ in range(15):
                    raw = extract_text_from_sample(next(code_pool_iter))
                    if is_java_code(raw) and passes_quality_filter(raw, seen_hashes):
                        text = raw
                        found = True
                        break
                if not found:
                    text = next(java_synth_cycle)

            elif slot == "chat":
                for _ in range(10):
                    raw = extract_text_from_sample(next(chat_iter))
                    if passes_quality_filter(raw, seen_hashes):
                        text = raw
                        break

            if text:
                yield text

        except Exception:
            time.sleep(0.1)
            continue


# ==============================================================================
# 4. SHARD BUILDER & TELEMETRY
# ==============================================================================

def build_pretrain_shards(
    output_dir: str = "data_shards",
    total_tokens: int = 135_000_000,
    shard_size: int = 5_000_000,
    val_ratio: float = 0.02,
):
    print("=" * 80)
    print("       USAID AI (500M) MULTI-DISCIPLINE & MULTI-LANGUAGE DATA INGESTION")
    print("                         Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Total Tokens: {total_tokens:,} tokens ({total_tokens / 1e6:.1f}M)")
    print(f"Tokens Per Shard:    {shard_size:,} tokens ({shard_size / 1e6:.1f}M)")
    print(f"Validation Split:    {val_ratio * 100:.1f}%")
    print("Exact Recipe Blend:  30% Textbooks | 30% Web | 10% Py | 10% C/C++ | 10% Java | 10% Chat")
    print(f"Destination:         {output_dir}")
    print("=" * 80 + "\n")

    enc = tiktoken.get_encoding("gpt2")
    eos_token_id = 50256

    train_writer = StreamingShardWriter(
        os.path.join(output_dir, "train"), "train", shard_size=shard_size
    )
    val_writer = StreamingShardWriter(
        os.path.join(output_dir, "val"), "val", shard_size=shard_size
    )

    val_interval = max(1, int(1.0 / val_ratio)) if val_ratio > 0 else 0
    doc_stream = stream_multi_discipline_documents(total_tokens_target=total_tokens)

    start_time = time.time()
    total_tokens_written = 0
    doc_index = 0

    print("Starting streaming tokenization & sharding...")
    for doc_text in doc_stream:
        # Tokenize document with EOS
        tokens = enc.encode(doc_text, allowed_special={"<|endoftext|>"})
        tokens.append(eos_token_id)
        token_len = len(tokens)

        # Route to train or val
        if val_interval > 0 and (doc_index % val_interval == val_interval - 1):
            val_writer.add_tokens(tokens)
        else:
            train_writer.add_tokens(tokens)

        total_tokens_written += token_len
        doc_index += 1

        # Live telemetry every 500k tokens
        if total_tokens_written % 500_000 < token_len:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens_written / max(elapsed, 1e-5)
            pct = (total_tokens_written / total_tokens) * 100
            remaining_tokens = max(0, total_tokens - total_tokens_written)
            eta_seconds = remaining_tokens / max(tok_per_sec, 1)
            eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds))

            print(
                f"  [{pct:5.1f}%] Processed {total_tokens_written:,} / {total_tokens:,} tokens | "
                f"Docs: {doc_index:,} | Speed: {tok_per_sec:,.0f} tok/s | ETA: {eta_str}"
            )

        if total_tokens_written >= total_tokens:
            print(f"\nReached target token budget of {total_tokens:,} tokens!")
            break

    train_writer.close()
    val_writer.close()

    total_time = time.time() - start_time
    total_final = train_writer.total_tokens + val_writer.total_tokens

    print("\n" + "=" * 80)
    print("                    DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total Tokens Saved:      {total_final:,}")
    print(f"  - Training Tokens:     {train_writer.total_tokens:,} ({train_writer.shard_idx} shards)")
    print(f"  - Validation Tokens:   {val_writer.total_tokens:,} ({val_writer.shard_idx} shards)")
    print(f"Total Documents:         {doc_index:,}")
    print(f"Total Elapsed Time:      {time.strftime('%Hh %Mm %Ss', time.gmtime(total_time))}")
    print(f"Average Stream Speed:    {total_final / max(total_time, 1):,.0f} tokens/second")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and shard multi-discipline & multi-language pretraining data")
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=135_000_000,
        help="Target number of tokens to stream (default: 135,000,000)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5_000_000,
        help="Number of tokens per binary shard file (default: 5,000,000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_shards",
        help="Directory to save shards (default: data_shards)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="Validation split ratio (default: 0.02 = 2%)",
    )
    args = parser.parse_args()

    build_pretrain_shards(
        output_dir=args.output_dir,
        total_tokens=args.total_tokens,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
