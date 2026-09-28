"""
End-to-End Test Suite for Usaid AI (500M) — Created by Mohamed Usaid.
Tests:
  1. SFT Dataset generation & prompt loss masking (-100)
  2. RAG KnowledgeBase parsing & BM25 passage retrieval
  3. Grounded RAG prompt assembly
"""

import os
import sys
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from data.tokenizer import GPT2Tokenizer
from sft.sft_dataset import SFTDataset
from scripts.download_sft_data import build_sft_dataset
from rag.knowledge_base import KnowledgeBase
from rag.retriever import BM25Retriever
from rag.pipeline import RAGPipeline


def test_sft_pipeline():
    test_file = "tests/test_sft_suite.jsonl"
    try:
        # 1. Build test SFT dataset
        build_sft_dataset(output_file=test_file, num_samples=50, identity_repeat=10)
        assert os.path.exists(test_file)

        # 2. Load with SFTDataset
        tokenizer = GPT2Tokenizer()
        dataset = SFTDataset(test_file, tokenizer=tokenizer, sequence_length=512)
        assert len(dataset) > 0

        # 3. Verify Prompt Loss Masking
        x, y = dataset[0]
        assert x.shape[0] == 511
        assert y.shape[0] == 511

        # Must have prompt tokens masked with -100
        masked_count = (y == -100).sum().item()
        assert masked_count > 0, "Prompt tokens must be masked with -100 for proper SFT backprop"

        # Must have unmasked target tokens
        unmasked_count = (y != -100).sum().item()
        assert unmasked_count > 0, "Assistant response tokens must be unmasked"

        print(f"[TEST PASSED] SFT pipeline verified: {len(dataset)} examples, {masked_count} prompt tokens masked.")

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_rag_pipeline():
    # 1. Ingest KnowledgeBase
    kb = KnowledgeBase("knowledge_base")
    assert len(kb.chunks) > 0, "KnowledgeBase must index chunks from knowledge_base/"

    # 2. Test BM25 Retrieval
    retriever = BM25Retriever(kb)
    results = retriever.retrieve("Who created Usaid AI and what is the architecture?", top_k=2)

    assert len(results) > 0, "Retriever must return top-k passages"
    top_chunk, score = results[0]
    assert score > 0, "BM25 score must be positive for matching query"
    assert "Mohamed Usaid" in top_chunk.text, "Retrieved chunk must contain Mohamed Usaid"

    # 3. Test RAG Prompt Formatting
    pipeline = RAGPipeline(model=None, tokenizer=GPT2Tokenizer(), retriever=retriever)
    prompt = pipeline.format_rag_prompt("Who created Usaid AI?", results)

    assert "Mohamed Usaid" in prompt
    assert "Reference Context:" in prompt
    assert "User: Who created Usaid AI?" in prompt
    assert "Assistant: " in prompt

    print(f"[TEST PASSED] RAG retrieval & prompt assembly verified (BM25 score: {score:.2f}).")


def test_language_detectors_and_multi_turn():
    from scripts.download_pretrain_data import is_c_cpp_code, is_java_code, is_python_code

    # 1. Test Language Detectors
    c_sample = "#include <stdio.h>\nint main() { printf(\"Hello C\"); return 0; }"
    cpp_sample = "#include <iostream>\nint main() { std::cout << \"Hello C++\"; return 0; }"
    java_sample = "public class Solution { public static void main(String[] args) { System.out.println(\"Java\"); } }"
    py_sample = "def calculate_factorial(n):\n    if n <= 1: return 1\n    return n * calculate_factorial(n - 1)"

    assert is_c_cpp_code(c_sample), "C sample must be identified"
    assert is_c_cpp_code(cpp_sample), "C++ sample must be identified"
    assert is_java_code(java_sample), "Java sample must be identified"
    assert is_python_code(py_sample), "Python sample must be identified"
    print("[TEST PASSED] Language verification filters (C, C++, Java, Python) verified.")

    # 2. Test Multi-Turn Masking
    multi_turn_example = {
        "messages": [
            {"role": "user", "content": "What is C?"},
            {"role": "assistant", "content": "C is a low-level programming language."},
            {"role": "user", "content": "Show me a hello world."},
            {"role": "assistant", "content": "#include <stdio.h>\nint main() { printf(\"Hello\"); }"}
        ]
    }
    temp_multi_file = "tests/test_multi_turn.jsonl"
    try:
        import json
        with open(temp_multi_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(multi_turn_example) + "\n")

        tokenizer = GPT2Tokenizer()
        ds = SFTDataset(temp_multi_file, tokenizer=tokenizer, sequence_length=256)
        x, y = ds[0]

        # Verify both turns exist and both user turns are masked
        masked_tokens = (y == -100).sum().item()
        supervised_tokens = (y != -100).sum().item()
        assert masked_tokens > 0 and supervised_tokens > 0, "Both user and assistant tokens must be present"
        print(f"[TEST PASSED] Multi-turn SFT conversation masking verified: {masked_tokens} masked, {supervised_tokens} supervised.")

    finally:
        if os.path.exists(temp_multi_file):
            os.remove(temp_multi_file)


if __name__ == "__main__":
    test_sft_pipeline()
    test_rag_pipeline()
    test_language_detectors_and_multi_turn()
    print("\nALL USAID AI AUDITED TESTS PASSED WITH 100% COMPLIANCE! 🚀")
