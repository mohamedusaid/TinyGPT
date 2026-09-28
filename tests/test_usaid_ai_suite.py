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


if __name__ == "__main__":
    test_sft_pipeline()
    test_rag_pipeline()
    print("\nALL USAID AI TESTS PASSED SUCCESSFULLY! 🚀")
