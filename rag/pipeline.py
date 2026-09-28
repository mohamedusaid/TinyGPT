"""
RAG Prompt Pipeline & Inference Orchestrator for Usaid AI (500M).
Created by Mohamed Usaid.
"""

from typing import List, Tuple, Optional
import torch

from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from inference.generate import generate
from rag.retriever import BM25Retriever
from rag.knowledge_base import KnowledgeChunk


class RAGPipeline:
    def __init__(
        self,
        model: TinyGPT500M,
        tokenizer: GPT2Tokenizer,
        retriever: BM25Retriever,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.system_prompt = system_prompt or (
            "You are Usaid AI, an intelligent, helpful, and concise conversational AI assistant created by Mohamed Usaid. "
            "Use the verified reference context below to answer the user's inquiry factually and clearly."
        )

    def format_rag_prompt(self, query: str, retrieved_passages: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Constructs grounded context prompt for Usaid AI."""
        if not retrieved_passages:
            context_block = "No external reference documents found."
        else:
            chunks_formatted = []
            for rank, (chunk, score) in enumerate(retrieved_passages, 1):
                chunks_formatted.append(f"[{rank}] Source: {chunk.source}\n{chunk.text.strip()}")
            context_block = "\n\n".join(chunks_formatted)

        prompt = (
            f"{self.system_prompt}\n\n"
            f"Reference Context:\n"
            f"----------------------------------------\n"
            f"{context_block}\n"
            f"----------------------------------------\n\n"
            f"User: {query}\n\n"
            f"Assistant: "
        )
        return prompt

    def query(
        self,
        query_text: str,
        top_k: int = 2,
        max_new_tokens: int = 150,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Tuple[str, List[Tuple[KnowledgeChunk, float]], str]:
        """
        Executes complete RAG pipeline:
          1. Retrieves top_k relevant passages
          2. Formats prompt with system persona and references
          3. Generates autoregressive completion with KV cache
        """
        # 1. Retrieve
        retrieved_passages = self.retriever.retrieve(query_text, top_k=top_k)

        # 2. Format
        rag_prompt = self.format_rag_prompt(query_text, retrieved_passages)

        # 3. Generate
        completion, stop_reason = generate(
            model=self.model,
            prompt=rag_prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=40,
        )

        # Extract only the assistant answer
        if "Assistant: " in completion:
            answer = completion.split("Assistant: ")[-1].strip()
        else:
            answer = completion.strip()

        return answer, retrieved_passages, stop_reason
