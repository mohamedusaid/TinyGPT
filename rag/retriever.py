"""
BM25 & Semantic Passage Retriever for Usaid AI RAG.
Created by Mohamed Usaid.
"""

import math
import re
from collections import Counter
from typing import List, Tuple
from rag.knowledge_base import KnowledgeBase, KnowledgeChunk


def tokenize_query_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric terms for lexical matching."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Retriever:
    """
    Production BM25 ranker for zero-shot passage retrieval.
    Operates without heavy external vector DB dependencies.
    """

    def __init__(self, knowledge_base: KnowledgeBase, k1: float = 1.5, b: float = 0.75):
        self.kb = knowledge_base
        self.k1 = k1
        self.b = b

        self.corpus_size = len(self.kb.chunks)
        self.doc_lengths = []
        self.doc_frequencies = Counter()
        self.doc_term_freqs = []

        self._build_index()

    def _build_index(self):
        total_length = 0
        for chunk in self.kb.chunks:
            tokens = tokenize_query_text(chunk.text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf.keys():
                self.doc_frequencies[term] += 1

        self.avg_doc_length = total_length / max(1, self.corpus_size)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 3.0) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Retrieves the top_k most relevant KnowledgeChunks for the query, ranked by BM25 score.
        Filters out low-confidence matches (below min_score) to prevent context pollution.
        """
        query_tokens = tokenize_query_text(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            df = self.doc_frequencies.get(term, 0)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones IDF
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            for idx, tf_counter in enumerate(self.doc_term_freqs):
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                scores[idx] += idf * (numerator / denominator)

        # Pair with chunks, enforce minimum relevance score, and sort descending
        scored_chunks = [
            (self.kb.chunks[idx], scores[idx])
            for idx in range(self.corpus_size)
            if scores[idx] >= min_score
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]
