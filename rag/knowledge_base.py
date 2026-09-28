"""
Knowledge Base Ingestion & Chunking Engine for Usaid AI RAG.
Created by Mohamed Usaid.
"""

import os
import re
from typing import List, Dict, Any


class KnowledgeChunk:
    def __init__(self, chunk_id: int, text: str, source: str, metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<Chunk {self.chunk_id} from {os.path.basename(self.source)} ({len(self.text.split())} words)>"


class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge_base", chunk_words: int = 200, overlap_words: int = 40):
        self.knowledge_dir = knowledge_dir
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.chunks: List[KnowledgeChunk] = []

        if os.path.exists(self.knowledge_dir):
            self.load_directory(self.knowledge_dir)

    def load_directory(self, directory: str):
        """Recursively parses all supported text and code files."""
        self.chunks.clear()
        chunk_counter = 0

        supported_exts = {".txt", ".md", ".json", ".py", ".c", ".cpp", ".java"}

        for root, _, files in os.walk(directory):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        file_chunks = self._chunk_document(content, file_path, start_id=chunk_counter)
                        self.chunks.extend(file_chunks)
                        chunk_counter += len(file_chunks)
                    except Exception as e:
                        print(f"Warning: Failed to read {file_path}: {e}")

        print(f"[KnowledgeBase] Loaded {len(self.chunks)} knowledge chunks from '{directory}'.")

    def add_document(self, text: str, source: str = "memory"):
        """Adds a raw text string dynamically into the knowledge base."""
        start_id = len(self.chunks)
        new_chunks = self._chunk_document(text, source, start_id=start_id)
        self.chunks.extend(new_chunks)

    def _chunk_document(self, content: str, source: str, start_id: int) -> List[KnowledgeChunk]:
        """Splits document content into overlapping word windows."""
        # Paragraph or section-based splitting first
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        chunks = []
        current_words = []
        current_id = start_id

        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue

            if len(current_words) + len(para_words) <= self.chunk_words:
                current_words.extend(para_words)
            else:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(KnowledgeChunk(chunk_id=current_id, text=chunk_text, source=source))
                    current_id += 1
                    # Keep overlap from previous chunk
                    current_words = current_words[-self.overlap_words:] if self.overlap_words > 0 else []

                # If paragraph itself is longer than chunk_words, split it by chunks
                while len(para_words) > self.chunk_words:
                    chunk_slice = para_words[: self.chunk_words]
                    chunk_text = " ".join(chunk_slice)
                    chunks.append(KnowledgeChunk(chunk_id=current_id, text=chunk_text, source=source))
                    current_id += 1
                    para_words = para_words[self.chunk_words - self.overlap_words:]

                current_words.extend(para_words)

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(KnowledgeChunk(chunk_id=current_id, text=chunk_text, source=source))

        return chunks
