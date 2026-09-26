"""
Memory-Mapped Binary Sharded Dataset for TinyGPT-500M.
Reads tokenized datasets stored in uint16 binary format (.bin) via numpy memmap.
Allows streaming and training on millions/billions of tokens with near-zero RAM footprint.
"""

import glob
import os
import numpy as np
import torch


class BinaryShardedDataset:
    def __init__(self, data_path: str, sequence_length: int = 1024):
        """
        Args:
            data_path: Directory containing .bin shard files, or path to a single .bin file.
            sequence_length: Target context length (e.g. 1024 tokens).
        """
        self.sequence_length = sequence_length
        self.files = []

        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.bin")))
        elif os.path.isfile(data_path):
            self.files = [data_path]

        if not self.files:
            raise FileNotFoundError(
                f"No .bin shard files found in path: {data_path}"
            )

        # Index shards and calculate total token count
        self.shards = []
        self.shard_lengths = []
        self.total_tokens = 0

        for f in self.files:
            # uint16 uses 2 bytes per token
            token_count = os.path.getsize(f) // 2
            self.shard_lengths.append(token_count)
            self.total_tokens += token_count

        self.current_shard_idx = 0
        self.current_mmap = None
        self._load_shard(0)

    def _load_shard(self, shard_idx: int):
        self.current_shard_idx = shard_idx
        self.current_mmap = np.memmap(
            self.files[shard_idx], dtype=np.uint16, mode="r"
        )

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of contiguous sequence chunks for causal LM training.

        Args:
            batch_size: Number of sequences in the batch (micro-batch size).
            device: Target device ('cpu' or 'cuda').

        Returns:
            Tuple of (x, y) where:
                x has shape (batch_size, sequence_length)
                y has shape (batch_size, sequence_length) shifted by +1 token.
        """
        shard_len = len(self.current_mmap)
        req_len = self.sequence_length + 1

        if shard_len <= req_len:
            # Switch to next shard if current is exhausted
            next_idx = (self.current_shard_idx + 1) % len(self.files)
            self._load_shard(next_idx)
            shard_len = len(self.current_mmap)

        # Sample random starting indices
        max_start = shard_len - req_len
        starts = np.random.randint(0, max_start, size=batch_size)

        batch_x = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        batch_y = np.empty((batch_size, self.sequence_length), dtype=np.int64)

        for i, start in enumerate(starts):
            chunk = self.current_mmap[start : start + req_len].astype(np.int64)
            batch_x[i] = chunk[:-1]
            batch_y[i] = chunk[1:]

        x_tensor = torch.from_numpy(batch_x).to(device)
        y_tensor = torch.from_numpy(batch_y).to(device)

        return x_tensor, y_tensor
