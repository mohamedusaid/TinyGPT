"""
Memory-Mapped Binary Sharded Dataset for TinyGPT-500M.
Reads tokenized datasets stored in uint16 binary format (.bin) via numpy memmap.
Allows streaming and training across multiple shards with true multi-shard sampling
and zero RAM exhaustion.
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
        self.req_len = sequence_length + 1
        self.files = []

        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.bin")))
        elif os.path.isfile(data_path):
            self.files = [data_path]

        if not self.files:
            raise FileNotFoundError(
                f"No .bin shard files found in path: {data_path}"
            )

        # Open memmap for each shard and compute capacities
        self.mmaps = []
        self.shard_lengths = []
        self.valid_shard_indices = []
        self.total_tokens = 0

        for idx, f in enumerate(self.files):
            m = np.memmap(f, dtype=np.uint16, mode="r")
            self.mmaps.append(m)
            token_count = len(m)
            self.shard_lengths.append(token_count)
            self.total_tokens += token_count
            if token_count >= self.req_len:
                self.valid_shard_indices.append(idx)

        if not self.valid_shard_indices:
            raise ValueError(
                f"All shards in {data_path} are smaller than required sequence length {self.req_len}"
            )

        # Sampling weights proportional to number of valid starting positions in each shard
        weights = [
            self.shard_lengths[i] - self.req_len + 1
            for i in self.valid_shard_indices
        ]
        total_weight = sum(weights)
        self.sampling_probs = [w / total_weight for w in weights]

        # Tracking for telemetry/auditing
        self.last_sampled_shards = []

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch across ALL available shards.
        Each sequence in the batch can come from a different shard, weighted by shard length.

        Args:
            batch_size: Number of sequences (micro-batch size).
            device: Target device ('cpu' or 'cuda').

        Returns:
            Tuple of (x, y) where x is (batch_size, sequence_length)
            and y is x shifted by +1 token.
        """
        # Sample shard indices for each element in batch according to probability
        chosen_shard_indices = np.random.choice(
            self.valid_shard_indices, size=batch_size, p=self.sampling_probs
        )
        self.last_sampled_shards = [int(s) for s in chosen_shard_indices]

        batch_x = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        batch_y = np.empty((batch_size, self.sequence_length), dtype=np.int64)

        for i, shard_idx in enumerate(chosen_shard_indices):
            mmap = self.mmaps[shard_idx]
            max_start = len(mmap) - self.req_len
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

            chunk = mmap[start : start + self.req_len].astype(np.int64)
            batch_x[i] = chunk[:-1]
            batch_y[i] = chunk[1:]

        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and device.startswith("cuda"))
        if is_cuda:
            x_tensor = torch.from_numpy(batch_x).pin_memory().to(device, non_blocking=True)
            y_tensor = torch.from_numpy(batch_y).pin_memory().to(device, non_blocking=True)
        else:
            x_tensor = torch.from_numpy(batch_x).to(device)
            y_tensor = torch.from_numpy(batch_y).to(device)

        return x_tensor, y_tensor

    def get_deterministic_batch(
        self,
        batch_idx: int,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a 100% deterministic batch across shards for evaluation.
        Given the same batch_idx and batch_size, this always produces the identical sequences.
        """
        # Use pseudo-random generator with fixed seed based on batch_idx
        rng = np.random.RandomState(seed=42 + batch_idx * 1000)
        chosen_shard_indices = rng.choice(
            self.valid_shard_indices, size=batch_size, p=self.sampling_probs
        )

        batch_x = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        batch_y = np.empty((batch_size, self.sequence_length), dtype=np.int64)

        for i, shard_idx in enumerate(chosen_shard_indices):
            mmap = self.mmaps[shard_idx]
            max_start = len(mmap) - self.req_len
            start = rng.randint(0, max_start + 1) if max_start > 0 else 0

            chunk = mmap[start : start + self.req_len].astype(np.int64)
            batch_x[i] = chunk[:-1]
            batch_y[i] = chunk[1:]

        x_tensor = torch.from_numpy(batch_x).to(device)
        y_tensor = torch.from_numpy(batch_y).to(device)

        return x_tensor, y_tensor
