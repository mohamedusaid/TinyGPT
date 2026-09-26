"""
DataLoader for TinyGPT-500M.
Provides an asynchronous or synchronized batch iteration interface for pretraining.
"""

import torch
from data.dataset import BinaryShardedDataset


class DataLoader:
    def __init__(
        self,
        dataset: BinaryShardedDataset,
        batch_size: int,
        device: torch.device | str = "cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches the next batch of (x, y) tensors."""
        return self.dataset.get_batch(
            batch_size=self.batch_size, device=self.device
        )

    def __iter__(self):
        while True:
            yield self.next_batch()
