"""
Perplexity Evaluation for TinyGPT-500M.
Computes cross-entropy loss and token perplexity over validation binary shards.
"""

import math
import torch
from data.dataset import BinaryShardedDataset
from model.transformer import TinyGPT500M


@torch.no_grad()
def evaluate_perplexity(
    model: TinyGPT500M,
    dataset: BinaryShardedDataset,
    num_batches: int = 50,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict:
    model.eval()
    total_loss = 0.0
    device_obj = torch.device(device)

    for _ in range(num_batches):
        x, y = dataset.get_batch(batch_size=batch_size, device=device_obj)
        out = model(x, targets=y)
        total_loss += out.loss.item()

    avg_loss = total_loss / max(1, num_batches)
    ppl = math.exp(min(avg_loss, 20.0))

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "batches_evaluated": num_batches,
    }
