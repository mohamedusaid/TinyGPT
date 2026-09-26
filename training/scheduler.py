"""
Learning Rate Scheduler for TinyGPT-500M.
Implements Cosine Annealing with Linear Warmup.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler that increases linearly from 0 to 1 over warmup_steps,
    then decreases to min_lr_ratio following a cosine curve over (max_steps - warmup_steps).

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of initial steps for linear warmup.
        max_steps: Total number of training steps.
        min_lr_ratio: Ratio of minimum LR to peak LR (default 0.1, i.e. decay to 10% peak LR).
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if current_step >= max_steps:
            return min_lr_ratio

        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
