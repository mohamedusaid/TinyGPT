"""
Optimizer setup for TinyGPT-500M.
Implements decoupled weight decay filtering for AdamW.
Applies weight decay (e.g. 0.1) exclusively to 2D weight matrices (Linear layers)
and disables weight decay for 1D vectors (RMSNorm scales and biases).
"""

import inspect
import torch
import torch.nn as nn


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Separates model parameters into decayed and non-decayed groups.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D parameters (RMSNorm weight, any biases) should not be decayed
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    decay_count = sum(p.numel() for p in decay_params)
    no_decay_count = sum(p.numel() for p in no_decay_params)
    print(
        f"AdamW Parameter Grouping: {decay_count:,} params decayed (weight_decay={weight_decay}), "
        f"{no_decay_count:,} params non-decayed."
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()

    kwargs = {"lr": learning_rate, "betas": betas, "eps": eps}
    if use_fused:
        kwargs["fused"] = True
        print("  - Fast CUDA Fused AdamW Kernel: Enabled")

    optimizer = torch.optim.AdamW(optim_groups, **kwargs)
    return optimizer
