"""
Shared checkpoint resolution & model loading helpers for TinyGPT-500M tools.
Used by the token-probability explorer and the embedding projector scripts.
"""

import os
import torch

from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M

CHECKPOINT_CANDIDATES = [
    "sft_checkpoints/usaid_ai_500m.pt",
    "checkpoints_sft/usaid_ai_500m.pt",
    "checkpoints/usaid_ai_500m.pt",
    "checkpoints/best_usaid_ai_500m.pt",
    "checkpoints/best_tinygpt_500m.pt",
]


def resolve_checkpoint(path: str | None) -> str:
    """Returns `path` if it exists, otherwise the first existing known candidate."""
    if path and os.path.exists(path):
        return path
    for c in CHECKPOINT_CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Checkpoint not found at '{path}' or any of: {CHECKPOINT_CANDIDATES}. "
        "Train a model first using scripts/run_pretrain.py or scripts/run_sft.py."
    )


def load_raw_checkpoint(path: str) -> tuple[dict, dict]:
    """Loads a checkpoint file and returns (state_dict, config_dict)."""
    try:
        # mmap avoids copying the whole 2 GB file into RAM up-front
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {}) or {}
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    raise ValueError(f"Unrecognized checkpoint format in: {path}")


def resolve_device_dtype(device: str | None, dtype: str = "auto") -> tuple[torch.device, torch.dtype]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype in table:
        return dev, table[dtype]
    return dev, (torch.float16 if dev.type == "cuda" else torch.float32)


def load_model(
    checkpoint: str | None,
    device: str | None = None,
    dtype: str = "auto",
) -> tuple[TinyGPT500M, str]:
    """Builds TinyGPT500M from a checkpoint. Returns (model, resolved_checkpoint_path)."""
    path = resolve_checkpoint(checkpoint)
    dev, target_dtype = resolve_device_dtype(device, dtype)

    print(f"Loading checkpoint from: {path} ...")
    state_dict, config_dict = load_raw_checkpoint(path)
    config = TinyGPTConfig.from_dict(config_dict) if config_dict else TinyGPTConfig()

    model = TinyGPT500M(config)
    model.load_state_dict(state_dict)
    model.to(dtype=target_dtype, device=dev)
    model.eval()
    print(f"Model loaded ({model.count_parameters():,} params) on {dev} [{target_dtype}].")
    return model, path
