"""
Checkpoint Manager for TinyGPT-500M.
Handles atomic saving and resuming of model weights, optimizer state,
scaler state, scheduler state, step counter, and RNG seeds.
"""

import os
import random
import torch
import numpy as np


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", max_to_keep: int = 1):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Scan for existing periodic checkpoints to support clean pruning across resumes
        self.saved_checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            existing = [
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("checkpoint_step_") and f.endswith(".pt")
            ]
            self.saved_checkpoints = sorted(existing)

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.cuda.amp.GradScaler | None,
        val_loss: float,
        tokens_trained: int,
        is_best: bool = False,
        filename: str | None = None,
    ) -> str:
        """Saves complete training checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{step:06d}.pt"

        save_path = os.path.join(self.checkpoint_dir, filename)

        raw_model = model.module if hasattr(model, "module") else model
        checkpoint_dict = {
            "step": step,
            "val_loss": val_loss,
            "tokens_trained": tokens_trained,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": (
                raw_model.config.to_dict()
                if hasattr(raw_model, "config") and hasattr(raw_model.config, "to_dict")
                else (getattr(raw_model, "config", {}))
            ),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

        # Save to temporary file first for atomic write safety
        temp_path = save_path + ".tmp"
        torch.save(checkpoint_dict, temp_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(temp_path, save_path)

        # Track and prune older periodic checkpoints to conserve disk space
        if filename.startswith("checkpoint_step_") and self.max_to_keep > 0:
            if save_path not in self.saved_checkpoints:
                self.saved_checkpoints.append(save_path)
            while len(self.saved_checkpoints) > self.max_to_keep:
                oldest = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest) and oldest != save_path:
                    try:
                        os.remove(oldest)
                    except OSError:
                        pass

        # Save best model as lightweight inference/SFT artifact (~1.86 GB vs 5.59 GB)
        # SFT and chat only need model weights + config, not 3.73 GB optimizer buffers
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_tinygpt_500m.pt")
            best_dict = {
                "step": step,
                "val_loss": val_loss,
                "tokens_trained": tokens_trained,
                "model_state_dict": raw_model.state_dict(),
                "config": checkpoint_dict["config"],
            }
            best_temp = best_path + ".tmp"
            torch.save(best_dict, best_temp)
            if os.path.exists(best_path):
                os.remove(best_path)
            os.rename(best_temp, best_path)

        return save_path

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        device: str = "cpu",
    ) -> dict:
        """Restores model and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "rng" in checkpoint:
            rng = checkpoint["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"])

        print(
            f"Successfully restored checkpoint from step {checkpoint['step']:,} "
            f"(Val Loss: {checkpoint.get('val_loss', 0.0):.4f})"
        )
        return checkpoint
