"""
Supervised Fine-Tuning (SFT) Trainer for TinyGPT-500M.
Aligns pre-trained base model checkpoints on conversational instruction data.
"""

import math
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from model.transformer import TinyGPT500M
from sft.sft_dataset import SFTDataset
from training.optim import configure_optimizers
from training.scheduler import get_cosine_schedule_with_warmup
from training.checkpoint import CheckpointManager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class SFTTrainer:
    def __init__(
        self,
        model: TinyGPT500M,
        train_dataset: SFTDataset,
        val_dataset: SFTDataset | None = None,
        config: dict | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                self.use_scaler = False
                self.precision_desc = "AMP BF16"
            else:
                self.dtype = torch.float16
                self.use_scaler = True
                self.precision_desc = "AMP FP16 with GradScaler"
        else:
            self.dtype = torch.float32
            self.use_scaler = False
            self.precision_desc = "Float32 (CPU)"

        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        self.batch_size = self.cfg.get("micro_batch_size", 2)
        self.grad_accum_steps = self.cfg.get("gradient_accumulation_steps", 16)
        self.learning_rate = self.cfg.get("learning_rate", 2e-5)
        self.min_lr = self.cfg.get("min_lr", 2e-6)
        self.epochs = self.cfg.get("epochs", 3)
        self.weight_decay = self.cfg.get("weight_decay", 0.01)
        self.max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        self.checkpoint_dir = self.cfg.get("checkpoint_dir", "checkpoints_sft")

        self.train_loader = TorchDataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        total_steps = (len(self.train_loader) // self.grad_accum_steps) * self.epochs
        self.total_steps = max(1, total_steps)
        self.warmup_steps = int(self.total_steps * self.cfg.get("warmup_ratio", 0.05))

        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
        )

        min_lr_ratio = self.min_lr / self.learning_rate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

    def train(self):
        print("=" * 80)
        print("                   TINYGPT-500M SFT INSTRUCTION TUNING")
        print("=" * 80)
        print(f"Device:                 {self.device} ({self.precision_desc})")
        print(f"Dataset Size:           {len(self.train_dataset):,} instruction pairs")
        print(f"Total Epochs:           {self.epochs}")
        print(f"Planned Steps:          {self.total_steps:,} optimizer steps")
        print(f"Peak Learning Rate:     {self.learning_rate:.2e}")
        print("=" * 80 + "\n")

        self.model.train()
        step = 0
        best_loss = float("inf")

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.device.type == "cuda"),
                ):
                    out = self.model(x, targets=y)
                    loss = out.loss / self.grad_accum_steps

                accum_loss += loss.item()

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % self.grad_accum_steps == 0:
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    if self.use_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    step += 1
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch + 1} | Step {step:4d}/{self.total_steps} | "
                        f"SFT Loss: {accum_loss:.4f} | Grad Norm: {grad_norm:.2f} | "
                        f"LR: {current_lr:.2e}"
                    )

                    if accum_loss < best_loss:
                        best_loss = accum_loss
                        self.checkpoint_manager.save(
                            step=step,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler if self.use_scaler else None,
                            val_loss=best_loss,
                            tokens_trained=0,
                            is_best=True,
                            filename="best_tinygpt_500m_sft.pt",
                        )

                    accum_loss = 0.0

        print(f"\nSFT Alignment complete! Best Loss: {best_loss:.4f}")
