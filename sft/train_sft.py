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

        # DDP Multi-GPU Support
        self.is_ddp = int(os.environ.get("RANK", -1)) != -1
        if self.is_ddp:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            major, _ = torch.cuda.get_device_capability(self.device)
            has_native_bf16 = (major >= 8) and torch.cuda.is_bf16_supported()
            if has_native_bf16:
                self.dtype = torch.bfloat16
                self.use_scaler = False
                self.precision_desc = f"AMP BF16 ({self.world_size}x GPU)" if self.is_ddp else "AMP BF16"
            else:
                self.dtype = torch.float16
                self.use_scaler = True
                self.precision_desc = f"AMP FP16 with GradScaler ({self.world_size}x GPU)" if self.is_ddp else "AMP FP16 with GradScaler"
        else:
            self.dtype = torch.float32
            self.use_scaler = False
            self.precision_desc = "Float32 (CPU)"

        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        if self.is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.batch_size = self.cfg.get("micro_batch_size", 2)
        total_grad_accum = self.cfg.get("gradient_accumulation_steps", 16)
        if self.is_ddp and self.world_size > 1:
            self.grad_accum_steps = max(1, total_grad_accum // self.world_size)
        else:
            self.grad_accum_steps = total_grad_accum

        self.learning_rate = self.cfg.get("learning_rate", 2e-5)
        self.min_lr = self.cfg.get("min_lr", 2e-6)
        self.epochs = self.cfg.get("epochs", 3)
        self.weight_decay = self.cfg.get("weight_decay", 0.01)
        self.max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        self.checkpoint_dir = self.cfg.get("checkpoint_dir", "checkpoints")

        if self.is_ddp:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.train_loader = TorchDataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=sampler
            )
        else:
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

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, max_to_keep=1)

    def train(self):
        if self.rank == 0:
            print("=" * 80)
            print("              USAID AI (500M) SUPERVISED FINE-TUNING (SFT)")
            print("                       Created by Mohamed Usaid")
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
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            if self.rank == 0:
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
                    if self.rank == 0:
                        print(
                            f"Epoch {epoch + 1} | Step {step:4d}/{self.total_steps} | "
                            f"SFT Loss: {accum_loss:.4f} | Grad Norm: {grad_norm:.2f} | "
                            f"LR: {current_lr:.2e}"
                        )

                    if accum_loss < best_loss:
                        best_loss = accum_loss
                        if self.rank == 0:
                            self.checkpoint_manager.save(
                                step=step,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                scaler=self.scaler if self.use_scaler else None,
                                val_loss=best_loss,
                                tokens_trained=0,
                                is_best=True,
                                filename="best_usaid_ai_500m.pt",
                            )

                    accum_loss = 0.0

        if self.rank == 0:
            final_model_path = os.path.join(self.checkpoint_dir, "usaid_ai_500m.pt")
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(
                {
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "config": raw_model.config.to_dict(),
                    "val_loss": best_loss,
                    "model_name": "Usaid AI",
                    "author": "Mohamed Usaid",
                },
                final_model_path,
            )
            print("\n" + "=" * 80)
            print(f"SFT Alignment Complete! Best Loss: {best_loss:.4f}")
            print(f"Usaid AI Aligned Model saved to: {final_model_path}")
            print("=" * 80 + "\n")

        if self.is_ddp:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
