"""
Production Pretraining Engine for TinyGPT-500M.
Optimized for Google Colab Tesla T4 (FP16 + GradScaler) and Ampere/Hopper GPU clusters (BF16).
Features gradient accumulation, gradient clipping, Cosine-Warmup scheduling, and live telemetry.
"""

import math
import os
import sys
import random
import time
from contextlib import nullcontext
import numpy as np

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.dataset import BinaryShardedDataset
from training.optim import configure_optimizers
from training.scheduler import get_cosine_schedule_with_warmup
from training.checkpoint import CheckpointManager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Trainer:
    def __init__(
        self,
        model: TinyGPT500M,
        train_dataset: BinaryShardedDataset,
        val_dataset: BinaryShardedDataset | None = None,
        train_config: dict | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = train_config or {}

        # 1. Device, DDP, and Precision Configuration
        self.is_ddp = int(os.environ.get("RANK", -1)) != -1
        if self.is_ddp:
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

        # Rank-distinct seeding: ensures GPU 0 and GPU 1 sample completely different sequence batches
        base_seed = self.cfg.get("seed", 42)
        rank_seed = base_seed + self.rank * 10007
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        self.model.to(self.device)

        if self.device.type == "cuda":
            # Hardware BF16 Tensor Cores are only available on Ampere (sm_80+) or Hopper (sm_90+)
            # Turing (Tesla T4, sm_75) does NOT have hardware BF16; it has fast FP16 Tensor Cores.
            major, _ = torch.cuda.get_device_capability(self.device)
            has_native_bf16 = (major >= 8) and torch.cuda.is_bf16_supported()
            if has_native_bf16:
                self.dtype = torch.bfloat16
                self.use_scaler = False
                self.precision_desc = (
                    f"AMP BF16 ({self.world_size}x GPU DDP)"
                    if self.is_ddp
                    else "AMP BF16 (Native Ampere/Hopper)"
                )
            else:
                self.dtype = torch.float16
                self.use_scaler = True
                self.precision_desc = (
                    f"AMP FP16 with GradScaler ({self.world_size}x GPU DDP)"
                    if self.is_ddp
                    else "AMP FP16 with GradScaler (Tesla T4 / Turing)"
                )
        else:
            self.dtype = torch.float32
            self.use_scaler = False
            self.precision_desc = "Float32 (CPU Mode)"

        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        # Gradient Checkpointing (Activation Checkpointing for 16GB GPUs)
        self.gradient_checkpointing = self.cfg.get("gradient_checkpointing", True)
        self.model.config.gradient_checkpointing = self.gradient_checkpointing

        # Wrap with DistributedDataParallel if running multi-GPU
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # 2. Hyperparameters
        self.micro_batch_size = self.cfg.get("micro_batch_size", 1)
        total_grad_accum = self.cfg.get("gradient_accumulation_steps", 64)
        if self.is_ddp and self.world_size > 1:
            self.gradient_accumulation_steps = max(1, total_grad_accum // self.world_size)
        else:
            self.gradient_accumulation_steps = total_grad_accum

        self.sequence_length = self.cfg.get("sequence_length", 1024)
        self.max_steps = self.cfg.get("max_steps", 2000)
        self.learning_rate = self.cfg.get("learning_rate", 3e-4)
        self.min_lr = self.cfg.get("min_lr", 3e-5)
        self.warmup_steps = self.cfg.get("warmup_steps", 200)
        self.weight_decay = self.cfg.get("weight_decay", 0.1)
        self.max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        self.eval_interval = self.cfg.get("eval_interval", 100)
        self.eval_batches = self.cfg.get("eval_batches", 20)
        self.save_interval = self.cfg.get("save_interval", 100)
        self.checkpoint_dir = self.cfg.get("checkpoint_dir", "checkpoints")

        # 3. Optimizer & Scheduler
        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=(self.cfg.get("beta1", 0.9), self.cfg.get("beta2", 0.95)),
            eps=self.cfg.get("eps", 1e-8),
        )

        min_lr_ratio = self.min_lr / self.learning_rate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr_ratio=min_lr_ratio,
        )

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.tokens_per_step = (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
            * self.sequence_length
        )

    def print_startup_banner(self):
        if self.rank != 0:
            return

        print("=" * 80)
        print("                    TINYGPT-500M TRAINING INITIALIZATION")
        print("=" * 80)
        print(f"Device:                  {self.device} (World Size: {self.world_size})")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            print(f"Primary GPU:             {torch.cuda.get_device_name(self.device)}")
            print(f"Total VRAM:              {props.total_memory / (1024**3):.2f} GiB")
        print(f"Precision Mode:          {self.precision_desc}")
        print(f"Sequence Length:         {self.sequence_length:,} tokens")
        print(f"Micro-Batch Size:        {self.micro_batch_size} sequences per GPU")
        print(f"Grad Accumulation Steps: {self.gradient_accumulation_steps} (Effective Total: {self.gradient_accumulation_steps * self.world_size})")
        print(f"Gradient Checkpointing:  {self.gradient_checkpointing}")
        print(f"Effective Tokens / Step: {self.tokens_per_step:,} tokens")
        print(f"Target Training Steps:   {self.max_steps:,} steps")
        print(
            f"Total Tokens Planned:    {(self.tokens_per_step * self.max_steps) / 1e6:.2f} Million tokens"
        )
        print("=" * 80 + "\n")

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluates model validation loss deterministically across validation shards.
        The exact same sequence batches are evaluated on every evaluation step.
        """
        if self.val_dataset is None:
            return 0.0

        eval_model = self.model.module if hasattr(self.model, "module") else self.model
        eval_model.eval()
        total_loss = 0.0

        for i in range(self.eval_batches):
            x, y = self.val_dataset.get_deterministic_batch(
                batch_idx=i,
                batch_size=self.micro_batch_size,
                device=self.device,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=(self.device.type == "cuda"),
            ):
                out = eval_model(x, targets=y)
                total_loss += out.loss.item()

        eval_model.train()
        return total_loss / self.eval_batches

    def train(self, start_step: int = 0, best_val_loss: float = float("inf")):
        """
        Runs the main training loop.

        Args:
            start_step: Starting optimizer step (e.g. from resumed checkpoint).
            best_val_loss: Restored best validation loss to prevent overwriting best checkpoint.
        """
        self.print_startup_banner()
        self.model.train()

        tokens_trained = start_step * self.tokens_per_step
        start_time = time.time()

        if self.rank == 0:
            print(f"Starting training loop at step {start_step + 1}...\n")

        for step in range(start_step, self.max_steps):
            step_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            step_shards = set()

            # Gradient accumulation loop
            for micro_step in range(self.gradient_accumulation_steps):
                x, y = self.train_dataset.get_batch(
                    self.micro_batch_size, device=self.device
                )
                step_shards.update(self.train_dataset.last_sampled_shards)

                is_accumulating = micro_step < self.gradient_accumulation_steps - 1
                sync_context = (
                    self.model.no_sync()
                    if (self.is_ddp and is_accumulating)
                    else nullcontext()
                )

                with sync_context:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.dtype,
                        enabled=(self.device.type == "cuda"),
                    ):
                        out = self.model(x, targets=y)
                        loss = out.loss / self.gradient_accumulation_steps

                    accum_loss += loss.item()

                    if self.use_scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            # Unscale before clipping
            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Optimizer and scaler step
            if self.use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()

            # Telemetry calculations
            step_time = time.time() - step_start
            tokens_trained += self.tokens_per_step
            tok_per_sec = self.tokens_per_step / max(step_time, 1e-5)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # VRAM tracking
            if self.device.type == "cuda":
                vram_used = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                vram_str = f" | VRAM: {vram_used:.1f} GiB"
            else:
                vram_str = ""

            # Shard info string
            shard_str = f" | Shards: {sorted(list(step_shards))}"

            # Remaining time estimation
            remaining_steps = self.max_steps - (step + 1)
            eta_seconds = remaining_steps * step_time
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_str = f"{eta_hours}h {eta_mins:02d}m"

            # Print step status on primary rank
            if self.rank == 0:
                print(
                    f"Step {step + 1:5d}/{self.max_steps} | "
                    f"Train Loss: {accum_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {tok_per_sec:,.0f} tok/s | "
                    f"Step Time: {step_time:.2f}s"
                    f"{vram_str}"
                    f"{shard_str} | "
                    f"ETA: {eta_str}"
                )

            # Evaluation & Checkpointing on primary rank
            if (step + 1) % self.eval_interval == 0 or (step + 1) == self.max_steps:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20.0))
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                if self.rank == 0:
                    print("-" * 80)
                    print(
                        f"[EVAL] Step {step + 1} | Val Loss: {val_loss:.4f} | "
                        f"Val PPL: {val_ppl:.2f} | Best Val Loss: {best_val_loss:.4f}"
                    )
                    saved_path = self.checkpoint_manager.save(
                        step=step + 1,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler if self.use_scaler else None,
                        val_loss=val_loss,
                        tokens_trained=tokens_trained,
                        is_best=is_best,
                    )
                    print(f"  Checkpoint saved: {saved_path}")
                    print("-" * 80)

        total_time = (time.time() - start_time) / 60
        if self.rank == 0:
            print("\n" + "=" * 80)
            print(f"Training Complete! Total time: {total_time:.2f} minutes.")
            print(f"Total tokens trained: {tokens_trained:,}")
            print(f"Best Validation Loss: {best_val_loss:.4f}")
            print("=" * 80)

        if self.is_ddp and dist.is_initialized():
            dist.destroy_process_group()
