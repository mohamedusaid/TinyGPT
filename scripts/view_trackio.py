"""
Interactive Training Telemetry Dashboard for Usaid AI (500M)
Powered by Trackio (gradio-app/trackio) by Hugging Face & Gradio.
Created by Mohamed Usaid.

Usage:
    python scripts/view_trackio.py            # Syncs logs and opens local interactive dashboard
    python scripts/view_trackio.py --share    # Generates a shareable public Gradio URL
    python scripts/view_trackio.py --port 7860
"""

import argparse
import os
import re
import sys


def sync_logs_to_trackio(log_path: str = "training_logs.txt", project_name: str = "UsaidAI-500M"):
    try:
        import trackio
    except ImportError:
        print("\n[!] Trackio is not installed. Please install it via pip:")
        print("    pip install trackio\n")
        sys.exit(1)

    if not os.path.exists(log_path):
        print(f"\n[!] Log file not found at: {log_path}")
        sys.exit(1)

    print("=" * 80)
    print("        USAID AI (500M) — TRACKIO EXPERIMENT SYNCHRONIZER")
    print("                     Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Source Log File:    {log_path}")
    print(f"Trackio Project:    {project_name}")
    print("-" * 80)

    # 1. Parse and log Pretraining Run
    print("Ingesting Pretraining Telemetry (131M Tokens across 2,000 Steps)...")
    run1 = trackio.init(
        project=project_name,
        name="pretraining-131M-tokens",
        config={
            "model": "Usaid AI (500M)",
            "architecture": "Dense Causal LM (GQA 16/4, RoPE, SwiGLU, RMSNorm)",
            "parameters": 500136960,
            "hidden_dim": 1024,
            "layers": 30,
            "heads_q": 16,
            "heads_kv": 4,
            "intermediate_dim": 3456,
            "context_length": 1024,
            "total_tokens": 131072000,
            "total_steps": 2000,
            "hardware": "Dual Cloud GPUs (2x Tesla T4, 32GB total)",
            "optimizer": "CUDA Fused AdamW",
            "peak_lr": 3.0e-4,
            "batch_geometry": "65,536 tokens / step",
        },
    )

    pretrain_count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_train = re.search(
                r"Step\s+(\d+)/2000\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Grad Norm:\s+([0-9.]+)\s+\|\s+LR:\s+([0-9.e+-]+)\s+\|\s+Speed:\s+([0-9,]+)\s+tok/s",
                line,
            )
            if m_train:
                step = int(m_train.group(1))
                trackio.log(
                    {
                        "train_loss": float(m_train.group(2)),
                        "grad_norm": float(m_train.group(3)),
                        "learning_rate": float(m_train.group(4)),
                        "throughput_tok_sec": float(m_train.group(5).replace(",", "")),
                    },
                    step=step,
                )
                pretrain_count += 1

            m_eval = re.search(
                r"\[EVAL\]\s+Step\s+(\d+)\s+\|\s+Val Loss:\s+([0-9.]+)\s+\|\s+Val PPL:\s+([0-9.]+)",
                line,
            )
            if m_eval:
                step = int(m_eval.group(1))
                trackio.log(
                    {
                        "val_loss": float(m_eval.group(2)),
                        "val_perplexity": float(m_eval.group(3)),
                    },
                    step=step,
                )

    trackio.finish()
    print(f"Successfully synced {pretrain_count} pretraining steps + evaluation checkpoints.")

    # 2. Parse and log SFT Run
    print("Ingesting SFT Alignment Telemetry (4,731 Pairs across 441 Steps)...")
    run2 = trackio.init(
        project=project_name,
        name="sft-alignment-4731-pairs",
        config={
            "model": "Usaid AI (500M Aligned)",
            "base_checkpoint": "best_tinygpt_500m.pt",
            "supervision_method": "Prompt Loss Masking (-100)",
            "dataset_pairs": 4731,
            "epochs": 3,
            "total_steps": 441,
            "peak_lr": 2.0e-5,
            "best_sft_loss": 1.3938,
            "final_sft_loss": 1.4715,
        },
    )

    sft_count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        # Find start of final SFT training run
        start_idx = 0
        for i, line in enumerate(lines):
            if "USAID AI (500M) SUPERVISED FINE-TUNING" in line:
                start_idx = i

        seen_steps = set()
        for line in lines[start_idx:]:
            m_sft = re.search(
                r"Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)/441\s+\|\s+SFT Loss:\s+([0-9.]+)\s+\|\s+Grad Norm:\s+([0-9.]+)\s+\|\s+LR:\s+([0-9.e+-]+)",
                line,
            )
            if m_sft:
                epoch = int(m_sft.group(1))
                step = int(m_sft.group(2))
                if step not in seen_steps:
                    seen_steps.add(step)
                    trackio.log(
                        {
                            "sft_loss": float(m_sft.group(3)),
                            "grad_norm": float(m_sft.group(4)),
                            "learning_rate": float(m_sft.group(5)),
                            "epoch": epoch,
                        },
                        step=step,
                    )
                    sft_count += 1

    trackio.finish()
    print(f"Successfully synced {sft_count} SFT alignment steps.")
    print("-" * 80)
    print("Synchronization Complete!")
    print("=" * 80)


def launch_dashboard(project_name: str = "UsaidAI-500M", port: int = 7860, share: bool = False):
    import trackio

    print(f"\nLaunching Trackio Dashboard for project '{project_name}'...")
    print(f"Terminal Command: trackio show --project \"{project_name}\"\n")
    trackio.show(
        project=project_name,
        server_port=port,
        share=share,
        open_browser=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View and sync Usaid AI (500M) training telemetry with Trackio"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="training_logs.txt",
        help="Path to training logs text file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="UsaidAI-500M",
        help="Trackio project name",
    )
    parser.add_argument(
        "--sync_only",
        action="store_true",
        help="Only parse and sync logs without launching dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve Trackio dashboard (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a shareable public Gradio link",
    )
    args = parser.parse_args()

    # Sync logs
    sync_logs_to_trackio(log_path=args.log_file, project_name=args.project)

    # Launch dashboard if requested
    if not args.sync_only:
        launch_dashboard(project_name=args.project, port=args.port, share=args.share)
