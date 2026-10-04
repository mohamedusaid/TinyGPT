"""
Private Archive Backup Utility for TinyGPT-500M Training Checkpoints & Shards.
Created by Mohamed Usaid.
"""

import argparse
import os
import sys
import time
from huggingface_hub import HfApi, login


ARCHIVE_REPO_ID = "Usaidddddddddddddd/TinyGPT-500M-Archive"

FILES_TO_BACKUP = [
    {
        "key": "readme",
        "description": "Archive Documentation & Metadata",
        "local_path": None,  # Generated dynamically
        "repo_path": "README.md",
    },
    {
        "key": "shards",
        "description": "Tokenized Training & Validation Shards",
        "local_path": r"C:\Users\Unique\Downloads\data_shards_download.zip",
        "repo_path": "data_shards_download.zip",
    },
    {
        "key": "best",
        "description": "Base Pretrained Model Checkpoint (Step 1300, Val Loss 2.89)",
        "local_path": r"D:\python_project\tiny-gpt-500m\checkpoints\best_tinygpt_500m.pt",
        "repo_path": "best_tinygpt_500m.pt",
    },
    {
        "key": "step2000",
        "description": "Full Optimizer State Checkpoint at Step 2000 (Model + AdamW + Scheduler + Scaler)",
        "local_path": r"C:\Users\Unique\Downloads\checkpoint_step_002000.zip",
        "repo_path": "checkpoint_step_002000.pt",
    },
]


README_CONTENT = """---
language:
- en
license: mit
tags:
- tinygpt
- pytorch
- checkpoint-archive
- pretraining
---

# TinyGPT-500M Private Training Archive

This repository is a private cloud backup containing raw PyTorch pretraining checkpoints, optimizer states, and tokenized datasets for **TinyGPT-500M / Usaid AI (500M)** developed by **Mohamed Usaid**.

---

## Repository Contents

| File | Size | Description |
|---|---|---|
| `best_tinygpt_500m.pt` | ~1.86 GB | Best pretraining checkpoint (Step 1300, validation loss: 2.89) prior to Supervised Fine-Tuning. |
| `checkpoint_step_002000.pt` | ~5.59 GB | Full pretraining state at step 2000 containing: model weights, AdamW optimizer states (momentum & variance buffers), cosine learning rate scheduler state, GradScaler state, and RNG seed. |
| `data_shards_download.zip` | ~165 MB | 27 tokenized training shards + validation shard (~2.7B tokens) formatted for high-throughput memory-mapped loading. |

---

## How to Resume Pretraining

```python
import torch

checkpoint = torch.load("checkpoint_step_002000.pt", map_location="cpu", weights_only=False)

print("Resuming from Step:", checkpoint["step"])
print("Tokens trained:", checkpoint["tokens_trained"])
print("Val Loss at save:", checkpoint["val_loss"])

# Load into model & optimizer:
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
scaler.load_state_dict(checkpoint["scaler_state_dict"])
```
"""


def format_size(bytes_num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"


def run_upload(target_keys: list[str], token: str | None = None):
    if token:
        login(token=token)

    api = HfApi()

    print("=" * 80)
    print("      TINYGPT-500M PRIVATE CLOUD BACKUP (HUGGING FACE HUB)")
    print("                 Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Repository:  https://huggingface.co/{ARCHIVE_REPO_ID}")
    print(f"Repository Privacy: PRIVATE [CONFIDENTIAL]")
    print("-" * 80)

    # Ensure repository exists and is private
    print("Verifying private repository on Hugging Face...")
    api.create_repo(repo_id=ARCHIVE_REPO_ID, repo_type="model", private=True, exist_ok=True)

    # Filter target items
    items_to_upload = [item for item in FILES_TO_BACKUP if item["key"] in target_keys]

    # Pre-flight check
    print("\nPre-flight Check:")
    total_upload_bytes = 0
    for item in items_to_upload:
        if item["key"] == "readme":
            size = len(README_CONTENT.encode("utf-8"))
            print(f"  [OK] {item['repo_path']} ({format_size(size)}) - {item['description']}")
            total_upload_bytes += size
        else:
            p = item["local_path"]
            if not os.path.exists(p):
                print(f"  [MISSING] {p}")
                sys.exit(1)
            size = os.path.getsize(p)
            total_upload_bytes += size
            print(f"  [OK] {item['repo_path']} ({format_size(size)}) <- {p}")

    print(f"\nTotal Upload Size: {format_size(total_upload_bytes)}")
    print("-" * 80)

    start_time = time.time()
    for idx, item in enumerate(items_to_upload, 1):
        print(f"\n[{idx}/{len(items_to_upload)}] Uploading {item['repo_path']}...")
        if item["key"] == "readme":
            api.upload_file(
                path_or_fileobj=README_CONTENT.encode("utf-8"),
                path_in_repo=item["repo_path"],
                repo_id=ARCHIVE_REPO_ID,
                repo_type="model",
                commit_message="Add archive documentation README",
            )
        else:
            api.upload_file(
                path_or_fileobj=item["local_path"],
                path_in_repo=item["repo_path"],
                repo_id=ARCHIVE_REPO_ID,
                repo_type="model",
                commit_message=f"Backup {item['repo_path']}",
            )
        print(f"  --> Successfully committed {item['repo_path']}!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("                    BACKUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Repository URL: https://huggingface.co/{ARCHIVE_REPO_ID}")
    print(f"Total time:     {elapsed / 60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload TinyGPT-500M archive to private Hugging Face repo")
    parser.add_argument(
        "--file",
        choices=["readme", "shards", "best", "step2000", "all"],
        default="all",
        help="Specify a single file to upload or 'all' to upload everything (default: all)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (optional if already logged in)",
    )
    args = parser.parse_args()

    targets = (
        ["readme", "shards", "best", "step2000"]
        if args.file == "all"
        else [args.file]
    )

    run_upload(targets, args.token)
