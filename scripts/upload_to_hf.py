"""
One-click upload utility to push Usaid AI (500M) to Hugging Face Hub.
Created by Mohamed Usaid.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, login


def upload_model(
    repo_id: str,
    local_folder: str = "exported_usaid_ai_aligned",
    token: str | None = None,
):
    if token:
        login(token=token)

    api = HfApi()
    print("=" * 80)
    print("           USAID AI (500M) — HUGGING FACE HUB PUBLISHER")
    print("                       Created by Mohamed Usaid")
    print("=" * 80)
    print(f"Target Repository:  https://huggingface.co/{repo_id}")
    print(f"Local Directory:    {local_folder}")
    print("-" * 80)

    if not os.path.exists(local_folder):
        print(f"Error: Directory '{local_folder}' does not exist.")
        sys.exit(1)

    print("Verifying / Creating repository on Hugging Face Hub...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading artifacts ({os.listdir(local_folder)})...")
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",
    )

    print("\n" + "=" * 80)
    print("                        PUBLISHING SUCCESSFUL!")
    print("=" * 80)
    print(f"Your model is officially live on Hugging Face Hub:")
    print(f"  --> https://huggingface.co/{repo_id}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Usaid AI to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="mohamedusaid/UsaidAI-500M",
        help="Hugging Face repository ID (e.g. mohamedusaid/UsaidAI-500M)",
    )
    parser.add_argument(
        "--local_folder",
        type=str,
        default="exported_usaid_ai_aligned",
        help="Path to folder containing safetensors, config, tokenizer, and README",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Write Token (or run 'huggingface-cli login' first)",
    )
    args = parser.parse_args()

    upload_model(args.repo_id, args.local_folder, args.token)
