"""
Supervised Fine-Tuning (SFT) Dataset with Prompt Loss Masking for TinyGPT-500M.
Masks prompt tokens with -100 so that backpropagation loss is computed
exclusively on the assistant response tokens.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from data.tokenizer import GPT2Tokenizer


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: GPT2Tokenizer,
        sequence_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.examples = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SFT dataset file not found: {data_path}")

        # Load JSON or JSONL
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    self.examples = content
                elif isinstance(content, dict) and "examples" in content:
                    self.examples = content["examples"]

        print(f"Loaded {len(self.examples):,} instruction examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]

        # Extract instruction and response
        if "messages" in ex:
            user_msg = ""
            assistant_msg = ""
            for m in ex["messages"]:
                if m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]
        elif "instruction" in ex:
            user_msg = ex["instruction"]
            assistant_msg = ex.get("response", ex.get("output", ""))
        elif "question" in ex:
            user_msg = ex["question"]
            assistant_msg = ex.get("answer", "")
        else:
            user_msg = str(ex)
            assistant_msg = ""

        # Format prompt and completion
        prompt_text = f"User: {user_msg}\n\nAssistant: "
        response_text = f"{assistant_msg}<|endoftext|>"

        prompt_tokens = self.tokenizer.encode(prompt_text)
        response_tokens = self.tokenizer.encode(response_text)

        input_ids = prompt_tokens + response_tokens
        # Mask out prompt tokens with -100 (ignored by PyTorch cross_entropy)
        target_ids = ([-100] * len(prompt_tokens)) + response_tokens

        # Truncate if exceeds max length
        if len(input_ids) > self.sequence_length:
            input_ids = input_ids[: self.sequence_length]
            target_ids = target_ids[: self.sequence_length]

        # Pad to sequence length
        pad_len = self.sequence_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            target_ids = target_ids + [-100] * pad_len

        # Shift targets by 1 for causal next-token prediction
        # Input: tokens[0 : seq_len-1], Target: tokens[1 : seq_len]
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(target_ids[1:], dtype=torch.long)

        return x, y
