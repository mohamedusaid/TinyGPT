"""
KV-Cache Numerical Equivalence Test for TinyGPT-500M.
Mathematically verifies that step-by-step autoregressive generation using the KV cache
yields identical output logits compared to full-sequence forward pass re-computation.
"""

import os
import sys
import unittest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from model.kv_cache import KVCache


class TestKVCache(unittest.TestCase):
    def test_kv_cache_equivalence(self):
        """
        Generates 5 tokens step-by-step using KVCache and compares against
        full-sequence forward pass.
        """
        torch.manual_seed(42)
        test_config = TinyGPTConfig(num_hidden_layers=3)
        model = TinyGPT500M(test_config)
        model.eval()

        batch_size = 1
        prompt_len = 8
        generate_steps = 4

        # Initial prompt
        prompt = torch.randint(0, test_config.vocab_size, (batch_size, prompt_len))

        # --- Method 1: Full sequence recomputation at every step ---
        full_tokens = prompt.clone()
        recompute_logits = []

        with torch.no_grad():
            for _ in range(generate_steps):
                out = model(full_tokens)
                next_logit = out.logits[:, -1, :]
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True)
                full_tokens = torch.cat([full_tokens, next_token], dim=1)
                recompute_logits.append(next_logit)

        # --- Method 2: Step-by-step generation with KVCache ---
        kv_cache = KVCache(num_layers=test_config.num_hidden_layers)
        cache_tokens = prompt.clone()
        cache_logits = []

        with torch.no_grad():
            # Prefill step
            out = model(prompt, kv_cache=kv_cache)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            cache_logits.append(out.logits[:, -1, :])
            cache_tokens = torch.cat([cache_tokens, next_token], dim=1)

            # Decode steps: feed only 1 token at a time
            for _ in range(generate_steps - 1):
                out = model(next_token, kv_cache=kv_cache)
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                cache_logits.append(out.logits[:, -1, :])
                cache_tokens = torch.cat([cache_tokens, next_token], dim=1)

        # Assert token sequences match identically
        self.assertTrue(
            torch.equal(full_tokens, cache_tokens),
            "Generated token sequence with KVCache must match full recomputation",
        )

        # Assert logits match closely (within floating point precision)
        for i in range(generate_steps):
            diff = torch.max(torch.abs(recompute_logits[i] - cache_logits[i])).item()
            self.assertLess(
                diff,
                1e-4,
                f"Logit difference at step {i} is {diff} (exceeds threshold 1e-4)",
            )


if __name__ == "__main__":
    unittest.main()
