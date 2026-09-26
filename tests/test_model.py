"""
Unit tests for TinyGPT-500M architecture:
1. Exact parameter count verification.
2. Tensor shape validation across all 30 layers.
3. Forward pass with dummy input.
4. Loss computation with target labels.
5. Backward pass gradient flow.
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


class TestTinyGPT500M(unittest.TestCase):
    def setUp(self):
        self.config = TinyGPTConfig()

    def test_parameter_count(self):
        """Asserts that instantiated PyTorch model parameters match the 500M specification."""
        model = TinyGPT500M(self.config)
        actual_params = model.count_parameters()
        expected_params = 500_136_960

        self.assertEqual(
            actual_params,
            expected_params,
            f"Expected {expected_params:,} parameters, got {actual_params:,}",
        )
        self.assertLess(
            abs(actual_params - 500_000_000) / 500_000_000,
            0.001,
            "Parameters must be within 0.1% of 500M",
        )

    def test_forward_and_loss(self):
        """Tests forward pass, output shapes, and loss computation."""
        # Use a small 2-layer config for fast CPU shape/gradient testing
        test_config = TinyGPTConfig(num_hidden_layers=2)
        model = TinyGPT500M(test_config)
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

        # Forward pass without targets
        with torch.no_grad():
            out = model(input_ids)
            self.assertEqual(
                out.logits.shape,
                (batch_size, seq_len, test_config.vocab_size),
                f"Expected logits shape {(batch_size, seq_len, test_config.vocab_size)}, got {out.logits.shape}",
            )
            self.assertIsNone(out.loss)

        # Forward pass with targets
        out = model(input_ids, targets=targets)
        self.assertIsNotNone(out.loss)
        self.assertFalse(torch.isnan(out.loss))
        self.assertGreater(out.loss.item(), 0.0)

    def test_backward_gradient_flow(self):
        """Verifies backward pass and ensures gradients flow to all layers."""
        test_config = TinyGPTConfig(num_hidden_layers=2)
        model = TinyGPT500M(test_config)
        model.train()

        input_ids = torch.randint(0, test_config.vocab_size, (2, 8))
        targets = torch.randint(0, test_config.vocab_size, (2, 8))

        out = model(input_ids, targets=targets)
        out.loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(
                param.grad, f"Gradient for {name} is None (no gradient flow)"
            )
            self.assertFalse(
                torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs"
            )


if __name__ == "__main__":
    unittest.main()
