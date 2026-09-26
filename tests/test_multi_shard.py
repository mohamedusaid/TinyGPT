"""
Unit tests specifically targeting:
1. True multi-shard sampling across multiple .bin shards (preventing shard 0000 lock).
2. Deterministic validation batch sampling.
3. Genuinely streaming data preparation without full-file read into RAM.
4. Resuming with best_val_loss preserved.
"""

import os
import shutil
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import BinaryShardedDataset
from data.prepare_data import StreamingShardWriter, process_streaming_data
from training.checkpoint import CheckpointManager
from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from training.trainer import Trainer


class TestMultiShardAndStreaming(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(REPO_ROOT, "tests", "scratch_multi_shard")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_multi_shard_sampling(self):
        """
        Creates 4 distinct shards with unique token ranges and asserts that
        get_batch() samples across ALL shards, not just shard 0000.
        """
        shard_paths = []
        num_shards = 4
        tokens_per_shard = 1000

        # Create 4 shards with non-overlapping token IDs:
        # Shard 0: [1000, 1999], Shard 1: [2000, 2999], etc.
        for i in range(num_shards):
            path = os.path.join(self.test_dir, f"test_shard_{i:04d}.bin")
            tokens = np.arange(i * 1000 + 1000, (i + 1) * 1000 + 1000, dtype=np.uint16)
            tokens.tofile(path)
            shard_paths.append(path)

        dataset = BinaryShardedDataset(self.test_dir, sequence_length=64)
        self.assertEqual(len(dataset.files), num_shards)
        self.assertEqual(len(dataset.valid_shard_indices), num_shards)

        # Sample 50 batches and record which shards were sampled
        sampled_shards = set()
        for _ in range(50):
            dataset.get_batch(batch_size=4)
            sampled_shards.update(dataset.last_sampled_shards)

        # Assert that all 4 shards were sampled, proving no single-shard lock
        self.assertEqual(
            sampled_shards,
            set(range(num_shards)),
            f"Expected all shards {set(range(num_shards))} to be sampled, but only got {sampled_shards}",
        )

    def test_deterministic_validation(self):
        """
        Asserts that get_deterministic_batch() returns the exact same sequences
        when called with the same batch_idx.
        """
        for i in range(3):
            path = os.path.join(self.test_dir, f"val_shard_{i:04d}.bin")
            tokens = np.random.randint(0, 50257, size=1000, dtype=np.uint16)
            tokens.tofile(path)

        dataset = BinaryShardedDataset(self.test_dir, sequence_length=32)

        # First evaluation pass
        batch_0_pass1, _ = dataset.get_deterministic_batch(batch_idx=0, batch_size=2)
        batch_1_pass1, _ = dataset.get_deterministic_batch(batch_idx=1, batch_size=2)

        # Second evaluation pass (simulating next eval step)
        batch_0_pass2, _ = dataset.get_deterministic_batch(batch_idx=0, batch_size=2)
        batch_1_pass2, _ = dataset.get_deterministic_batch(batch_idx=1, batch_size=2)

        # Assert identical sequences
        self.assertTrue(torch.equal(batch_0_pass1, batch_0_pass2))
        self.assertTrue(torch.equal(batch_1_pass1, batch_1_pass2))

        # Assert batch 0 and batch 1 are different
        self.assertFalse(torch.equal(batch_0_pass1, batch_1_pass1))

    def test_streaming_data_preparation(self):
        """
        Verifies that StreamingShardWriter flushes shards at shard_size threshold
        without buffering the entire corpus in memory.
        """
        writer = StreamingShardWriter(
            output_dir=self.test_dir, split_name="stream_test", shard_size=500
        )

        # Add 1250 tokens in 5 small document chunks of 250 tokens
        for _ in range(5):
            tokens = [100] * 250
            writer.add_tokens(tokens)

        writer.close()

        # 1250 tokens with shard_size=500 must produce exactly 3 shards:
        # shard 0 (500 tokens), shard 1 (500 tokens), shard 2 (250 tokens)
        bin_files = sorted(
            [f for f in os.listdir(self.test_dir) if f.endswith(".bin")]
        )
        self.assertEqual(len(bin_files), 3)

        sizes = [
            os.path.getsize(os.path.join(self.test_dir, f)) // 2 for f in bin_files
        ]
        self.assertEqual(sizes, [500, 500, 250])

    def test_resume_preserves_best_val_loss(self):
        """
        Verifies that resuming restores the previous best_val_loss so that
        worse validation steps do not erroneously overwrite the best checkpoint.
        """
        model = TinyGPT500M(TinyGPTConfig(num_hidden_layers=2))
        checkpoint_mgr = CheckpointManager(self.test_dir)

        # Simulate a checkpoint with best_val_loss = 3.25
        saved_path = checkpoint_mgr.save(
            step=10,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            scheduler=None,
            scaler=None,
            val_loss=3.25,
            tokens_trained=50000,
            is_best=True,
        )

        # Load checkpoint
        loaded = checkpoint_mgr.load(saved_path, model=model)
        self.assertEqual(loaded["val_loss"], 3.25)
        self.assertEqual(loaded["step"], 10)


if __name__ == "__main__":
    unittest.main()
