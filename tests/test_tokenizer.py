"""
Unit tests for GPT-2 Tokenizer and BinaryShardedDataset.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.tokenizer import GPT2Tokenizer
from data.dataset import BinaryShardedDataset


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT2Tokenizer()

    def test_tokenizer_properties(self):
        """Validates standard GPT-2 vocabulary size and special token IDs."""
        self.assertEqual(self.tokenizer.vocab_size, 50257)
        self.assertEqual(self.tokenizer.eos_token_id, 50256)
        self.assertEqual(self.tokenizer.eos_token, "<|endoftext|>")

    def test_tokenizer_round_trip(self):
        """Validates that encode -> decode is lossless."""
        sample_text = "The Transformer is a deep learning architecture introduced in 2017."
        token_ids = self.tokenizer.encode(sample_text)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

        decoded_text = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded_text, sample_text)

    def test_binary_sharded_dataset(self):
        """Creates a temporary binary shard and verifies batch sampling."""
        test_dir = os.path.join(REPO_ROOT, "tests", "scratch_data")
        os.makedirs(test_dir, exist_ok=True)
        shard_path = os.path.join(test_dir, "test_shard.bin")

        # Generate 2048 dummy tokens
        tokens = np.random.randint(0, 50257, size=2048, dtype=np.uint16)
        tokens.tofile(shard_path)

        dataset = BinaryShardedDataset(shard_path, sequence_length=64)
        self.assertEqual(dataset.total_tokens, 2048)

        x, y = dataset.get_batch(batch_size=4)
        self.assertEqual(x.shape, (4, 64))
        self.assertEqual(y.shape, (4, 64))

        # Assert y is x shifted by 1 token
        self.assertTrue((x[:, 1:] == y[:, :-1]).all())

        # Cleanup
        try:
            os.remove(shard_path)
            os.rmdir(test_dir)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
