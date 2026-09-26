"""
Standard GPT-2 Tokenizer Wrapper for TinyGPT-500M.
Uses tiktoken with 50,257 vocabulary tokens.
Clean base model pretraining without ChatML special token pollution.
"""

import tiktoken


class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.eos_token_id = 50256
        self.bos_token_id = 50256
        self.pad_token_id = 50256
        self.eos_token = "<|endoftext|>"

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:
        """Encodes text string into a list of GPT-2 token IDs."""
        if allowed_special is None:
            allowed_special = {self.eos_token}
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a list of token IDs back into a UTF-8 text string."""
        return self.enc.decode(token_ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of strings into a list of token ID lists."""
        return [self.encode(t) for t in texts]

    def batch_decode(self, token_batches: list[list[int]]) -> list[str]:
        """Decodes a batch of token ID lists into strings."""
        return [self.decode(ids) for ids in token_batches]
