"""
Key-Value (KV) Cache for TinyGPT-500M.
Caches past Key and Value activations during autoregressive generation to avoid O(T^2) recomputation.
"""

import torch


class KVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: list[torch.Tensor | None] = [None] * num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * num_layers
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache for a given layer.

        Args:
            layer_idx: Index of the current transformer layer (0 to num_layers-1).
            key_states: New key states of shape (batch, num_kv_heads, new_tokens, head_dim).
            value_states: New value states of shape (batch, num_kv_heads, new_tokens, head_dim).

        Returns:
            Tuple of concatenated full (keys, values) up to the current sequence length.
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key_states
            self.v_cache[layer_idx] = value_states
        else:
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], key_states], dim=2
            )
            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], value_states], dim=2
            )

        if layer_idx == 0:
            self._seq_len = self.k_cache[0].shape[2]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        """Clears all cached activations."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self._seq_len = 0
