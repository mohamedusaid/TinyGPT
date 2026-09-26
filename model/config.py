"""
Configuration class for TinyGPT-500M.
Defines model hyperparameters and calculates exact parameter counts.
"""

from dataclasses import dataclass, asdict
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class TinyGPTConfig:
    vocab_size: int = 50257
    context_size: int = 1024
    max_position_embeddings: int = 2048
    hidden_size: int = 1024
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 3456
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False
    target_parameters: int = 500_000_000

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def calculate_parameter_breakdown(self) -> dict:
        """
        Computes exact integer parameter counts for each component.
        """
        # Embeddings
        token_embed = self.vocab_size * self.hidden_size
        lm_head = 0 if self.tie_word_embeddings else (self.vocab_size * self.hidden_size)
        rope = 0  # Fixed rotary formula has no learnable parameters

        # Per-layer Attention: W_q, W_k, W_v, W_o
        w_q = self.hidden_size * (self.num_attention_heads * self.head_dim)
        w_k = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        w_v = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        w_o = self.hidden_size * self.hidden_size
        attn_per_layer = w_q + w_k + w_v + w_o
        attn_total = attn_per_layer * self.num_hidden_layers

        # Per-layer SwiGLU MLP: Gate, Up, Down
        w_gate = self.hidden_size * self.intermediate_size
        w_up = self.hidden_size * self.intermediate_size
        w_down = self.intermediate_size * self.hidden_size
        mlp_per_layer = w_gate + w_up + w_down
        mlp_total = mlp_per_layer * self.num_hidden_layers

        # RMSNorms: 2 per layer (input norm, post-attention norm) + 1 final norm
        norm_per_layer = 2 * self.hidden_size
        norm_final = self.hidden_size
        norm_total = (norm_per_layer * self.num_hidden_layers) + norm_final

        # Total model parameters
        total = token_embed + lm_head + attn_total + mlp_total + norm_total
        diff = total - self.target_parameters

        return {
            "target": self.target_parameters,
            "actual_total": total,
            "difference": diff,
            "diff_percent": (diff / self.target_parameters) * 100.0,
            "token_embed": token_embed,
            "lm_head": lm_head,
            "rope": rope,
            "attn_per_layer": {
                "w_q": w_q,
                "w_k": w_k,
                "w_v": w_v,
                "w_o": w_o,
                "total": attn_per_layer,
            },
            "attn_total": attn_total,
            "mlp_per_layer": {
                "w_gate": w_gate,
                "w_up": w_up,
                "w_down": w_down,
                "total": mlp_per_layer,
            },
            "mlp_total": mlp_total,
            "norm_total": norm_total,
            "num_norms": (2 * self.num_hidden_layers) + 1,
        }

    def print_audit(self):
        """Prints a human-readable table of parameter allocation."""
        b = self.calculate_parameter_breakdown()
        print("=" * 80)
        print("                 TINYGPT-500M EXACT PARAMETER AUDIT")
        print("=" * 80)
        print(f"Target Parameters:             {b['target']:,}")
        print(f"Actual Total Parameters:       {b['actual_total']:,}")
        print(f"Parameter Difference:          {b['difference']:+,} ({b['diff_percent']:+.3f}%)")
        print("-" * 80)
        print("COMPONENT BREAKDOWN:")
        print(f"  Token Embedding (V={self.vocab_size}, d={self.hidden_size}):       {b['token_embed']:,} ({b['token_embed']/b['actual_total']*100:.2f}%)")
        if self.tie_word_embeddings:
            print(f"  LM Head:                                0 (Shared with Token Embedding)")
        else:
            print(f"  LM Head (Untied, d={self.hidden_size}, V={self.vocab_size}):       {b['lm_head']:,} ({b['lm_head']/b['actual_total']*100:.2f}%)")
        print(f"  RoPE Positional Encoding:               0 (Formula-based, 0 params)")
        print(f"  Attention Layers ({self.num_hidden_layers} layers):            {b['attn_total']:,} ({b['attn_total']/b['actual_total']*100:.2f}%)")
        print(f"    - W_q Proj ({self.hidden_size} -> {self.num_attention_heads * self.head_dim}):               {b['attn_per_layer']['w_q'] * self.num_hidden_layers:,}")
        print(f"    - W_k Proj ({self.hidden_size} -> {self.num_key_value_heads * self.head_dim}):                 {b['attn_per_layer']['w_k'] * self.num_hidden_layers:,}")
        print(f"    - W_v Proj ({self.hidden_size} -> {self.num_key_value_heads * self.head_dim}):                 {b['attn_per_layer']['w_v'] * self.num_hidden_layers:,}")
        print(f"    - W_o Proj ({self.hidden_size} -> {self.hidden_size}):               {b['attn_per_layer']['w_o'] * self.num_hidden_layers:,}")
        print(f"  SwiGLU MLP Layers ({self.num_hidden_layers} layers):          {b['mlp_total']:,} ({b['mlp_total']/b['actual_total']*100:.2f}%)")
        print(f"    - W_gate Proj ({self.hidden_size} -> {self.intermediate_size}):           {b['mlp_per_layer']['w_gate'] * self.num_hidden_layers:,}")
        print(f"    - W_up Proj   ({self.hidden_size} -> {self.intermediate_size}):           {b['mlp_per_layer']['w_up'] * self.num_hidden_layers:,}")
        print(f"    - W_down Proj ({self.intermediate_size} -> {self.hidden_size}):           {b['mlp_per_layer']['w_down'] * self.num_hidden_layers:,}")
        print(f"  RMSNorm Layers ({b['num_norms']} norms):                   {b['norm_total']:,} ({b['norm_total']/b['actual_total']*100:.2f}%)")
        print("=" * 80)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TinyGPTConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "TinyGPTConfig":
        if os.path.isdir(pretrained_path):
            config_path = os.path.join(pretrained_path, "config.json")
        else:
            config_path = pretrained_path
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
