"""
Fast KV-Cached Autoregressive Generation Engine for TinyGPT-500M.
Supports temperature, top-k, top-p (nucleus sampling), repetition penalty, and token streaming.
"""

from typing import Callable
import torch
import torch.nn.functional as F

from model.transformer import TinyGPT500M
from model.kv_cache import KVCache
from data.tokenizer import GPT2Tokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_tokens: list[int] | None = None,
) -> int:
    """
    Applies temperature, repetition penalty, top-k, and nucleus (top-p) sampling to logits.

    Args:
        logits: 1D Tensor of unnormalized log-probabilities over vocabulary of shape (vocab_size,).
        temperature: Sampling temperature (higher = more random, lower = more deterministic).
        top_k: Filters out tokens outside the top-k highest logits.
        top_p: Nucleus sampling cumulative probability threshold.
        repetition_penalty: Penalizes logits of previously generated tokens.
        generated_tokens: List of previously generated token IDs.

    Returns:
        Next sampled token ID as an integer.
    """
    # 1. Repetition Penalty
    if repetition_penalty != 1.0 and generated_tokens:
        for token_id in set(generated_tokens):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

    # 2. Greedy sampling if temperature <= 0
    if temperature <= 0.0:
        return torch.argmax(logits).item()

    # 3. Apply Temperature
    logits = logits / max(temperature, 1e-5)

    # 4. Top-K Filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # 5. Top-P (Nucleus) Filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 6. Multinomial categorical sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


@torch.no_grad()
def generate(
    model: TinyGPT500M,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream_callback: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """
    Generates text continuation from a prompt using KV-cached autoregressive decoding.

    Args:
        model: TinyGPT500M instance.
        tokenizer: GPT2Tokenizer instance.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        top_p: Top-p nucleus filtering threshold.
        repetition_penalty: Repetition penalty factor.
        stream_callback: Optional callback function called with each newly decoded token chunk.

    Returns:
        Tuple of (full_generated_text, stop_reason).
    """
    model.eval()
    device = next(model.parameters()).device

    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.bos_token_id]

    # Initialize KV Cache
    kv_cache = KVCache(num_layers=model.config.num_hidden_layers)

    # Prefill step: process entire prompt
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    out = model(input_ids, kv_cache=kv_cache)

    last_logits = out.logits[0, -1, :]
    generated_tokens = list(prompt_tokens)
    new_token_ids = []

    # Sample first new token
    next_token = sample_next_token(
        last_logits.clone(),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generated_tokens=generated_tokens,
    )

    stop_reason = "max_tokens"

    # Autoregressive decoding loop (1 token per step with KV cache)
    for _ in range(max_new_tokens):
        if next_token == tokenizer.eos_token_id:
            stop_reason = "eos_token"
            break

        generated_tokens.append(next_token)
        new_token_ids.append(next_token)

        if stream_callback:
            token_text = tokenizer.decode([next_token])
            stream_callback(token_text)

        # Feed only the single next token to model
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        out = model(token_tensor, kv_cache=kv_cache)
        last_logits = out.logits[0, -1, :]

        next_token = sample_next_token(
            last_logits.clone(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated_tokens,
        )

    full_text = tokenizer.decode(generated_tokens)
    return full_text, stop_reason
