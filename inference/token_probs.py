"""
Step-by-step token probability inspection for TinyGPT-500M.

`TokenProbabilityStepper` runs KV-cached decoding one token at a time and exposes,
for every step, the model's full next-token distribution:
  * raw probability   -> softmax(logits)               (what the model "believes")
  * sample probability -> after repetition penalty, temperature, top-k, top-p
                          (what the sampler actually draws from)

The caller decides which token to commit (sampled one, or any alternative),
which makes it easy to build interactive "press Enter for next token" tools.
"""

from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F

from model.transformer import TinyGPT500M
from model.kv_cache import KVCache
from data.tokenizer import GPT2Tokenizer


@dataclass
class Candidate:
    rank: int            # 1-based rank by raw model probability
    token_id: int
    text: str
    raw_prob: float      # softmax of raw logits (temperature 1, no filtering)
    sample_prob: float   # probability under the actual sampling distribution
    logit: float


@dataclass
class StepInfo:
    step: int
    candidates: list[Candidate]
    sampled: Candidate              # token drawn from the sampling distribution
    entropy_bits: float             # entropy of the raw distribution (uncertainty)
    raw_probs: torch.Tensor = field(repr=False)
    sample_probs: torch.Tensor = field(repr=False)

    def describe(self, token_id: int, tokenizer: GPT2Tokenizer) -> Candidate:
        """Returns a Candidate record for any token id (rank computed on the fly)."""
        for c in self.candidates:
            if c.token_id == token_id:
                return c
        raw_p = float(self.raw_probs[token_id])
        rank = int((self.raw_probs > self.raw_probs[token_id]).sum().item()) + 1
        return Candidate(
            rank=rank,
            token_id=token_id,
            text=tokenizer.decode([token_id]),
            raw_prob=raw_p,
            sample_prob=float(self.sample_probs[token_id]),
            logit=float("nan"),
        )


def sampling_distribution(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_tokens: list[int] | None = None,
) -> torch.Tensor:
    """
    Mirrors `inference.generate.sample_next_token` but returns the full probability
    vector the sampler draws from instead of a single sampled id.
    """
    logits = logits.float().clone()

    if repetition_penalty != 1.0 and generated_tokens:
        idx = torch.tensor(sorted(set(generated_tokens)), device=logits.device)
        vals = logits[idx]
        logits[idx] = torch.where(vals < 0, vals * repetition_penalty, vals / repetition_penalty)

    if temperature <= 0.0:  # greedy -> one-hot
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs

    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k)[0][..., -1, None]
        logits[logits < kth] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits[sorted_idx[remove]] = float("-inf")

    return F.softmax(logits, dim=-1)


class TokenProbabilityStepper:
    def __init__(
        self,
        model: TinyGPT500M,
        tokenizer: GPT2Tokenizer,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        show_top: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.show_top = show_top
        self.device = next(model.parameters()).device

        self.kv_cache: KVCache | None = None
        self.all_tokens: list[int] = []
        self.new_tokens: list[int] = []
        self.chosen: list[Candidate] = []
        self._last_logits: torch.Tensor | None = None
        self._step = 0

    @torch.no_grad()
    def prefill(self, prompt: str) -> None:
        """Processes the whole prompt and prepares the first next-token distribution."""
        self.model.eval()
        prompt_tokens = self.tokenizer.encode(prompt) or [self.tokenizer.bos_token_id]
        self.kv_cache = KVCache(num_layers=self.model.config.num_hidden_layers)
        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        out = self.model(ids, kv_cache=self.kv_cache)
        self._last_logits = out.logits[0, -1, :]
        self.all_tokens = list(prompt_tokens)
        self.new_tokens, self.chosen, self._step = [], [], 0

    def peek(self) -> StepInfo:
        """Computes the distribution over the next token (does not advance)."""
        assert self._last_logits is not None, "Call prefill() first."
        logits = self._last_logits.float()
        raw = F.softmax(logits, dim=-1)
        samp = sampling_distribution(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            generated_tokens=self.all_tokens,
        )

        top_p_vals, top_ids = torch.topk(raw, self.show_top)
        candidates = [
            Candidate(
                rank=i + 1,
                token_id=int(tid),
                text=self.tokenizer.decode([int(tid)]),
                raw_prob=float(p),
                sample_prob=float(samp[tid]),
                logit=float(logits[tid]),
            )
            for i, (p, tid) in enumerate(zip(top_p_vals, top_ids))
        ]

        sampled_id = int(torch.multinomial(samp, 1).item())
        entropy = float(-(raw * torch.log2(raw.clamp_min(1e-12))).sum())

        info = StepInfo(
            step=self._step + 1,
            candidates=candidates,
            sampled=None,  # type: ignore[arg-type]
            entropy_bits=entropy,
            raw_probs=raw.cpu(),
            sample_probs=samp.cpu(),
        )
        info.sampled = info.describe(sampled_id, self.tokenizer)
        return info

    @torch.no_grad()
    def commit(self, info: StepInfo, token_id: int) -> Candidate:
        """Appends `token_id` to the sequence and runs one KV-cached forward step."""
        cand = info.describe(token_id, self.tokenizer)
        self.all_tokens.append(token_id)
        self.new_tokens.append(token_id)
        self.chosen.append(cand)
        self._step += 1

        t = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        out = self.model(t, kv_cache=self.kv_cache)
        self._last_logits = out.logits[0, -1, :]
        return cand

    @property
    def generated_text(self) -> str:
        return self.tokenizer.decode(self.new_tokens)

    def summary(self) -> dict:
        """Aggregate stats over committed tokens (raw model probabilities)."""
        if not self.chosen:
            return {"tokens": 0}
        logps = [math.log(max(c.raw_prob, 1e-12)) for c in self.chosen]
        return {
            "tokens": len(self.chosen),
            "avg_prob": sum(c.raw_prob for c in self.chosen) / len(self.chosen),
            "min_prob": min(c.raw_prob for c in self.chosen),
            "perplexity": math.exp(-sum(logps) / len(logps)),
            "top1_rate": sum(1 for c in self.chosen if c.rank == 1) / len(self.chosen),
        }
