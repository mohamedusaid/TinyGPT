"""
Zero-Shot Multiple Choice Benchmark Evaluator for TinyGPT-500M.
Computes completion log-likelihoods to evaluate accuracy on multiple-choice reasoning tasks.
"""

import torch
import torch.nn.functional as F
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer


@torch.no_grad()
def evaluate_multiple_choice(
    model: TinyGPT500M,
    tokenizer: GPT2Tokenizer,
    questions: list[dict],
    device: str = "cpu",
) -> dict:
    """
    Evaluates zero-shot accuracy on multiple choice questions.

    Args:
        questions: List of dicts, each with:
            {
                "prompt": "What is the capital of France?",
                "choices": ["Berlin", "Madrid", "Paris", "Rome"],
                "answer_idx": 2
            }
    """
    model.eval()
    device_obj = torch.device(device)
    correct = 0
    total = len(questions)

    for item in questions:
        prompt = item["prompt"]
        choices = item["choices"]
        answer_idx = item["answer_idx"]

        choice_log_probs = []

        for choice in choices:
            full_text = prompt + " " + choice
            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device_obj)
            out = model(input_ids)
            logits = out.logits[0, :-1, :]  # (T-1, vocab)
            targets = input_ids[0, 1:]  # (T-1,)

            # Log-softmax over vocabulary
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(targets)), targets]

            # Sum log-probabilities only for the choice tokens
            choice_token_count = len(full_ids) - len(prompt_ids)
            if choice_token_count > 0:
                choice_log_prob = target_log_probs[-choice_token_count:].sum().item()
                # Length-normalized log-probability
                norm_log_prob = choice_log_prob / choice_token_count
            else:
                norm_log_prob = float("-inf")

            choice_log_probs.append(norm_log_prob)

        predicted_idx = int(torch.argmax(torch.tensor(choice_log_probs)).item())
        if predicted_idx == answer_idx:
            correct += 1

    accuracy = (correct / max(1, total)) * 100.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }
