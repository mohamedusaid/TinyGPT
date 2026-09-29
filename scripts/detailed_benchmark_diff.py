"""
Detailed question-by-question comparative benchmark evaluator.
Outputs the exact prompt, choices, ground truth, and predictions for both:
  1. Pre-SFT Base Model (best_tinygpt_500m.pt)
  2. Post-SFT Aligned Model (usaid_ai_500m.pt)
"""

import os
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.config import TinyGPTConfig
from model.transformer import TinyGPT500M
from data.tokenizer import GPT2Tokenizer
from scripts.run_eval import BENCHMARK_CATEGORIES


@torch.no_grad()
def evaluate_detailed(model, tokenizer, questions, device="cpu"):
    device_obj = torch.device(device)
    model.eval()
    results = []

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
            logits = out.logits[0, :-1, :]
            targets = input_ids[0, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(targets)), targets]

            choice_token_count = len(full_ids) - len(prompt_ids)
            if choice_token_count > 0:
                choice_log_prob = target_log_probs[-choice_token_count:].sum().item()
                norm_log_prob = choice_log_prob / choice_token_count
            else:
                norm_log_prob = float("-inf")
            choice_log_probs.append(norm_log_prob)

        pred_idx = int(torch.argmax(torch.tensor(choice_log_probs)).item())
        results.append({
            "prompt": prompt,
            "choices": choices,
            "correct_idx": answer_idx,
            "correct_text": choices[answer_idx],
            "pred_idx": pred_idx,
            "pred_text": choices[pred_idx],
            "is_correct": (pred_idx == answer_idx),
            "log_probs": choice_log_probs,
        })
    return results


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TinyGPTConfig.from_dict(checkpoint.get("config", {}))
    model = TinyGPT500M(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer()

    base_ckpt = "checkpoints/best_tinygpt_500m.pt"
    sft_ckpt = "sft_checkpoints/usaid_ai_500m.pt" if os.path.exists("sft_checkpoints/usaid_ai_500m.pt") else "checkpoints/usaid_ai_500m.pt"

    print("Loading Pre-SFT Base Model...")
    base_model = load_model(base_ckpt, device)

    print("Evaluating Pre-SFT Base Model...")
    all_questions = []
    category_map = []
    for cat_name, q_list in BENCHMARK_CATEGORIES.items():
        for q in q_list:
            all_questions.append(q)
            category_map.append(cat_name)

    base_results = evaluate_detailed(base_model, tokenizer, all_questions, device)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Post-SFT Aligned Model...")
    sft_model = load_model(sft_ckpt, device)

    print("Evaluating Post-SFT Aligned Model...")
    sft_results = evaluate_detailed(sft_model, tokenizer, all_questions, device)
    del sft_model

    print("\n" + "=" * 100)
    print("      USAID AI (500M) — COMPLETE QUESTION-BY-QUESTION BENCHMARK AUDIT")
    print("=" * 100)

    for i, (b_res, s_res, cat) in enumerate(zip(base_results, sft_results, category_map), 1):
        prompt = b_res["prompt"]
        correct = b_res["correct_text"]
        b_pred = b_res["pred_text"]
        s_pred = s_res["pred_text"]
        b_status = "CORRECT [O]" if b_res["is_correct"] else "WRONG [X]"
        s_status = "CORRECT [O]" if s_res["is_correct"] else "WRONG [X]"

        if not b_res["is_correct"] and s_res["is_correct"]:
            change = "IMPROVED (+)"
        elif b_res["is_correct"] and not s_res["is_correct"]:
            change = "REGRESSED (-)"
        elif b_res["is_correct"] and s_res["is_correct"]:
            change = "MAINTAINED (O)"
        else:
            change = "UNRESOLVED (X)"

        print(f"\n[Q{i:02d}] Category: {cat}")
        print(f"  {prompt}")
        print(f"  Choices:      {b_res['choices']}")
        print(f"  Ground Truth: {correct}")
        print(f"  Pre-SFT Base: {b_pred:<25} -> {b_status}")
        print(f"  Post-SFT AI:  {s_pred:<25} -> {s_status}   [{change}]")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
