"""
Interactive Token-Probability Explorer for Usaid AI (TinyGPT-500M).

For every generated token it shows the TOP-5 candidate next tokens with their
probabilities, then waits for you to press Enter before moving to the next token.

Controls at each step:
    Enter   -> accept the sampled token and move on
    1..N    -> force candidate #N instead (explore "what if" branches)
    a       -> auto-run the rest of the answer (still prints probabilities)
    q       -> stop this answer

Examples:
    python scripts/run_token_probs.py
    python scripts/run_token_probs.py --prompt "What is machine learning?"
    python scripts/run_token_probs.py --prompt "The capital of France is" --raw --temperature 0
    python scripts/run_token_probs.py --prompt "Hello" --auto --save_json probs.json
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.tokenizer import GPT2Tokenizer
from inference.model_loader import load_model
from inference.token_probs import TokenProbabilityStepper, StepInfo, Candidate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if os.name == "nt":
    os.system("")  # enables ANSI escape codes in the Windows console


class C:
    """ANSI colors (disabled with --no_color)."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    INV = "\033[7m"

    @classmethod
    def disable(cls):
        for k in ("RESET", "BOLD", "DIM", "GREEN", "YELLOW", "CYAN", "MAGENTA", "INV"):
            setattr(cls, k, "")


def parse_args():
    p = argparse.ArgumentParser(description="Step through generation and see top-k token probabilities")
    p.add_argument("--checkpoint", type=str, default="checkpoints/usaid_ai_500m.pt")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt (otherwise interactive loop)")
    p.add_argument("--raw", action="store_true", help="Feed prompt as-is (no 'User:/Assistant:' chat wrapper)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--show_top", type=int, default=5, help="How many candidates to display per step")
    p.add_argument("--auto", action="store_true", help="Don't wait for Enter between tokens")
    p.add_argument("--save_json", type=str, default=None, help="Write per-step probabilities to a JSON file")
    p.add_argument("--no_color", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return p.parse_args()


def fmt_tok(text: str, width: int = 16) -> str:
    """Quoted, escaped token text so spaces/newlines are visible."""
    s = repr(text)[1:-1]
    s = f'"{s}"'
    return s if len(s) >= width else s + " " * (width - len(s))


def bar(p: float, width: int = 30) -> str:
    full = p * width
    n = int(full)
    partial = " ▏▎▍▌▋▊▉"[int((full - n) * 8)] if n < width else ""
    return ("█" * n + partial).ljust(width)


def print_step(info: StepInfo, context_text: str, show_sample_col: bool):
    print(f"\n{C.DIM}{'─' * 78}{C.RESET}")
    print(f"{C.BOLD}Step {info.step}{C.RESET}   "
          f"{C.DIM}uncertainty (entropy): {info.entropy_bits:.2f} bits{C.RESET}")
    tail = context_text[-240:]
    print(f"{C.DIM}Text so far:{C.RESET} {tail}{C.INV} ? {C.RESET}")
    print()
    hdr = f"   {'#':>2}  {'token':<16}  {'prob':>7}"
    if show_sample_col:
        hdr += f"  {'sample p':>8}"
    print(C.DIM + hdr + C.RESET)

    for c in info.candidates:
        is_pick = c.token_id == info.sampled.token_id
        color = C.GREEN if is_pick else (C.CYAN if c.rank == 1 else "")
        marker = "▶" if is_pick else " "
        line = f" {marker} {c.rank:>2}  {fmt_tok(c.text)}  {c.raw_prob * 100:6.2f}%"
        if show_sample_col:
            line += f"  {c.sample_prob * 100:7.2f}%"
        line += f"  {bar(c.raw_prob)}"
        print(color + line + C.RESET)

    shown = sum(c.raw_prob for c in info.candidates)
    print(f"{C.DIM}      (top-{len(info.candidates)} cover {shown * 100:.1f}% of probability mass){C.RESET}")

    s = info.sampled
    if all(c.token_id != s.token_id for c in info.candidates):
        print(f"{C.YELLOW} ▶ sampled outside top-{len(info.candidates)}: {fmt_tok(s.text, 0)} "
              f"rank #{s.rank}, prob {s.raw_prob * 100:.2f}%{C.RESET}")


def describe_choice(c: Candidate) -> str:
    return f"{fmt_tok(c.text, 0)} (rank #{c.rank}, {c.raw_prob * 100:.2f}%)"


def run_one(stepper: TokenProbabilityStepper, prompt: str, args, log: list):
    tok = stepper.tokenizer
    stepper.prefill(prompt)
    auto = args.auto
    stop_reason = "max_tokens"
    steps_log = []

    for _ in range(args.max_tokens):
        info = stepper.peek()
        print_step(info, stepper.generated_text, show_sample_col=True)

        choice_id = info.sampled.token_id
        if not auto:
            n = len(info.candidates)
            ans = input(f"{C.MAGENTA}[Enter]=accept  1-{n}=pick  a=auto  q=stop > {C.RESET}").strip().lower()
            if ans == "q":
                stop_reason = "user_stop"
                break
            if ans == "a":
                auto = True
            elif ans.isdigit() and 1 <= int(ans) <= n:
                choice_id = info.candidates[int(ans) - 1].token_id

        chosen = info.describe(choice_id, tok)
        steps_log.append({
            "step": info.step,
            "entropy_bits": round(info.entropy_bits, 4),
            "top": [
                {"rank": c.rank, "id": c.token_id, "token": c.text,
                 "prob": round(c.raw_prob, 6), "sample_prob": round(c.sample_prob, 6)}
                for c in info.candidates
            ],
            "chosen": {"id": chosen.token_id, "token": chosen.text,
                       "rank": chosen.rank, "prob": round(chosen.raw_prob, 6)},
        })

        if choice_id == tok.eos_token_id:
            print(f"{C.GREEN}✔ chose <|endoftext|> {describe_choice(chosen)}{C.RESET}")
            stop_reason = "eos_token"
            break

        stepper.commit(info, choice_id)
        print(f"{C.GREEN}✔ chose {describe_choice(chosen)}{C.RESET}")

        if "User:" in stepper.generated_text:
            stop_reason = "stop_sequence"
            break

    print(f"\n{C.BOLD}{'=' * 78}{C.RESET}")
    print(f"{C.BOLD}Final answer:{C.RESET} {stepper.generated_text.split('User:')[0].strip()}")
    s = stepper.summary()
    if s["tokens"]:
        print(f"{C.DIM}tokens={s['tokens']}  avg p={s['avg_prob'] * 100:.1f}%  "
              f"min p={s['min_prob'] * 100:.2f}%  top-1 picks={s['top1_rate'] * 100:.0f}%  "
              f"perplexity={s['perplexity']:.2f}  stop={stop_reason}{C.RESET}")
    log.append({"prompt": prompt, "steps": steps_log, "summary": s, "stop_reason": stop_reason})


def main():
    args = parse_args()
    if args.no_color:
        C.disable()

    tokenizer = GPT2Tokenizer()
    model, _ = load_model(args.checkpoint, args.device, args.dtype)
    stepper = TokenProbabilityStepper(
        model, tokenizer,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, show_top=args.show_top,
    )

    print(f"\n{C.DIM}'prob' = model's raw softmax probability.  "
          f"'sample p' = probability after temperature={args.temperature}, "
          f"top_k={args.top_k}, top_p={args.top_p}, rep_penalty={args.repetition_penalty}.{C.RESET}")

    def wrap(p: str) -> str:
        if args.raw or p.startswith("User:") or "Assistant:" in p:
            return p
        return f"User: {p}\n\nAssistant: "

    log: list = []
    try:
        if args.prompt:
            run_one(stepper, wrap(args.prompt), args, log)
        else:
            print("Type a prompt (or 'quit').")
            while True:
                p = input(f"\n{C.BOLD}User > {C.RESET}").strip()
                if not p:
                    continue
                if p.lower() in ("quit", "exit", "q"):
                    break
                run_one(stepper, wrap(p), args, log)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        if args.save_json and log:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
            print(f"Saved per-step probabilities to {args.save_json}")


if __name__ == "__main__":
    main()
