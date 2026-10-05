"""
Token Embedding Projector for Usaid AI (TinyGPT-500M).

Builds an interactive 3D/2D map of the model's token embedding vectors — similar to
https://projector.tensorflow.org/ — so you can see which words live close together.

What it does:
  1. Reads `lm_head.weight` (default) or `embed_tokens.weight` straight from the checkpoint
     (no full model needed — fast and light on RAM).
  2. Picks the N most frequent word-like tokens (+ any words you force with --words).
  3. Computes PCA and t-SNE (pure PyTorch, no sklearn needed). UMAP too if `umap-learn` is installed.
  4. Finds the k nearest neighbors of every token by cosine similarity in the original 1024-D space.
  5. Clusters tokens with k-means for coloring.
  6. Writes a standalone HTML page (open in any browser) and optionally TSV files you can
     upload directly to projector.tensorflow.org ("Load" button).

Examples:
    python scripts/visualize_embeddings.py --open
    python scripts/visualize_embeddings.py --num_tokens 2000 --words "king,queen,man,woman,python,java"
    python scripts/visualize_embeddings.py --methods pca --num_tokens 8000          # fast, no t-SNE
    python scripts/visualize_embeddings.py --source lm_head --export_tsv embedding_viz/tsv
"""

import argparse
import json
import math
import os
import re
import sys
import time
import webbrowser
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from data.tokenizer import GPT2Tokenizer
from inference.model_loader import resolve_checkpoint, load_raw_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEMPLATE_PATH = os.path.join(REPO_ROOT, "visualization", "projector_template.html")
WORD_RE = re.compile(r"^ [A-Za-z][A-Za-z']+$")  # whole words only (GPT-2 marks word starts with a space)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive token-embedding projector")
    p.add_argument("--checkpoint", type=str, default="checkpoints/usaid_ai_500m.pt")
    p.add_argument("--source", choices=["embed", "lm_head"], default="lm_head",
                   help="'lm_head' = output (unembedding) vectors — default, much more semantic for this "
                        "checkpoint; 'embed' = input token embeddings (still close to random init here)")
    p.add_argument("--num_tokens", type=int, default=3000, help="How many tokens to plot")
    p.add_argument("--filter", choices=["words", "all"], default="words",
                   help="'words' = only alphabetic word tokens (cleaner); 'all' = any token")
    p.add_argument("--no_dedupe", action="store_true",
                   help="Keep both ' word' and 'word' / 'Word' variants (default merges them)")
    p.add_argument("--words", type=str, default="",
                   help="Comma-separated words that must be included, e.g. 'king,queen,apple'")
    p.add_argument("--methods", type=str, default="pca,tsne", help="Any of: pca,tsne,umap")
    p.add_argument("--neighbors", type=int, default=15, help="Nearest neighbors stored per token")
    p.add_argument("--clusters", type=int, default=12, help="k-means clusters used for coloring")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="embedding_viz/embedding_projector.html")
    p.add_argument("--export_tsv", type=str, default=None,
                   help="Directory to write vectors.tsv + metadata.tsv for projector.tensorflow.org")
    p.add_argument("--open", action="store_true", help="Open the HTML in your browser when done")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def load_embedding_matrix(path: str, source: str) -> torch.Tensor:
    state_dict, _ = load_raw_checkpoint(path)
    key = "embed_tokens.weight" if source == "embed" else "lm_head.weight"
    for k, v in state_dict.items():
        if k == key or k.endswith("." + key):
            return v.detach().float().clone()
    raise KeyError(f"'{key}' not found in checkpoint. Available keys start with: {list(state_dict)[:5]}")


def display_label(text: str) -> str:
    s = text.strip()
    return s if s else repr(text)[1:-1]


def select_tokens(tok: GPT2Tokenizer, vocab: int, num: int, mode: str,
                  dedupe: bool, forced_words: list[str]) -> list[int]:
    """
    GPT-2 BPE ids are (roughly) ordered by merge frequency, so scanning ids from
    low to high yields the most common tokens first.
    """
    chosen, seen = [], set()

    def key_of(t: str) -> str:
        return t.strip().lower() if dedupe else t

    for w in forced_words:
        for variant in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = tok.encode(variant)
            if len(ids) == 1 and ids[0] < vocab:
                k = key_of(tok.decode(ids))
                if k not in seen:
                    seen.add(k)
                    chosen.append(ids[0])
                break
        else:
            pieces = [tok.decode([i]) for i in tok.encode(" " + w)]
            print(f"  ! '{w}' is not a single token (splits into {pieces}) — skipped")

    for tid in range(vocab):
        if len(chosen) >= num:
            break
        try:
            text = tok.decode([tid])
        except Exception:
            continue
        if "\ufffd" in text or not text.strip():
            continue
        if mode == "words" and not WORD_RE.match(text):
            continue
        k = key_of(text)
        if k in seen:
            continue
        seen.add(k)
        chosen.append(tid)
    return chosen


# ----------------------------------------------------------------------------- math
def pca(X: torch.Tensor, dims: int) -> tuple[torch.Tensor, list[float]]:
    Xc = X - X.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = S.pow(2)
    explained = (var[:dims] / var.sum()).tolist()
    return Xc @ Vh[:dims].T, explained


def _conditional_probs(D: torch.Tensor, perplexity: float, steps: int = 64) -> torch.Tensor:
    """Vectorized binary search for per-point Gaussian bandwidths matching `perplexity`."""
    N = D.shape[0]
    eye = torch.eye(N, dtype=torch.bool)
    D = D / D[~eye].mean()
    Dmin = D.masked_fill(eye, float("inf")).min(1, keepdim=True).values
    Ds = (D - Dmin).masked_fill(eye, 0.0)

    target = math.log(perplexity)
    beta = torch.ones(N)
    lo = torch.zeros(N)
    hi = torch.full((N,), float("inf"))
    for _ in range(steps):
        P = torch.exp(-Ds * beta[:, None]).masked_fill(eye, 0.0)
        sumP = P.sum(1).clamp_min(1e-12)
        H = torch.log(sumP) + beta * (Ds * P).sum(1) / sumP
        too_flat = H > target  # entropy too high -> narrow the Gaussian (bigger beta)
        lo = torch.where(too_flat, beta, lo)
        hi = torch.where(too_flat, hi, beta)
        beta = torch.where(torch.isinf(hi), beta * 2.0, (lo + hi) / 2.0)
    P = torch.exp(-Ds * beta[:, None]).masked_fill(eye, 0.0)
    return P / P.sum(1, keepdim=True).clamp_min(1e-12)


def tsne(X: torch.Tensor, dims: int = 3, perplexity: float = 30.0, iters: int = 1000,
         seed: int = 42) -> torch.Tensor:
    """Exact t-SNE (van der Maaten 2008) in PyTorch. Fine for a few thousand points."""
    N = X.shape[0]
    perplexity = min(perplexity, (N - 1) / 3)
    X50, _ = pca(X, min(50, X.shape[1], N))
    P = _conditional_probs(torch.cdist(X50, X50).pow(2), perplexity)
    P = (P + P.T)
    P = (P / P.sum()).clamp_min(1e-12)

    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(N, dims, generator=g) * 1e-4
    vel = torch.zeros_like(Y)
    gains = torch.ones_like(Y)
    lr = max(N / 12.0 / 4.0, 50.0)  # sklearn's "auto" learning rate
    exag_iters = min(250, iters // 4)
    t0 = time.time()

    for it in range(iters):
        exag = 12.0 if it < exag_iters else 1.0
        mom = 0.5 if it < exag_iters else 0.8
        sq = (Y * Y).sum(1)
        num = 1.0 / (1.0 + sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T))
        num.fill_diagonal_(0.0)
        Q = (num / num.sum()).clamp_min(1e-12)
        PQ = (exag * P - Q) * num
        grad = 4.0 * (PQ.sum(1, keepdim=True) * Y - PQ @ Y)

        flip = (grad > 0) != (vel > 0)
        gains = torch.where(flip, gains + 0.2, gains * 0.8).clamp_min(0.01)
        vel = mom * vel - lr * gains * grad
        Y = Y + vel
        Y = Y - Y.mean(0, keepdim=True)

        if (it + 1) % 100 == 0 or it == iters - 1:
            kl = float((P * torch.log(P / Q)).sum())
            print(f"    t-SNE iter {it + 1:>4}/{iters}  KL={kl:.4f}  ({time.time() - t0:.1f}s)")
    return Y


def run_umap(X: torch.Tensor, dims: int, seed: int) -> torch.Tensor | None:
    try:
        import umap  # type: ignore
    except ImportError:
        print("  ! UMAP skipped — install with: pip install umap-learn")
        return None
    reducer = umap.UMAP(n_components=dims, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=seed)
    return torch.from_numpy(reducer.fit_transform(X.numpy())).float()


def knn_cosine(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    Xn = F.normalize(X, dim=1)
    S = Xn @ Xn.T
    S.fill_diagonal_(-float("inf"))
    sims, idx = torch.topk(S, k=min(k, X.shape[0] - 1), dim=1)
    return idx, sims


def kmeans_cosine(X: torch.Tensor, k: int, iters: int = 40, seed: int = 42) -> torch.Tensor:
    Xn = F.normalize(X, dim=1)
    g = torch.Generator().manual_seed(seed)
    # k-means++ init
    centers = [Xn[torch.randint(len(Xn), (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):
        d = 1 - (Xn @ torch.stack(centers).T).max(1).values
        centers.append(Xn[torch.multinomial(d.clamp_min(1e-9), 1, generator=g)].squeeze(0))
    C = torch.stack(centers)
    for _ in range(iters):
        assign = (Xn @ C.T).argmax(1)
        for c in range(k):
            m = assign == c
            if m.any():
                C[c] = F.normalize(Xn[m].mean(0), dim=0)
    return (Xn @ C.T).argmax(1)


def normalize_coords(Y: torch.Tensor) -> list[list[float]]:
    Y = Y - Y.mean(0, keepdim=True)
    Y = Y / Y.abs().max().clamp_min(1e-9)
    return [[round(float(v), 4) for v in row] for row in Y]


# ----------------------------------------------------------------------------- output
def tsv_safe(s: str) -> str:
    return repr(s)[1:-1].replace("\t", "\\t") or "<empty>"


def export_tsv(out_dir: str, X: torch.Tensor, tokens: list[str], ids: list[int], clusters: list[int]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vectors.tsv"), "w", encoding="utf-8") as f:
        for row in X.tolist():
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")
    with open(os.path.join(out_dir, "metadata.tsv"), "w", encoding="utf-8") as f:
        f.write("token\ttoken_id\tcluster\n")
        for t, i, c in zip(tokens, ids, clusters):
            f.write(f"{tsv_safe(t.strip() or t)}\t{i}\t{c}\n")
    print(f"  TSV for projector.tensorflow.org -> {out_dir}/vectors.tsv + metadata.tsv")


def main():
    args = parse_args()
    t_start = time.time()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"[1/6] Reading '{args.source}' matrix from {ckpt} ...")
    W = load_embedding_matrix(ckpt, args.source)
    vocab, dim = W.shape
    print(f"      matrix shape: {vocab} x {dim}")

    print(f"[2/6] Selecting up to {args.num_tokens} tokens (filter={args.filter}) ...")
    tok = GPT2Tokenizer()
    forced = [w.strip() for w in args.words.split(",") if w.strip()]
    ids = select_tokens(tok, vocab, args.num_tokens, args.filter, not args.no_dedupe, forced)
    tokens = [tok.decode([i]) for i in ids]
    labels = [display_label(t) for t in tokens]
    X = W[ids]
    print(f"      selected {len(ids)} tokens")

    print(f"[3/6] Nearest neighbors (cosine, k={args.neighbors}) + k-means ({args.clusters} clusters) ...")
    nb_idx, nb_sims = knn_cosine(X, args.neighbors)
    clusters = kmeans_cosine(X, min(args.clusters, len(ids)), seed=args.seed).tolist()

    print(f"[4/6] Projections: {methods}")
    proj, meta_methods = {}, {}
    if "pca" in methods or not methods:
        Y, ev = pca(X, 3)
        proj["pca"] = normalize_coords(Y)
        meta_methods["pca"] = {"note": f"PCA: 3 components explain <b>{sum(ev) * 100:.1f}%</b> of variance "
                                       f"({', '.join(f'{e * 100:.1f}%' for e in ev)}). Linear &amp; global structure."}
        print(f"    PCA explained variance: {[round(e, 4) for e in ev]}")
    if "tsne" in methods:
        n = len(ids)
        print(f"    t-SNE on {n} points (exact, O(N^2) — ~{n * n * 4 * 6 / 1e9:.1f} GB peak) ...")
        proj["tsne"] = normalize_coords(tsne(X, 3, args.perplexity, args.tsne_iters, args.seed))
        meta_methods["tsne"] = {"note": f"t-SNE (perplexity {args.perplexity:g}, {args.tsne_iters} iters): "
                                        "preserves local neighborhoods — best for seeing word groups. "
                                        "Distances between far-apart clusters are not meaningful."}
    if "umap" in methods:
        Y = run_umap(X, 3, args.seed)
        if Y is not None:
            proj["umap"] = normalize_coords(Y)
            meta_methods["umap"] = {"note": "UMAP (cosine, 15 neighbors): balances local and global structure."}
    if not proj:
        raise SystemExit("No projection computed — check --methods.")
    default = "tsne" if "tsne" in proj else next(iter(proj))

    print("[5/6] Writing HTML ...")
    data = {
        "meta": {
            "checkpoint": ckpt.replace("\\", "/"), "source": f"{args.source} ({'embed_tokens' if args.source == 'embed' else 'lm_head'}.weight)",
            "dim": dim, "methods": meta_methods,
            "generated": f"generated {datetime.now():%Y-%m-%d %H:%M} · {len(ids)} tokens · seed {args.seed}",
        },
        "default_proj": default,
        "tokens": tokens, "labels": labels, "ids": ids, "clusters": clusters,
        "proj": proj,
        "neighbors": nb_idx.tolist(),
        "sims": [[round(float(s), 3) for s in row] for row in nb_sims],
    }
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", payload, 1)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"      -> {os.path.abspath(args.out)}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    print("[6/6] Extras ...")
    if args.export_tsv:
        export_tsv(args.export_tsv, X, tokens, ids, clusters)

    # Quick text preview so you see something useful right in the terminal
    preview = forced[:5] or ["king", "good", "computer", "water", "love"]
    lab_index = {l.lower(): i for i, l in enumerate(labels)}
    for w in preview:
        i = lab_index.get(w.lower())
        if i is None:
            continue
        nn = ", ".join(f"{labels[j]} ({s:.2f})" for j, s in zip(nb_idx[i][:6].tolist(), nb_sims[i][:6].tolist()))
        print(f"    {labels[i]:>10} -> {nn}")

    print(f"Done in {time.time() - t_start:.1f}s.")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(args.out).replace("\\", "/"))


if __name__ == "__main__":
    main()
