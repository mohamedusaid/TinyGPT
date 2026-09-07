"""
TinyGPT v2 Question Answering

Uses:

    tiny_gpt_qa_v2.pt
    content.txt
    qa_content.txt

The QA dataset is used as a lightweight knowledge/retrieval
layer. TinyGPT is then used for generation.
"""

import os
import re
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_FILE = "tiny_gpt_qa_v2.pt"

CONTENT_FILE = (
    "content.txt"
)

QA_FILE = (
    "qa_content.txt"
)

CONTEXT_SIZE = 128

MAX_GENERATION_TOKENS = 80

TEMPERATURE = 0.65

TOP_K = 30

EOS_TOKEN_ID = 50256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if hasattr(
    sys.stdout,
    "reconfigure"
):

    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# 2. STARTUP
# ============================================================

print(
    f"Device: {DEVICE}"
)

print(
    f"Model file: {MODEL_FILE}"
)


# ============================================================
# 3. TOKENIZER
# ============================================================

print(
    "Loading tokenizer..."
)

tokenizer = tiktoken.get_encoding(
    "gpt2"
)

print(
    f"Tokenizer vocabulary: "
    f"{tokenizer.n_vocab}"
)

print(
    f"EOS Token ID: "
    f"{EOS_TOKEN_ID}"
)


# ============================================================
# 4. LOAD QA DATA
# ============================================================

print(
    "Loading QA data..."
)

if not os.path.exists(QA_FILE):

    raise FileNotFoundError(
        f"Missing {QA_FILE}"
    )


with open(
    QA_FILE,
    "r",
    encoding="utf-8"
) as f:

    qa_text = f.read()


qa_blocks = [
    block.strip()
    for block in qa_text.split(
        "\n\n"
    )
    if block.strip()
]


# ============================================================
# 5. PARSE QA PAIRS
# ============================================================

qa_pairs = []


for block in qa_blocks:

    question_match = re.search(
        r"Question:\s*(.*?)\s*Answer:",
        block,
        re.IGNORECASE |
        re.DOTALL
    )

    answer_match = re.search(
        r"Answer:\s*(.*?)(?:<\|endoftext\|>|$)",
        block,
        re.IGNORECASE |
        re.DOTALL
    )

    if (
        question_match
        and answer_match
    ):

        question = (
            question_match
            .group(1)
            .strip()
        )

        answer = (
            answer_match
            .group(1)
            .strip()
        )

        if question and answer:

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer
                }
            )


print(
    f"QA pairs loaded: "
    f"{len(qa_pairs):,}"
)


# ============================================================
# 6. TEXT NORMALIZATION
# ============================================================

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


STOP_WORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "was",
    "were",
    "are",
    "who",
    "where",
    "when",
    "how",
    "did",
    "do",
    "does",
    "can",
    "could",
    "would",
    "tell",
    "me",
    "about",
    "your",
    "you"
}


def useful_words(text):

    words = normalize(text).split()

    return {
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    }


# ============================================================
# 7. QA RETRIEVAL
# ============================================================

def retrieve_qa(
    question,
    min_score=0.30
):

    query_words = useful_words(
        question
    )

    if not query_words:

        return []


    scored = []


    for pair in qa_pairs:

        candidate_words = useful_words(
            pair["question"]
        )

        if not candidate_words:
            continue


        intersection = (
            query_words
            & candidate_words
        )


        union = (
            query_words
            | candidate_words
        )


        jaccard = (
            len(intersection)
            / max(
                len(union),
                1
            )
        )


        overlap = (
            len(intersection)
            / max(
                len(query_words),
                1
            )
        )


        score = (
            0.45 * jaccard
            + 0.55 * overlap
        )


        # Strong bonus for important exact phrases.

        normalized_query = normalize(
            question
        )

        normalized_candidate = normalize(
            pair["question"]
        )


        if normalized_candidate == normalized_query:

            score += 1.0


        if (
            normalized_query
            in normalized_candidate
            or normalized_candidate
            in normalized_query
        ):

            score += 0.35


        scored.append(
            (
                score,
                pair
            )
        )


    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )


    results = []

    for score, pair in scored[:5]:

        if score >= min_score:

            results.append(
                {
                    "score": score,
                    "question": pair[
                        "question"
                    ],
                    "answer": pair[
                        "answer"
                    ]
                }
            )


    return results


# ============================================================
# 8. SPECIAL RESPONSES
# ============================================================

def special_response(
    question
):

    q = normalize(
        question
    )


    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good evening"
    }


    if q in greetings:

        return (
            "Hi! I'm TinyGPT. "
            "Ask me something."
        )


    if (
        "your name" in q
        or "who are you" in q
        or q == "your name"
    ):

        return (
            "I'm TinyGPT, "
            "a small language model."
        )


    if q in {
        "bye",
        "goodbye",
        "see you"
    }:

        return (
            "Goodbye!"
        )


    return None


# ============================================================
# 9. MODEL COMPONENTS
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_size
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = (
            embed_dim // num_heads
        )

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_size,
                    context_size
                )
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)


        scores = (
            q
            @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )


        mask = (
            self.mask[
                :T,
                :T
            ]
            .bool()
        )


        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )


        weights = F.softmax(
            scores,
            dim=-1
        )


        out = weights @ v


        out = out.transpose(
            1,
            2
        ).contiguous()


        out = out.view(
            B,
            T,
            C
        )


        return self.proj(
            out
        )


# ============================================================
# 10. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_size
    ):
        super().__init__()

        self.attention = (
            CausalSelfAttention(
                embed_dim,
                num_heads,
                context_size
            )
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            )
        )


    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


# ============================================================
# 11. TINYGPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_size,
        embedding_size,
        num_heads,
        num_layers
    ):
        super().__init__()

        self.context_size = (
            context_size
        )

        self.token_embedding = (
            nn.Embedding(
                vocab_size,
                embedding_size
            )
        )

        self.position_embedding = (
            nn.Embedding(
                context_size,
                embedding_size
            )
        )


        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    context_size
                )
                for _ in range(num_layers)
            ]
        )


        self.final_norm = (
            nn.LayerNorm(
                embedding_size
            )
        )


        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )


        self.lm_head.weight = (
            self.token_embedding.weight
        )


    def forward(
        self,
        idx
    ):

        B, T = idx.shape

        positions = torch.arange(
            T,
            device=idx.device
        )


        x = (
            self.token_embedding(idx)
            + self.position_embedding(
                positions
            )
        )


        for block in self.blocks:

            x = block(x)


        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# 12. LOAD CHECKPOINT
# ============================================================

print()
print(
    "Loading model..."
)


if not os.path.exists(
    MODEL_FILE
):

    raise FileNotFoundError(
        f"Could not find {MODEL_FILE}."
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


config = checkpoint[
    "config"
]


print()
print(
    "Model configuration:"
)

print(
    f"  vocab_size: "
    f"{config['vocab_size']}"
)

print(
    f"  context_size: "
    f"{config['context_size']}"
)

print(
    f"  embedding_size: "
    f"{config['embedding_size']}"
)

print(
    f"  num_heads: "
    f"{config['num_heads']}"
)

print(
    f"  num_layers: "
    f"{config['num_layers']}"
)


model = TinyGPT(
    vocab_size=config[
        "vocab_size"
    ],

    context_size=config[
        "context_size"
    ],

    embedding_size=config[
        "embedding_size"
    ],

    num_heads=config[
        "num_heads"
    ],

    num_layers=config[
        "num_layers"
    ]
).to(DEVICE)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    "Model loaded successfully."
)

print(
    f"Parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# 13. GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.65,
    top_k=30
):

    idx = torch.tensor(
        [
            tokenizer.encode(
                prompt
            )
        ],
        dtype=torch.long,
        device=DEVICE
    )


    stop_reason = (
        "maximum length"
    )


    for _ in range(
        max_new_tokens
    ):

        idx_cond = (
            idx[
                :,
                -CONTEXT_SIZE:
            ]
        )


        logits = model(
            idx_cond
        )


        logits = (
            logits[:, -1, :]
        )


        if temperature <= 0:

            next_token = (
                torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
            )

        else:

            logits = (
                logits
                / temperature
            )


            if (
                top_k is not None
                and top_k > 0
            ):

                values, indices = (
                    torch.topk(
                        logits,
                        min(
                            top_k,
                            logits.size(-1)
                        )
                    )
                )


                filtered = (
                    torch.full_like(
                        logits,
                        float("-inf")
                    )
                )


                filtered.scatter_(
                    1,
                    indices,
                    values
                )


                logits = filtered


            probabilities = (
                F.softmax(
                    logits,
                    dim=-1
                )
            )


            next_token = (
                torch.multinomial(
                    probabilities,
                    1
                )
            )


        token_id = (
            next_token.item()
        )


        if token_id == EOS_TOKEN_ID:

            stop_reason = "EOS"

            break


        idx = torch.cat(
            [
                idx,
                next_token
            ],
            dim=1
        )


    generated = (
        tokenizer.decode(
            idx[0].tolist()
        )
    )


    return (
        generated,
        stop_reason
    )


# ============================================================
# 14. ANSWER QUESTION
# ============================================================

def answer_question(
    question
):

    question = question.strip()


    if not question:

        return


    # --------------------------------------------------------
    # Special responses
    # --------------------------------------------------------

    special = special_response(
        question
    )


    if special is not None:

        print()
        print(
            f"TinyGPT: {special}"
        )

        return


    # --------------------------------------------------------
    # Retrieve QA knowledge
    # --------------------------------------------------------

    results = retrieve_qa(
        question
    )


    print()


    if results:

        best = results[0]

        print(
            "Retrieved knowledge:"
        )

        print(
            f"  Match: "
            f"{best['question']}"
        )

        print(
            f"  Score: "
            f"{best['score']:.3f}"
        )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # For high-confidence QA matches we return the known
        # answer directly.
        #
        # This prevents a tiny 11M parameter model from
        # corrupting an answer it already has in the dataset.
        # ----------------------------------------------------

        if best["score"] >= 0.65:

            print()

            print(
                f"TinyGPT: "
                f"{best['answer']}"
            )

            return


        # ----------------------------------------------------
        # Medium-confidence result
        #
        # Give the model the retrieved knowledge.
        # ----------------------------------------------------

        context = best[
            "answer"
        ]


        prompt = (
            "Question: "
            + question
            + "\n"
            + "Relevant information: "
            + context
            + "\n"
            + "Answer:"
        )


    else:

        # ----------------------------------------------------
        # No QA match.
        #
        # Let the language model attempt a response.
        # ----------------------------------------------------

        prompt = (
            "Question: "
            + question
            + "\n"
            + "Answer:"
        )


    generated, reason = (
        generate(
            prompt,
            max_new_tokens=MAX_GENERATION_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K
        )
    )


    # Remove the prompt from output.

    if generated.startswith(
        prompt
    ):

        answer = generated[
            len(prompt):
        ].strip()

    else:

        answer = generated.strip()


    # Remove accidental EOS text.

    answer = answer.replace(
        "<|endoftext|>",
        ""
    ).strip()


    print(
        f"Generation stopped: "
        f"{reason}"
    )


    if not answer:

        print(
            "TinyGPT: "
            "I couldn't generate an answer."
        )

    else:

        print(
            f"TinyGPT: {answer}"
        )


# ============================================================
# 15. MAIN LOOP
# ============================================================

print()
print("=" * 60)
print("TinyGPT Question Answering")
print("=" * 60)

print()
print(
    "Ask a question."
)

print(
    "Type 'quit' or 'exit' to stop."
)


while True:

    try:

        question = input(
            "\nYou: "
        )

    except (
        EOFError,
        KeyboardInterrupt
    ):

        print()
        print(
            "Goodbye!"
        )

        break


    question = question.strip()


    if question.lower() in {
        "quit",
        "exit"
    }:

        print(
            "Goodbye!"
        )

        break


    answer_question(
        question
    )