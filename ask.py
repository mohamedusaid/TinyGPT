# ask.py
#
# TinyGPT Question Answering
#
# Uses:
#   content.txt   -> story knowledge
#   tiny_gpt.pt   -> trained TinyGPT
#
# The program:
#   1. Loads the story
#   2. Splits it into chunks
#   3. Finds the most relevant chunks for a question
#   4. Gives the relevant context to TinyGPT
#   5. Generates an answer
#
# CPU-friendly version

import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "tiny_gpt.pt"
STORY_FILE = "content.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_CONTEXT = 64

TEMPERATURE = 0.7
TOP_K = 40

MAX_NEW_TOKENS = 60

# Number of story chunks retrieved
TOP_CHUNKS = 3


# ============================================================
# DEVICE
# ============================================================

print(f"Device: {DEVICE}")


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

try:
    import tiktoken
except ImportError:
    print()
    print("ERROR: tiktoken is not installed.")
    print()
    print("Run:")
    print("    pip install tiktoken")
    print()
    raise SystemExit


tokenizer = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = tokenizer.n_vocab

print(f"Tokenizer vocabulary: {VOCAB_SIZE}")


# ============================================================
# LOAD STORY
# ============================================================

if not os.path.exists(STORY_FILE):
    print(f"ERROR: {STORY_FILE} not found.")
    raise SystemExit

with open(STORY_FILE, "r", encoding="utf-8") as f:
    story = f.read().strip()


print(f"Story characters: {len(story)}")


# ============================================================
# SPLIT STORY INTO PARAGRAPHS / CHUNKS
# ============================================================

def split_story(text, max_words=90):
    """
    Split story into reasonably sized chunks.

    We prefer paragraph boundaries, then combine small
    paragraphs together.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = []
    current_words = 0

    for paragraph in paragraphs:

        words = paragraph.split()

        # If one paragraph itself is very large,
        # split it into smaller pieces.
        if len(words) > max_words:

            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0

            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)

            continue

        if current_words + len(words) > max_words:

            if current:
                chunks.append(" ".join(current))

            current = [paragraph]
            current_words = len(words)

        else:
            current.append(paragraph)
            current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


chunks = split_story(story)

print(f"Story chunks: {len(chunks)}")


# ============================================================
# SIMPLE TF-IDF RETRIEVAL
# ============================================================

def tokenize_words(text):
    """
    Basic word tokenizer for retrieval.
    """
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def build_document_frequency(chunks):
    df = {}

    for chunk in chunks:

        words = set(tokenize_words(chunk))

        for word in words:
            df[word] = df.get(word, 0) + 1

    return df


document_frequency = build_document_frequency(chunks)

NUM_DOCUMENTS = len(chunks)


def tfidf_score(query, document):
    """
    Calculate a simple TF-IDF similarity score.
    """

    query_words = tokenize_words(query)
    document_words = tokenize_words(document)

    if not query_words or not document_words:
        return 0.0

    document_counts = {}

    for word in document_words:
        document_counts[word] = document_counts.get(word, 0) + 1

    document_length = len(document_words)

    score = 0.0

    for word in query_words:

        if word not in document_counts:
            continue

        tf = document_counts[word] / document_length

        df = document_frequency.get(word, 0)

        if df == 0:
            continue

        idf = math.log(
            (NUM_DOCUMENTS + 1)
            / (df + 1)
        ) + 1

        score += tf * idf

    return score


def retrieve(question, top_n=TOP_CHUNKS):

    scored = []

    for index, chunk in enumerate(chunks):

        score = tfidf_score(question, chunk)

        scored.append(
            (score, index, chunk)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored[:top_n]


# ============================================================
# MODEL
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, context_length):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=True
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.split(
            self.embed_dim,
            dim=2
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

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        mask = self.mask[:T, :T]

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        out = attention_weights @ v

        out = out.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        out = self.proj(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        hidden_dim = 4 * embed_dim

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embed_dim
            )
        )

    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim,
            num_heads,
            context_length
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(
            embed_dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        num_heads,
        num_layers
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                context_length
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(
            embed_dim
        )

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.context_length = context_length

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        token_embeddings = (
            self.token_embedding(x)
        )

        position_embeddings = (
            self.position_embedding(positions)
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

print("Checkpoint keys:")

for key in checkpoint.keys():
    print(f"  {key}")


config = checkpoint["config"]

print()
print("Model configuration:")

for key, value in config.items():
    print(f"  {key}: {value}")


model = TinyGPT(
    vocab_size=config["vocab_size"],
    context_length=config["context_size"],
    embed_dim=config["embedding_size"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"]
)


# ============================================================
# HANDLE CHECKPOINT MASK COMPATIBILITY
# ============================================================

state_dict = checkpoint["model_state_dict"]

# Older/newer versions of the model may save
# the causal mask as [T,T] or [1,1,T,T].
#
# The mask is not learned, so we simply remove it
# from the checkpoint and allow the model to create it.

keys_to_remove = []

for key in state_dict.keys():

    if key.endswith("attention.mask"):

        keys_to_remove.append(key)

for key in keys_to_remove:
    del state_dict[key]


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

if missing:
    print()
    print("Missing keys:")
    for key in missing:
        print(" ", key)

if unexpected:
    print()
    print("Unexpected keys:")
    for key in unexpected:
        print(" ", key)


model.to(DEVICE)

model.eval()

print()
print("Model loaded successfully.")


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {parameter_count:,}"
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_tokens(
    input_tokens,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K
):

    tokens = input_tokens.clone()

    for _ in range(max_new_tokens):

        # Keep only model context
        context = tokens[:, -MAX_CONTEXT:]

        logits = model(context)

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:

            values, indices = torch.topk(
                logits,
                min(top_k, logits.shape[-1])
            )

            filtered = torch.full_like(
                logits,
                float("-inf")
            )

            filtered.scatter_(
                1,
                indices,
                values
            )

            logits = filtered

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

    return tokens


# ============================================================
# TOKEN ENCODING
# ============================================================

def encode(text):

    tokens = tokenizer.encode(
        text
    )

    if len(tokens) == 0:
        return torch.empty(
            (1, 0),
            dtype=torch.long,
            device=DEVICE
        )

    return torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )


def decode(tokens):

    return tokenizer.decode(
        tokens
    )


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def clean_answer(text):

    text = text.strip()

    # Remove obvious prompt remnants
    prefixes = [
        "Answer:",
        "Answer",
        "A:",
        "AI:",
        "TinyGPT:"
    ]

    for prefix in prefixes:

        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Stop if the model starts creating another question
    stop_patterns = [
        "\nQuestion:",
        "\nQ:",
        "\nYou:",
        "\nUser:"
    ]

    for pattern in stop_patterns:

        if pattern in text:

            text = text.split(
                pattern,
                1
            )[0]

    return text.strip()


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(question):

    results = retrieve(question)

    # --------------------------------------------------------
    # Display retrieval information
    # --------------------------------------------------------

    print()
    print("Relevant story sections:")

    for rank, (score, index, chunk) in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"[{rank}] relevance={score:.4f}"
        )

        preview = chunk.replace(
            "\n",
            " "
        )

        print(
            preview[:250]
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for _, _, chunk in results:

        context_parts.append(
            chunk
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Your TinyGPT was trained on normal story text,
    # not instruction/QA examples.
    #
    # Therefore we use a very simple prompt so that the
    # model has a chance to continue naturally.
    # --------------------------------------------------------

    prompt = (
        "Question: "
        + question
        + "\n"
        + "Context:\n"
        + context
        + "\n"
        + "Answer:"
    )

    prompt_tokens = tokenizer.encode(
        prompt
    )

    # TinyGPT only accepts 64 tokens.
    #
    # Keep the question and beginning of context.
    if len(prompt_tokens) > MAX_CONTEXT - 1:

        question_tokens = tokenizer.encode(
            "Question: " + question + "\nAnswer:"
        )

        remaining = (
            MAX_CONTEXT
            - len(question_tokens)
            - 1
        )

        if remaining < 1:
            remaining = 1

        context_tokens = tokenizer.encode(
            context
        )[:remaining]

        prompt_tokens = (
            question_tokens
            + context_tokens
        )

    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated = generate_tokens(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    generated_text = decode(
        generated[0].tolist()
    )

    # Only take newly generated text
    original_length = len(
        decode(
            input_ids[0].tolist()
        )
    )

    answer = generated_text[
        original_length:
    ]

    answer = clean_answer(
        answer
    )

    return answer


# ============================================================
# INTERACTIVE QA
# ============================================================

print()
print("=" * 60)
print("TinyGPT Question Answering")
print("=" * 60)

print()
print("Ask a question about the story.")
print("Type 'exit' to quit.")
print()


while True:

    try:

        question = input("You: ").strip()

    except KeyboardInterrupt:

        print()
        print("Goodbye!")

        break

    except EOFError:

        print()
        print("Goodbye!")

        break


    if not question:
        continue


    if question.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print("Goodbye!")

        break


    try:

        answer = answer_question(
            question
        )

        print()
        print("TinyGPT:", answer)
        print()

    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        print()