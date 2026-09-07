import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "tiny_gpt.pt"
CONTENT_FILE = "content.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_K_CHUNKS = 3
MAX_GENERATION_TOKENS = 150

TEMPERATURE = 0.8
TOP_K = 40


print(f"Device: {DEVICE}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = tokenizer.n_vocab
EOS_TOKEN_ID = 50256

print(f"Tokenizer vocabulary: {VOCAB_SIZE}")
print(f"EOS Token ID: {EOS_TOKEN_ID}")


# ============================================================
# LOAD STORY
# ============================================================

print("Loading story...")

with open(CONTENT_FILE, "r", encoding="utf-8") as f:
    story = f.read().strip()

print(f"Story characters: {len(story)}")


# ============================================================
# STORY CHUNKING
# ============================================================

def split_story(text, chunk_words=120, overlap_words=30):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_words, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


story_chunks = split_story(story)

print(f"Story chunks: {len(story_chunks)}")


# ============================================================
# SIMPLE TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "to", "of", "and", "or", "in", "on", "at",
    "for", "from", "with", "who", "what", "where",
    "when", "why", "how", "did", "does", "do",
    "has", "have", "had", "he", "she", "it",
    "they", "them", "his", "her", "their",
    "this", "that", "these", "those",
    "about", "tell", "me", "please"
}


def normalize_words(text):

    words = re.findall(r"[a-zA-Z0-9']+", text.lower())

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 1
    ]


# ============================================================
# PREPARE CHUNK WORDS
# ============================================================

chunk_words = []

for chunk in story_chunks:
    chunk_words.append(set(normalize_words(chunk)))


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k=TOP_K_CHUNKS):

    query_words = set(normalize_words(question))

    if not query_words:
        return []

    scored = []

    for i, words in enumerate(chunk_words):

        overlap = query_words.intersection(words)

        score = len(overlap)

        # Slight bonus when important question words
        # occur multiple times in the chunk.
        raw_lower = story_chunks[i].lower()

        frequency_bonus = 0

        for word in query_words:
            count = raw_lower.count(word)

            if count > 1:
                frequency_bonus += min(count, 3) * 0.1

        score += frequency_bonus

        scored.append((score, i))

    scored.sort(reverse=True)

    results = []

    for score, index in scored[:top_k]:

        if score > 0:

            results.append({
                "score": score,
                "index": index,
                "text": story_chunks[index]
            })

    return results


# ============================================================
# MODEL
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, context_size):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # IMPORTANT:
        # checkpoint has qkv WITHOUT bias.
        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=False
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        # IMPORTANT:
        # checkpoint contains mask with shape [64, 64]
        mask = torch.tril(
            torch.ones(
                context_size,
                context_size
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

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
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = weights @ v

        output = output.transpose(
            1, 2
        ).contiguous()

        output = output.view(
            B,
            T,
            C
        )

        output = self.proj(output)

        return output


class FeedForward(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        hidden_dim = embed_dim * 4

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
        context_size
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim,
            num_heads,
            context_size
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
        context_size,
        embed_dim,
        num_heads,
        num_layers
    ):

        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embed_dim
        )

        self.blocks = nn.ModuleList([

            TransformerBlock(
                embed_dim,
                num_heads,
                context_size
            )

            for _ in range(num_layers)

        ])

        self.ln_f = nn.LayerNorm(
            embed_dim
        )

        # IMPORTANT:
        # The training model tied lm_head weights
        # to token_embedding.
        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(self, idx):

        B, T = idx.shape

        if T > self.context_size:
            idx = idx[:, -self.context_size:]

            T = self.context_size

        positions = torch.arange(
            T,
            device=idx.device
        )

        token_emb = self.token_embedding(idx)

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

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


# ------------------------------------------------------------
# Support both names used in previous model.py versions
# ------------------------------------------------------------

vocab_size = config.get(
    "vocab_size",
    VOCAB_SIZE
)

context_size = config.get(
    "context_size",
    config.get("context_length", 64)
)

embed_dim = config.get(
    "embedding_size",
    config.get("embed_dim", 128)
)

num_heads = config.get(
    "num_heads",
    4
)

num_layers = config.get(
    "num_layers",
    2
)


model = TinyGPT(
    vocab_size=vocab_size,
    context_size=context_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers
)


state_dict = checkpoint["model_state_dict"]


# ============================================================
# CLEAN OLD CHECKPOINT DIFFERENCES
# ============================================================

# Some versions of the model saved attention masks as
# parameters/buffers. They are deterministic and don't need
# to be restored from the checkpoint.

state_dict = {
    key: value
    for key, value in state_dict.items()
    if not key.endswith(".attention.mask")
}


missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)


if missing:

    print()
    print("Missing keys:")

    for key in missing:
        print(f"  {key}")


if unexpected:

    print()
    print("Unexpected keys:")

    for key in unexpected:
        print(f"  {key}")


model.to(DEVICE)

model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    f"Model loaded successfully."
)

print(
    f"Parameters: {parameter_count:,}"
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=MAX_GENERATION_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    stream=True
):

    tokens = tokenizer.encode(
        prompt
    )

    if len(tokens) == 0:
        return ""

    idx = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    stop_reason = "max_length"

    if stream:
        print("TinyGPT: ", end="", flush=True)

    generated_token_ids = []

    for step in range(max_new_tokens):

        idx_cond = idx[
            :, -context_size:
        ]

        logits = model(
            idx_cond
        )

        logits = logits[:, -1, :]

        if temperature > 0:

            logits = logits / temperature

            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
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

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

        else:

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        # ========================================================
        # EOS STOP
        # ========================================================

        if next_token.item() == EOS_TOKEN_ID:
            stop_reason = "EOS"
            break

        token_id = next_token.item()
        generated_token_ids.append(token_id)

        # Stream token live as it is generated
        if stream:
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)

        # Add token
        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    # ============================================================
    # SHOW WHY GENERATION STOPPED (AFTER OUTPUT COMPLETES)
    # ============================================================

    if stream:
        print("\n")

    if stop_reason == "EOS":
        print("Generation stopped: EOS token generated.")
    else:
        print(
            f"Generation stopped: maximum length reached "
            f"({max_new_tokens} new tokens)."
        )

    if stream:
        print()

    return tokenizer.decode(generated_token_ids)


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 0
    ]


def extract_answer(question, chunks):

    if not chunks:
        return None

    question_words = set(
        normalize_words(question)
    )

    candidates = []

    for chunk in chunks:

        sentences = split_sentences(
            chunk["text"]
        )

        for sentence in sentences:

            sentence_words = set(
                normalize_words(sentence)
            )

            overlap = (
                question_words
                & sentence_words
            )

            if not overlap:
                continue

            score = len(overlap)

            # Prefer sentences containing likely
            # entity/name information.
            capitalized_words = re.findall(
                r"\b[A-Z][a-z]+\b",
                sentence
            )

            score += min(
                len(capitalized_words) * 0.05,
                0.5
            )

            candidates.append(
                (
                    score,
                    sentence
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    answer = candidates[0][1]

    return answer


# ============================================================
# QUESTION TYPE
# ============================================================

def is_greeting(question):

    q = question.lower().strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good evening",
        "good afternoon",
        "ok",
        "okay",
        "thanks",
        "thank you"
    }

    return q in greetings


def is_identity_question(question):

    q = question.lower()

    phrases = [
        "who are you",
        "what are you",
        "what is your name",
        "who is this"
    ]

    return any(
        phrase in q
        for phrase in phrases
    )


# ============================================================
# ANSWER FUNCTION
# ============================================================

def answer_question(question):

    q = question.strip()

    if not q:
        return "Please ask me a question about the story."

    if is_greeting(q):

        print(
            "\nTinyGPT: Hi! I'm TinyGPT. "
            "Ask me something about the story.\n"
        )
        return

    if is_identity_question(q):

        print(
            "\nTinyGPT: I'm TinyGPT, a small language model "
            "trained on this story.\n"
        )
        return

    results = retrieve_chunks(
        q,
        TOP_K_CHUNKS
    )

    if not results:

        print(
            "\nTinyGPT: I couldn't find anything relevant "
            "in the story.\n"
        )
        return

    # --------------------------------------------------------
    # TinyGPT generation
    # --------------------------------------------------------

    context = "\n\n".join(
        r["text"]
        for r in results
    )

    prompt = (
        "Story:\n"
        + context
        + "\n\nQuestion: "
        + q
        + "\nAnswer:"
    )

    print()
    generate(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        top_k=30,
        stream=True
    )


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

    if question.lower() in {
        "exit",
        "quit"
    }:

        print("Goodbye!")

        break

    if not question:
        continue

    answer_question(
        question
    )