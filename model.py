"""
TinyGPT v2
-----------
Small decoder-only Transformer for learning.

Training data:
    content.txt
    qa_content.txt

Architecture:
    context_size = 128
    embedding_size = 192
    num_heads = 6
    num_layers = 3

The trained checkpoint is saved as:
    tiny_gpt_qa_v2.pt
"""

import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 1. SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_SIZE = 128
EMBEDDING_SIZE = 192
NUM_HEADS = 6
NUM_LAYERS = 3

BATCH_SIZE = 8

LEARNING_RATE = 3e-4

MAX_STEPS = 2000

EVAL_INTERVAL = 100
EVAL_BATCHES = 5

EARLY_STOPPING_PATIENCE = 4

QA_REPEAT = 3

EOS_TOKEN_ID = 50256

CHECKPOINT_FILE = "tiny_gpt_qa_v2.pt"

GENERAL_FILE = "content.txt"
QA_FILE = "qa_content.txt"

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DEVICE
# ============================================================

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# ============================================================
# 3. TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

enc = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = enc.n_vocab

print(f"Tokenizer vocabulary: {VOCAB_SIZE}")
print(f"EOS Token ID: {EOS_TOKEN_ID}")


# ============================================================
# 4. LOAD GENERAL DATA
# ============================================================

if not os.path.exists(GENERAL_FILE):
    raise FileNotFoundError(
        f"Could not find {GENERAL_FILE}"
    )

if not os.path.exists(QA_FILE):
    raise FileNotFoundError(
        f"Could not find {QA_FILE}"
    )

print()
print("Loading datasets...")

with open(
    GENERAL_FILE,
    "r",
    encoding="utf-8"
) as f:
    general_text = f.read()


with open(
    QA_FILE,
    "r",
    encoding="utf-8"
) as f:
    qa_text = f.read()


general_words = len(general_text.split())
qa_words = len(qa_text.split())


# ============================================================
# 5. SPLIT GENERAL TEXT INTO BLOCKS
# ============================================================

general_blocks = [
    block.strip()
    for block in general_text.split("\n\n")
    if block.strip()
]


# ============================================================
# 6. SPLIT QA DATA INTO BLOCKS
# ============================================================

qa_blocks = [
    block.strip()
    for block in qa_text.split("\n\n")
    if block.strip()
]


print()
print("=" * 60)
print("TinyGPT v2 Dataset")
print("=" * 60)

print(
    f"General text: {general_words:,} words"
)

print(
    f"General blocks: {len(general_blocks):,}"
)

print(
    f"QA examples: {len(qa_blocks):,}"
)

print(
    f"QA words: {qa_words:,}"
)


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

rng = random.Random(SEED)


# General text
general_shuffled = general_blocks[:]
rng.shuffle(general_shuffled)

general_train_count = int(
    len(general_shuffled) * 0.90
)

general_train_blocks = general_shuffled[
    :general_train_count
]

general_val_blocks = general_shuffled[
    general_train_count:
]


# QA
qa_shuffled = qa_blocks[:]
rng.shuffle(qa_shuffled)

qa_train_count = int(
    len(qa_shuffled) * 0.85
)

qa_train_blocks = qa_shuffled[
    :qa_train_count
]

qa_val_blocks = qa_shuffled[
    qa_train_count:
]


# ============================================================
# 8. BUILD TRAINING DATA
# ============================================================

# Repeat QA examples several times.
#
# Why?
#
# General text is tens of thousands of words.
# QA data is much smaller.
#
# Repeating QA examples gives the model more exposure
# to the Question -> Answer pattern.

train_blocks = (
    general_train_blocks
    + qa_train_blocks * QA_REPEAT
)

rng.shuffle(train_blocks)


# Validation data is NOT repeated.
#
# This is important because we want validation to represent
# unseen data rather than duplicated training examples.

val_blocks = (
    general_val_blocks
    + qa_val_blocks
)

rng.shuffle(val_blocks)


# ============================================================
# 9. TOKENIZATION
# ============================================================

def blocks_to_tokens(blocks):
    """
    Convert text blocks into token IDs.

    Each block receives an EOS token so the model can learn
    where one independent piece of text ends.
    """

    result = []

    for block in blocks:

        tokens = enc.encode(
            block,
            allowed_special={
                "<|endoftext|>"
            }
        )

        result.extend(tokens)

        if (
            len(tokens) == 0
            or tokens[-1] != EOS_TOKEN_ID
        ):
            result.append(EOS_TOKEN_ID)

    return result


print()
print("Tokenizing training data...")

train_tokens = blocks_to_tokens(
    train_blocks
)

print("Tokenizing validation data...")

val_tokens = blocks_to_tokens(
    val_blocks
)


train_data = torch.tensor(
    train_tokens,
    dtype=torch.long
)

val_data = torch.tensor(
    val_tokens,
    dtype=torch.long
)


print()
print("=" * 60)

print(
    f"Training tokens:   {len(train_data):,}"
)

print(
    f"Validation tokens: {len(val_data):,}"
)

print(
    f"QA repeat factor:   {QA_REPEAT}x"
)

print("=" * 60)


# ============================================================
# 10. BATCH CREATION
# ============================================================

def get_batch(data):

    if len(data) <= CONTEXT_SIZE:
        raise ValueError(
            "Dataset is smaller than CONTEXT_SIZE."
        )

    starts = torch.randint(
        0,
        len(data) - CONTEXT_SIZE - 1,
        (
            BATCH_SIZE,
        )
    )

    x = torch.stack(
        [
            data[i:i + CONTEXT_SIZE]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + CONTEXT_SIZE + 1]
            for i in starts
        ]
    )

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# 11. SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_size
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embedding_size must be divisible "
                "by num_heads"
            )

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
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        causal_mask = (
            self.mask[:T, :T]
            .bool()
        )

        scores = scores.masked_fill(
            ~causal_mask,
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

        return self.proj(out)


# ============================================================
# 12. TRANSFORMER BLOCK
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
# 13. TINYGPT MODEL
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

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_size
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

        self.final_norm = nn.LayerNorm(
            embedding_size
        )

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )

        # Weight tying.
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > self.context_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context size {self.context_size}"
            )

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

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss


# ============================================================
# 14. CREATE MODEL
# ============================================================

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    embedding_size=EMBEDDING_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("=" * 60)
print("Model Configuration")
print("=" * 60)

print(
    f"Vocabulary:       {VOCAB_SIZE:,}"
)

print(
    f"Context size:     {CONTEXT_SIZE}"
)

print(
    f"Embedding size:   {EMBEDDING_SIZE}"
)

print(
    f"Attention heads:  {NUM_HEADS}"
)

print(
    f"Transformer layers: {NUM_LAYERS}"
)

print(
    f"Parameters:       {parameter_count:,}"
)

print("=" * 60)


# ============================================================
# 15. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 16. EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(data)

            _, loss = model(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# 17. SPEED BENCHMARK
# ============================================================

print()
print("Benchmarking training speed...")

benchmark_steps = 20

model.train()

start = time.time()

for _ in range(benchmark_steps):

    x, y = get_batch(
        train_data
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(
        x,
        y
    )

    loss.backward()

    optimizer.step()


elapsed = time.time() - start

seconds_per_100 = (
    elapsed
    / benchmark_steps
    * 100
)

minutes_for_max = (
    seconds_per_100
    * MAX_STEPS
    / 60
)

print()
print("=" * 60)

print(
    f"Benchmark: {seconds_per_100:.1f} "
    f"seconds / 100 steps"
)

print(
    f"Estimated {MAX_STEPS:,} steps: "
    f"{minutes_for_max:.1f} minutes"
)

print("=" * 60)


# ============================================================
# 18. RESET MODEL AFTER BENCHMARK
# ============================================================

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    embedding_size=EMBEDDING_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 19. TRAINING
# ============================================================

print()
print("=" * 60)
print("Starting training...")
print("=" * 60)


best_val_loss = float("inf")

steps_without_improvement = 0

training_start = time.time()

for step in range(MAX_STEPS):

    model.train()

    x, y = get_batch(
        train_data
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(
        x,
        y
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        step % EVAL_INTERVAL == 0
        or step == MAX_STEPS - 1
    ):

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"Step {step:4d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            steps_without_improvement = 0

        else:

            steps_without_improvement += 1

            if (
                steps_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                print(
                    f"Best validation loss: "
                    f"{best_val_loss:.4f}"
                )

                break


# ============================================================
# 20. FINAL EVALUATION
# ============================================================

final_losses = estimate_loss()

training_time = (
    time.time()
    - training_start
)


print()
print("=" * 60)
print("Training finished")
print("=" * 60)

print(
    f"Final Train Loss: "
    f"{final_losses['train']:.4f}"
)

print(
    f"Final Validation Loss: "
    f"{final_losses['val']:.4f}"
)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 21. GENERATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):

    model.eval()

    idx = torch.tensor(
        [enc.encode(prompt)],
        dtype=torch.long,
        device=DEVICE
    )

    stop_reason = "maximum length"

    for _ in range(max_new_tokens):

        idx_cond = idx[
            :,
            -CONTEXT_SIZE:
        ]

        logits, _ = model(
            idx_cond
        )

        logits = logits[:, -1, :]

        if temperature <= 0:

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        else:

            logits = (
                logits
                / temperature
            )

            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
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
                1
            )

        token_id = next_token.item()

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

    return (
        enc.decode(
            idx[0].tolist()
        ),
        stop_reason
    )


# ============================================================
# 22. TEST GENERATION
# ============================================================

print()
print("=" * 60)
print("Testing generation")
print("=" * 60)


test_prompts = [
    "Question: Who is Erasmus Thorne?\nAnswer:",
    "Question: What is the name of the village?\nAnswer:",
    "Question: Who is Constable Whitmore?\nAnswer:",
    "Question: What is Python?\nAnswer:",
    "Question: What is an algorithm?\nAnswer:",
]


for prompt in test_prompts:

    print()
    print(
        f"Prompt: {prompt!r}"
    )

    generated, reason = generate(
        model,
        prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=30
    )

    print(
        f"Stopped because: {reason}"
    )

    print(
        generated
    )


# ============================================================
# 23. SAVE CHECKPOINT
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),

    "config": {
        "vocab_size": VOCAB_SIZE,
        "context_size": CONTEXT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS
    },

    "tokenizer": "gpt2",

    "eos_token_id": EOS_TOKEN_ID,

    "training": {
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "qa_repeat": QA_REPEAT
    }
}


torch.save(
    checkpoint,
    CHECKPOINT_FILE
)


print()
print("=" * 60)

print(
    f"Model saved as {CHECKPOINT_FILE}"
)

print(
    "Existing .pt files were not modified."
)

print("=" * 60)