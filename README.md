# 🚀 Usaid AI (TinyGPT-500M)

<div align="center">

**A 500M Parameter Causal Language Model Architected, Pretrained & SFT Aligned from Scratch**  
*Engineered by **Mohamed Usaid***

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-UsaidAI--500M-blue.svg)](https://huggingface.co/Usaidddddddddddddd/UsaidAI-500M)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Interactive%20Chat-orange.svg)](https://huggingface.co/spaces/Usaidddddddddddddd/UsaidAI-500M-Chat)
[![GitHub License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-500.14M-blueviolet.svg)](#1-architectural-specification)
[![Hardware](https://img.shields.io/badge/Pretrained%20On-Kaggle%20Dual%20Tesla%20T4%20(DDP)-yellow.svg)](#3-empirical-training--convergence-results)

[**Live Interactive Chat Demo**](https://huggingface.co/spaces/Usaidddddddddddddd/UsaidAI-500M-Chat) • [**Hugging Face Model Weights**](https://huggingface.co/Usaidddddddddddddd/UsaidAI-500M) • [**Architecture Specs**](#1-architectural-specification) • [**CLI & Usage Guide**](#4-comprehensive-usage-guide)

</div>

---

## 🌟 Overview

**Usaid AI (TinyGPT-500M)** is a half-billion parameter autoregressive causal language model designed to demonstrate how modern frontier large language model architectures can be built, trained, aligned, and deployed completely from scratch.

Scaling from foundational 6.8M and 15.3M prototypes, this 500M system incorporates modern architectural innovations:
- **Rotary Position Embeddings (RoPE)** for relative position-aware attention without static parameter bloat.
- **Grouped-Query Attention (GQA)** with a 4:1 query-to-KV head ratio for high-throughput, memory-efficient KV-caching.
- **SwiGLU Activation Function** in a 3-matrix Feed-Forward Network ($d_{ff} = 3,456$).
- **RMSNorm Pre-Normalization** with zero-mean centering and no bias terms for numerical gradient stability.
- **PyTorch Native Scaled Dot-Product Attention (SDPA)** leveraging FlashAttention/Memory-Efficient kernels.
- **Supervised Fine-Tuning (SFT)** with prompt loss masking ($-100$) for conversational identity, docstrings, and multi-language code generation.
- **Retrieval-Augmented Generation (RAG)** grounding engine using BM25 semantic retrieval over custom knowledge documents.
- **Interactive Interpretability Suite**: Step-by-step top-5 token probability branch explorer and 3D/2D embedding projector.

---

## 1. Architectural Specification

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Target Parameters** | **$500,000,000$** | Architecture scale goal |
| **Actual Parameters** | **`500,136,960`** | Exact parameter count (**$+0.027\%$ delta**) |
| **Vocabulary Size ($V$)** | **`50,257`** | Byte-Pair Encoding (`tiktoken` / GPT-2 standard) |
| **Hidden Size ($d_{\text{model}}$)** | **`1,024`** | Power-of-2 dimension ($2^{10}$) for optimal Tensor Core tiling |
| **Transformer Layers ($L$)** | **`30`** | Deep hierarchical reasoning depth |
| **Query Attention Heads ($H_q$)** | **`16`** | Head dimension $d_{\text{head}} = 64$ ($16 \times 64 = 1,024$) |
| **Key/Value Heads ($H_{kv}$)** | **`4`** | **Grouped-Query Attention (GQA 4:1 ratio)** |
| **Intermediate Size ($d_{ff}$)** | **`3,456`** | **SwiGLU** 3-matrix FFN ($27 \times 128 = 3,456 \approx 3.375 \times d_{\text{model}}$) |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings ($\theta = 10,000$, 0 static parameters) |
| **Normalization** | **RMSNorm** | Pre-norm with learnable scale $\gamma$ (zero-mean, no bias parameters) |
| **LM Head Tying** | **False** | Untied weights ($51.46\text{M}$ input embeddings + $51.46\text{M}$ output LM head) |
| **Attention Kernel** | **SDPA** | PyTorch native `scaled_dot_product_attention` |
| **Context Length** | **`1,024`** | Native training context length (extensible to `2,048`) |

### Parameter Audit Breakdown
Run `python scripts/print_params.py` to inspect the exact tensor parameter allocation:

```text
================================================================================
COMPONENT BREAKDOWN:
  Token Embedding (V=50257, d=1024):       51,463,168 (10.29%)
  LM Head (Untied, d=1024, V=50257):       51,463,168 (10.29%)
  RoPE Positional Encoding:                         0  (0.00%)
  Attention Layers (30 layers):            78,643,200 (15.72%)
    - W_q Proj (1024 -> 1024):             31,457,280
    - W_k Proj (1024 -> 256):               7,864,320
    - W_v Proj (1024 -> 256):               7,864,320
    - W_o Proj (1024 -> 1024):             31,457,280
  SwiGLU MLP Layers (30 layers):          318,504,960 (63.68%)
    - W_gate Proj (1024 -> 3456):         106,168,320
    - W_up Proj   (1024 -> 3456):         106,168,320
    - W_down Proj (3456 -> 1024):         106,168,320
  RMSNorm Layers (61 norms):                   62,464  (0.01%)
--------------------------------------------------------------------------------
TOTAL ACTUAL PARAMETERS:                  500,136,960 (100.00%)
================================================================================
```

---

## 2. Note on Hugging Face Ecosystem Compatibility

> [!NOTE]
> **Independent Custom Architecture**: This model was written, pretrained, and aligned from scratch by **Mohamed Usaid** using pure PyTorch (`model/transformer.py`).
>
> When exporting the model weights to the Hugging Face `transformers` format (`export/export_hf.py`), the model metadata in `config.json` is set to standard `"architectures": ["LlamaForCausalLM"]`.
> 
> **Why?** Modern frontier architectures (RoPE, GQA, SwiGLU, RMSNorm without bias) share the exact structural tensor layout of the open LLaMA format. Utilizing this standard specification allows **Usaid AI (500M)** to be loaded directly by any library in the AI ecosystem—including Hugging Face `AutoModelForCausalLM`, vLLM, Ollama, Gradio ZeroGPU, and TGI—out-of-the-box without requiring custom untrusted scripts (`trust_remote_code=True`).

---

## 3. Empirical Training & Convergence Results

### Pretraining Phase (Foundational Base Model)
- **Hardware**: Kaggle Dual Tesla T4 GPUs ($2 \times 16\text{ GB} = 32\text{ GB}$ GDDR6)
- **Distributed Strategy**: PyTorch DistributedDataParallel (DDP, `torchrun`, `world_size=2`)
- **Optimizer**: CUDA Fused AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.1$ with 1D bias/norm exclusion)
- **Learning Rate Schedule**: Cosine Annealing with Linear Warmup ($1.0 \times 10^{-6} \to 3.0 \times 10^{-4} \to 3.0 \times 10^{-5}$)
- **Batch Geometry**: Micro-batch $2 \times$ Grad Accum $16 \times 1024$ seq len $\times 2$ GPUs = **$65,536\text{ tokens / step}$**
- **Tokens Ingested**: **$131,072,000$ tokens** (~131.07 Million tokens across 2,000 steps)
- **Loss Progression**:
  - Step 0: Initial loss $> 10.5$ (random initialization)
  - Step 800 (40%): Train Loss: `3.6849` | Val Loss: `3.5274` | Val PPL: `34.04`
  - Step 1600 (80%): Train Loss: `3.1250` | Val Loss: `3.0410` | Val PPL: `20.93`
  - **Step 2000 (100%)**: Train Loss: `2.9623` | **Val Loss: `2.9091`** | **Val PPL: `18.34`**
- **Artifact**: `checkpoints/best_tinygpt_500m.pt` (1.86 GB model weights)

### Supervised Fine-Tuning Phase (SFT Alignment)
- **Base Model**: `checkpoints/best_tinygpt_500m.pt`
- **Dataset**: `data/sft_instructions.jsonl` ($4,731$ curated instruction-response pairs)
- **Tokens Backpropagated**: $\approx 28.9\text{ Million tokens}$ across 3 epochs (441 optimizer steps)
- **Supervision Technique**: Strict **Prompt Loss Masking** (tokens corresponding to `User: ...` are labeled with `target = -100` so gradient updates occur exclusively on assistant responses)
- **Loss Progression**:
  - Step 1: Initial SFT loss `2.9302`
  - Step 147 (Epoch 1): SFT loss `2.5299`
  - Step 294 (Epoch 2): SFT loss `2.2036`
  - **Step 384**: **Best SFT Loss `1.3938`** (**>52% drop from baseline**)
  - Final Step (Step 441): SFT loss `1.4715`
- **Artifact**: `sft_checkpoints/usaid_ai_500m.pt`

### Zero-Shot Multi-Domain Diagnostic Benchmark
Evaluated using length-normalized completion log-likelihood over 4 candidate choices:

| Domain Category | Pre-SFT Base Model (`best_tinygpt_500m.pt`) | Post-SFT Aligned Model (`usaid_ai_500m.pt`) | Empirical Delta |
| :--- | :---: | :---: | :---: |
| **ML & Transformer Architecture** | $2 / 5$ (40.0%) | **$3 / 5$ (60.0%)** | **+20.0%** |
| **World Knowledge & Science** | $2 / 5$ (40.0%) | **$3 / 5$ (60.0%)** | **+20.0%** |
| **Python & Software Engineering** | $1 / 5$ (20.0%) | $1 / 5$ (20.0%) | Baseline |
| **Logic & Arithmetic** | $1 / 5$ (20.0%) | $1 / 5$ (20.0%) | Baseline |
| **OVERALL ACCURACY** | **$6 / 20$ (30.0%)** | **$8 / 20$ (40.0%)** | **+10.0%** |

---

## 4. Comprehensive Usage Guide

### Prerequisites
Clone the repository and install the dependencies:
```bash
git clone https://github.com/mohamedusaid/TinyGPT.git
cd TinyGPT
pip install -r requirements.txt
```

---

### Mode 1: Continuous Multi-Turn Chat Console (SFT Model)
Run a full conversational terminal session with live token streaming and multi-turn context memory:

```bash
# Launch interactive chat (automatically detects usaid_ai_500m.pt)
python scripts/run_chat_loop.py --checkpoint sft_checkpoints/usaid_ai_500m.pt

# Tune generation creativity
python scripts/run_chat_loop.py --checkpoint sft_checkpoints/usaid_ai_500m.pt --temperature 0.6 --top_p 0.9
```

**In-Session Console Commands:**
- Type any message and press `[Enter]` to converse.
- `clear` or `reset` — Wipe session history and restart conversation context.
- `rag on` / `rag off` — Toggle RAG dynamic knowledge grounding on/off in real time.
- `exit` or `quit` — Exit the chat console.

---

### Mode 2: Single-Prompt Generation
Generate completions for a single prompt directly from the CLI:

```bash
# Test SFT conversational model:
python scripts/run_chat.py --checkpoint sft_checkpoints/usaid_ai_500m.pt --prompt "Who are you and who created you?"

# Test Base Pretrained Model (raw statistical text continuation):
python scripts/run_chat.py --checkpoint checkpoints/best_tinygpt_500m.pt --prompt "The theory of general relativity states that"
```

---

### Mode 3: RAG Knowledge Grounded Chat
Ground responses in factual knowledge documents from `knowledge_base/` using BM25 retrieval:

```bash
# Interactive RAG conversation session:
python scripts/run_rag_chat.py --checkpoint sft_checkpoints/usaid_ai_500m.pt

# Single query with RAG retrieval:
python scripts/run_rag_chat.py --checkpoint sft_checkpoints/usaid_ai_500m.pt --prompt "What is Usaid AI and what architecture does it use?" --top_k 2
```

---

### Mode 4: Top-5 Token Probability Explorer
Step through autoregressive generation one token at a time. Inspect the model's internal probability distribution, analyze token entropy, and explore counterfactual branching paths:

```bash
# Interactive loop (type prompts freely)
python scripts/run_token_probs.py

# Inspect a specific prompt
python scripts/run_token_probs.py --prompt "What is machine learning?"

# Greedy decoding with raw prompt (no chat template wrapper)
python scripts/run_token_probs.py --prompt "The capital of France is" --raw --temperature 0

# Auto-run without pausing, exporting full step data to JSON
python scripts/run_token_probs.py --prompt "Hello" --auto --save_json probs.json
```

**Interactive Controls at Each Step:**
- `Enter` — Accept the top sampled candidate and proceed to the next token.
- `1` .. `5` — Force alternative candidate #N (explore "what if" model branches).
- `a` — Auto-run the rest of the generation (continues displaying probabilities).
- `q` — Halt generation for the current prompt.

*Files involved: `scripts/run_token_probs.py`, `inference/token_probs.py`, `inference/model_loader.py`.*

---

### Mode 5: 3D/2D Embedding Projector (TensorFlow Projector Style)
Extract the model's 1024-dimensional token embedding space (`lm_head.weight` or `embed_tokens.weight`), perform dimensionality reduction (PCA, t-SNE, UMAP), compute cosine nearest neighbors, and explore token semantic clusters:

```bash
# Generate the visualization and automatically launch it in your browser:
python scripts/visualize_embeddings.py --open

# Project specific focal words and export TSVs for projector.tensorflow.org:
python scripts/visualize_embeddings.py --words "king,queen,man,woman,doctor,python" --export_tsv embedding_viz/tsv

# Fast projection (PCA only, 4,000 tokens):
python scripts/visualize_embeddings.py --num_tokens 4000 --methods pca
```

**Key Deliverables:**
- **Interactive Web App**: Generates a self-contained HTML page at `embedding_viz/embedding_projector.html`. Open it in any browser for full 3D orbital camera exploration, search, and nearest neighbor inspection.
- **TensorFlow Projector TSVs**: Creates `tensors.tsv` and `metadata.tsv` ready to load at [projector.tensorflow.org](https://projector.tensorflow.org).

*Files involved: `scripts/visualize_embeddings.py`, `visualization/projector_template.html`, `inference/model_loader.py`.*

---

### Mode 6: Using Hugging Face `transformers`

You can load and query the published model weights directly using Hugging Face:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Usaidddddddddddddd/UsaidAI-500M"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# SFT Conversational Template
prompt = "User: Who created you and what is your architecture?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response.strip())
```

---

### Mode 7: Training & Export Pipeline

#### 1. Hardware Verification
```bash
python scripts/check_env.py
```

#### 2. Shard Dataset Preparation
Tokenize text documents into memory-mapped uint16 `.bin` binary shards:
```bash
python data/prepare_data.py --output_dir data_shards --shard_size 1000000
```

#### 3. Pretraining
```bash
# Smoke test on CPU:
python scripts/run_pretrain.py --dry_run

# Full pretraining run:
python scripts/run_pretrain.py --data_dir data_shards

# Resume from checkpoint:
python scripts/run_pretrain.py --resume checkpoints/checkpoint_step_000500.pt
```

#### 4. Supervised Fine-Tuning (SFT)
```bash
python scripts/run_sft.py --base_model checkpoints/best_tinygpt_500m.pt --data_path data/sft_instructions.jsonl
```

#### 5. Benchmark Evaluation
```bash
# Evaluate Base Model:
python scripts/run_eval.py --checkpoint checkpoints/best_tinygpt_500m.pt

# Evaluate SFT Aligned Model:
python scripts/run_eval.py --checkpoint sft_checkpoints/usaid_ai_500m.pt
```

#### 6. Export to Hugging Face Safetensors
Export raw PyTorch checkpoint into Hugging Face `model.safetensors` + `config.json`:
```bash
python export/export_hf.py --checkpoint sft_checkpoints/usaid_ai_500m.pt --output_dir exported_usaid_ai_aligned
```

---

## 5. Repository Structure

```text
tiny-gpt-500m/
├── configs/
│   ├── model_500m.json                 # 500M architecture hyperparameters
│   ├── train_pretrain.json             # Pretraining settings (LR, warmup, accum)
│   └── train_sft.json                  # SFT instruction tuning settings
│
├── model/                              # Pure PyTorch Model (From-Scratch)
│   ├── config.py                       # Dataclass config & parameter math
│   ├── rms_norm.py                     # Pre-norm RMSNorm layer (zero-mean, no bias)
│   ├── rotary.py                       # Vectorized RoPE with precomputed cos/sin cache
│   ├── attention.py                    # Grouped-Query Attention (GQA 4:1) + SDPA
│   ├── mlp.py                          # SwiGLU 3-matrix Feed-Forward Network
│   ├── transformer.py                  # TransformerBlock & full TinyGPT500M model
│   └── kv_cache.py                     # Dynamic KV Cache for autoregressive decoding
│
├── data/                               # Tokenization & Binary Streaming
│   ├── tokenizer.py                    # GPT-2 tokenizer wrapper (tiktoken 50,257 vocab)
│   ├── dataset.py                      # Memory-mapped uint16 binary dataset (.bin)
│   ├── dataloader.py                   # Distributed batch sampler
│   └── prepare_data.py                 # Sharding & document packing pipeline
│
├── training/                           # Pretraining Engine
│   ├── trainer.py                      # Training loop (AMP, GradScaler, clip, DDP)
│   ├── scheduler.py                    # Cosine Annealing with Linear Warmup
│   ├── optim.py                        # Fused AdamW with 1D weight decay exclusion
│   └── checkpoint.py                   # Checkpointing (weights, optimizer, RNG state)
│
├── sft/                                # Supervised Fine-Tuning
│   ├── sft_dataset.py                  # Instruction dataset with -100 prompt masking
│   └── train_sft.py                    # SFT alignment engine
│
├── rag/                                # Retrieval-Augmented Generation
│   ├── knowledge_base.py               # Document parser & chunker
│   ├── retriever.py                    # BM25 semantic retriever
│   └── pipeline.py                     # Context formatting & grounding pipeline
│
├── inference/                          # Inference & Interpretability Core
│   ├── generate.py                     # KV-cached sampler (temp, top-k, top-p, rep penalty)
│   ├── chat.py                         # Single-turn interactive chat streamer
│   ├── model_loader.py                 # Unified checkpoint resolver & mmap loader
│   └── token_probs.py                  # Top-k candidate stepper & probability tracker
│
├── visualization/                      # Interpretability Visualization Templates
│   └── projector_template.html         # Standalone 3D/2D embedding projector template
│
├── embedding_viz/                      # Generated Embedding Projector Artifacts
│   ├── embedding_projector.html        # Interactive 3D embedding browser
│   └── tsv/                            # TSV exports for projector.tensorflow.org
│
├── hf_space_chat/                      # Hugging Face Space App
│   ├── app.py                          # ZeroGPU Gradio chat interface
│   ├── requirements.txt                # Space dependencies
│   └── README.md                       # Space card metadata
│
├── eval/                               # Evaluation & Diagnostics
│   ├── perplexity.py                   # Validation loss & token perplexity
│   └── benchmarks.py                   # Zero-shot 4-choice benchmark suite
│
├── export/                             # Ecosystem Export
│   ├── export_hf.py                    # Converts PyTorch .pt to HF safetensors
│   └── export_gguf.py                  # GGUF / Ollama conversion recipe
│
├── scripts/                            # CLI Executables
│   ├── check_env.py                    # Hardware & precision probe
│   ├── print_params.py                 # Parameter breakdown audit
│   ├── run_pretrain.py                 # Pretraining runner
│   ├── run_sft.py                      # SFT alignment runner
│   ├── run_chat.py                     # Single prompt generation
│   ├── run_chat_loop.py                # Multi-turn continuous chat console
│   ├── run_rag_chat.py                 # Interactive RAG chat CLI
│   ├── run_token_probs.py              # Top-5 token probability explorer
│   ├── visualize_embeddings.py         # Embedding projector generator
│   └── run_eval.py                     # Multi-domain benchmark evaluation
│
├── knowledge_base/                     # Grounding documents for RAG
├── checkpoints/                        # Pretrained model checkpoints
├── sft_checkpoints/                    # Aligned model checkpoints
├── exported_usaid_ai_aligned/          # Safetensors export directory
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 6. Author & Citation

**Architected and Built by:** Mohamed Usaid  
- GitHub: [@mohamedusaid](https://github.com/mohamedusaid)  
- Hugging Face: [@Usaidddddddddddddd](https://huggingface.co/Usaidddddddddddddd)  
- Project Repository: [TinyGPT](https://github.com/mohamedusaid/TinyGPT)

If you find this codebase or model useful in your research or learning journey, please star ⭐ the repository!
