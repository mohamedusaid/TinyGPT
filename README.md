# TinyGPT (Foundational Prototype)

<div align="center">

**A Minimal From-Scratch Causal Language Model (~6.84M Parameters)**  
*Engineered by **Mohamed Usaid***

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-6.84M-green.svg)](#1-architectural-specifications)
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20Core%20i7%20(CPU)-orange.svg)]()
[![Branch](https://img.shields.io/badge/Branch-main-purple.svg)]()

</div>

---

## 🌟 Overview

**TinyGPT (~6.84M)** is the foundational prototype in the TinyGPT series. Designed to demystify large language model mechanics, it implements an autoregressive decoder-only Transformer entirely from scratch in PyTorch without high-level wrapper libraries.

This prototype was built for rapid CPU experimentation, educational clarity, and validating the fundamental mechanics of self-attention, tokenization, and causal language modeling before scaling to larger architectures.

---

## 1. Architectural Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **`6,837,120`** | ~6.84 Million parameters |
| **Vocabulary Size (V)** | **`50,257`** | Byte-Pair Encoding (`tiktoken` / GPT-2 BPE) |
| **Context Length** | **`64`** | Maximum sequence length |
| **Hidden Size (d_model)** | **`128`** | Feature dimension |
| **Transformer Layers (L)** | **`2`** | Number of stacked Transformer blocks |
| **Attention Heads (H)** | **`4`** | Standard Multi-Head Attention (head dimension: 32) |
| **Intermediate Size (d_ff)** | **`512`** | 2-layer GELU Feed-Forward Network (4 x d_model) |
| **Positional Encoding** | **Learned** | Absolute positional embedding table (64 tokens) |
| **Normalization** | **LayerNorm** | Pre-norm `nn.LayerNorm` with learnable bias |
| **LM Head Tying** | **True** | Output projection tied to input token embeddings |

### Exact Parameter Allocation
- **Token Embeddings** (50,257 x 128): `6,432,896` (94.09%)
- **Positional Embeddings** (64 x 128): `8,192` (0.12%)
- **Transformer Layers** (2 blocks): `395,776` (5.79%)
  - *Per Layer:* LayerNorm 1: `256`, QKV Proj: `49,152`, Out Proj: `16,512`, LayerNorm 2: `256`, FFN 1: `66,048`, FFN 2: `65,664`
- **Final LayerNorm** (weight + bias): `256` (0.00%)
- **LM Head** (Weight-Tied): `0` (Shared with input embeddings)
- **Total Parameters:** **`6,837,120`** (100.00%)

---

## 2. Empirical Training & Verification

- **Compute Hardware:** Intel Core i7 CPU (Windows 11)
- **Dataset:** `content.txt` (1,072 tokens: 964 train / 108 val)
- **Batch Geometry:** Batch Size 8 x 64 context = **512 tokens / step**
- **Optimizer:** AdamW (lr = 3e-4, weight_decay = 0.01, grad_clip = 1.0)
- **Progression Telemetry (3,000 Steps):**
  - Step 0: Train Loss `10.7542` | Val Loss `10.7268` *(Matches theoretical uniform random baseline ln(50257) = 10.82)*
  - Step 100: Train Loss `5.6421` | Val Loss `7.8928`
  - **Step 200 (Optimal Checkpoint):** **Train Loss `4.0445` | Val Loss `7.5187`**
  - Step 1000: Train Loss `0.0505` | Val Loss `8.5753`
  - Step 3000: Train Loss `0.0249` | Val Loss `9.0916` *(Memorization drift on small corpus)*
- **Saved Model Checkpoint:** `tiny_gpt.pt` (6,837,120 parameters)

For the full step-by-step tensor audit, bug resolutions, and RAG evaluation probes, see [**`Result_documentation.txt`**](Result_documentation.txt).

---

## 3. File Structure

```text
tiny-gpt/
├── model.py                    # From-scratch PyTorch model, tokenizer, and training loop
├── ask.py                      # Interactive text generation and retrieval-assisted QA CLI
├── content.txt                 # Training corpus text (1,072 tokens)
├── tiny_gpt.pt                 # Saved model checkpoint weights (~27.4 MB)
├── tokenizer_test.py           # Tokenizer verification test
├── Result_documentation.txt    # Complete experimental log and tensor mechanics audit
├── LICENSE                     # Apache 2.0 license
└── .gitignore                  # Git ignore configuration
```

---

## 4. How to Run

### Prerequisites
```bash
pip install torch tiktoken
```

### 1. Train the Model
Train the 6.84M prototype directly:
```bash
python model.py
```
*Loads `content.txt`, tokenizes text via `tiktoken`, runs the training loop, and saves weights to `tiny_gpt.pt`.*

### 2. Generate Text / Interactive QA
Prompt the model or ask questions based on the ingested content:
```bash
python ask.py
```

### 3. Verify Tokenizer
Check Byte-Pair Encoding functionality:
```bash
python tokenizer_test.py
```

---

## 5. The TinyGPT Scaling Journey

This repository represents the initial step in a 3-stage scaling progression:

1. **`main` (This Branch):** **TinyGPT (6.84M)** — 2 layers, 128 dim, 4 heads, 64 context.
2. [**`scaling-up` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up): **TinyGPT-15.3M** — 3 layers, 256 dim, 8 heads, 256 context.
3. [**`scaling-up-500m` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up-500m): **✨ Usaid AI (500.14M)** — Modern frontier architecture (RoPE, GQA, SwiGLU, RMSNorm, DDP pretraining on 131M tokens, SFT alignment, and RAG).

---

## 📜 License

Licensed under the Apache License, Version 2.0. Copyright 2026 Mohamed Usaid.
