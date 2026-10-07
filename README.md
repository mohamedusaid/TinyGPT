# TinyGPT (Foundational Prototype)

<div align="center">

**A Minimal From-Scratch Causal Language Model (~6.84M Parameters)**  
*Engineered by **Mohamed Usaid***

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-6.84M-green.svg)](#1-architectural-specifications)
[![Branch](https://img.shields.io/badge/Branch-main-purple.svg)]()

</div>

---

## 🌟 Overview

**TinyGPT (~6.84M)** is the initial foundational prototype in the TinyGPT series. Designed to demystify large language model mechanics, it implements an autoregressive decoder-only Transformer entirely from scratch in PyTorch without high-level wrapper libraries.

This prototype was built for rapid CPU/GPU experimentation, educational clarity, and validating the fundamental mechanics of self-attention, tokenization, and causal language modeling before scaling to larger architectures.

---

## 1. Architectural Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **`6,837,120`** | ~6.84 Million parameters |
| **Vocabulary Size ($V$)** | **`50,257`** | Byte-Pair Encoding (`tiktoken` / GPT-2 BPE) |
| **Context Length** | **`64`** | Maximum sequence length |
| **Hidden Size ($d_{\text{model}}$)** | **`128`** | Feature dimension |
| **Transformer Layers ($L$)** | **`2`** | Number of stacked Transformer blocks |
| **Attention Heads ($H$)** | **`4`** | Standard Multi-Head Attention (head dimension: 32) |
| **Intermediate Size ($d_{\text{ff}}$)** | **`512`** | 2-layer GELU Feed-Forward Network ($4 \times d_{\text{model}}$) |
| **Positional Encoding** | **Learned** | Absolute positional embedding table (64 tokens) |
| **Normalization** | **LayerNorm** | Pre-norm `nn.LayerNorm` with learnable bias |
| **LM Head Tying** | **True** | Output projection tied to input token embeddings |

### Parameter Allocation
- **Token Embeddings** ($50,257 \times 128$): `6,432,896` (94.09%)
- **Positional Embeddings** ($64 \times 128$): `8,192` (0.12%)
- **Attention Layers** (2 blocks): `98,816` (1.45%)
- **FFN Layers** (2 blocks): `263,168` (3.85%)
- **LayerNorm Layers** (5 norms): `1,280` (0.02%)
- **Output Head** (Weight-Tied): `0` (Shared with input embeddings)
- **Total Parameters:** **`6,837,120`** (100.00%)

---

## 2. File Structure

```text
tiny-gpt/
├── model.py            # Complete from-scratch PyTorch model, tokenizer, and training loop
├── ask.py              # Interactive text generation and retrieval-assisted QA CLI
├── content.txt         # Training corpus text
├── tiny_gpt.pt         # Saved model checkpoint weights (~27.4 MB)
├── tokenizer_test.py   # Tokenizer verification test
└── .gitignore          # Git ignore configuration
```

---

## 3. How to Run

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

## 4. The TinyGPT Scaling Journey

This repository represents the initial step in a 3-stage scaling progression:

1. **`main` (This Branch):** **TinyGPT (6.84M)** — 2 layers, 128 dim, 4 heads, 64 context.
2. [**`scaling-up` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up): **TinyGPT-15.3M** — 3 layers, 256 dim, 8 heads, 256 context.
3. [**`scaling-up-500m` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up-500m): **Usaid AI (500.14M)** — Modern frontier architecture (RoPE, GQA, SwiGLU, RMSNorm, DDP pretraining on 131M tokens, SFT alignment, and RAG).

---

## 📜 License

Licensed under the Apache License, Version 2.0. Copyright 2026 Mohamed Usaid.
