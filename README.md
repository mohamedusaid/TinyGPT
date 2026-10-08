# TinyGPT-15.3M (Scaling Up Prototype)

<div align="center">

**A Scaled-Up Decoder-Only Transformer Architecture (~15.3M Parameters)**  
*Engineered by **Mohamed Usaid***

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-15.30M-green.svg)](#1-architectural-specifications)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Tesla%20T4-green.svg)]()
[![Branch](https://img.shields.io/badge/Branch-scaling--up-purple.svg)]()

</div>

---

## 🌟 Overview

**TinyGPT-15.3M** represents the second milestone in the TinyGPT development journey. It scales the foundational 6.84M prototype by **2.24x**, expanding the hidden representation dimension, attention head count, layer depth, and context window.

This prototype served as an empirical research baseline to evaluate training dynamics, context scaling, loss convergence inflection points, and dataset memorization thresholds before transitioning to modern frontier architectures at the 500M scale.

---

## 1. Architectural Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **`15,301,120`** | Exact parameter count (~15.3 Million) |
| **Vocabulary Size (V)** | **`50,257`** | Byte-Pair Encoding (`tiktoken` / GPT-2 BPE) |
| **Context Length** | **`256`** | Expanded sequence length (4x larger than 6.84M prototype) |
| **Hidden Size (d_model)** | **`256`** | Feature dimension (2x larger than 6.84M prototype) |
| **Transformer Layers (L)** | **`3`** | Hierarchical layer depth |
| **Attention Heads (H)** | **`8`** | Standard Multi-Head Attention (head dimension: 32) |
| **Intermediate Size (d_ff)** | **`1,024`** | 2-layer GELU Feed-Forward Network (4 x d_model) |
| **Positional Encoding** | **Learned** | Absolute positional embedding table (256 tokens) |
| **Normalization** | **LayerNorm** | Pre-norm `nn.LayerNorm` with learnable bias |
| **LM Head Tying** | **True** | Output projection tied to input token embeddings |

### Exact Parameter Allocation
- **Token Embeddings** (50,257 x 256): `12,865,792` (84.08%)
- **Positional Embeddings** (256 x 256): `65,536` (0.43%)
- **Transformer Layers** (3 blocks): `2,369,280` (15.48%)
  - *Per Layer:* LayerNorm 1: `512`, QKV Proj: `196,608`, Out Proj: `65,792`, LayerNorm 2: `512`, FFN 1: `263,168`, FFN 2: `262,656`
- **Final LayerNorm** (weight + bias): `512` (0.00%)
- **LM Head** (Weight-Tied): `0` (Shared with input embeddings)
- **Total Parameters:** **`15,301,120`** (100.00%)

---

## 2. Empirical Training & Verification

- **Compute Hardware:** Cloud GPU Environment (NVIDIA Tesla T4, 15.0 GiB VRAM, CUDA 12.8)
- **Training Throughput:** 17.7 seconds / 100 steps (**~23,140 tokens / second**)
- **Batch Geometry:** Batch Size 16 x 256 context = **4,096 tokens / step**
- **Optimizer:** AdamW (Learning Rate: 3e-4, weight_decay = 0.01)
- **Total Tokens Seen:** 5,000 steps x 4,096 tokens = **20,480,000 tokens** across 130,798 unique training tokens (~156.5 epochs)
- **Convergence Progression (5,000 Steps, 14.03 Minutes):**
  - Step 0: Train Loss `10.5574` | Val Loss `10.5697`
  - Step 250: Train Loss `4.7624` | Val Loss `5.8446`
  - Step 500: Train Loss `3.5097` | Val Loss `5.6621`
  - **Step 750 (Optimal Validation Inflection Point):** **Train Loss `2.5365` | Val Loss `5.6054`** (~23.5 epochs)
  - Step 1000: Train Loss `1.7372` | Val Loss `5.6134`
  - Step 1250: Train Loss `1.1201` | Val Loss `6.1955` *(Generalization boundary)*
  - Step 1500: Train Loss `0.7519` | Val Loss `6.5557` *(Overfitting begins)*
  - Step 2000: Train Loss `0.2984` | Val Loss `6.6914`
  - Step 3000: Train Loss `0.1205` | Val Loss `7.1991`
  - Step 4000: Train Loss `0.0835` | Val Loss `7.7507`
  - Step 4999: Train Loss `0.0757` | Val Loss `7.9367` *(Memorization drift: +2.33 val loss increase)*
- **Saved Checkpoint:** `tiny_gpt_qa_v2.pt` (Restored optimal state from Step 750 weights)

### Key Research Learning:
Training a 15.3M parameter model on a ~130k token corpus demonstrated that validation loss reaches its global minimum at Step 750 (~23 epochs) before memorization occurs. This proved that scaling to 500M parameters required continuous large-scale streaming datasets (130M+ unique tokens) and modern regularizations (RoPE, GQA, RMSNorm, SwiGLU).

For the complete step-by-step telemetry, 5 instruction probe evaluations, and scaling comparison matrix, see [**`Result_documentation.txt`**](Result_documentation.txt).

---

## 3. File Structure & Checkpoints

```text
tiny-gpt/
├── model.py                    # Scaled-up model definition, tokenizer, and 5000-step training loop
├── ask.py                      # Interactive text generation and story-retrieval QA CLI
├── content.txt                 # General training text corpus
├── qa_content.txt              # Question-Answer pairs dataset
├── fetch_imdb.py               # Dataset ingestion utility for IMDb reviews
├── append_stem_text.py         # Dataset expansion script for STEM texts
├── tokenizer_test.py           # BPE tokenizer verification
├── Result_documentation.txt    # Complete 15.3M telemetry log, probes & scaling audit
├── LICENSE                     # Apache 2.0 license
│
├── tiny_gpt.pt                 # Baseline weights from initial prototype (~27.4 MB)
├── tiny_gpt_scaling_up.pt      # Scaled-up base checkpoint (~61.3 MB)
├── tiny_gpt_scaling-up_qa.pt   # Checkpoint trained with QA content (~61.3 MB)
└── tiny_gpt_qa_v2.pt           # Refined v2 QA checkpoint restored from Step 750 (~61.3 MB)
```

---

## 4. How to Run

### Prerequisites
```bash
pip install torch tiktoken
```

### 1. Train the 15.3M Model
```bash
python model.py
```
*Trains on combined general and QA text, saving optimal weights to `tiny_gpt_qa_v2.pt`.*

### 2. Interactive QA & Generation
```bash
python ask.py
```

### 3. Expand Training Data
```bash
python fetch_imdb.py
python append_stem_text.py
```

---

## 5. The TinyGPT Scaling Journey

1. [**`main` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/main): **TinyGPT (6.84M)** — 2 layers, 128 dim, 4 heads, 64 context (Intel Core i7 CPU).
2. **`scaling-up` (This Branch):** **TinyGPT-15.3M** — 3 layers, 256 dim, 8 heads, 256 context (Cloud GPU NVIDIA Tesla T4).
3. [**`scaling-up-500m` Branch**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up-500m): [**✨ Usaid AI (500.14M)**](https://github.com/mohamedusaid/TinyGPT/tree/scaling-up-500m) — Modern frontier architecture (RoPE, GQA, SwiGLU, RMSNorm, DDP pretraining on 131M tokens, SFT alignment, and RAG).

---

## 📜 License

Licensed under the Apache License, Version 2.0. Copyright 2026 Mohamed Usaid.
