# TinyGPT-500M

**TinyGPT-500M** is a ~500M parameter Causal Language Model designed to bridge deep, pedagogical from-scratch clarity with modern frontier MNC architecture (RoPE, GQA, SwiGLU, RMSNorm, PyTorch native SDPA).

It scales the foundational concepts from the earlier 6M/11M/15M TinyGPT prototypes into a half-billion parameter system engineered to train within the 16 GB VRAM budget of a single Google Colab Tesla T4 GPU while remaining seamlessly scalable to multi-GPU clusters.

---

## 1. Architectural Specification

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Target Parameters** | **$500,000,000$** | Goal scale |
| **Actual Parameters** | **`500,136,960`** | Exact parameter count (**$+0.027\%$ diff**) |
| **Vocabulary Size ($V$)** | **`50,257`** | Standard GPT-2 Byte-Pair Encoding (`tiktoken`) |
| **Hidden Size ($d_{\text{model}}$)** | **`1,024`** | Power-of-2 dimension ($2^{10}$) for Tensor Core efficiency |
| **Transformer Layers ($L$)** | **`30`** | Deep hierarchical reasoning depth |
| **Query Attention Heads ($H_q$)** | **`16`** | Head dimension $d_{\text{head}} = 64$ |
| **Key/Value Heads ($H_{kv}$)** | **`4`** | **Grouped-Query Attention (GQA 4:1 ratio)** |
| **Intermediate Size ($d_{ff}$)** | **`3,456`** | **SwiGLU** 3-matrix FFN ($27 \times 128 = 3,456 \approx 3.375 \times d_{\text{model}}$) |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings ($\theta = 10,000$, 0 parameters) |
| **Normalization** | **RMSNorm** | Pre-norm with learnable scale $\gamma$ (zero-mean, no bias) |
| **LM Head Tying** | **False** | Untied embeddings ($51.46\text{M}$ embed + $51.46\text{M}$ LM head) |
| **Attention Kernel** | **SDPA** | PyTorch native `scaled_dot_product_attention` (auto-backend) |
| **Context Length** | **`1,024`** | Extensible up to `2,048` |

### Parameter Audit Breakdown
Run `python scripts/print_params.py` to view the exact component audit:
```text
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
```

---

## 2. Google Colab Tesla T4 (16 GB VRAM) Optimization

* **Mixed Precision (AMP)**: Automatically detects hardware:
  * Tesla T4 (`sm_75`): Uses **FP16 + GradScaler** (Turing has fast FP16 Tensor Cores).
  * Ampere / Hopper (`sm_80+`): Uses **BF16** without scaler.
  * CPU: Uses **Float32**.
* **Memory Budget**:
  * FP16 weights: $\approx 1.0\text{ GB}$
  * AdamW optimizer (FP32 states + master weights) + Gradients: $\approx 7.0 - 8.0\text{ GB}$
  * Available headroom for activations: $\approx 7.0\text{ GB}$
* **Batch Geometry for T4**:
  * Micro-batch size: `2` sequences of length `1,024` ($2,048$ tokens/micro-batch).
  * Gradient accumulation steps: `32`
  * Effective batch size: **$65,536\text{ tokens / step}$**.
  * Peak VRAM: $\approx 10.5\text{ GB}$ (safely within 16 GB).
  * Step Time: $\approx 10.5\text{ seconds}$ per step ($\approx 6,240\text{ tokens/sec}$).

---

## 3. Project Structure

```text
tiny-gpt-500m/
├── configs/
│   ├── model_500m.json           # 500M architecture hyperparameters
│   ├── train_pretrain.json       # Pretraining hyperparameters (LR, warmup, accum)
│   └── train_sft.json            # SFT instruction tuning hyperparameters
│
├── model/                        # Core PyTorch model (clean, from-scratch)
│   ├── __init__.py
│   ├── config.py                 # Dataclass config & parameter breakdown logic
│   ├── rms_norm.py               # Pre-norm RMSNorm layer
│   ├── rotary.py                 # Vectorized RoPE with precomputed cos/sin cache
│   ├── attention.py              # Grouped-Query Attention (GQA 4:1) + native SDPA
│   ├── mlp.py                    # SwiGLU 3-matrix Feed-Forward Network
│   ├── transformer.py            # TransformerBlock & full TinyGPT500M model
│   └── kv_cache.py               # KV Cache for fast autoregressive decoding
│
├── data/                         # Tokenization & streaming binary datasets
│   ├── __init__.py
│   ├── tokenizer.py              # GPT-2 tokenizer wrapper (tiktoken 50,257 vocab)
│   ├── dataset.py                # Memory-mapped uint16 binary dataset (.bin)
│   ├── dataloader.py             # Batch sampler
│   └── prepare_data.py           # Sharding & document packing pipeline
│
├── training/                     # Pretraining infrastructure
│   ├── __init__.py
│   ├── trainer.py                # Pretraining loop (AMP, GradScaler, accum, clip, eval)
│   ├── scheduler.py              # Cosine Annealing with Linear Warmup
│   ├── optim.py                  # AdamW with 1D weight decay exclusion (biases/norms)
│   └── checkpoint.py             # Resumable checkpointing (weights, optim, scaler, RNG)
│
├── inference/                    # Autoregressive generation & CLI
│   ├── __init__.py
│   ├── generate.py               # KV-cached sampler (temperature, top-k, top-p)
│   └── chat.py                   # Interactive terminal streaming generation
│
├── sft/                          # Supervised Fine-Tuning (Alignment)
│   ├── __init__.py
│   ├── sft_dataset.py            # SFT dataset with prompt loss masking (-100)
│   └── train_sft.py              # SFT instruction tuning engine
│
├── eval/                         # Evaluation & benchmarks
│   ├── __init__.py
│   ├── perplexity.py             # Validation loss & token perplexity
│   └── benchmarks.py             # Zero-shot multiple choice reasoning evaluation
│
├── export/                       # Ecosystem export
│   ├── __init__.py
│   ├── export_hf.py              # Converts checkpoint to Hugging Face Transformers
│   └── export_gguf.py            # GGUF / Ollama conversion recipe
│
├── tests/                        # Automated unit tests
│   ├── test_model.py             # Exact parameter count, shapes, backward pass
│   ├── test_kv_cache.py          # Verifies KV cache numerical identity
│   └── test_tokenizer.py         # Verifies tokenization and binary datasets
│
├── scripts/                      # CLI entrypoints
│   ├── check_env.py              # Hardware & precision probe
│   ├── print_params.py           # Parameter audit script
│   ├── run_pretrain.py           # Pretraining runner
│   ├── run_sft.py                # SFT runner
│   ├── run_chat.py               # Generation / Chat runner
│   └── run_eval.py               # Benchmark evaluation runner
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 4. Quickstart Guide

### 1. Hardware Probe
Verify your environment and detected GPU precision:
```bash
python scripts/check_env.py
```

### 2. Verify Parameter Audit
Verify the exact 500M parameter allocation:
```bash
python scripts/print_params.py
```

### 3. Run Unit Tests
Run the automated test suite:
```bash
python -m unittest discover tests
```

### 4. Prepare Training Shards
Tokenize text documents into memory-mapped `.bin` shards:
```bash
python data/prepare_data.py --output_dir data_shards --shard_size 1000000
```

### 5. Launch Pretraining
Run pretraining on GPU (or dry-run on CPU):
```bash
# Dry-run verification (5 steps on CPU):
python scripts/run_pretrain.py --dry_run

# Full pretraining run on Google Colab GPU:
python scripts/run_pretrain.py --data_dir data_shards

# Resume from an existing checkpoint:
python scripts/run_pretrain.py --resume checkpoints/checkpoint_step_000100.pt
```

### 6. Interactive Generation
Chat with a trained checkpoint:
```bash
python scripts/run_chat.py --checkpoint checkpoints/best_tinygpt_500m.pt
```
Or generate for a single prompt:
```bash
python scripts/run_chat.py --checkpoint checkpoints/best_tinygpt_500m.pt --prompt "The theory of general relativity states that"
```

### 7. Run SFT Instruction Tuning
Fine-tune on conversational instruction datasets:
```bash
python scripts/run_sft.py --base_model checkpoints/best_tinygpt_500m.pt
```

### 8. Export to Hugging Face & Ollama
Export to Hugging Face `transformers` format (`config.json` + `model.safetensors`):
```bash
python export/export_hf.py --checkpoint checkpoints/best_tinygpt_500m.pt --output_dir exported_hf_model
```
View GGUF conversion instructions for Ollama:
```bash
python export/export_gguf.py
```
