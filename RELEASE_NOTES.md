# ✨ Usaid AI (500M) — Experimental Release (v1.0.0)

An independent, 500M parameter causal language model architected, pretrained, and SFT aligned completely from scratch in PyTorch by **Mohamed Usaid**.

---

### 🚀 Key Highlights & Ecosystem Links
- 🤗 **Hugging Face Model Card:** [Usaidddddddddddddd/UsaidAI-500M](https://huggingface.co/Usaidddddddddddddd/UsaidAI-500M) (SafeTensors weights)
- 💬 **Interactive Live Chat Space:** [Usaidddddddddddddd/UsaidAI-500M-Chat](https://huggingface.co/spaces/Usaidddddddddddddd/UsaidAI-500M-Chat)
- 🧩 **Community GGUF Quantizations (llama.cpp):** [mradermacher/UsaidAI-500M-GGUF](https://huggingface.co/mradermacher/UsaidAI-500M-GGUF)
- 📊 **Telemetry & Experiment Tracker:** Integrated with [`gradio-app/trackio`](https://github.com/gradio-app/trackio)

---

### 📐 Architectural Specifications
| Hyperparameter / Feature | Specification |
| :--- | :--- |
| **Total Trainable Parameters** | **500,136,960** (100% dense, non-MoE) |
| **Transformer Layers (L)** | **30 layers** |
| **Hidden Dimension ($d_{\text{model}}$)** | **1,024** |
| **Query Attention Heads ($H_q$)** | **16 heads** ($d_{\text{head}} = 64$) |
| **Key/Value Attention Heads ($H_{kv}$)** | **4 heads** (**Grouped-Query Attention 4:1**) |
| **KV Cache Compression** | **75% memory reduction** |
| **Feed-Forward Network (FFN)** | **SwiGLU** ($d_{\text{ff}} = 3,456$) |
| **Positional Encoding** | **RoPE** (Rotary Position Embeddings) |
| **Layer Normalization** | **RMSNorm** (Pre-LN, bias-free) |
| **Vocabulary Size ($V$)** | **50,257** (BPE / GPT-2 standard) |
| **Context Length ($T$)** | **1,024 tokens** |

---

### 🔬 Empirical Training & Alignment Milestones
1. **Foundational Pretraining (Dual Cloud GPUs - 2x NVIDIA Tesla T4):**
   - Ingested **131.07 Million tokens** across 2,000 DDP steps.
   - Batch Geometry: Micro-batch 2 × 16 grad accum × 1,024 seq len × 2 GPUs = **65,536 tokens / step**.
   - Final Validation Loss: **`2.9091`** (Validation Perplexity: **`18.34`**).
   - Artifact: `checkpoints/best_tinygpt_500m.pt` (1.86 GB).

2. **Supervised Fine-Tuning (SFT Alignment):**
   - Aligned across **4,731 instruction pairs** (5.62 MB) with strict Prompt Loss Masking (`target = -100` on user tokens).
   - Convergence: SFT loss dropped monotonically from `2.9302` down to all-time best **`1.3938`** (>52% drop).
   - Artifact: `sft_checkpoints/usaid_ai_500m.pt`.

---

### 📊 Exact SFT Dataset Audit & Source Matrix
| Dataset / Source | Site / URL | Split | Exact Row Range | HF Pages | Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolTalk: Everyday Conversations** | [Hugging Face SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/everyday-conversations/train) | `train` | Row 1 to Row 2,260 (100% split) | Pages 1–23 | 2,260 |
| **SmolTalk: OpenHermes-100k** | [Hugging Face OpenHermes-100k](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/openhermes-100k/train) | `train` | Row 1 to Row 2,260 (out of 95k) | Pages 1–23 | 2,260 |
| **Curated Identity & Creator Persona** | `scripts/download_sft_data.py` | Custom | 34 distinct questions (4x repeats) | N/A | 136 |
| **Curated Multi-Language Code** | `scripts/download_sft_data.py` | Custom | 5 distinct questions (15x repeats) | N/A | 75 |
| **TOTAL SFT TRAINING SET** | — | — | — | — | **4,731** |

---

### 🤝 Research Exploration & Collaboration
All training telemetry, model weights, and tokenizer configs are preserved for reproducible research. If you are an AI researcher, student, or engineer interested in exploring intermediate checkpoints (`checkpoint_step_002000.pt`, `best_tinygpt_500m.pt`), analyzing pre-tokenized binary data shards, or collaborating on future scaling and architectural experiments, feel free to open an issue or connect on Hugging Face!
