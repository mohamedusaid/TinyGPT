---
title: UsaidAI 500M Chat
emoji: ✨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Interactive chat demo for Usaid AI (500M) causal lm
---

# ✨ Usaid AI (500M) — Interactive Chat & Code Assistant

Welcome to the live interactive Hugging Face Space for **Usaid AI (500M)**, created and engineered by **Mohamed Usaid**.

---

### Model Architecture & Specs
* **Parameter Count:** 500,136,960 parameters (~500M)
* **Architecture:** Decoder-Only Transformer (30 Layers, 1024 Dim, 16 Q / 4 KV Heads - Grouped-Query Attention 4:1)
* **Non-Linearity:** SwiGLU Feed-Forward Networks (Intermediate Dim: 3,456)
* **Positional Encoding:** Vectorized Rotary Position Embeddings (RoPE)
* **Normalization:** Root Mean Square Normalization (RMSNorm)
* **Context Capacity:** 1,024 tokens (capable up to 2,048)
* **Pretraining Corpus:** 131M high-quality educational tokens (Cosmopedia-v2, FineWeb-Edu, Python-Edu, C/C++, Java, SmolTalk)
* **Supervised Fine-Tuning (SFT):** 441 steps across 3 epochs (best loss: 1.3938) on 4,731 multi-turn dialogues

---

### Resources & Links
* 📦 **Hugging Face Model Hub:** [Usaidddddddddddddd/UsaidAI-500M](https://huggingface.co/Usaidddddddddddddd/UsaidAI-500M)
* 💻 **GitHub Repository:** [mohamedusaid/TinyGPT](https://github.com/mohamedusaid/TinyGPT)
* 🔒 **Private Checkpoint Archive:** [Usaidddddddddddddd/TinyGPT-500M-Archive](https://huggingface.co/Usaidddddddddddddd/TinyGPT-500M-Archive)
