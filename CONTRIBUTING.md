# Contributing to TinyGPT & Usaid AI

Thank you for your interest in contributing to **TinyGPT** and **Usaid AI**! We welcome contributions from researchers, machine learning engineers, and open-source enthusiasts.

---

## 🌟 How You Can Contribute

1. **Bug Reports & Issues:** Identify bugs in tensor implementations, tokenizer handling, or distributed training setups.
2. **Architecture Improvements:** Experiment with attention optimizations (FlashAttention, SDPA variants), alternative activations, or tokenizer extensions.
3. **Training & Fine-Tuning:** Share results from training runs, SFT instruction datasets, or downstream domain fine-tuning.
4. **Documentation & Benchmarks:** Improve documentation, add zero-shot evaluation benchmarks, or provide conversion utilities (ONNX, llama.cpp, vLLM).

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/mohamedusaid/TinyGPT.git
   cd TinyGPT
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Unit & Dry-Run Tests:**
   ```bash
   python tests/dry_run_train.py
   ```

---

## 🐛 Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When opening an issue, use our **Bug Report Template** and provide:
- A clear, descriptive title.
- Step-by-step instructions to reproduce the behavior.
- Expected behavior vs. actual behavior (including full stack traces).
- Your hardware environment (CPU/GPU, OS, Python version, PyTorch version).

---

## 💡 Proposing Architectural Features & Enhancements

Have an idea for architectural scaling, positional encoding variants, or dataset curation?
- Open an issue using the **Feature Request Template**.
- Describe the motivation, theoretical grounding, and proposed implementation details.

---

## 🔄 Submitting a Pull Request (PR)

1. **Branch Naming:** Create a feature branch from `scaling-up-500m`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Code Style:**
   - Adhere to PEP 8 guidelines.
   - Maintain clear docstrings and comments on mathematical tensor transformations.
3. **Verification:**
   - Ensure all model configurations (`configs/`) instantiate cleanly without tensor shape mismatches.
4. **Commit Messages:** Follow standard conventional commit formats (`feat: ...`, `fix: ...`, `docs: ...`).
5. **Open PR:** Submit your pull request targeting `scaling-up-500m` with a detailed description of changes.

---

## 📜 Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.
