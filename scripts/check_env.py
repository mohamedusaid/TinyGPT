"""
Hardware and Environment Probe for TinyGPT-500M
Detects GPU, VRAM, CUDA version, precision support (FP16/BF16), and SDPA backends.
"""

import sys
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check_environment():
    print("=" * 80)
    print("           TINYGPT-500M HARDWARE & ENVIRONMENT AUDIT")
    print("=" * 80)

    # 1. Python & PyTorch version
    print(f"Python Version:          {sys.version.split()[0]}")
    print(f"PyTorch Version:         {torch.__version__}")

    # 2. CUDA Device Detection
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:          {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = props.total_memory / (1024 ** 3)

        print(f"Device Count:            {device_count}")
        print(f"Primary GPU:             {device_name}")
        print(f"Compute Capability:      {capability[0]}.{capability[1]}")
        print(f"Total VRAM:              {vram_total_gb:.2f} GiB ({props.total_memory / (1024 ** 2):.0f} MiB)")

        # 3. Precision Support
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"Native BF16 Support:     {bf16_supported}")
        if bf16_supported:
            recommended_dtype = "torch.bfloat16 (Ampere/Hopper native)"
            use_scaler = False
        else:
            recommended_dtype = "torch.float16 with GradScaler (Turing T4 / Volta)"
            use_scaler = True
        print(f"Recommended Precision:   {recommended_dtype}")
        print(f"GradScaler Required:     {use_scaler}")
    else:
        print("Running Mode:            CPU Mode (Development & Unit Testing)")
        print("Recommended Precision:   torch.float32")
        use_scaler = False

    # 4. SDPA Backend Verification
    print("-" * 80)
    print("SDPA (Scaled Dot-Product Attention) Backend Probe:")
    try:
        # Test SDPA execution with a micro tensor
        q = torch.randn(1, 4, 16, 64)
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        if cuda_available:
            q, k, v = q.cuda(), k.cuda(), v.cuda()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("  [OK] PyTorch SDPA is fully functional and responsive.")
    except Exception as e:
        print(f"  [FAIL] SDPA Probe failed: {e}")

    # 5. Tokenizer Check
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print(f"  [OK] tiktoken GPT-2 Tokenizer loaded successfully (Vocab: {enc.n_vocab:,})")
    except ImportError:
        print("  [FAIL] tiktoken is not installed. Please install via: pip install tiktoken")

    print("=" * 80)


if __name__ == "__main__":
    check_environment()
