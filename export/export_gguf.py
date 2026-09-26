"""
GGUF Conversion & Quantization Guide for TinyGPT-500M.
Allows running TinyGPT-500M locally in Ollama, LM Studio, or llama.cpp.
"""

import argparse
import os
import subprocess
import sys


def print_gguf_instructions(hf_model_dir: str, output_gguf_path: str):
    print("=" * 80)
    print("           TINYGPT-500M GGUF & OLLAMA CONVERSION PIPELINE")
    print("=" * 80)
    print("To convert your exported Hugging Face TinyGPT-500M model to GGUF format:")
    print()
    print("1. Clone llama.cpp if not already installed:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   pip install -r llama.cpp/requirements.txt")
    print()
    print("2. Convert Hugging Face directory to FP16 GGUF:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {hf_model_dir} --outfile {output_gguf_path} --outtype f16")
    print()
    print("3. Quantize to 4-bit (Q4_K_M) for high-speed local inference:")
    q4_path = output_gguf_path.replace(".gguf", "_q4_k_m.gguf")
    print(f"   ./llama.cpp/llama-quantize {output_gguf_path} {q4_path} Q4_K_M")
    print()
    print("4. Serve via Ollama:")
    print("   Create a Modelfile:")
    print(f"     FROM ./{q4_path}")
    print("     PARAMETER temperature 0.7")
    print("     PARAMETER top_p 0.9")
    print("   Run:")
    print("     ollama create tinygpt-500m -f Modelfile")
    print("     ollama run tinygpt-500m")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Convert TinyGPT-500M to GGUF")
    parser.add_argument("--hf_model_dir", type=str, default="exported_hf_model")
    parser.add_argument("--output_gguf", type=str, default="tinygpt_500m.gguf")
    args = parser.parse_args()

    print_gguf_instructions(args.hf_model_dir, args.output_gguf)


if __name__ == "__main__":
    main()
