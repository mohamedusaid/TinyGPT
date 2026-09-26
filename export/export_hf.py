"""
Hugging Face Exporter for TinyGPT-500M.
Converts custom .pt checkpoints into standard Hugging Face Transformers format
with safetensors weights (compatible with AutoModelForCausalLM and AutoTokenizer).
"""

import argparse
import json
import os
import sys
import torch

try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def convert_state_dict_to_hf(state_dict: dict) -> dict:
    """
    Maps TinyGPT-500M internal parameter names to Hugging Face Llama/Qwen architecture format.
    """
    hf_state_dict = {}

    for name, param in state_dict.items():
        hf_name = name

        # Embeddings
        if name == "embed_tokens.weight":
            hf_name = "model.embed_tokens.weight"
        elif name == "norm.weight":
            hf_name = "model.norm.weight"
        elif name == "lm_head.weight":
            hf_name = "lm_head.weight"
        elif name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            submodule = parts[2]
            tail = ".".join(parts[3:])

            if submodule == "input_layernorm":
                hf_name = f"model.layers.{layer_idx}.input_layernorm.{tail}"
            elif submodule == "post_attention_layernorm":
                hf_name = f"model.layers.{layer_idx}.post_attention_layernorm.{tail}"
            elif submodule == "self_attn":
                proj = parts[3]
                proj_map = {
                    "w_q": "q_proj",
                    "w_k": "k_proj",
                    "w_v": "v_proj",
                    "w_o": "o_proj",
                }
                hf_proj = proj_map.get(proj, proj)
                hf_name = f"model.layers.{layer_idx}.self_attn.{hf_proj}.weight"
            elif submodule == "mlp":
                proj = parts[3]
                proj_map = {
                    "w_gate": "gate_proj",
                    "w_up": "up_proj",
                    "w_down": "down_proj",
                }
                hf_proj = proj_map.get(proj, proj)
                hf_name = f"model.layers.{layer_idx}.mlp.{hf_proj}.weight"

        # Ensure contiguous half-precision tensor for export
        hf_state_dict[hf_name] = param.contiguous().half()

    return hf_state_dict


def export_to_huggingface(checkpoint_path: str, output_dir: str):
    """
    Exports a checkpoint to Hugging Face format.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading checkpoint from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_state = checkpoint["model_state_dict"]
    config_dict = checkpoint.get("config", {})

    print("Converting state dict to Hugging Face format...")
    hf_state_dict = convert_state_dict_to_hf(model_state)

    # Build Hugging Face config.json
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config_dict.get("vocab_size", 50257),
        "hidden_size": config_dict.get("hidden_size", 1024),
        "intermediate_size": config_dict.get("intermediate_size", 3456),
        "num_hidden_layers": config_dict.get("num_hidden_layers", 30),
        "num_attention_heads": config_dict.get("num_attention_heads", 16),
        "num_key_value_heads": config_dict.get("num_key_value_heads", 4),
        "max_position_embeddings": config_dict.get("max_position_embeddings", 2048),
        "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-5),
        "rope_theta": config_dict.get("rope_theta", 10000.0),
        "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
        "torch_dtype": "float16",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Saved config.json to: {config_path}")

    # Generation config
    generation_config = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "max_length": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)

    # Save weights
    if SAFETENSORS_AVAILABLE:
        weights_path = os.path.join(output_dir, "model.safetensors")
        save_safetensors(hf_state_dict, weights_path)
        print(f"Saved safetensors weights to: {weights_path}")
    else:
        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(hf_state_dict, weights_path)
        print(f"Saved pytorch weights to: {weights_path}")

    print(f"\nExport complete! Hugging Face model directory ready at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export TinyGPT-500M to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_tinygpt_500m.pt")
    parser.add_argument("--output_dir", type=str, default="exported_hf_model")
    args = parser.parse_args()

    export_to_huggingface(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
