import os
import sys
import threading
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

MODEL_ID = "Usaidddddddddddddd/UsaidAI-500M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading tokenizer and model from {MODEL_ID} on {DEVICE} ({DTYPE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
).to(DEVICE)
model.eval()
print("Model loaded successfully!")


def extract_text(content):
    """Safely extracts plain string text from any Gradio content representation."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text_val = item.get("text", "")
                if text_val:
                    parts.append(str(text_val).strip())
                elif "content" in item:
                    parts.append(str(item["content"]).strip())
            else:
                parts.append(str(item).strip())
        return " ".join([p for p in parts if p]).strip()
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", ""))).strip()
    return str(content).strip() if content is not None else ""


def build_prompt(message, history, system_prompt):
    prompt = ""
    if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
        prompt += f"{system_prompt.strip()}\n\n"

    for turn in history or []:
        role = ""
        raw_content = ""
        if isinstance(turn, dict):
            role = turn.get("role", "")
            raw_content = turn.get("content", "")
            role_str = "User" if str(role).lower() == "user" else "Assistant"
            content_text = extract_text(raw_content)
            if content_text:
                if role_str == "Assistant":
                    prompt += f"Assistant: {content_text}<|endoftext|>\n\n"
                else:
                    prompt += f"User: {content_text}\n\n"
            continue
        elif hasattr(turn, "role") and hasattr(turn, "content"):
            role_str = "User" if str(turn.role).lower() == "user" else "Assistant"
            content_text = extract_text(turn.content)
            if content_text:
                if role_str == "Assistant":
                    prompt += f"Assistant: {content_text}<|endoftext|>\n\n"
                else:
                    prompt += f"User: {content_text}\n\n"
            continue
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            if u:
                u_text = extract_text(u)
                if u_text:
                    prompt += f"User: {u_text}\n\n"
            if a:
                a_text = extract_text(a)
                if a_text:
                    prompt += f"Assistant: {a_text}<|endoftext|>\n\n"
            continue

        role_str = "User" if str(role).lower() == "user" else "Assistant"
        content_text = extract_text(raw_content)
        if content_text:
            if role_str == "Assistant":
                prompt += f"Assistant: {content_text}<|endoftext|>\n\n"
            else:
                prompt += f"User: {content_text}\n\n"

    user_msg = extract_text(message)
    prompt += f"User: {user_msg}\n\nAssistant:"
    return prompt


@gpu_decorator
def predict(
    message,
    history: list,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
):
    user_msg = extract_text(message)
    if not user_msg:
        yield ""
        return

    full_prompt = build_prompt(user_msg, history, system_prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0.0,
        temperature=float(max(temperature, 0.01)),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    accumulated = ""
    for token in streamer:
        accumulated += token
        if "\nUser:" in accumulated:
            accumulated = accumulated.split("\nUser:")[0]
            yield accumulated.strip()
            break
        yield accumulated.strip()

    thread.join()


DESCRIPTION_HTML = """
<div style="text-align: center; max-width: 800px; margin: 0 auto 1.5rem auto;">
    <p style="font-size: 1.05rem; color: #4b5563;">
        Architected, Pretrained & SFT Aligned from Scratch by <strong>Mohamed Usaid</strong>
    </p>
    <div style="display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
        <a href="https://huggingface.co/Usaidddddddddddddd/UsaidAI-500M" target="_blank">
            <img src="https://img.shields.io/badge/HF_Model-UsaidAI--500M-blue?logo=huggingface" alt="Hugging Face">
        </a>
        <a href="https://github.com/mohamedusaid/TinyGPT" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-TinyGPT-black?logo=github" alt="GitHub">
        </a>
        <img src="https://img.shields.io/badge/Parameters-500.14M-green" alt="500M">
        <img src="https://img.shields.io/badge/Architecture-GQA_·_RoPE_·_SwiGLU-purple" alt="Arch">
        <img src="https://img.shields.io/badge/Hardware-NVIDIA_A10G_ZeroGPU-orange" alt="Hardware">
    </div>
</div>
"""

additional_inputs = [
    gr.Textbox(
        value="",
        label="System Prompt (Optional)",
        placeholder="Leave blank to use model's native identity. SFT model was trained with direct User/Assistant prompts.",
        lines=2,
    ),
    gr.Slider(
        minimum=0.0,
        maximum=1.5,
        value=0.6,
        step=0.05,
        label="Temperature",
        info="Default 0.6 matches SFT training evaluation. Lower values are more deterministic.",
    ),
    gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.9,
        step=0.05,
        label="Top-P (Nucleus Sampling)",
    ),
    gr.Slider(
        minimum=32,
        maximum=512,
        value=256,
        step=32,
        label="Max New Tokens",
    ),
    gr.Slider(
        minimum=1.0,
        maximum=1.5,
        value=1.15,
        step=0.05,
        label="Repetition Penalty",
    ),
]

demo = gr.ChatInterface(
    fn=predict,
    additional_inputs=additional_inputs,
    title="✨ Usaid AI (500M) — Interactive Chat & Code Assistant",
    description=DESCRIPTION_HTML,
    examples=[
        ["Who are you?"],
        ["Who created you and what is your architecture?"],
        ["Write a Python function to compute the Fibonacci sequence with memoization."],
        ["Write a C function to reverse an integer array in-place."],
        ["Explain the difference between Multi-Head Attention and Grouped-Query Attention."],
    ],
)

if __name__ == "__main__":
    demo.launch(show_error=True)
