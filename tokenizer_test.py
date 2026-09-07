"""Tokenizer test script for tiny-gpt."""

import tiktoken

# Load the GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Read our story
with open("content.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the story
tokens = tokenizer.encode(text)

print("Total characters:", len(text))
print("Total tokens:", len(tokens))
print()

print("First 30 tokens:")
print("-" * 50)

for i, token_id in enumerate(tokens[:30]):
    token_text = tokenizer.decode([token_id])
    print(f"T{i + 1:2d} | ID: {token_id:5d} | {token_text!r}")