"""
Fetch text from Hugging Face stanfordnlp/imdb dataset and save to content.txt.
"""

import sys
import requests
import tiktoken

TARGET_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
OUTPUT_FILE = "content.txt"

print(f"Fetching IMDb reviews targeting ~{TARGET_WORDS:,} words...")

enc = tiktoken.get_encoding("gpt2")
all_reviews = []
total_words = 0
offset = 0

while total_words < TARGET_WORDS:
    url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset=stanfordnlp%2Fimdb&config=plain_text&split=train&offset={offset}&limit=100"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching offset {offset}: {response.status_code}")
        break

    data = response.json()
    rows = data.get("rows", [])
    if not rows:
        break

    for item in rows:
        text = item["row"]["text"]
        # Clean HTML line breaks into standard newlines
        clean_text = text.replace("<br />", "\n").replace("<br/>", "\n").strip()
        words = len(clean_text.split())
        all_reviews.append(clean_text)
        total_words += words

        if total_words >= TARGET_WORDS:
            break

    offset += 100

combined_text = "\n\n".join(all_reviews)
total_words_final = len(combined_text.split())
total_tokens_final = len(enc.encode(combined_text))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"Saved {len(all_reviews)} reviews to '{OUTPUT_FILE}'.")
print(f"Total words: {total_words_final:,}")
print(f"Total GPT-2 tokens: {total_tokens_final:,}")
